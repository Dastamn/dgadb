import math
import logging
import time
import torch
import numpy as np
from .SLADE_TGN import SLADE_TGN
from .utils.utils import get_neighbor_finder
from .utils.data_processing import SLADEData
from .evaluation.evaluation import eval_anomaly_node_detection
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SLADEModel:
    def __init__(self, device, hyperparams, epoch_evaluation_metric):

        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric

        logger.info(
            f"Initializing SLADEModel with device={self.device} and hyperparams={hyperparams}")

        self.batch_size = hyperparams.get("batch_size", 100)
        self.num_neighbors = hyperparams.get("num_neighbors", 20)
        self.num_epoch = hyperparams.get("num_epoch", 10)
        self.num_heads = hyperparams.get("num_heads", 2)
        self.drop_out = hyperparams.get("drop_out", 0.1)
        self.gpu = hyperparams.get("gpu", 0)
        self.num_layer = hyperparams.get("num_layer", 1)
        self.learning_rate = hyperparams.get("learning_rate", 3e-6)
        self.message_dim = hyperparams.get("message_dim", 128)
        self.memory_dim = hyperparams.get("memory_dim", 256)
        self.lr_decay = hyperparams.get("lr_decay", 0.8)
        self.weight_decay = hyperparams.get("weight_decay", 0.0001)
        self.memory_agg_type = hyperparams.get("memory_agg_type", 'TGAT')
        self.negative_memory_type = hyperparams.get(
            "negative_memory_type", 'train')
        self.message_updater = hyperparams.get("message_updater", 'mlp')
        self.memory_updater = hyperparams.get("memory_updater", 'gru')
        self.srf = hyperparams.get("srf", 0.1)
        self.drf = hyperparams.get("drf", 0.1)
        self.only_drift_loss_score = hyperparams.get(
            "only_drift_loss_score", False)
        self.only_recovery_loss_score = hyperparams.get(
            "only_recovery_loss_score", False)
        self.only_drift_score = hyperparams.get("only_drift_score", False)
        self.only_rec_score = hyperparams.get("only_rec_score", False)
        self.test_inference_time = hyperparams.get(
            "test_inference_time", False)
        self.n_runs = hyperparams.get("n_runs", 1)
        self.seed = hyperparams.get("seed", 0)

    def setup(self, df: pd.DataFrame):

        sources = df["src"].values
        destinations = df["tgt"].values
        edge_idxs = df["edge_id"].values
        labels = df["label"].values
        timestamps = df["timestamp"].values

        train_mask = df["train_mask"].values
        test_mask = df["test_mask"].values
        self.has_val = "val_mask" in df.columns
        if self.has_val:
            val_mask = df["val_mask"].values

        self.full_data = SLADEData(
            sources, destinations, timestamps, edge_idxs, labels)
        self.train_data = SLADEData(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                                    edge_idxs[train_mask], labels[train_mask])

        self.test_data = SLADEData(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                                   edge_idxs[test_mask], labels[test_mask])

        if self.has_val:
            self.val_data = SLADEData(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                                      edge_idxs[val_mask], labels[val_mask])
            train_val_mask = train_mask | val_mask
            self.train_val_data = SLADEData(sources[train_val_mask], destinations[train_val_mask], timestamps[train_val_mask],
                                            edge_idxs[train_val_mask], labels[train_val_mask])

        max_idx = max(self.full_data.unique_nodes)

        self.train_ngh_finder = get_neighbor_finder(
            self.train_data, uniform=False, max_node_idx=max_idx)
        if self.has_val:
            self.val_ngh_finder = get_neighbor_finder(
                self.train_val_data, uniform=False, max_node_idx=max_idx)
            # should be safe based on how neighbor finder is implemented
        self.full_ngh_finder = get_neighbor_finder(
            self.full_data, uniform=False, max_node_idx=max_idx)

        self.src_neighbors, _, self.src_neighbors_time = self.train_ngh_finder.get_temporal_neighbor_tqdm(
            self.train_data.sources, self.train_data.timestamps, self.num_neighbors)
        self.dst_neighbors, _, self.dst_neighbors_time = self.train_ngh_finder.get_temporal_neighbor_tqdm(
            self.train_data.destinations, self.train_data.timestamps, self.num_neighbors)

        self.dcl_tgn = SLADE_TGN(neighbor_finder=self.train_ngh_finder, n_nodes=self.full_data.n_unique_nodes, n_edges=self.full_data.n_interactions,
                                 device=self.device, n_layers=self.num_layer, n_heads=self.num_heads,
                                 dropout=self.drop_out, message_dimension=self.message_dim, memory_dimension=self.memory_dim, n_neighbors=self.num_neighbors,
                                 memory_agg_type=self.memory_agg_type, negative_memory_type=self.negative_memory_type, message_updater=self.message_updater,
                                 memory_updater=self.memory_updater, src_reg_factor=self.srf, dst_reg_factor=self.drf,
                                 only_drift_loss=self.only_drift_loss_score, only_recovery_loss=self.only_recovery_loss_score)

        self.dcl_tgn = self.dcl_tgn.to(self.device)

        self.train_data_sources = torch.from_numpy(
            self.train_data.sources).long().to(self.device)
        self.train_data_destinations = torch.from_numpy(
            self.train_data.destinations).long().to(self.device)
        self.train_data_timestamps = torch.from_numpy(
            self.train_data.timestamps).float().to(self.device)
        self.train_data_src_neighbors = torch.from_numpy(
            self.src_neighbors).long().to(self.device)
        self.train_data_dst_neighbors = torch.from_numpy(
            self.dst_neighbors).long().to(self.device)
        self.train_data_src_neighbors_time = torch.from_numpy(
            self.src_neighbors_time).long().to(self.device)
        self.train_data_dst_neighbors_time = torch.from_numpy(
            self.dst_neighbors_time).long().to(self.device)

        self.num_instance = len(self.train_data.sources)
        self.num_batch = math.ceil(self.num_instance / self.batch_size)

        self.optimizer = torch.optim.Adam(self.dcl_tgn.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.lr_decay)

        self.negative_train_nodes = torch.from_numpy(np.array(list(set(
            self.train_data.destinations) | set(self.train_data.sources)))).long().to(self.device)

    def train(self):
        logger.info(f"Starting training for {self.num_epoch} epochs...")
        for epoch in range(self.num_epoch):
            self.dcl_tgn.memory.__init_memory__()
            self.dcl_tgn.set_neighbor_finder(self.train_ngh_finder)
            m_loss = []

            train_start = time.time()
            for k in tqdm(range(self.num_batch)):
                loss = 0
                self.optimizer.zero_grad()
                s_idx = k * self.batch_size
                e_idx = min(self.num_instance, s_idx + self.batch_size)

                sources_batch = self.train_data_sources[s_idx: e_idx]
                destinations_batch = self.train_data_destinations[s_idx: e_idx]
                timestamps_batch = self.train_data_timestamps[s_idx: e_idx]
                src_neighbors_batch = self.train_data_src_neighbors[s_idx: e_idx]
                dst_neighbors_batch = self.train_data_dst_neighbors[s_idx: e_idx]
                src_neighbors_time_batch = self.train_data_src_neighbors_time[s_idx: e_idx]
                dst_neighbors_time_batch = self.train_data_dst_neighbors_time[s_idx: e_idx]

                self.dcl_tgn.train()
                _, _, _, _, contrastive_loss = self.dcl_tgn.compute_node_diff_score(sources_batch,
                                                                                    destinations_batch,
                                                                                    timestamps_batch,
                                                                                    src_neighbors_batch,
                                                                                    dst_neighbors_batch,
                                                                                    src_neighbors_time_batch,
                                                                                    dst_neighbors_time_batch,
                                                                                    self.num_neighbors,
                                                                                    self.negative_train_nodes)  # train_data.n_unique_nodes
                loss += contrastive_loss
                if self.only_drift_loss_score and k == 0:
                    continue
                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())
                self.dcl_tgn.memory.detach_memory()

            train_end = time.time()

            self.train_time = train_end - train_start
            logger.info(f"Training finished in {self.train_time:.2f} seconds.")

            self.scheduler.step()

            pred_score, labels, _ = self.inference("val")
            epoch_score = self.epoch_evaluation_metric(labels, pred_score)

            logger.info("Epoch {} - mloss: {:.4f} {} auc: {:.4f}".format(
                str(epoch), sum(m_loss)/len(m_loss), "val", epoch_score))

    def inference(self, split: str):
        inf_start = time.time()

        if split == "train":
            self.dcl_tgn.set_neighbor_finder(self.train_ngh_finder)
            data = self.train_data
        elif split == "val" and not self.has_val:
            logger.warning(
                "Tried to evaluate on validation split, but none was found.")
            raise ValueError("No validation split available.")
        elif split == "val":
            self.dcl_tgn.set_neighbor_finder(self.val_ngh_finder)
            data = self.val_data
        elif split == "test":
            self.dcl_tgn.set_neighbor_finder(self.full_ngh_finder)
            data = self.test_data
        else:
            raise ValueError("No such split available.")

        _, pred_score, labels = eval_anomaly_node_detection(self.dcl_tgn, data, self.batch_size,
                                                            n_neighbors=self.num_neighbors, device=self.device, only_rec_score=self.only_rec_score or self.only_recovery_loss_score,
                                                            only_drift_score=self.only_drift_loss_score or self.only_drift_score, test_inference_time=self.test_inference_time)

        inf_end = time.time()
        inf_time = inf_end-inf_start

        return pred_score, labels, inf_time
