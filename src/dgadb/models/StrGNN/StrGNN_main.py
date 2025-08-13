import torch
import numpy as np
import polars as pl
import sys
import copy
import math
import time
import pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os
import os.path
import random
import argparse
import pickle
from src.dgadb.models.StrGNN.pytorch_DGCNN.main import *
from src.dgadb.models.StrGNN.detection.util_functions import *
from os import path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# TODO: StrGNN works if given a dataset with labels. I have not implemented negative/positive sampling because I believe this should be done similarly for all methods.
# TODO: Get it working with validation sets
class STRGNNModel:
    def __init__(
        self,
        device: torch.device,
        meta_dict: dict,
        hyperparams: dict,
        epoch_evaluation_metric,
    ):
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric

        self.batch_size = hyperparams.get("batch_size", 32)
        self.num_epoch = hyperparams.get("num_epoch", 20)
        self.learning_rate = hyperparams.get("learning_rate", 1e-4)
        self.hop = hyperparams.get("hop", 1)
        self.window = hyperparams.get("window", 5)
        self.use_embedding = hyperparams.get("use_embedding", True)
        self.use_attribute = hyperparams.get("use_attribute", False)
        self.classifier = None
        self.optimizer = None
        self.data_dict = {}

        self.latent_dim = hyperparams.get("latent_dim", [32])
        self.out_dim = hyperparams.get("out_dim", 0)
        self.edge_feat_dim = hyperparams.get("edge_feat_dim", 0)
        self.sortpooling_k = hyperparams.get("sortpooling_k", 30)  # was 0.6 in og
        self.conv1d_activation = hyperparams.get("conv1d_activation", "relu")
        self.hidden = hyperparams.get("hidden", 50)
        self.num_class = hyperparams.get("num_class", 2)
        self.dropout = hyperparams.get("dropout", 0.5)
        self.regression = hyperparams.get("regression", False)
        self.gm = hyperparams.get("gm", "DGCNN")
        self.window_size = hyperparams.get("window_size", 5)

    def setup(self, df: pl.DataFrame) -> None:
        logger.info("STRGNN setup started...")

        # Node mapping
        all_nodes = np.unique(np.concatenate([df["src"].to_numpy(), df["tgt"].to_numpy()]))
        num_nodes = len(all_nodes)

        # Split edges into train/test/val
        train_df = df.filter(pl.col("train_mask"))
        test_df = df.filter(pl.col("test_mask"))
        val_df = df.filter(pl.col("val_mask")) if "val_mask" in df.columns else None

        train_labels = train_df["label"].to_numpy()
        test_labels = test_df["label"].to_numpy()
        val_labels = val_df["label"].to_numpy() if val_df is not None else None

        def get_edge_indices(sub_df):
            src_idx = sub_df["src"].to_numpy()
            tgt_idx = sub_df["tgt"].to_numpy()
            snap_idx = sub_df["snapshot_id"].to_numpy()
            return src_idx, tgt_idx, snap_idx

        train_src, train_tgt, train_snap = get_edge_indices(train_df)
        test_src, test_tgt, test_snap = get_edge_indices(test_df)

        # Get unique snapshots
        snapshot_ids = df["snapshot_id"].unique().sort().to_numpy()
        num_snapshots = len(snapshot_ids)
        num_train_snapshots = len(df.filter(pl.col("train_mask"))["snapshot_id"].unique())
        net = []
        for snap_id in snapshot_ids:
            # Get edges for this snapshot
            snap_edges = df.filter(pl.col("snapshot_id") == snap_id)

            if len(snap_edges) > 0:
                # Map node IDs for this snapshot
                snap_row = snap_edges["src"].to_numpy()
                snap_col = snap_edges["tgt"].to_numpy()
                snap_data = np.ones_like(snap_row, dtype=np.int32)

                # Create adjacency matrix for this snapshot
                A_snap = ssp.csr_matrix((snap_data, (snap_row, snap_col)), shape=(num_nodes, num_nodes))
                A_snap = A_snap + A_snap.transpose()  # symmetric
                A_snap.setdiag(0)  # remove self-loops
            else:
                # Empty snapshot
                A_snap = ssp.csr_matrix((num_nodes, num_nodes))

            net.append(A_snap)

        # Optional val split
        if val_df is not None:
            val_src, val_tgt, val_snap = get_edge_indices(val_df)
            val_labels = val_df["label"].to_numpy()

        A = net[0]  # only get embeddings from training graph

        # Node2Vec embeddings (optional)
        if self.use_embedding:
            embeddings = generate_node2vec_embeddings(A, 128)
            self.data_dict["embeddings"] = embeddings
        else:
            embeddings = None

        self.data_dict["train_pos"] = (train_src[train_labels == 1], train_tgt[train_labels == 1])
        self.data_dict["train_neg"] = (train_src[train_labels == 0], train_tgt[train_labels == 0])
        self.data_dict["train_pos_id"] = train_snap[train_labels == 1]
        self.data_dict["train_neg_id"] = train_snap[train_labels == 0]

        self.data_dict["test_pos"] = (test_src[test_labels == 1], test_tgt[test_labels == 1])
        self.data_dict["test_neg"] = (test_src[test_labels == 0], test_tgt[test_labels == 0])
        self.data_dict["test_pos_id"] = test_snap[test_labels == 1]
        self.data_dict["test_neg_id"] = test_snap[test_labels == 0]

        # Subgraph extraction
        logger.info("Extracting subgraphs for STRGNN...")
        train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(
            net,
            self.window_size,
            self.data_dict["train_pos_id"],
            self.data_dict["train_pos"],
            self.data_dict["train_neg_id"],
            self.data_dict["train_neg"],
            self.data_dict["test_pos_id"],
            self.data_dict["test_pos"],
            self.data_dict["test_neg_id"],
            self.data_dict["test_neg"],
            h=self.hop,
            max_nodes_per_hop=None,
            node_information=self.data_dict.get("embeddings")
            if self.data_dict.get("embeddings") is not None
            else self.data_dict.get("attributes"),
        )
        self.data_dict.update(
            {
                "train_graphs": train_graphs,
                "test_graphs": test_graphs,
                "max_n_label": max_n_label,
            }
        )

        self.feat_dim = max_n_label + 1  # Structural labels dimension
        self.attr_dim = 0
        if embeddings is not None:
            self.attr_dim = embeddings.shape[1]  # node2vec dimension (128)

        print(f"DEBUG: feat_dim={self.feat_dim}, attr_dim={self.attr_dim}, total={self.feat_dim + self.attr_dim}")

        # DGCNN model setup
        self.classifier = Classifier(
            gm=self.gm,
            latent_dim=self.latent_dim,
            out_dim=self.out_dim,
            feat_dim=self.feat_dim,
            attr_dim=self.attr_dim,
            edge_feat_dim=self.edge_feat_dim,
            sortpooling_k=self.sortpooling_k,
            conv1d_activation=self.conv1d_activation,
            hidden=self.hidden,
            num_class=self.num_class,
            dropout=self.dropout,
            mode="gpu" if self.device == "cuda" else "cpu",
            regression=self.regression,
        )
        if self.device == "cuda":
            self.classifier = self.classifier.cuda()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def train(self) -> None:
        logger.info(f"Starting STRGNN training for {self.num_epoch} epochs...")
        train_graphs = self.data_dict["train_graphs"]
        test_graphs = self.data_dict["test_graphs"]
        train_idxes = list(range(len(train_graphs)))
        for epoch in range(self.num_epoch):
            np.random.shuffle(train_idxes)
            self.classifier.train()
            avg_loss, labels, preds = loop_dataset(
                train_graphs, self.classifier, train_idxes, optimizer=self.optimizer, bsize=self.batch_size
            )
            score = self.epoch_evaluation_metric(labels, preds)
            logger.info(f"Epoch {epoch}: train loss={avg_loss[0]:.5f}")
            if (epoch + 1) % 5 == 0:
                logger.info(f"Total AUC: {score:.5f}")

    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, float]:
        start_time = time.time()
        self.classifier.eval()
        if split == "train":
            graphs = self.data_dict["train_graphs"]
        else:
            graphs = self.data_dict["test_graphs"]
        avg_loss, labels, preds = loop_dataset(graphs, self.classifier, list(range(len(graphs))), bsize=self.batch_size)
        inf_time = time.time() - start_time
        return preds, labels, inf_time
