from dgadb.storage import TemporalGraph
import logging
import math
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Any

from dgadb.models.SLADE.evaluation.evaluation import eval_anomaly_node_detection
from dgadb.models.SLADE.SLADE_TGN import SLADE_TGN
from dgadb.models.SLADE.utils.data_processing import SLADEData
from dgadb.models.SLADE.utils.utils import NeighborFinder, get_neighbor_finder

logger = logging.getLogger(__name__)


class SLADEModel:
    """Wrap SLADE model.

    This class provides a standardized interface for the SLADE temporal gnn model.

    Attributes:
        device (torch.device): Device to run computations on.
        epoch_evaluation_metric (callable): Function to evaluate model performance in epochs.
        dcl_tgn (SLADE_TGN): The underlying SLADE model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        has_val (bool): Whether validation data is available.

    Args:
        device (torch.device): Device to run computations on.
        hyperparams (dict): Dictionary containing model hyperparameters including
            batch_size, num_neighbors, num_epoch, learning_rate, memory settings, etc.
        epoch_evaluation_metric (callable): Function that takes (labels, predictions)
            and returns a scalar evaluation metric.

    """

    def __init__(
        self,
        device: torch.device,
        meta_dict,
        hyperparams: dict[str, Any],
        epoch_evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialize SLADEModel with device, hyperparameters, and evaluation metric.

        Sets up the model configuration and stores hyperparameters learning.

        Args:
            device (torch.device): Device to run computations on.
            hyperparams (dict): Dictionary containing model hyperparameters such as
                batch_size, num_neighbors, learning_rate, memory_dim, etc.
            epoch_evaluation_metric (callable): Function that takes (labels, predictions)
                and returns a scalar metric for evaluation during training.

        """
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
        self.memory_agg_type = hyperparams.get("memory_agg_type", "TGAT")
        self.negative_memory_type = hyperparams.get(
            "negative_memory_type", "train")
        self.message_updater = hyperparams.get("message_updater", "mlp")
        self.memory_updater = hyperparams.get("memory_updater", "gru")
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

        # Initialize attributes for other methods
        self.has_val: bool | None = None
        self.full_data: SLADEData | None = None
        self.train_data: SLADEData | None = None
        self.test_data: SLADEData | None = None
        self.val_data: SLADEData | None = None
        self.train_val_data: SLADEData | None = None

        # Neighbor finders
        self.train_ngh_finder: NeighborFinder | None = None
        self.val_ngh_finder: NeighborFinder | None = None
        self.full_ngh_finder: NeighborFinder | None = None

        # Neighbor data
        self.src_neighbors: np.ndarray | None = None
        self.src_neighbors_time: np.ndarray | None = None
        self.dst_neighbors: np.ndarray | None = None
        self.dst_neighbors_time: np.ndarray | None = None

        # Model and training components
        self.dcl_tgn: SLADE_TGN | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # Training data tensors
        self.train_data_sources: torch.Tensor | None = None
        self.train_data_destinations: torch.Tensor | None = None
        self.train_data_timestamps: torch.Tensor | None = None
        self.train_data_src_neighbors: torch.Tensor | None = None
        self.train_data_dst_neighbors: torch.Tensor | None = None
        self.train_data_src_neighbors_time: torch.Tensor | None = None
        self.train_data_dst_neighbors_time: torch.Tensor | None = None

        # Training metadata
        self.num_instance: int | None = None
        self.num_batch: int | None = None
        self.negative_train_nodes: torch.Tensor | None = None

        # Training timing
        self.train_time: float | None = None

    def setup(self, temporal_graph: TemporalGraph):

        sources = temporal_graph.src.numpy()
        destinations = temporal_graph.tgt.numpy()
        edge_idxs = np.arange(len(sources), dtype=np.int64)
        labels = temporal_graph.edge_labels.numpy()
        timestamps = temporal_graph.t.numpy()
        # timestamps = np.sort(temporal_graph.t.numpy())
        # assert np.all(timestamps[:-1] <= timestamps[1:])

        train_mask = temporal_graph.train_mask.numpy()
        test_mask = temporal_graph.test_mask.numpy()
        val_mask = temporal_graph.val_mask.numpy()

        # sources = df["src"].to_numpy()
        # destinations = df["tgt"].to_numpy()
        # edge_idxs = df["edge_id"].to_numpy()
        # labels = df["label"].to_numpy()
        # timestamps = df["timestamp"].to_numpy()

        # train_mask = df["train_mask"].to_numpy()
        # test_mask = df["test_mask"].to_numpy()
        # self.has_val = "val_mask" in df.columns
        # if self.has_val:
        #     val_mask = df["val_mask"].to_numpy()

        self.full_data = SLADEData(
            sources, destinations, timestamps, edge_idxs, labels)
        self.train_data = SLADEData(
            sources[train_mask],
            destinations[train_mask],
            timestamps[train_mask],
            edge_idxs[train_mask],
            labels[train_mask],
        )

        self.test_data = SLADEData(
            sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask]
        )

        if self.has_val:
            self.val_data = SLADEData(
                sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask]
            )
            train_val_mask = train_mask | val_mask
            self.train_val_data = SLADEData(
                sources[train_val_mask],
                destinations[train_val_mask],
                timestamps[train_val_mask],
                edge_idxs[train_val_mask],
                labels[train_val_mask],
            )

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
            self.train_data.sources, self.train_data.timestamps, self.num_neighbors
        )
        self.dst_neighbors, _, self.dst_neighbors_time = self.train_ngh_finder.get_temporal_neighbor_tqdm(
            self.train_data.destinations, self.train_data.timestamps, self.num_neighbors
        )

        self.dcl_tgn = SLADE_TGN(
            neighbor_finder=self.train_ngh_finder,
            n_nodes=self.full_data.n_unique_nodes,
            n_edges=self.full_data.n_interactions,
            device=self.device,
            n_layers=self.num_layer,
            n_heads=self.num_heads,
            dropout=self.drop_out,
            message_dimension=self.message_dim,
            memory_dimension=self.memory_dim,
            n_neighbors=self.num_neighbors,
            memory_agg_type=self.memory_agg_type,
            negative_memory_type=self.negative_memory_type,
            message_updater=self.message_updater,
            memory_updater=self.memory_updater,
            src_reg_factor=self.srf,
            dst_reg_factor=self.drf,
            only_drift_loss=self.only_drift_loss_score,
            only_recovery_loss=self.only_recovery_loss_score,
        )

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

        self.optimizer = torch.optim.Adam(
            self.dcl_tgn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.lr_decay)

        self.negative_train_nodes = (
            torch.from_numpy(np.array(
                list(set(self.train_data.destinations) | set(self.train_data.sources))))
            .long()
            .to(self.device)
        )

    def setup_old(self, df: pd.DataFrame) -> None:
        """Set up data processing and initialize the SLADE model.

        Processes the input DataFrame into SLADE data format, creates neighbor finders
        for different data splits, initializes the SLADE_TGN model, and sets up
        optimizers and schedulers.

        Args:
            df (pd.DataFrame): Input DataFrame containing columns: src, tgt, edge_id,
                label, timestamp, train_mask, test_mask, and optionally val_mask.

        """
        sources = df["src"].to_numpy()
        destinations = df["tgt"].to_numpy()
        edge_idxs = df["edge_id"].to_numpy()
        labels = df["label"].to_numpy()
        timestamps = df["timestamp"].to_numpy()

        train_mask = df["train_mask"].to_numpy()
        test_mask = df["test_mask"].to_numpy()
        self.has_val = "val_mask" in df.columns
        if self.has_val:
            val_mask = df["val_mask"].to_numpy()

        self.full_data = SLADEData(
            sources, destinations, timestamps, edge_idxs, labels)
        self.train_data = SLADEData(
            sources[train_mask],
            destinations[train_mask],
            timestamps[train_mask],
            edge_idxs[train_mask],
            labels[train_mask],
        )

        self.test_data = SLADEData(
            sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask]
        )

        if self.has_val:
            self.val_data = SLADEData(
                sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask]
            )
            train_val_mask = train_mask | val_mask
            self.train_val_data = SLADEData(
                sources[train_val_mask],
                destinations[train_val_mask],
                timestamps[train_val_mask],
                edge_idxs[train_val_mask],
                labels[train_val_mask],
            )

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
            self.train_data.sources, self.train_data.timestamps, self.num_neighbors
        )
        self.dst_neighbors, _, self.dst_neighbors_time = self.train_ngh_finder.get_temporal_neighbor_tqdm(
            self.train_data.destinations, self.train_data.timestamps, self.num_neighbors
        )

        self.dcl_tgn = SLADE_TGN(
            neighbor_finder=self.train_ngh_finder,
            n_nodes=self.full_data.n_unique_nodes,
            n_edges=self.full_data.n_interactions,
            device=self.device,
            n_layers=self.num_layer,
            n_heads=self.num_heads,
            dropout=self.drop_out,
            message_dimension=self.message_dim,
            memory_dimension=self.memory_dim,
            n_neighbors=self.num_neighbors,
            memory_agg_type=self.memory_agg_type,
            negative_memory_type=self.negative_memory_type,
            message_updater=self.message_updater,
            memory_updater=self.memory_updater,
            src_reg_factor=self.srf,
            dst_reg_factor=self.drf,
            only_drift_loss=self.only_drift_loss_score,
            only_recovery_loss=self.only_recovery_loss_score,
        )

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

        self.optimizer = torch.optim.Adam(
            self.dcl_tgn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.lr_decay)

        self.negative_train_nodes = (
            torch.from_numpy(np.array(
                list(set(self.train_data.destinations) | set(self.train_data.sources))))
            .long()
            .to(self.device)
        )

    def _ensure_setup(self) -> None:
        """Ensure setup() has been called before using the model."""
        if any(
            attr is None
            for attr in [
                self.dcl_tgn,
                self.optimizer,
                self.scheduler,
                self.num_batch,
                self.train_data_sources,
                self.train_data_destinations,
                self.train_data_timestamps,
                self.train_data_src_neighbors,
                self.train_data_dst_neighbors,
                self.train_data_src_neighbors_time,
                self.train_data_dst_neighbors_time,
                self.negative_train_nodes,
                self.train_ngh_finder,
            ]
        ):
            raise RuntimeError(
                "Model not properly initialized. Call setup() before train() or inference().")

    def train(self, runnable=None) -> None:
        """Train the SLADE model.

        Performs epoch-based training with mini batches. Computes contrastive loss
        between node embeddings. Updates memory states and evaluates on validation
        set (if available) after each epoch.
        """
        self._ensure_setup()
        logger.info(f"Starting training for {self.num_epoch} epochs...")
        train_start = time.time()
        for epoch in range(self.num_epoch):
            self.dcl_tgn.memory.__init_memory__()
            self.dcl_tgn.set_neighbor_finder(self.train_ngh_finder)
            m_loss = []

            for k in tqdm(range(self.num_batch)):
                loss = 0
                self.optimizer.zero_grad()
                s_idx = k * self.batch_size
                e_idx = min(self.num_instance, s_idx + self.batch_size)

                sources_batch = self.train_data_sources[s_idx:e_idx]
                destinations_batch = self.train_data_destinations[s_idx:e_idx]
                timestamps_batch = self.train_data_timestamps[s_idx:e_idx]
                src_neighbors_batch = self.train_data_src_neighbors[s_idx:e_idx]
                dst_neighbors_batch = self.train_data_dst_neighbors[s_idx:e_idx]
                src_neighbors_time_batch = self.train_data_src_neighbors_time[s_idx:e_idx]
                dst_neighbors_time_batch = self.train_data_dst_neighbors_time[s_idx:e_idx]

                self.dcl_tgn.train()
                _, _, _, _, contrastive_loss = self.dcl_tgn.compute_node_diff_score(
                    sources_batch,
                    destinations_batch,
                    timestamps_batch,
                    src_neighbors_batch,
                    dst_neighbors_batch,
                    src_neighbors_time_batch,
                    dst_neighbors_time_batch,
                    self.num_neighbors,
                    self.negative_train_nodes,
                )  # train_data.n_unique_nodes
                loss += contrastive_loss
                if self.only_drift_loss_score and k == 0:
                    continue
                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())
                self.dcl_tgn.memory.detach_memory()

            self.scheduler.step()
            if self.has_val:
                split = "val"
            else:
                split = "test"

            pred_score, labels = self.inference(split=split)
            epoch_score = self.epoch_evaluation_metric(labels, pred_score)

            if runnable is not None:
                runnable(epoch_score, self, epoch)

            logger.info(
                "Epoch {} - mloss: {:.4f} {} auc: {:.4f}".format(
                    str(epoch), sum(m_loss) / len(m_loss), split, epoch_score
                )
            )

        train_end = time.time()

        self.train_time = train_end - train_start
        logger.info(f"Training finished in {self.train_time:.2f} seconds.")

    def inference(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on the specified data split.

        Evaluates the trained model on train, validation, or test data and returns
        anomaly scores, ground truth labels, and inference timing.

        Args:
            split (str): Data split to evaluate on. Options are "train", "val", "test".

        Returns:
            tuple: A tuple containing:
                - pred_score (np.ndarray): Predicted anomaly scores
                - labels (np.ndarray): Ground truth binary labels
                - inf_time (float): Inference time in seconds

        Raises:
            ValueError: If the specified split is not available.

        """
        self._ensure_setup()
        inf_start = time.time()
        self.dcl_tgn.memory.__init_memory__()  # Memory needs to be cleared!

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

        _, pred_score, labels = eval_anomaly_node_detection(
            self.dcl_tgn,
            data,
            self.batch_size,
            n_neighbors=self.num_neighbors,
            device=self.device,
            only_rec_score=self.only_rec_score or self.only_recovery_loss_score,
            only_drift_score=self.only_drift_loss_score or self.only_drift_score,
            test_inference_time=self.test_inference_time,
        )

        inf_end = time.time()
        inf_time = inf_end - inf_start

        return pred_score, labels
        return pred_score, labels, inf_time


if __name__ == "__main__":
    from dgadb.storage.temporal_graph import TemporalGraphLoaderNew

    loader = TemporalGraphLoaderNew()
    tg = loader.load("bitcoin-alpha", create_if_not_found=True)
    model = SLADEModel("cpu", {}, None)
    model.setup(tg)
    model.train()
