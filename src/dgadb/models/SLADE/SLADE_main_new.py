from tqdm import tqdm
import numpy as np
import torch
from src.dgadb.experiment.callbacks import ExperimentCallback
from src.dgadb.models.base import BaseADModel, BaseADModelComponents, TrainingState
from src.dgadb.storage.temporal_graph import TemporalGraph
from src.dgadb.storage.temporal_snapshot import TemporalGraphSnapshot, TemporalGraphSnapshotLoader
from typing import Any
from dataclasses import dataclass
from .utils.utils import NeighborFinder, get_neighbor_finder
from .SLADE_TGN import SLADE_TGN
from .utils.data_processing import SLADEData

import math
from src.dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler
from sklearn.metrics import roc_auc_score


@dataclass
class SLADEComponents(BaseADModelComponents):
    dcl_tgn: SLADE_TGN
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


class SLADEModelNew(BaseADModel[SLADEComponents]):
    def __init__(self,
                 batch_size=100,
                 num_neighbors=20,
                 num_epoch=10,
                 num_heads=2,
                 drop_out=0.1,
                 num_layer=1,
                 learning_rate=3e-6,
                 message_dim=128,
                 memory_dim=256,
                 lr_decay=0.8,
                 weight_decay=0.0001,
                 memory_agg_type="TGAT",
                 negative_memory_type="train",
                 message_updater="mlp",
                 memory_updater="gru",
                 srf=0.1,
                 drf=0.1,
                 only_drift_loss_score=False,
                 only_recovery_loss_score=False,
                 only_drift_score=False,
                 only_rec_score=False,
                 seed=0,
                 device: torch.device | str = "cpu"
                 ) -> None:
        super().__init__(device)
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_epoch = num_epoch
        self.num_heads = num_heads
        self.drop_out = drop_out
        self.num_layer = num_layer
        self.learning_rate = learning_rate
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.memory_agg_type = memory_agg_type
        self.negative_memory_type = negative_memory_type
        self.message_updater = message_updater
        self.memory_updater = memory_updater
        self.srf = srf
        self.drf = drf
        self.only_drift_loss_score = only_drift_loss_score
        self.only_recovery_loss_score = only_recovery_loss_score
        self.only_drift_score = only_drift_score
        self.only_rec_score = only_rec_score
        self.seed = seed

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
        # self.dcl_tgn: SLADE_TGN | None = None
        # self.optimizer: torch.optim.Optimizer | None = None
        # self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        sources = data.src.numpy()
        destinations = data.tgt.numpy()
        edge_idxs = np.arange(len(sources), dtype=np.int64)
        labels = data.edge_labels.numpy()
        timestamps = data.t.numpy()

        train_mask = data.train_mask.numpy()
        test_mask = data.test_mask.numpy()
        val_mask = data.val_mask.numpy()

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

        if val_mask:
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
        if val_mask:
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

        # removed .to(device)
        self.train_data_sources = torch.from_numpy(
            self.train_data.sources).long()  # .to(self.device)
        self.train_data_destinations = torch.from_numpy(
            self.train_data.destinations).long()  # .to(self.device)
        self.train_data_timestamps = torch.from_numpy(
            self.train_data.timestamps).float()  # .to(self.device)
        self.train_data_src_neighbors = torch.from_numpy(
            self.src_neighbors).long()  # .to(self.device)
        self.train_data_dst_neighbors = torch.from_numpy(
            self.dst_neighbors).long()  # .to(self.device)
        self.train_data_src_neighbors_time = torch.from_numpy(
            self.src_neighbors_time).long()  # .to(self.device)
        self.train_data_dst_neighbors_time = torch.from_numpy(
            self.dst_neighbors_time).long()  # .to(self.device)

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
            # .to(self.device)
        )

        self._components = SLADEComponents(
            dcl_tgn=self.dcl_tgn,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        ).to(self.device)

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        device = self.device
        dcl_tgn = self.components.dcl_tgn
        optimizer = self.components.optimizer

        loss = 0
        self.optimizer.zero_grad()

        return 0

    def train(self, epochs: int, train_loader: TemporalGraphSnapshotLoader, val_loader: TemporalGraphSnapshotLoader | None = None, callbacks: list[ExperimentCallback] | None = None):
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        for epoch in range(epochs):
            self.set_training_mode(True)
            state.epoch = epoch
            handler.on_train_epoch_begin(state)

            # From original impl
            self.components.dcl_tgn.memory.__init_memory__()
            self.components.dcl_tgn.set_neighbor_finder(self.train_ngh_finder)
            m_loss = []

            # Method is in fact continuous,
            # We treat snapshots as batches
            for i, train_snapshot in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
                state.step_in_epoch = i
                state.total_steps += 1
                handler.on_train_step_begin(state)
                state.loss = self._train_step(train_snapshot)
                handler.on_train_step_end(state)

            if val_loader:
                self.set_training_mode(False)
                val_labels, val_probs = self.run_inference(val_loader)
                val_auc = roc_auc_score(
                    val_labels.cpu().numpy(), val_probs.cpu().numpy())
                state.val_metrics = {'roc_auc': val_auc}
                print(val_auc)

            handler.on_train_epoch_end(state)

        handler.on_train_end(state)

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        return super()._predict(snapshot, **kwargs)
