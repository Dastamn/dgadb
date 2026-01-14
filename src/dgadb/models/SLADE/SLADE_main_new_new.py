import os
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Self

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.dgadb.models.base import BaseADModel, BaseADModelComponents
from src.dgadb.storage.temporal_graph import TemporalGraph
from src.dgadb.storage.temporal_snapshot import TemporalGraphSnapshot

from src.dgadb.models.SLADE.SLADE_TGN import SLADE_TGN
from src.dgadb.models.SLADE.utils.data_processing import SLADEData
from src.dgadb.models.SLADE.utils.utils import NeighborFinder, get_neighbor_finder


@dataclass
class SLADEComponents(BaseADModelComponents):
    dcl_tgn: SLADE_TGN
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


class SLADE(BaseADModel[SLADEComponents]):
    def __init__(
        self,
        batch_size: int = 100,
        num_neighbors: int = 20,
        num_layer: int = 1,
        num_heads: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 3e-6,
        message_dim: int = 128,
        memory_dim: int = 256,
        lr_decay: float = 0.8,
        weight_decay: float = 0.0001,
        memory_agg_type: str = "TGAT",
        negative_memory_type: str = "train",
        message_updater: str = "mlp",
        memory_updater: str = "gru",
        srf: float = 0.1,
        drf: float = 0.1,
        only_drift_loss: bool = False,
        only_recovery_loss: bool = False,
        device: torch.device | str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(device)
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.dropout = dropout
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
        self.only_drift_loss = only_drift_loss
        self.only_recovery_loss = only_recovery_loss

        # State holders for neighbor finders initialized in setup
        self.train_ngh_finder: Optional[NeighborFinder] = None
        self.full_ngh_finder: Optional[NeighborFinder] = None
        self.negative_train_nodes: Optional[torch.Tensor] = None

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        if self._components is not None:
            return

        sources = data.src.cpu().numpy()
        destinations = data.tgt.cpu().numpy()
        timestamps = data.t.cpu().float().numpy()
        edge_idxs = np.arange(len(sources), dtype=np.int64)
        labels = data.edge_labels.cpu().numpy()

        # Create SLADEData objects
        train_mask = data.train_mask.cpu().numpy()

        # Full data is used for testing/inference
        full_slade_data = SLADEData(
            sources, destinations, timestamps, edge_idxs, labels)

        # Train data is used for training neighbor finder (avoids information leakage)
        train_slade_data = SLADEData(
            sources[train_mask],
            destinations[train_mask],
            timestamps[train_mask],
            edge_idxs[train_mask],
            labels[train_mask]
        )

        # 2. Initialize NeighborFinders
        max_idx = max(full_slade_data.unique_nodes)

        self.train_ngh_finder = get_neighbor_finder(
            train_slade_data, uniform=False, max_node_idx=max_idx
        )
        self.full_ngh_finder = get_neighbor_finder(
            full_slade_data, uniform=False, max_node_idx=max_idx
        )

        # 3. Initialize Model
        dcl_tgn = SLADE_TGN(
            neighbor_finder=self.train_ngh_finder,
            n_nodes=full_slade_data.n_unique_nodes,
            n_edges=full_slade_data.n_interactions,
            device=self.device,
            n_layers=self.num_layer,
            n_heads=self.num_heads,
            dropout=self.dropout,
            message_dimension=self.message_dim,
            memory_dimension=self.memory_dim,
            n_neighbors=self.num_neighbors,
            memory_agg_type=self.memory_agg_type,
            negative_memory_type=self.negative_memory_type,
            message_updater=self.message_updater,
            memory_updater=self.memory_updater,
            src_reg_factor=self.srf,
            dst_reg_factor=self.drf,
            only_drift_loss=self.only_drift_loss,
            only_recovery_loss=self.only_recovery_loss,
        )

        train_nodes = set(train_slade_data.sources) | set(
            train_slade_data.destinations)
        self.negative_train_nodes = torch.tensor(
            list(train_nodes), dtype=torch.long, device=self.device)

        # 5. Optimization
        optimizer = torch.optim.Adam(
            dcl_tgn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.lr_decay
        )

        self._components = SLADEComponents(
            dcl_tgn=dcl_tgn,
            optimizer=optimizer,
            scheduler=scheduler
        ).to(self.device)

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        model = self.components.dcl_tgn
        optimizer = self.components.optimizer

        if snapshot.snapshot_id == 0:
            model.memory.__init_memory__()
            model.set_neighbor_finder(self.train_ngh_finder)

        model.train()
        optimizer.zero_grad()

        src = snapshot.current.src
        dst = snapshot.current.tgt
        t = snapshot.current.t

        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        t_np = t.cpu().numpy()

        src_ngh, _, src_ngh_t = model.neighbor_finder.get_temporal_neighbor(
            src_np, t_np, self.num_neighbors
        )
        dst_ngh, _, dst_ngh_t = model.neighbor_finder.get_temporal_neighbor(
            dst_np, t_np, self.num_neighbors
        )

        src_ngh = torch.from_numpy(src_ngh).long().to(self.device)
        dst_ngh = torch.from_numpy(dst_ngh).long().to(self.device)
        src_ngh_t = torch.from_numpy(src_ngh_t).long().to(self.device)
        dst_ngh_t = torch.from_numpy(dst_ngh_t).long().to(self.device)

        _, _, _, _, loss = model.compute_node_diff_score(
            src, dst, t.float(),
            src_ngh, dst_ngh,
            src_ngh_t.float(), dst_ngh_t.float(),
            self.num_neighbors,
            self.negative_train_nodes
        )

        loss.backward()
        optimizer.step()

        # Detach memory to prevent backprop through time indefinitely
        model.memory.detach_memory()

        return loss.item()

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        model = self.components.dcl_tgn
        if snapshot.snapshot_id == 0:
            model.memory.__init_memory__()
            model.set_neighbor_finder(self.full_ngh_finder)

        model.eval()

        src = snapshot.current.src
        dst = snapshot.current.tgt
        t = snapshot.current.t

        # Neighbor Finder Query
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        t_np = t.cpu().numpy()

        src_ngh, _, src_ngh_t = model.neighbor_finder.get_temporal_neighbor(
            src_np, t_np, self.num_neighbors
        )
        dst_ngh, _, dst_ngh_t = model.neighbor_finder.get_temporal_neighbor(
            dst_np, t_np, self.num_neighbors
        )

        src_ngh = torch.from_numpy(src_ngh).long().to(self.device)
        dst_ngh = torch.from_numpy(dst_ngh).long().to(self.device)
        src_ngh_t = torch.from_numpy(src_ngh_t).long().to(self.device)
        dst_ngh_t = torch.from_numpy(dst_ngh_t).long().to(self.device)

        with torch.no_grad():
            pos_mem_score, drift_score, _, _ = model.compute_anomaly_score(
                src, dst, t.float(),
                src_ngh, dst_ngh,
                src_ngh_t.float(), dst_ngh_t.float(),
                self.num_neighbors
            )

        score = (-(drift_score).cpu() - (pos_mem_score).cpu() + 2) / 4

        return score.to(self.device)

    def save(self, save_dir: str) -> None:
        return None

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        return None


if __name__ == "__main__":
    from src.dgadb.storage import TemporalGraphLoader

    loader = TemporalGraphLoader()
    tg = loader.load("bitcoin-alpha", create_if_not_found=True)

    ad_model = SLADE()
    print(ad_model)


if __name__ == "__main__":
    from src.dgadb.storage import TemporalGraphLoader
    from src.dgadb.experiment.runner import ExperimentRunner
    from src.dgadb.experiment.callbacks import ResourceMonitor

    anom_config = {
        "anom_type": "structural",
        # "anom_train_ratio": 0.5,
        "anom_test_ratio": 0.1,
        "anom_val_ratio": 0.1,
    }
    snapshot_config = {
        "strategy": "window",
        "window_size": 100,
        "include_cumulative": False
    }

    model = SLADE()
    print(model)

    loader = TemporalGraphLoader()
    tg = loader.load(
        "digg-homo", **anom_config, create_if_not_found=True)
    # tg = loader.load(
    #     "digg-homo", create_if_not_found=True)

    # from src.dgadb.preprocessing.add_graph_temporary import inject_anomalies_addgraph_style

    # tg = inject_anomalies_addgraph_style(
    #     tg,
    #     anom_train_ratio=0.0,
    #     anom_val_ratio=0.1,
    #     anom_test_ratio=0.1,
    #     noise_ratio=0.0)

    runner = ExperimentRunner(model, tg)
    resource_monitor = ResourceMonitor(runner.output_dir, step_interval=10)
    runner.run(50, snapshot_config, [resource_monitor])
