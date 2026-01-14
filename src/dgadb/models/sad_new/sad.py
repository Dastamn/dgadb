from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..base import BaseADModel, BaseADModelComponents, TrainingState
from ..SAD.model.tgat import TGAT
from ..SAD import datasets as ds
from ..SAD.utils import get_neighbor_finder
from src.dgadb.experiment.callbacks import ExperimentCallbackHandler
from src.dgadb.storage.temporal_graph import TemporalGraph
from src.dgadb.storage.temporal_snapshot import TemporalGraphSnapshot, TemporalGraphSnapshotLoader


logger = logging.getLogger(__name__)


@dataclass
class SADComponents(BaseADModelComponents):
    model: TGAT
    optimizer: torch.optim.Optimizer
    ngh_finder: Any  # NeighborFinder
    collate_fn: Any  # Collate
    edge_features: np.ndarray
    node_features: np.ndarray


class SADAD(BaseADModel[SADComponents]):
    """SAD (Structural Anomaly Detection) model adapted to BaseADModel interface.

    This class wraps the SAD model to provide a standardized interface
    compatible with the DGADB framework.

    Args:
        bipartite: Whether to use bipartite graph mode
        mode: Model mode - "origin", "gdn", or "sad"
        add_scl: Whether to add supervised contrastive learning
        module_type: Type of graph module - "graph_attention" or "graph_sum"
        mask_label: Whether to mask labels during training
        mask_ratio: Ratio of labels to mask
        dev_alpha: Weight for deviation loss
        dev_beta: Weight for deviation beta term
        anomaly_alpha: Weight for anomaly loss
        supc_alpha: Weight for supervised contrastive loss
        memory_size: Size of memory buffer
        sample_size: Size of sampling buffer
        n_neighbors: Number of neighbors to sample
        batch_size: Batch size for training
        num_data_workers: Number of data loading workers
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        n_heads: Number of attention heads
        drop_out: Dropout rate
        n_layer: Number of layers
        learning_rate: Learning rate for optimizer
        device: Device to run computations on
    """

    def __init__(
        self,
        bipartite: bool = True,
        mode: str = "sad",
        add_scl: bool = False,
        module_type: str = "graph_attention",
        mask_label: bool = False,
        mask_ratio: float = 0.5,
        dev_alpha: float = 1.0,
        dev_beta: float = 1.0,
        anomaly_alpha: float = 1e-1,
        supc_alpha: float = 5e-3,
        memory_size: int = 5000,
        sample_size: int = 2000,
        n_neighbors: int = 20,
        batch_size: int = 256,
        num_data_workers: int = 1,
        input_dim: int = 1,
        hidden_dim: int = 128,
        n_heads: int = 2,
        drop_out: float = 0.2,
        n_layer: int = 2,
        learning_rate: float = 5e-4,
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)

        # Model configuration
        self.bipartite = bipartite
        self.mode = mode
        self.add_scl = add_scl
        self.module_type = module_type
        self.mask_label = mask_label
        self.mask_ratio = mask_ratio

        # Loss weights
        self.dev_alpha = dev_alpha
        self.dev_beta = dev_beta
        self.anomaly_alpha = anomaly_alpha
        self.supc_alpha = supc_alpha

        # Memory / sampling
        self.memory_size = memory_size
        self.sample_size = sample_size

        # Data loading
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers

        # Model architecture
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.n_layer = n_layer
        self.learning_rate = learning_rate

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        """Initialize the SAD model with the temporal graph data.

        Args:
            data: TemporalGraph object containing the graph data
            **kwargs: Additional arguments
        """
        if self._components is not None:
            logger.info("Model already initialized, skipping setup")
            return

        print("[SAD] Setting up model...", flush=True)

        # Prepare data for SAD
        # We need to construct SADData object for the neighbor finder
        src = data.src.detach().cpu().numpy()
        tgt = data.tgt.detach().cpu().numpy()
        timestamps = data.t.detach().cpu().numpy()
        labels = data.edge_labels.detach().cpu().numpy()
        edge_ids = np.arange(len(src))

        full_data = ds.SADData(src, tgt, timestamps, edge_ids, labels)

        # Initialize neighbor finder
        ngh_finder = get_neighbor_finder(full_data, uniform=False)

        # Get features
        edge_features = data.edge_attr.detach().cpu().numpy().astype(
            np.float32) if data.edge_attr is not None else np.zeros((len(src), 1), dtype=np.float32)
        node_features = data.node_attr.detach().cpu().numpy().astype(
            np.float32) if data.node_attr is not None else np.zeros((data.num_nodes, 1), dtype=np.float32)

        # Initialize collate function
        collate_fn = ds.Collate(node_features)

        # Create model
        arg_dict = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "drop_out": self.drop_out,
            "n_layer": self.n_layer,
            "module_type": self.module_type,
            "mode": self.mode,
            "memory_size": self.memory_size,
            "sample_size": self.sample_size,
        }

        backbone = TGAT(arg_dict, self.device)
        model = backbone.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        self._components = SADComponents(
            model=model,
            optimizer=optimizer,
            ngh_finder=ngh_finder,
            collate_fn=collate_fn,
            edge_features=edge_features,
            node_features=node_features
        ).to(self.device)

        print("[SAD] Model setup complete", flush=True)

    def _criterion(self, prediction_dict, labels):
        """Compute loss for SAD model."""
        # Filter predictions and labels
        # SAD logic: filter out masked labels (-1)
        valid_mask = labels > -1

        # Clone to avoid modifying original dict in place if needed,
        # but here we just extract what we need.
        logits = prediction_dict["logits"][valid_mask]
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), \
                torch.tensor(0.0, device=self.device), \
                torch.tensor(0.0, device=self.device), \
                torch.tensor(0.0, device=self.device)

        loss_classify = F.binary_cross_entropy_with_logits(
            logits, valid_labels.float(), reduction="none"
        )
        loss_classify = torch.mean(loss_classify)

        loss = loss_classify.clone()
        loss_anomaly = torch.tensor(0.0).to(self.device)
        loss_supc = torch.tensor(0.0).to(self.device)

        if self.mode == "sad":
            loss_anomaly = self.components.model.gdn.dev_loss(
                torch.squeeze(valid_labels),
                torch.squeeze(prediction_dict["anom_score"][valid_mask]),
                torch.squeeze(prediction_dict["time"][valid_mask]),
            )
            loss_supc = self.components.model.suploss(
                # SupConLoss handles its own filtering/logic usually
                prediction_dict["root_embedding"],
                prediction_dict["group"],
                prediction_dict["dev"]
            )
            loss += self.anomaly_alpha * loss_anomaly + self.supc_alpha * loss_supc

        return loss, loss_classify, loss_anomaly, loss_supc

    def _process_batch_data(self, sources, timestamps, labels):
        """Prepare batch data using neighbor finder and collate function."""
        # This logic mimics DygDataset.__getitem__ but for a batch of indices

        batch_items = []
        for i in range(len(sources)):
            source_node = sources[i]
            current_time = timestamps[i]
            label = labels[i]

            # Find neighbors
            src_neigh_edge, src_neigh_time, src_neigh_idx = self.components.ngh_finder.get_temporal_neighbor_all(
                source_node, current_time, self.n_layer, self.n_neighbors
            )

            src_edge_feature = self.components.edge_features[src_neigh_idx].astype(
                np.float32)
            src_edge_to_time = current_time - src_neigh_time
            src_center_node_idx = np.reshape(source_node, [-1])

            if src_neigh_edge.shape[0] == 0:
                # Padding
                # We need to access the helper method from DygDataset or reimplement it.
                # Since it's a method of DygDataset, we'll reimplement it here briefly.
                src_neigh_edge = np.concatenate((src_neigh_edge, np.tile(
                    src_center_node_idx.reshape(-1, 1), (1, 2))), axis=0)
                src_neigh_time = np.concatenate(
                    (src_neigh_time, np.zeros([1], dtype=src_neigh_time.dtype)), axis=0)
                src_edge_feature = np.concatenate((src_edge_feature, np.zeros(
                    [1, src_edge_feature.shape[1]], dtype=src_edge_feature.dtype)), axis=0)
                src_neigh_idx = np.concatenate(
                    (src_neigh_idx, np.zeros([1], dtype=src_neigh_idx.dtype)), axis=0)

                # Re-calculate time diff after padding
                src_edge_to_time = current_time - src_neigh_time

            label = np.reshape(label, [-1])
            current_time = np.reshape(current_time, [-1])

            batch_items.append({
                "src_center_node_idx": src_center_node_idx,
                "src_neigh_edge": torch.from_numpy(src_neigh_edge),
                "src_edge_feature": torch.from_numpy(src_edge_feature),
                "src_edge_to_time": torch.from_numpy(src_edge_to_time.astype(np.float32)),
                "init_edge_index": torch.from_numpy(src_neigh_idx),
                "current_time": torch.from_numpy(current_time),
                "label": torch.from_numpy(label),
            })

        return self.components.collate_fn.dyg_collate_fn(batch_items)

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        """Perform a single training step on a snapshot."""
        model = self.components.model
        optimizer = self.components.optimizer

        # Get edges for this snapshot
        current_graph = snapshot.current

        # We need numpy arrays for the finder
        src = current_graph.src.cpu().numpy()
        # tgt = current_graph.tgt.cpu().numpy() # Not used in get_temporal_neighbor_all for source
        t = current_graph.t.cpu().numpy()
        labels = current_graph.edge_labels.cpu().numpy()

        num_edges = len(src)
        if num_edges == 0:
            return 0.0

        # Mask labels if configured (mimic DygDataset behavior)
        if self.mask_label:
            # Simple random masking for this batch
            # Note: This is slightly different from DygDataset which masks globally once.
            # But for streaming/snapshot training, we mask per batch/snapshot.
            num_mask = int(num_edges * self.mask_ratio)
            if num_mask > 0:
                mask_indices = np.random.choice(
                    num_edges, num_mask, replace=False)
                labels[mask_indices] = -1

        # Process in batches to avoid OOM if snapshot is large
        total_loss = 0.0
        num_batches = 0

        indices = np.arange(num_edges)
        # Shuffle for training
        np.random.shuffle(indices)

        for start_idx in range(0, num_edges, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_edges)
            batch_indices = indices[start_idx:end_idx]

            batch_src = src[batch_indices]
            batch_t = t[batch_indices]
            batch_labels = labels[batch_indices]

            batch_sample = self._process_batch_data(
                batch_src, batch_t, batch_labels)

            optimizer.zero_grad()

            # Forward pass
            x = model(
                batch_sample["src_edge_feat"].to(self.device),
                batch_sample["src_edge_to_time"].to(self.device),
                batch_sample["src_center_node_idx"].to(self.device),
                batch_sample["src_neigh_edge"].to(self.device),
                batch_sample["src_node_features"].to(self.device),
                batch_sample["current_time"].to(self.device),
                batch_sample["labels"].to(self.device),
            )

            y = batch_sample["labels"].to(self.device)

            # Compute loss
            loss, _, _, _ = self._criterion(x, y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1, norm_type=2
            )
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        # print(f"[DEBUG] Snapshot Loss: {avg_loss:.4f}")
        return avg_loss

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        """Predict anomaly scores for edges in a snapshot."""
        model = self.components.model
        self.set_training_mode(False)

        current_graph = snapshot.current
        src = current_graph.src.cpu().numpy()
        t = current_graph.t.cpu().numpy()
        # Needed for shape/device, though we don't use them for training
        labels = current_graph.edge_labels.cpu().numpy()

        num_edges = len(src)
        if num_edges == 0:
            return torch.empty(0, device=self.device)

        all_scores = []

        # Process in batches
        for start_idx in range(0, num_edges, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_edges)

            batch_src = src[start_idx:end_idx]
            batch_t = t[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            batch_sample = self._process_batch_data(
                batch_src, batch_t, batch_labels)

            with torch.no_grad():
                x = model(
                    batch_sample["src_edge_feat"].to(self.device),
                    batch_sample["src_edge_to_time"].to(self.device),
                    batch_sample["src_center_node_idx"].to(self.device),
                    batch_sample["src_neigh_edge"].to(self.device),
                    batch_sample["src_node_features"].to(self.device),
                    batch_sample["current_time"].to(self.device),
                    batch_sample["labels"].to(self.device),
                )

                # Use logits or anom_score?
                # In SAD code: pred_score = x["logits"].sigmoid().cpu().numpy().flatten()
                # But x["anom_score"] is also available.
                # The original code uses logits for ROC AUC in _inference_internal.
                # scores = x["logits"].sigmoid()
                scores = torch.abs(x["dev"]).view(-1)

                if scores.shape[0] == 2 * len(batch_src):
                    scores = scores[:len(batch_src)]

                all_scores.append(scores)

        result = torch.cat(all_scores) if all_scores else torch.empty(
            0, device=self.device)

        # Debug stats
        if len(result) > 0:
            mean_score = result.mean().item()
            std_score = result.std().item()
            # print(f"[DEBUG] Predict Scores: Mean={mean_score:.4f}, Std={std_score:.4f}")

        return result

    def save(self, save_dir: str) -> None:
        """Save model checkpoint and configuration."""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.components.model.state_dict(),
            'optimizer_state_dict': self.components.optimizer.state_dict(),
        }

        config = {
            'bipartite': self.bipartite,
            'mode': self.mode,
            'add_scl': self.add_scl,
            'module_type': self.module_type,
            'mask_label': self.mask_label,
            'mask_ratio': self.mask_ratio,
            'dev_alpha': self.dev_alpha,
            'dev_beta': self.dev_beta,
            'anomaly_alpha': self.anomaly_alpha,
            'supc_alpha': self.supc_alpha,
            'memory_size': self.memory_size,
            'sample_size': self.sample_size,
            'n_neighbors': self.n_neighbors,
            'batch_size': self.batch_size,
            'num_data_workers': self.num_data_workers,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_heads': self.n_heads,
            'drop_out': self.drop_out,
            'n_layer': self.n_layer,
            'learning_rate': self.learning_rate,
        }

        model_path = os.path.join(save_dir, "model.pt")
        torch.save(checkpoint, model_path)

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs):
        """Load model from checkpoint."""
        config_path = os.path.join(load_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at '{config_path}'")

        with open(config_path, 'r') as f:
            config = json.load(f)

        instance = cls(device=device, **config)

        model_path = os.path.join(load_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'")

        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False)

        # Note: Need to call setup first with actual data before loading weights
        # This is a limitation of the current design
        logger.warning(
            "Loading SAD model requires calling setup() with data before "
            "loading weights. Weights will be loaded when available."
        )

        # We can't load state dict here because model is not initialized (setup not called)
        # We store the checkpoint to be loaded in setup if needed, or user must call load_weights after setup.
        # But BaseADModel.load returns an instance.
        # The standard pattern in this repo seems to be:
        # 1. load() returns instance
        # 2. user calls setup(data)
        # 3. setup() should check if there are weights to load?
        # OR, we monkey-patch setup to load weights after initialization.

        original_setup = instance.setup

        def setup_with_load(data: TemporalGraph, **kwargs):
            original_setup(data, **kwargs)
            instance.components.model.load_state_dict(
                checkpoint['model_state_dict'])
            instance.components.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            logger.info("Loaded model weights from checkpoint.")

        instance.setup = setup_with_load

        return instance
