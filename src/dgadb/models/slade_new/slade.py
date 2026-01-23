import logging
import os
import json
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dgadb.models.base import BaseADModel, BaseADModelComponents, TrainingState
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot, TemporalGraphSnapshotLoader
from dgadb.experiment.callbacks import ExperimentCallback, ExperimentCallbackHandler

from dgadb.models.SLADE.SLADE_TGN import SLADE_TGN
from dgadb.models.SLADE.utils.utils import get_neighbor_finder, NeighborFinder
from dgadb.models.SLADE.utils.data_processing import SLADEData


@dataclass
class SLADEComponents(BaseADModelComponents):
    model: SLADE_TGN
    full_ngh_finder: NeighborFinder
    train_ngh_finder: NeighborFinder
    negative_train_nodes: torch.Tensor
    num_neighbors: int
    optimizer: torch.optim.Optimizer


class SLADEAD(BaseADModel[SLADEComponents]):
    """SLADE (Structural Leap Anomaly DetEction) model adapted to BaseADModel interface.

    This class wraps the SLADE model to provide a standardized interface
    compatible with the DGADB framework.

    Args:
        num_neighbors: Number of neighbors to sample
        num_layer: Number of layers (always 1 for SLADE)
        num_heads: Number of attention heads
        drop_out: Dropout rate
        message_dim: Dimension of message embeddings
        memory_dim: Dimension of memory embeddings
        memory_agg_type: Type of memory aggregation (TGAT)
        negative_memory_type: Type of negative memory sampling (train/random)
        message_updater: Type of message updater (mlp/identity)
        memory_updater: Type of memory updater (gru/rnn)
        srf: Source regularization factor
        drf: Destination regularization factor
        only_drift_loss_score: Use only drift loss and score
        only_recovery_loss_score: Use only recovery loss and score
        only_drift_score: Use only drift score with both losses
        only_rec_score: Use only recovery score with both losses
        test_inference_time: Test with inference time mode
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to run computations on
    """

    def __init__(
        self,
        num_neighbors: int = 20,
        num_layer: int = 1,
        num_heads: int = 2,
        drop_out: float = 0.1,
        message_dim: int = 128,
        memory_dim: int = 256,
        memory_agg_type: str = "TGAT",
        negative_memory_type: str = "train",
        message_updater: str = "mlp",
        memory_updater: str = "gru",
        srf: float = 0.1,
        drf: float = 0.1,
        only_drift_loss_score: bool = False,
        only_recovery_loss_score: bool = False,
        only_drift_score: bool = False,
        only_rec_score: bool = False,
        test_inference_time: bool = False,
        learning_rate: float = 3e-6,
        weight_decay: float = 0.0001,
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)

        # Store hyperparameters
        self.num_neighbors = num_neighbors
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.drop_out = drop_out
        self.message_dim = message_dim
        self.memory_dim = memory_dim
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
        self.test_inference_time = test_inference_time
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.logger = logging.getLogger(__name__)

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        """Initialize the SLADE model with the temporal graph data.

        Args:
            data: TemporalGraph object containing the graph data
            **kwargs: Additional arguments (can override hyperparameters)
        """
        if self._components is not None:
            self.logger.info("Model already initialized, skipping setup")
            return

        self.logger.info("Setting up SLADE model...")

        # Allow kwargs to override init hyperparameters
        num_neighbors = kwargs.get("num_neighbors", self.num_neighbors)
        num_layer = kwargs.get("num_layer", self.num_layer)
        num_heads = kwargs.get("num_heads", self.num_heads)
        drop_out = kwargs.get("drop_out", self.drop_out)
        message_dim = kwargs.get("message_dim", self.message_dim)
        memory_dim = kwargs.get("memory_dim", self.memory_dim)
        memory_agg_type = kwargs.get("memory_agg_type", self.memory_agg_type)
        negative_memory_type = kwargs.get(
            "negative_memory_type", self.negative_memory_type)
        message_updater = kwargs.get("message_updater", self.message_updater)
        memory_updater = kwargs.get("memory_updater", self.memory_updater)
        srf = kwargs.get("srf", self.srf)
        drf = kwargs.get("drf", self.drf)
        only_drift_loss_score = kwargs.get(
            "only_drift_loss_score", self.only_drift_loss_score)
        only_recovery_loss_score = kwargs.get(
            "only_recovery_loss_score", self.only_recovery_loss_score)
        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        weight_decay = kwargs.get("weight_decay", self.weight_decay)

        # Prepare data for SLADE
        sources = data.src.cpu().numpy()
        destinations = data.tgt.cpu().numpy()
        timestamps = data.t.cpu().numpy()
        edge_idxs = np.arange(len(sources))
        labels = data.edge_labels.cpu().numpy()

        train_mask = data.train_mask.cpu().numpy()

        full_data = SLADEData(sources, destinations,
                              timestamps, edge_idxs, labels)
        train_data = SLADEData(
            sources[train_mask],
            destinations[train_mask],
            timestamps[train_mask],
            edge_idxs[train_mask],
            labels[train_mask],
        )

        max_idx = max(full_data.unique_nodes)

        # Initialize NeighborFinders
        full_ngh_finder = get_neighbor_finder(
            full_data, uniform=False, max_node_idx=max_idx)
        train_ngh_finder = get_neighbor_finder(
            train_data, uniform=False, max_node_idx=max_idx)

        # Initialize Model
        model = SLADE_TGN(
            neighbor_finder=train_ngh_finder,
            n_nodes=full_data.n_unique_nodes,
            n_edges=full_data.n_interactions,
            device=self.device,
            n_layers=num_layer,
            n_heads=num_heads,
            dropout=drop_out,
            message_dimension=message_dim,
            memory_dimension=memory_dim,
            n_neighbors=num_neighbors,
            memory_agg_type=memory_agg_type,
            negative_memory_type=negative_memory_type,
            message_updater=message_updater,
            memory_updater=memory_updater,
            src_reg_factor=srf,
            dst_reg_factor=drf,
            only_drift_loss=only_drift_loss_score,
            only_recovery_loss=only_recovery_loss_score,
        ).to(self.device)

        # Negative train nodes
        negative_train_nodes = (
            torch.from_numpy(
                np.array(list(set(train_data.destinations) | set(train_data.sources))))
            .long()
            .to(self.device)
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self._components = SLADEComponents(
            model=model,
            full_ngh_finder=full_ngh_finder,
            train_ngh_finder=train_ngh_finder,
            negative_train_nodes=negative_train_nodes,
            num_neighbors=num_neighbors,
            optimizer=optimizer
        )

        self.logger.info("SLADE model setup complete.")

    def train(
        self,
        epochs: int,
        train_loader: TemporalGraphSnapshotLoader,
        val_loader: Optional[TemporalGraphSnapshotLoader] = None,
        callbacks: Optional[list[ExperimentCallback]] = None
    ):
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        for epoch in range(epochs):
            # SLADE SPECIFIC: Reset memory at start of epoch
            self.components.model.memory.__init_memory__()
            # SLADE SPECIFIC: Set neighbor finder to train
            self.components.model.set_neighbor_finder(
                self.components.train_ngh_finder)

            self.set_training_mode(True)
            state.epoch = epoch
            handler.on_train_epoch_begin(state)

            for i, train_snapshot in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
                state.step_in_epoch = i
                state.total_steps += 1
                handler.on_train_step_begin(state)
                state.loss = self._train_step(train_snapshot)
                handler.on_train_step_end(state)

                # SLADE SPECIFIC: Detach memory after each batch
                self.components.model.memory.detach_memory()

            if val_loader:
                self.set_training_mode(False)
                val_labels, val_probs = self.run_inference(val_loader)
                val_auc = roc_auc_score(
                    val_labels.detach().cpu().numpy(), val_probs.detach().cpu().numpy())
                state.val_metrics = {'roc_auc': val_auc}
                print(f"Epoch {epoch+1} Val AUC: {val_auc:.4f}")

            handler.on_train_epoch_end(state)

        handler.on_train_end(state)

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        model = self.components.model
        current_graph = snapshot.current

        sources_batch = current_graph.src
        destinations_batch = current_graph.tgt
        # Ensure timestamps are float32, not long
        timestamps_batch = current_graph.t.float()

        # Get neighbors on the fly
        src_neighbors_batch_np, _, src_neighbors_time_batch_np = self.components.train_ngh_finder.get_temporal_neighbor(
            sources_batch.cpu().numpy(), timestamps_batch.cpu(
            ).numpy(), self.components.num_neighbors
        )
        dst_neighbors_batch_np, _, dst_neighbors_time_batch_np = self.components.train_ngh_finder.get_temporal_neighbor(
            destinations_batch.cpu().numpy(), timestamps_batch.cpu(
            ).numpy(), self.components.num_neighbors
        )

        src_neighbors_batch = torch.from_numpy(
            src_neighbors_batch_np).long().to(self.device)
        dst_neighbors_batch = torch.from_numpy(
            dst_neighbors_batch_np).long().to(self.device)
        # Ensure time batches are float, not long
        src_neighbors_time_batch = torch.from_numpy(
            src_neighbors_time_batch_np.astype(np.float32)).to(self.device)
        dst_neighbors_time_batch = torch.from_numpy(
            dst_neighbors_time_batch_np.astype(np.float32)).to(self.device)

        _, _, _, _, contrastive_loss = model.compute_node_diff_score(
            sources_batch,
            destinations_batch,
            timestamps_batch,
            src_neighbors_batch,
            dst_neighbors_batch,
            src_neighbors_time_batch,
            dst_neighbors_time_batch,
            self.components.num_neighbors,
            self.components.negative_train_nodes,
        )

        self.components.optimizer.zero_grad()
        contrastive_loss.backward()
        self.components.optimizer.step()

        return contrastive_loss.item()

    def run_inference(self, loader: TemporalGraphSnapshotLoader) -> tuple[torch.Tensor, torch.Tensor]:
        # SLADE SPECIFIC: Reset memory
        self.components.model.memory.__init_memory__()
        # SLADE SPECIFIC: Set neighbor finder to full (safe for all splits if causal)
        self.components.model.set_neighbor_finder(
            self.components.full_ngh_finder)

        return super().run_inference(loader)

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        model = self.components.model
        current_graph = snapshot.current

        sources_batch = current_graph.src
        destinations_batch = current_graph.tgt
        timestamps_batch = current_graph.t.float()

        # Get neighbors on the fly using full_ngh_finder (set in run_inference)
        src_neighbors_batch_np, _, src_neighbors_time_batch_np = self.components.full_ngh_finder.get_temporal_neighbor(
            sources_batch.cpu().numpy(), timestamps_batch.cpu(
            ).numpy(), self.components.num_neighbors
        )
        dst_neighbors_batch_np, _, dst_neighbors_time_batch_np = self.components.full_ngh_finder.get_temporal_neighbor(
            destinations_batch.cpu().numpy(), timestamps_batch.cpu(
            ).numpy(), self.components.num_neighbors
        )

        src_neighbors_batch = torch.from_numpy(
            src_neighbors_batch_np).long().to(self.device)
        dst_neighbors_batch = torch.from_numpy(
            dst_neighbors_batch_np).long().to(self.device)
        # Ensure time batches are float, not long
        src_neighbors_time_batch = torch.from_numpy(
            src_neighbors_time_batch_np.astype(np.float32)).to(self.device)
        dst_neighbors_time_batch = torch.from_numpy(
            dst_neighbors_time_batch_np.astype(np.float32)).to(self.device)

        positive_memory_score, drift_score, _, _ = model.compute_anomaly_score(
            sources_batch,
            destinations_batch,
            timestamps_batch,
            src_neighbors_batch,
            dst_neighbors_batch,
            src_neighbors_time_batch,
            dst_neighbors_time_batch,
            self.components.num_neighbors
        )

        # Calculate final score (default method)
        # pred_score = (-(drift_score)-(positive_memory_score) + 2)/4
        # Note: scores are tensors here - detach to avoid gradient issues
        pred_score = (-(drift_score) - (positive_memory_score) + 2) / 4

        return pred_score.detach()

    def save(self, save_dir: str) -> None:
        """Save model checkpoint and configuration."""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.components.model.state_dict(),
            'optimizer_state_dict': self.components.optimizer.state_dict(),
        }

        config = {
            'num_neighbors': self.num_neighbors,
            'num_layer': self.num_layer,
            'num_heads': self.num_heads,
            'drop_out': self.drop_out,
            'message_dim': self.message_dim,
            'memory_dim': self.memory_dim,
            'memory_agg_type': self.memory_agg_type,
            'negative_memory_type': self.negative_memory_type,
            'message_updater': self.message_updater,
            'memory_updater': self.memory_updater,
            'srf': self.srf,
            'drf': self.drf,
            'only_drift_loss_score': self.only_drift_loss_score,
            'only_recovery_loss_score': self.only_recovery_loss_score,
            'only_drift_score': self.only_drift_score,
            'only_rec_score': self.only_rec_score,
            'test_inference_time': self.test_inference_time,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
        }

        model_path = os.path.join(save_dir, "model.pt")
        torch.save(checkpoint, model_path)

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        self.logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> 'SLADEAD':
        """Load model from checkpoint.

        Note: After loading, you must call setup(data) before using the model,
        as SLADE requires data-dependent initialization of neighbor finders.
        """
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

        # We need to monkey-patch setup to load weights after initialization
        # because SLADE requires data to be setup before we can load state dicts
        original_setup = instance.setup

        def setup_with_load(data: TemporalGraph, **setup_kwargs):
            original_setup(data, **setup_kwargs)
            instance.components.model.load_state_dict(
                checkpoint['model_state_dict'])
            instance.components.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            instance.logger.info("Loaded model weights from checkpoint.")

        instance.setup = setup_with_load

        return instance
