from __future__ import annotations

import os
import json
import pickle
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import splu

from ..base import BaseADModel, BaseADModelComponents, TrainingState
from ..TADDY.codes.Component import MyConfig
from ..TADDY.codes.DynADModel import DynADModel
from dgadb.experiment.callbacks import ExperimentCallbackHandler
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot, TemporalGraphSnapshotLoader
from dgadb.storage.utils import generate_temporal_graph_filename
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TADDYADComponents(BaseADModelComponents):
    model: DynADModel
    optimizer: torch.optim.Optimizer


class TADDYAD(BaseADModel[TADDYADComponents]):
    """TADDY model adapted to BaseADModel interface.

    This class wraps the TADDY (Temporal Anomaly Detection in Dynamic graphs) model
    to provide a standardized interface compatible with the DGADB framework.

    Args:
        batch_size: Batch size for training
        num_neighbors: Number of neighbors to sample
        num_heads: Number of attention heads
        drop_out: Dropout rate
        num_layer: Number of layers
        learning_rate: Learning rate for optimizer
        message_dim: Dimension of message embeddings
        memory_dim: Dimension of memory embeddings
        weight_decay: L2 regularization weight
        c: TADDY-specific hyperparameter for eigen computation
        window_size: Window size for temporal snapshots
        snap_size: Size of each snapshot
        device: Device to run computations on
    """

    def __init__(
        self,
        batch_size: int = 100,  # Original TADDYModel default (was 256)
        num_neighbors: int = 5,
        num_heads: int = 2,
        drop_out: float = 0.1,  # Original TADDYModel default (was 0.5)
        num_layer: int = 1,  # Match TADDYModel default (was 2)
        learning_rate: float = 0.001,
        message_dim: int = 128,  # Match TADDYModel default (was 32)
        memory_dim: int = 256,   # Match TADDYModel default (was 32)
        weight_decay: float = 5e-4,
        c: float = 0.15,
        window_size: int = 3,  # Match TADDYModel default (was 2)
        snap_size: int = 2000,
        device: torch.device | str = "cpu",
        cache_dir: str = "cache"
    ) -> None:
        super().__init__(device)

        # Store hyperparameters
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        self.drop_out = drop_out
        self.num_layer = num_layer
        self.learning_rate = learning_rate
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.weight_decay = weight_decay
        self.c = c
        self.window_size = window_size
        self.snap_size = snap_size

        self.cache_dir = Path(cache_dir)

        # Internal state
        self.data_dict: Optional[dict] = None
        self.embeddings: dict[str, np.ndarray] = {}

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        """Initialize the TADDY model with the temporal graph data.

        Args:
            data: TemporalGraph object containing the graph data
            **kwargs: Additional arguments (dataset_name, train_ratio, val_ratio, anom_val_ratio)
        """
        # if self._components is not None:
        #     logger.info("Model already initialized, skipping setup")
        #     return

        logger.info("Setting up TADDY model...")

        # Update snap_size if provided (e.g. from ExperimentRunner)
        if "snap_size" in kwargs and kwargs["snap_size"] is not None:
            self.snap_size = kwargs["snap_size"]

        # Build training adjacency matrix
        train_mask = data.train_mask
        train_src = data.src[train_mask]
        train_tgt = data.tgt[train_mask]
        num_nodes = data.num_nodes

        data_ones = np.ones_like(train_src, dtype=np.int32)
        train_adj = sp.csr_matrix(
            (data_ones, (train_src, train_tgt)), shape=(num_nodes, num_nodes)
        )
        train_adj = train_adj + sp.eye(num_nodes)
        train_adj_lil = train_adj.tolil()
        headtail = train_adj_lil.rows

        # Process snapshots
        rows, cols, labs, weis, edges = [], [], [], [], []

        # Process train snapshots (NO include_cumulative like original TADDYModel)
        train_snap_loader = TemporalGraphSnapshotLoader(
            data, window_size=self.snap_size, split="train", include_cumulative=False
        )
        for train_snap in train_snap_loader:
            train_curr = train_snap.current
            src_np = train_curr.src.numpy()
            tgt_np = train_curr.tgt.numpy()
            rows.append(src_np)
            cols.append(tgt_np)
            labs.append(train_curr.edge_labels)
            # Handle case where weights don't exist (default to ones)
            w_np = train_curr.w.numpy() if train_curr.w is not None else np.ones_like(
                src_np, dtype=np.float32)
            weis.append(w_np)
            edges.append(torch.vstack(
                [train_curr.src, train_curr.tgt]).T.numpy())

        # Process val snapshots (NO include_cumulative like original TADDYModel)
        val_snap_loader = TemporalGraphSnapshotLoader(
            data, window_size=self.snap_size, split="val", include_cumulative=False
        )
        for val_snap in val_snap_loader:
            val_curr = val_snap.current
            src_np = val_curr.src.numpy()
            tgt_np = val_curr.tgt.numpy()
            rows.append(src_np)
            cols.append(tgt_np)
            labs.append(val_curr.edge_labels)
            # Handle case where weights don't exist (default to ones)
            w_np = val_curr.w.numpy() if val_curr.w is not None else np.ones_like(
                src_np, dtype=np.float32)
            weis.append(w_np)
            edges.append(torch.vstack([val_curr.src, val_curr.tgt]).T.numpy())

        # Process test snapshots (NO include_cumulative like original TADDYModel)
        test_snap_loader = TemporalGraphSnapshotLoader(
            data, window_size=self.snap_size, split="test", include_cumulative=False
        )
        for test_snap in test_snap_loader:
            test_curr = test_snap.current
            src_np = test_curr.src.numpy()
            tgt_np = test_curr.tgt.numpy()
            rows.append(src_np)
            cols.append(tgt_np)
            labs.append(test_curr.edge_labels)
            # Handle case where weights don't exist (default to ones)
            w_np = test_curr.w.numpy() if test_curr.w is not None else np.ones_like(
                src_np, dtype=np.float32)
            weis.append(w_np)
            edges.append(torch.vstack(
                [test_curr.src, test_curr.tgt]).T.numpy())

        train_snap_size = len(train_snap_loader)
        val_snap_size = len(val_snap_loader)
        test_snap_size = len(test_snap_loader)

        degrees = np.array([len(x) for x in headtail])

        dataset_id = generate_temporal_graph_filename(data)

        # Build adjacency matrices
        adjs, eigen_adjs = self._get_adjs(
            rows, cols, weis, num_nodes, data.dataset_name, dataset_id, # self.snap_size
        )

        idx = list(range(num_nodes))
        index_id_map = {i: i for i in idx}
        idx = np.array(idx)

        num_snap = train_snap_size + val_snap_size + test_snap_size

        self.data_dict = {
            "X": None,
            "A": adjs,
            "S": eigen_adjs,
            "index_id_map": index_id_map,
            "edges": edges,
            "y": labs,
            "idx": idx,
            "snap_train": list(range(num_snap))[:train_snap_size],
            "snap_val": list(range(num_snap))[train_snap_size:(train_snap_size + val_snap_size)],
            "snap_test": list(range(num_snap))[(train_snap_size + val_snap_size):],
            "degrees": degrees,
            "num_snap": num_snap,
        }

        # Initialize TADDY model
        my_config = MyConfig(
            k=self.num_neighbors,
            window_size=self.window_size,
            hidden_size=self.message_dim,
            intermediate_size=self.message_dim,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layer,
            weight_decay=self.weight_decay,
        )

        # Create a minimal wrapper object for TADDY's expectations
        class ConfigWrapper:
            def __init__(self, device):
                self.device = device

        wrapper = ConfigWrapper(self.device)
        taddy_model = DynADModel(my_config, wrapper)
        taddy_model.data = self.data_dict
        taddy_model.spy_tag = True

        optimizer = torch.optim.Adam(
            params=taddy_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self._components = TADDYADComponents(
            model=taddy_model,
            optimizer=optimizer
        ).to(self.device)

        logger.info("TADDY model setup complete")

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        """Perform a single training step.

        Note: TADDY training is snapshot-based and handled in the train() method.
        This method is required by the interface but delegates to the full training logic.
        """
        # TADDY doesn't train on individual snapshots in the traditional sense
        # Instead, it processes all snapshots per epoch
        # This is a compatibility stub
        return 0.0

    def train(
        self,
        epochs: int,
        train_loader: TemporalGraphSnapshotLoader,
        val_loader: Optional[TemporalGraphSnapshotLoader] = None,
        callbacks: Optional[list] = None
    ):
        """Train the TADDY model.

        Note: TADDY has a custom training loop that operates on all snapshots
        rather than using the standard snapshot-by-snapshot training.
        """
        logger.info("Starting TADDY training for %d epochs...", epochs)

        # Setup callback handler
        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        # Compute embeddings once before training
        self._compute_embeddings()
        self.data_dict["raw_embeddings"] = None

        model = self.components.model
        optimizer = self.components.optimizer

        pbar = tqdm(range(epochs))

        for epoch in pbar:
            model.train()
            loss_train = 0
            state.epoch = epoch
            handler.on_train_epoch_begin(state)

            # Generate negative samples for this epoch
            import time
            start_neg_time = time.time()
            negatives = model.negative_sampling(
                self.data_dict["edges"][:max(self.data_dict["snap_train"]) + 1]
            )

            _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = model.generate_embedding(
                negatives
            )
            neg_time = time.time() - start_neg_time

            # Train on each snapshot
            num_valid_snaps = 0
            total_train_snaps = len(self.data_dict["snap_train"])
            for i, snap in enumerate(self.data_dict["snap_train"]):
                if self.embeddings["wl"][snap] is None:
                    continue

                state.step_in_epoch = i
                state.total_steps += 1
                handler.on_train_step_begin(state)

                # Positive samples
                int_embedding_pos = self.embeddings["int"][snap]
                hop_embedding_pos = self.embeddings["hop"][snap]
                time_embedding_pos = self.embeddings["time"][snap]
                # Convert to float like original TADDY
                y_pos = self.data_dict["y"][snap].float().to(self.device)

                # Negative samples
                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0], device=self.device)

                # Combine positive and negative
                int_embedding = torch.vstack(
                    (int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack(
                    (hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack(
                    (time_embedding_pos, time_embedding_neg))
                y = torch.hstack((y_pos, y_neg))

                optimizer.zero_grad()

                output = model.forward(
                    int_embedding, hop_embedding, time_embedding
                ).squeeze()

                loss = F.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()
                num_valid_snaps += 1

                state.loss = loss.item()
                handler.on_train_step_end(state)

            # Normalize loss like original TADDY (accounts for window_size)
            loss_train /= (len(self.data_dict["snap_train"]
                               ) - self.window_size + 1)
            epoch_time = time.time() - start_neg_time
            pbar.set_description(
                f"[TRAIN] Epoch {epoch + 1}/{epochs}: Loss={loss_train:.4f}, Time={epoch_time:.1f}s")

            # Validation
            if val_loader:
                self.set_training_mode(False)
                val_labels, val_scores = self.run_inference(val_loader)
                val_auc = roc_auc_score(
                        val_labels.cpu().numpy(), val_scores.cpu().numpy())
                self.logger.info(f"Epoch {epoch} Val AUC : {val_auc:.4f}")

            handler.on_train_epoch_end(state)

        handler.on_train_end(state)

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        """Predict anomaly scores for edges in a snapshot.

        Note: TADDY prediction requires the snapshot index in the data_dict.
        This is a compatibility method for the standard interface.
        """
        # This method is challenging because TADDY needs the snapshot index
        # For now, we'll raise an informative error
        raise NotImplementedError(
            "TADDY's _predict method requires full inference context. "
            "Use run_inference() with a TemporalGraphSnapshotLoader instead."
        )

    def run_inference(self, loader: TemporalGraphSnapshotLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a snapshot loader and return labels and scores.

        Args:
            loader: Snapshot loader for the data split (val or test)

        Returns:
            Tuple of (labels, scores) as tensors
        """
        self.components.model.eval()

        preds = []
        labels = []

        # Determine which snapshots to evaluate based on loader split
        if loader.split == "val":
            snap_ids = self.data_dict["snap_val"]
        elif loader.split == "test":
            snap_ids = self.data_dict["snap_test"]
        else:
            snap_ids = self.data_dict["snap_train"]

        for snap in tqdm(snap_ids, desc="TEST"):
            int_embedding = self.embeddings["int"][snap]
            hop_embedding = self.embeddings["hop"][snap]
            time_embedding = self.embeddings["time"][snap]

            with torch.no_grad():
                output = self.components.model.forward(
                    int_embedding, hop_embedding, time_embedding, None
                )
                output = torch.sigmoid(output)

            pred = output.squeeze().cpu()
            preds.append(pred)
            labels.append(self.data_dict["y"][snap].cpu())

        all_labels = torch.cat(labels)
        all_scores = torch.cat(preds)

        return all_labels, all_scores

    def _compute_embeddings(self) -> None:
        """Compute and cache embeddings for all graph snapshots."""
        import time
        print("[TADDY] Computing embeddings for all snapshots...", flush=True)
        start_time = time.time()
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = (
            self.components.model.generate_embedding(self.data_dict["edges"])
        )
        self.embeddings = {
            "raw": raw_embeddings,
            "wl": wl_embeddings,
            "hop": hop_embeddings,
            "int": int_embeddings,
            "time": time_embeddings,
        }
        elapsed_time = time.time() - start_time
        print(
            f"[TADDY] Embeddings computed in {elapsed_time:.2f}s", flush=True)

    def _normalize(self, mx: sp.spmatrix) -> sp.spmatrix:
        """Row-normalize sparse matrix."""
        rowsum = np.array(mx.sum(1))
        with np.errstate(divide='ignore', invalid='ignore'):
            r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def _normalize_adj(self, adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        with np.errstate(divide='ignore', invalid='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _adj_normalize(self, mx: sp.spmatrix) -> sp.spmatrix:
        """Row-normalize sparse matrix with sqrt."""
        rowsum = np.array(mx.sum(1))
        with np.errstate(divide='ignore', invalid='ignore'):
            r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx: sp.spmatrix) -> torch.Tensor:
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float, device=self.device)

    def _preprocess_adj(self, adj: sp.spmatrix) -> torch.Tensor:
        """Preprocessing of adjacency matrix for GCN and conversion to torch tensor."""
        adj = adj + adj.T.multiply(adj < adj.T) - adj.multiply(adj < adj.T)
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = self._sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized

    def _get_adjs_old(
        self,
        rows: list[np.ndarray],
        cols: list[np.ndarray],
        weights: list[np.ndarray],
        nb_nodes: int,
        dataset_name: str,
        dataset_id: str,
        # train_ratio: float,
        # val_ratio: float,
        # anom_val_ratio: float,
        snap_size: int,
    ) -> tuple[list[torch.Tensor], list[np.ndarray | None]]:
        """Build adjacency matrices and optionally compute/load eigen adjacencies."""
        current_cache_dir = self.cache_dir / Path(dataset_name)
        current_cache_dir.mkdir(parents=True, exist_ok=True)
        eigen_file_name = current_cache_dir / f"{dataset_id}_s{snap_size}.pkl"

        if not os.path.exists(eigen_file_name):
            generate_eigen = True
            logger.info(f"Generating eigen as: {eigen_file_name}")
        else:
            generate_eigen = False
            logger.info(f"Loading eigen from: {eigen_file_name}")
            with open(eigen_file_name, "rb") as f:
                eigen_adjs_sparse = pickle.load(f)
            eigen_adjs = []
            for eigen_adj_sparse in eigen_adjs_sparse:
                eigen_adjs.append(np.array(eigen_adj_sparse.todense()))

        adjs = []
        if generate_eigen:
            eigen_adjs = []
            eigen_adjs_sparse = []

        total_snapshots = len(rows)
        if generate_eigen:
            # print(
            #     f"[TADDY] Computing eigen-adjacencies for {total_snapshots} snapshots...", flush=True)

            for i in tqdm(range(total_snapshots), desc="[TADDY] Computing eigen-adjacencies"):
                adj = sp.csr_matrix(
                    (weights[i], (rows[i], cols[i])),
                    shape=(nb_nodes, nb_nodes),
                    dtype=np.float32
                )
                adjs.append(self._preprocess_adj(adj))

                if generate_eigen:
                    from numpy.linalg import inv
                    eigen_adj = self.c * inv(
                        (sp.eye(adj.shape[0]) - (1 - self.c)
                        * self._adj_normalize(adj)).toarray()
                    )
                    for p in range(adj.shape[0]):
                        eigen_adj[p, p] = 0.0
                    eigen_adj = self._normalize(eigen_adj)
                    eigen_adjs.append(eigen_adj)
                    eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

        # if generate_eigen:
            with open(eigen_file_name, "wb") as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

        return adjs, eigen_adjs

    def _get_adjs(self, rows, cols, weights, nb_nodes, dataset_name, dataset_id):
        current_cache_dir = self.cache_dir / Path(dataset_name)
        current_cache_dir.mkdir(parents=True, exist_ok=True)
        eigen_file_name = current_cache_dir / f"{dataset_id}_s{len(rows)}.pkl"

        # Precompute identity once
        I = sp.eye(nb_nodes, format="csr", dtype=np.float32)

        eigen_adjs = None
        eigen_adjs_sparse = None
        generate_eigen = False

        if os.path.exists(eigen_file_name):
            logger.info(f"Loading eigen from: {eigen_file_name}")
            with open(eigen_file_name, "rb") as f:
                eigen_adjs_sparse = pickle.load(f)

            eigen_adjs = [ea_sparse.toarray() for ea_sparse in eigen_adjs_sparse]
        else:
            generate_eigen = True
            logger.info(f"Generating eigen as: {eigen_file_name}")
            eigen_adjs = []
            eigen_adjs_sparse = []

        adjs = []
        for i in tqdm(range(len(rows))):
            # Fast sparse construction
            # COO is faster to build; convert to CSR for subsequent ops.
            adj = sp.coo_matrix(
                (weights[i], (rows[i], cols[i])),
                shape=(nb_nodes, nb_nodes),
                dtype=np.float32
            ).tocsr()

            adjs.append(self._preprocess_adj(adj))

            if generate_eigen:
                # Original:
                # eigen_adj = c * inv((I - (1-c) * adj_normalize(adj)).toarray())
                # Replace with sparse LU solve:
                #
                # M = I - (1-c) * adj_normalize(adj)   (sparse)
                # eigen_adj = c * M^{-1}               (dense output)
                #
                # This avoids forming M as dense and avoids np.linalg.inv dense.
                adj_norm = self._adj_normalize(adj).astype(np.float32)   # sparse
                M = (I - (1.0 - self.c) * adj_norm).tocsc()             # sparse CSC for LU

                # Sparse LU factorization and solve for identity (gives dense inverse)
                lu = splu(M)
                eigen_adj = self.c * lu.solve(np.eye(nb_nodes, dtype=np.float32))  # dense

                np.fill_diagonal(eigen_adj, 0.0)

                # Row-normalize
                eigen_adj = self._normalize(eigen_adj)

                eigen_adjs.append(eigen_adj)
                eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

        # Save
        if generate_eigen:
            with open(eigen_file_name, "wb") as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

        return adjs, eigen_adjs

    def save(self, save_dir: str) -> None:
        """Save model checkpoint and configuration."""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.components.model.state_dict(),
            'optimizer_state_dict': self.components.optimizer.state_dict(),
            'embeddings': self.embeddings,
            'data_dict': self.data_dict,
        }

        config = {
            'batch_size': self.batch_size,
            'num_neighbors': self.num_neighbors,
            'num_heads': self.num_heads,
            'drop_out': self.drop_out,
            'num_layer': self.num_layer,
            'learning_rate': self.learning_rate,
            'message_dim': self.message_dim,
            'memory_dim': self.memory_dim,
            'weight_decay': self.weight_decay,
            'c': self.c,
            'window_size': self.window_size,
            'snap_size': self.snap_size,
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

        # Restore data structures
        instance.embeddings = checkpoint['embeddings']
        instance.data_dict = checkpoint['data_dict']

        # Recreate model components
        my_config = MyConfig(
            k=instance.num_neighbors,
            window_size=instance.window_size,
            hidden_size=instance.message_dim,
            intermediate_size=instance.message_dim,
            num_attention_heads=instance.num_heads,
            num_hidden_layers=instance.num_layer,
            weight_decay=instance.weight_decay,
        )

        class ConfigWrapper:
            def __init__(self, device):
                self.device = device

        wrapper = ConfigWrapper(device)
        taddy_model = DynADModel(my_config, wrapper)
        taddy_model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(
            params=taddy_model.parameters(),
            lr=instance.learning_rate,
            weight_decay=instance.weight_decay,
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        instance._components = TADDYADComponents(
            model=taddy_model,
            optimizer=optimizer
        ).to(device)

        logger.info(f"Model loaded from {load_dir}")
        return instance
