import logging
import time
from collections.abc import Callable
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as fun

from src.dgadb.models.TADDY.codes.Component import MyConfig
from src.dgadb.models.TADDY.codes.DynADModel import DynADModel
from src.dgadb.storage.graph import Graph

logger = logging.getLogger(__name__)


class TADDYModel:
    """TADDY Graph Transformer model wrapper for DGADB.

    This class wraps the TADDY model to provide a standardized interface for benchmarking pipelines with init, setup,
    train, and inference methods.

    Attributes:
        device (torch.device): Device to run the model on.
        epoch_evaluation_metric (callable): Function to evaluate model performance during training.
        embeddings (dict): Cached embeddings for efficient inference.
        data_dict (dict): Processed graph data and metadata.
        method_obj (DynADModel): The underlying TADDY model instance.
        optimizer (torch.optim.Optimizer): Optimizer for training.

    Args:
        device (torch.device): Device to run computations on.
        hyperparams (dict): Dictionary containing model hyperparameters.
        epoch_evaluation_metric (callable): Function that takes (labels, predictions)
            and returns a scalar metric.

    """

    def __init__(
        self,
        device: torch.DeviceObjType,
        hyperparams: dict[str, int | float],
        epoch_evaluation_metric: Callable[[torch.FloatTensor, torch.FloatTensor], float],
    ) -> None:
        """Initialize TADDYModel with device, hyperparameters, and evaluation metric.

        Sets up the model configuration and stores hyperparameters for later use.

        Args:
            device (torch.device): Device to run computations on.
            hyperparams (dict): Dictionary containing model hyperparameters such as
                batch_size, num_neighbors, num_epoch, learning_rate, etc.
            epoch_evaluation_metric (callable): Function that takes (labels, predictions)
                and returns a scalar metric for training epoch evaluation.

        """
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        logger.info(f"Initializing TADDYModel with device={self.device} and hyperparams={hyperparams}")

        self.batch_size = hyperparams.get("batch_size", 100)
        self.num_neighbors = hyperparams.get("num_neighbors", 5)
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
        self.window_size: int | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.method_obj: DynADModel | None = None
        self.data_dict: dict | None = None

        # model specific
        self.c = hyperparams.get("c", 0.15)

        # initialize containers for embeddings
        self.embeddings = {}

    def setup(self, g: Graph) -> None:
        """Set up data preprocessing and initialize the model.

        Processes the input graph, builds adjacency matrices, and initializes the TADDY model with the specified
        configuration.

        Args:
            g (Graph): Input dynamic graph object containing data which has been already split into
                train/test and snapshots.

        """
        logger.info(f"Setup started...")

        n_nodes = g.num_nodes
        #self.window_size = g.window_size
        self.window_size = 3
        logger.info(f"Starting the processing of snapshots with window_size: {self.window_size} and num_snapshots: {g.num_snapshots}")

        # per-snapshot edge pairs and labels
        rows = []
        cols = []
        weights = []
        labels = []
        edges = []
        degrees = np.zeros(n_nodes, dtype=np.int32)
        
        for snap in tqdm(g.snapshots(all_nodes=True), desc="Processing snapshots", leave=True, total=g.num_snapshots):
            this_e_pairs = snap.e_pairs
            this_edges = this_e_pairs.T.cpu().numpy()
            rows.append(this_e_pairs[0])
            cols.append(this_e_pairs[1])

            # add weights if present otherwise just do 1s
            if hasattr(snap, "e_weight"):
                weights.append(snap.e_weight.cpu().numpy())
            else:
                weights.append(np.ones(this_edges.shape[0], dtype=np.float32))
            edges.append(this_edges)

            # add labels if present
            if hasattr(snap, "e_label"):
                labels.append(snap.e_label.cpu().numpy())
            else:
                labels.append(snap.n_label)

            # update degrees
            for n in this_e_pairs.flatten():
                degrees[n] += 1

        # build idx and index_id_map CHECK THIS
        #idx = g.n_id.cpu().numpy()
        idx = g.n_id.cpu().numpy()
        index_id_map = {i: i for i in idx}

        # get train/test split
        train_size = len(torch.unique(g.e_snapshot_id[g.e_train_mask]))
        snap_train = list(range(train_size))
        snap_test = list(range(train_size, g.num_snapshots))

        # Process adjacency matrices
        adjs, eigen_adjs = self._build_adjacencies(rows, cols, weights, n_nodes)

        # pack it all up
        self.data_dict = {
            "X": g.n_feat.cpu().numpy(),
            "A": adjs,
            "S": eigen_adjs,
            "index_id_map": index_id_map,
            "edges": edges,
            "y": labels,
            "idx": idx,
            "snap_train": snap_train,
            "degrees": degrees,
            "snap_test": snap_test,
            "num_snap": g.num_snapshots,
        }

        # prepare model
        my_config = MyConfig(
            k=self.num_neighbors,
            window_size=self.window_size,
            hidden_size=self.message_dim,
            intermediate_size=self.message_dim,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layer,
            weight_decay=self.weight_decay,
        )

        self.method_obj = DynADModel(my_config, self)
        self.method_obj.data = self.data_dict
        self.method_obj.spy_tag = True
        self.method_obj.max_epoch = self.num_epoch
        self.method_obj.lr = self.learning_rate

        self.optimizer = torch.optim.Adam(
            params=self.method_obj.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _build_adjacencies(
        self,
        rows: list[np.ndarray],
        cols: list[np.ndarray],
        weights: list[np.ndarray],
        n_nodes: int,
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        """Build and preprocess adjacency matrices for all graph snapshots."""

        def _preprocess_adj(adj: sp.csr_matrix) -> torch.Tensor:
            # add selfloop, symmetric normalize, torch sparse tensor
            adj = adj + adj.T.multiply(adj < adj.T) - adj.multiply(adj < adj.T)
            adj = adj + sp.eye(adj.shape[0])
            # symmetric normalization
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv_sqrt = np.power(rowsum, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
            # to torch sparse tensor
            indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64))
            values = torch.from_numpy(adj_normalized.data)
            shape = torch.Size(adj_normalized.shape)
            return torch.sparse_coo_tensor(indices, values, shape)

        adjs = []
        eigen_adjs = []

        for i in tqdm(range(len(rows)), desc="Building adjacencies", leave=True):
            adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(n_nodes, n_nodes), dtype=np.float32)
            adjs.append(_preprocess_adj(adj))
            eigen_adj = self.c * np.linalg.inv(np.eye(adj.shape[0]) - (1 - self.c) * adj.toarray())
            np.fill_diagonal(eigen_adj, 0.0)
            # row normalize
            rowsum = np.array(eigen_adj.sum(1)).flatten()
            r_inv = np.power(rowsum, -1)
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = np.diag(r_inv)
            eigen_adj = r_mat_inv @ eigen_adj
            eigen_adjs.append(eigen_adj)

        return adjs, eigen_adjs

    def _compute_embeddings(self) -> None:
        """Compute and cache embeddings for all graph snapshots."""
        logger.info("Computing embeddings...")
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = (
            self.method_obj.generate_embedding(
                self.data_dict["edges"],
            )
        )
        self.embeddings = {
            "raw": raw_embeddings,
            "wl": wl_embeddings,
            "hop": hop_embeddings,
            "int": int_embeddings,
            "time": time_embeddings,
        }

    def _ensure_setup(self) -> None:
        """Ensure setup() has been called."""
        if any(x is None for x in [self.optimizer, self.method_obj, self.data_dict, self.window_size]):
            raise RuntimeError("Model not properly initialized. Call setup() before train() or inference().")

    def train(self) -> None:
        """Train the TADDY model using the configured data and hyperparameters.

        Performs embedding computation, negative sampling, and iterative training
        with evaluation at each epoch.
        """
        self._ensure_setup()
        logger.info(f"Starting training for {self.num_epoch} epochs...")

        self._compute_embeddings()
        self.data_dict["raw_embeddings"] = None
        
        for epoch in range(self.num_epoch):
            t_epoch_begin = time.time()

            # -------------------------
            negatives = self.method_obj.negative_sampling(
                self.data_dict["edges"][: max(self.data_dict["snap_train"]) + 1]
            )
            
            _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.method_obj.generate_embedding(
                negatives
            )

            self.method_obj.train()
            loss_train = 0

            for snap in tqdm(self.data_dict["snap_train"], desc=f"Going through snapshots in epoch {epoch}", leave=True):
                if self.embeddings["wl"][snap] is None:
                    continue

                # positive samples
                int_embedding_pos = self.embeddings["int"][snap]
                hop_embedding_pos = self.embeddings["hop"][snap]
                time_embedding_pos = self.embeddings["time"][snap]
                y_pos = self.data_dict["y"][snap].float()

                # negative samples
                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                # combine positive and negative
                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
                y = torch.hstack((y_pos, y_neg))

                self.optimizer.zero_grad()

                output = self.method_obj.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                loss = fun.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                self.optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data_dict["snap_train"]) - self.method_obj.config.window_size + 1
            logger.info(f"Epoch: {epoch + 1}, loss:{loss_train:.4f}, Time: {time.time() - t_epoch_begin:.4f}s")

            preds_full, labels_full, _ = self.inference(split="test")  # do val here when implemented
            auc_full = self.epoch_evaluation_metric(labels_full, preds_full)
            logger.info(f"TOTAL AUC:{auc_full:.4f}")

    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, float]:
        """Run inference on the specified data split.

        Uses pre-computed embeddings to generate predictions for the test set
        and returns both predictions and ground truth labels.

        Args:
            split (str, optional): Data split to run inference on.
                Currently supports "test". Defaults to "test".

        Returns:
            tuple: A tuple containing:
                - preds_full (np.ndarray): Predicted probabilities
                - labels_full (np.ndarray): Ground truth labels
                - inf_time (float): Inference time in seconds

        Raises:
            ValueError: If the specified split is not supported.

        """
        self._ensure_setup()

        inf_start = time.time()
        self.method_obj.eval()

        preds = []
        split_map = {  # "val": self.data_dict["snap_val"], not implemented yet
            "test": self.data_dict["snap_test"]
        }
        if split not in split_map:
            error_message = f"Split '{split}' not supported. Available: {list(split_map.keys())}"
            logger.warning(error_message)
            raise ValueError(error_message)

        for snap in split_map[split]:
            int_embedding = self.embeddings["int"][snap]
            hop_embedding = self.embeddings["hop"][snap]
            time_embedding = self.embeddings["time"][snap]
            with torch.no_grad():
                output = self.method_obj.forward(int_embedding, hop_embedding, time_embedding, None)
                output = torch.sigmoid(output)
            pred = output.squeeze().numpy()
            preds.append(pred)

        labels = self.data_dict["y"][min(split_map[split]) : max(split_map[split]) + 1]
        labels = [y_snap.numpy() for y_snap in labels]

        labels_full = np.hstack(labels)
        preds_full = np.hstack(preds)
        inf_time = time.time() - inf_start

        return preds_full, labels_full, inf_time


# TODO @Tobias: add support for validation split
