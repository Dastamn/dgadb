import logging
import os
import pickle
import time
from collections.abc import Callable

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn.functional as fun
from numpy.linalg import inv
from tqdm import tqdm

from dgadb.models.TADDY.codes.Component import MyConfig
from dgadb.models.TADDY.codes.DynADModel import DynADModel

from dgadb.storage import TemporalGraph
from dgadb.storage import TemporalGraphSnapshotLoader

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
        meta_dict: dict[str, str | int | float],
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
        logger.info(
            f"Initializing TADDYModel with device={self.device} and hyperparams={hyperparams}")

        self.batch_size = hyperparams.get("batch_size", 100)
        self.num_neighbors = hyperparams.get("num_neighbors", 5)
        self.num_epoch = hyperparams.get("num_epoch", 20)
        self.num_heads = hyperparams.get("num_heads", 2)
        self.drop_out = hyperparams.get("drop_out", 0.1)
        self.gpu = hyperparams.get("gpu", 0)
        self.num_layer = hyperparams.get("num_layer", 1)
        self.learning_rate = hyperparams.get("learning_rate", 0.001)
        self.message_dim = hyperparams.get("message_dim", 128)
        self.memory_dim = hyperparams.get("memory_dim", 256)
        self.lr_decay = hyperparams.get("lr_decay", 0.8)
        self.weight_decay = hyperparams.get("weight_decay", 5e-4)
        self.print_freq = hyperparams.get("print_freq", 10)
        self.print_per_snap = hyperparams.get("print_per_snap", False)
        self.snap_size = hyperparams.get("snap_size", 2000)
        self.window_size: int | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.method_obj: DynADModel | None = None
        self.data_dict: dict | None = None

        # model specific
        self.c = hyperparams.get("c", 0.15)

        self.dataset_name = meta_dict["dataset_name"]
        self.train_per = meta_dict["train_ratio"]
        self.anomaly_per = meta_dict["anom_val_ratio"]
        self.val_per = meta_dict["val_ratio"]
        self.has_val = True if self.val_per > 0 else False

        self.compute_s = True
        self.window_size = 3

        # initialize containers for embeddings
        self.embeddings: dict[str, np.ndarray] = {}

    def setup(self, temporal_graph: TemporalGraph) -> None:
        logger.info("Setup started...")

        train_mask = temporal_graph.train_mask
        train_src = temporal_graph.src[train_mask]
        train_tgt = temporal_graph.tgt[train_mask]

        num_nodes = temporal_graph.num_nodes

        data = np.ones_like(train_src, dtype=np.int32)
        train_adj = sp.csr_matrix(
            (data, (train_src, train_tgt)), shape=(num_nodes, num_nodes))
        train_adj = train_adj + sp.eye(num_nodes)
        train_adj_lil = train_adj.tolil()
        headtail = train_adj_lil.rows

        rows, cols, labs, weis = [], [], [], []
        edges = []
        # snap_loader = TemporalGraphSnapshotLoader(
        #     temporal_graph, window_size=self.snap_size
        # )
        # for snap in snap_loader:
        #     curr = snap.current
        #     rows.append(curr.src)
        #     cols.append(curr.tgt)
        #     labs.append(curr.edge_labels)
        #     weis.append(curr.w)

        train_snap_loader = TemporalGraphSnapshotLoader(
            temporal_graph, window_size=self.snap_size, split="train")
        for train_snap in train_snap_loader:
            train_curr = train_snap.current
            rows.append(train_curr.src)
            cols.append(train_curr.tgt)
            labs.append(train_curr.edge_labels)
            # weis.append(train_curr.w)
            weis.append(torch.ones_like(train_curr.edge_labels))
            edges.append(torch.vstack(
                [train_curr.src, train_curr.tgt]).T.numpy())

        val_snap_loader = TemporalGraphSnapshotLoader(
            temporal_graph, window_size=self.snap_size, split="val")
        for val_snap in val_snap_loader:
            val_curr = val_snap.current
            rows.append(val_curr.src)
            cols.append(val_curr.tgt)
            labs.append(val_curr.edge_labels)
            # weis.append(val_curr.w)
            weis.append(torch.ones_like(val_curr.edge_labels))
            edges.append(torch.vstack([val_curr.src, val_curr.tgt]).T.numpy())

        test_snap_loader = TemporalGraphSnapshotLoader(
            temporal_graph, window_size=self.snap_size, split="test")
        for test_snap in test_snap_loader:
            test_curr = test_snap.current
            rows.append(test_curr.src)
            cols.append(test_curr.tgt)
            labs.append(test_curr.edge_labels)
            # weis.append(test_curr.w)
            weis.append(torch.ones_like(test_curr.edge_labels))
            edges.append(torch.vstack(
                [test_curr.src, test_curr.tgt]).T.numpy())

        train_snap_size = len(train_snap_loader)
        val_snap_size = len(val_snap_loader)
        test_snap_size = len(test_snap_loader)

        degrees = np.array([len(x) for x in headtail])

        adjs, eigen_adjs = self._get_adjs(rows, cols, weis, num_nodes)

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
            "snap_val": list(range(num_snap))[train_snap_size:(train_snap_size+val_snap_size)],
            "degrees": degrees,
            "snap_test": list(range(num_snap))[(train_snap_size+val_snap_size):],
            "num_snap": train_snap_size + val_snap_size + test_snap_size,
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
            params=self.method_obj.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def setup_(self, df: pl.DataFrame) -> None:
        """Set up data preprocessing and initialize the model.

        Processes the input graph, builds adjacency matrices, and initializes the TADDY model with the specified
        configuration.

        Args:
            g (Graph): Input dynamic graph object containing data which has been already split into
                train/test and snapshots.

        """
        logger.info("Setup started...")

        train_df = df.filter(pl.col("train_mask"))
        test_df = df.filter(pl.col("test_mask"))

        all_nodes = np.unique(np.concatenate(
            [df["src"].to_numpy(), df["tgt"].to_numpy()]))
        n = len(all_nodes)

        # build adj
        row = train_df["src"].to_numpy()
        col = train_df["tgt"].to_numpy()
        data = np.ones_like(row, dtype=np.int32)
        train_mat = sp.csr_matrix((data, (row, col)), shape=(n, n))
        # make undirected and add selfloops
        train_mat = train_mat + train_mat.transpose() + sp.eye(n)
        train_mat = train_mat.tolil()
        # Get headtail: list of neighbors for each node
        headtail = train_mat.rows

        snapshot_ids = df["snapshot_id"].unique().to_list()
        train_size = df.filter(pl.col("train_mask")).select(
            "snapshot_id").unique().height
        val_size = df.filter(pl.col("val_mask")).select(
            "snapshot_id").unique().height if self.has_val else 0
        test_size = df.filter(pl.col("test_mask")).select(
            "snapshot_id").unique().height

        rows, cols, labs, weis = [], [], [], []

        for snap_id in snapshot_ids:
            snap_df = df.filter(pl.col("snapshot_id") == snap_id)
            row = snap_df["src"].to_numpy().astype(np.int32)
            col = snap_df["tgt"].to_numpy().astype(np.int32)
            label = snap_df["label"].to_numpy().astype(np.int32)
            weight = np.ones_like(row, dtype=np.int32)
            rows.append(row)
            cols.append(col)
            labs.append(label)
            weis.append(weight)

        logger.debug(f"Length of snapshot IDs: {len(snapshot_ids)}")
        logger.debug(
            f"Number of snapshots {(test_size + train_size + val_size)}")
        val_labels = labs[train_size:(
            train_size + val_size)] if self.has_val else []
        test_labels = labs[(train_size + val_size):]
        n_test_edges = sum(len(lbl) for lbl in test_labels)
        n_anomalies = sum((lbl == 1).sum().item() for lbl in test_labels)
        logger.debug(
            f"Test edges: {n_test_edges}, Anomalies: {n_anomalies}, Ratio: {n_anomalies / n_test_edges:.4f}")

        degrees = np.array([len(x) for x in headtail])
        num_snap = test_size + train_size + val_size

        edges = [np.vstack((rows[i], cols[i])).T for i in range(
            num_snap)]  # [(2, num_edges), ...]

        # print(len(edges))
        # print(edges[0].shape)
        # return

        adjs, eigen_adjs = self._get_adjs(rows, cols, weis, n)

        labs = [torch.LongTensor(label) for label in labs]

        snap_train = list(range(num_snap))[:train_size]
        snap_val = list(range(num_snap))[train_size:(train_size+val_size)]
        snap_test = list(range(num_snap))[(train_size+val_size):]

        idx = list(range(n))
        index_id_map = {i: i for i in idx}
        idx = np.array(idx)

        # pack it all up
        self.data_dict = {
            "X": None,
            "A": adjs,
            "S": eigen_adjs,
            "index_id_map": index_id_map,
            "edges": edges,
            "y": labs,
            "idx": idx,
            "snap_train": snap_train,
            "snap_val": snap_val,
            "degrees": degrees,
            "snap_test": snap_test,
            # "num_snap": num_snap,
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
            params=self.method_obj.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _normalize(self, mx: sp.spmatrix) -> sp.spmatrix:
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def _normalize_adj(self, adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix. (0226)"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _adj_normalize(self, mx: sp.spmatrix) -> sp.spmatrix:
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx: sp.spmatrix) -> torch.Tensor:
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)

    def _encode_onehot(self, labels: list[int] | np.ndarray) -> np.ndarray:
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[
            i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(
            list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def _sparse_to_tuple(self, sparse_mx: sp.spmatrix | list[sp.spmatrix]) -> tuple | list[tuple]:
        """Convert sparse matrix to tuple representation. (0226)"""

        def _to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = _to_tuple(sparse_mx[i])
        else:
            sparse_mx = _to_tuple(sparse_mx)

        return sparse_mx

    def _preprocess_adj(self, adj: sp.spmatrix) -> torch.Tensor:
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
        adj = adj + adj.T.multiply(adj < adj.T) - adj.multiply(adj < adj.T)
        # adj_np = np.array(adj.todense())
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = self._sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized

    def _get_adjs(
        self,
        rows: list[np.ndarray],
        cols: list[np.ndarray],
        weights: list[np.ndarray],
        nb_nodes: int,
    ) -> tuple[list[torch.Tensor], list[np.ndarray | None]]:
        base_path = os.environ["BASE_PATH"]
        eigen_file_name = (
            "src/dgadb/models/TADDY/data/eigen/"
            + self.dataset_name
            + "_t"
            + str(self.train_per)
            + "_v"
            + str(self.val_per)
            + "_a"
            + str(self.anomaly_per)
            + ".pkl"
        )
        full_eigen_path = os.path.join(base_path, eigen_file_name)
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(full_eigen_path), exist_ok=True)
        if not os.path.exists(full_eigen_path):
            generate_eigen = True
            logger.info(f"Generating eigen as: {eigen_file_name}")
        else:
            generate_eigen = False
            logger.info(f"Loading eigen from: {eigen_file_name}")
            with open(full_eigen_path, "rb") as f:
                eigen_adjs_sparse = pickle.load(f)
            eigen_adjs = []
            for eigen_adj_sparse in eigen_adjs_sparse:
                eigen_adjs.append(np.array(eigen_adj_sparse.todense()))

        adjs = []
        if generate_eigen:
            eigen_adjs = []
            eigen_adjs_sparse = []

        for i in tqdm(range(len(rows)), desc="_get_adjs"):
            adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(
                nb_nodes, nb_nodes), dtype=np.float32)
            adjs.append(self._preprocess_adj(adj))
            if self.compute_s:
                if generate_eigen:
                    eigen_adj = self.c * \
                        inv((sp.eye(adj.shape[0]) - (1 - self.c)
                            * self._adj_normalize(adj)).toarray())
                    for p in range(adj.shape[0]):
                        eigen_adj[p, p] = 0.0
                    eigen_adj = self._normalize(eigen_adj)
                    eigen_adjs.append(eigen_adj)
                    eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

            else:
                eigen_adjs.append(None)

        if generate_eigen:
            with open(full_eigen_path, "wb") as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

        return adjs, eigen_adjs

    def _compute_embeddings(self) -> None:
        """Compute and cache embeddings for all graph snapshots."""
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
            raise RuntimeError(
                "Model not properly initialized. Call setup() before train() or inference().")

    def train(self, runnable) -> None:
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
                self.data_dict["edges"][: max(
                    self.data_dict["snap_train"]) + 1],
            )

            _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.method_obj.generate_embedding(
                negatives,
            )

            self.method_obj.train()
            loss_train = 0

            for snap in tqdm(
                self.data_dict["snap_train"],
                desc=f"Going through snapshots in epoch {epoch}",
                leave=True,
            ):
                # print("snap:", snap)
                # print(len(self.data_dict["edges"][snap]))
                if self.embeddings["wl"][snap] is None:
                    continue

                # positive samples
                int_embedding_pos = self.embeddings["int"][snap]
                hop_embedding_pos = self.embeddings["hop"][snap]
                time_embedding_pos = self.embeddings["time"][snap]
                y_pos = self.data_dict["y"][snap]

                # negative samples
                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                # print(f"{int_embedding_pos.shape=}")
                # print(f"{hop_embedding_pos.shape=}")
                # print(f"{time_embedding_pos.shape=}")

                # print(f"{int_embedding_neg.shape=}")
                # print(f"{hop_embedding_neg.shape=}")
                # print(f"{time_embedding_neg.shape=}")
                # print(f"{y_neg.shape=}")

                # combine positive and negative
                int_embedding = torch.vstack(
                    (int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack(
                    (hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack(
                    (time_embedding_pos, time_embedding_neg))

                # print(f"{int_embedding.shape=}")
                # print(f"{hop_embedding.shape=}")
                # print(f"{time_embedding.shape=}")

                y = torch.hstack((y_pos, y_neg))

                self.optimizer.zero_grad()

                output = self.method_obj.forward(
                    int_embedding, hop_embedding, time_embedding).squeeze()

                loss = fun.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                self.optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data_dict["snap_train"]) - \
                self.method_obj.config.window_size + 1
            logger.info(
                f"Epoch: {epoch + 1}, loss:{loss_train:.4f}, Time: {time.time() - t_epoch_begin:.4f}s")
            split = "val" if self.has_val else "test"

            if (((epoch + 1) % self.print_freq) == 0) or runnable:
                # _, _, preds_full, labels_full, _ = self.inference(
                #     split=split)  # do val here when implemented
                preds_full, labels_full = self.inference(
                    split=split)  # do val here when implemented
                auc_full = self.epoch_evaluation_metric(
                    labels_full, preds_full)
                logger.info(f"Total auc on {split}: {auc_full:.4f}")

                runnable(auc_full, self, epoch)

    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run inference on the specified data split and return predictions, labels, and inference time.

        Args:
            split (str): Which data split to use for inference ("test" or "train"). Default is "test".

        Returns:
            tuple:
                - preds_full (np.ndarray): Model predictions for the entire split.
                - labels_full (np.ndarray): Ground truth labels for the entire split.
                - inf_time (float): Total inference time in seconds.

        Notes:
            - Evaluates the model on each snapshot in the specified split.
            - If `self.print_per_snap` is True, logs per-snapshot evaluation metrics.
        """
        self._ensure_setup()
        start_time = time.time()
        self.method_obj.eval()

        preds = []
        labels = []
        snap_ids = self.data_dict["snap_" + split]

        for snap in snap_ids:
            int_embedding = self.embeddings["int"][snap]
            hop_embedding = self.embeddings["hop"][snap]
            time_embedding = self.embeddings["time"][snap]
            with torch.no_grad():
                output = self.method_obj.forward(
                    int_embedding, hop_embedding, time_embedding, None)
                output = torch.sigmoid(output)
            pred = output.squeeze().cpu().numpy()
            preds.append(pred)
            labels.append(self.data_dict["y"][snap].cpu().numpy())

        if self.print_per_snap:
            # Per-snapshot AUCs
            aucs = []
            for i in range(len(snap_ids)):
                if len(np.unique(labels[i])) > 1:
                    auc = self.epoch_evaluation_metric(labels[i], preds[i])
                else:
                    auc = float("nan")
                aucs.append(auc)
                logger.info(f"Snap: {snap_ids[i]:02d} | Score: {auc:.4f}")

        # Total AUC
        labels_full = np.hstack(labels)
        preds_full = np.hstack(preds)

        inf_time = time.time() - start_time
        return preds_full, labels_full
        return preds, labels, preds_full, labels_full, inf_time


# TODO @Tobias: add support for validation split
