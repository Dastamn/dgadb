import torch
import numpy as np
from dgadb.storage.temporal_graph import TemporalGraph
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import pickle
from dgadb.models.StrGNN.pytorch_DGCNN.main import *
from dgadb.models.StrGNN.detection.util_functions import *
from os import path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        self.sortpooling_k = hyperparams.get(
            "sortpooling_k", 30)  # was 0.6 in og
        self.conv1d_activation = hyperparams.get("conv1d_activation", "relu")
        self.hidden = hyperparams.get("hidden", 50)
        self.num_class = hyperparams.get("num_class", 2)
        self.dropout = hyperparams.get("dropout", 0.5)
        self.regression = hyperparams.get("regression", False)
        self.gm = hyperparams.get("gm", "DGCNN")
        self.window_size = hyperparams.get("window_size", 5)

    def setup(self, temporal_graph: TemporalGraph) -> None:
        logger.info("STRGNN setup started with Negative Sampling...")

        num_nodes = temporal_graph.num_nodes
        src = temporal_graph.src.cpu().numpy()
        tgt = temporal_graph.tgt.cpu().numpy()
        timestamps = temporal_graph.t.cpu().numpy()

        # We assume all edges in the graph provided are positive (Normal)
        train_mask = temporal_graph.train_mask.cpu().numpy()
        test_mask = temporal_graph.test_mask.cpu().numpy()

        # unique_timestamps = np.unique(timestamps)
        # snapshot_mapping = {ts: i for i, ts in enumerate(unique_timestamps)}
        # snapshot_ids = np.array([snapshot_mapping[ts] for ts in timestamps])
        # num_snapshots = len(unique_timestamps)
        # print("num snapshots", num_snapshots)

        # 2. Define Fixed Window Size (e.g., 1000 edges per snapshot)
        edges_per_snapshot = 2000
        num_total_edges = len(src)
        # Calculate snapshot ID for every edge: 0, 0, ... (1000 times), 1, 1, ...
        snapshot_ids = np.floor(
            np.arange(num_total_edges) / edges_per_snapshot).astype(int)
        num_snapshots = int(snapshot_ids.max() + 1)

        logger.info(
            f"Created {num_snapshots} snapshots with {edges_per_snapshot} edges each.")

        import sys
        # sys.exit(0)

        # 1. Create adjacency matrices for each snapshot
        net = []
        A_train_all = ssp.csr_matrix((num_nodes, num_nodes))
        max_train_snap = snapshot_ids[train_mask].max() if any(
            train_mask) else 0

        for snap_id in range(num_snapshots):
            snap_mask = snapshot_ids == snap_id
            s, t = src[snap_mask], tgt[snap_mask]

            if len(s) > 0:
                A_snap = ssp.csr_matrix(
                    (np.ones_like(s), (s, t)), shape=(num_nodes, num_nodes))
                A_snap = A_snap + A_snap.transpose()
                A_snap.setdiag(0)
                if snap_id <= max_train_snap:
                    A_train_all = A_train_all + A_snap
            else:
                A_snap = ssp.csr_matrix((num_nodes, num_nodes))
            net.append(A_snap)

        # 2. Node2Vec embeddings
        node_information = None
        if self.use_embedding:
            logger.info("Generating Node2Vec embeddings...")
            node_information = generate_node2vec_embeddings(A_train_all, 128)

        # 3. Separate Positive Train Edges
        train_pos_src = src[train_mask]
        train_pos_tgt = tgt[train_mask]
        train_pos_ids = snapshot_ids[train_mask]

        # 4. NEGATIVE SAMPLING (Crucial step)
        logger.info("Sampling negative training links...")
        train_neg_src = []
        train_neg_tgt = []
        train_neg_ids = []

        # For every positive edge at snapshot T, find a pair (u, v) that is NOT connected at T
        for i in range(len(train_pos_ids)):
            snap_id = train_pos_ids[i]
            A_snap = net[snap_id]

            # Keep trying random pairs until we find a non-edge
            while True:
                u = np.random.randint(0, num_nodes)
                v = np.random.randint(0, num_nodes)
                if u != v and A_snap[u, v] == 0:
                    train_neg_src.append(u)
                    train_neg_tgt.append(v)
                    train_neg_ids.append(snap_id)
                    break

        # Convert to numpy for consistency
        train_neg_src = np.array(train_neg_src)
        train_neg_tgt = np.array(train_neg_tgt)
        train_neg_ids = np.array(train_neg_ids)

        # 2. Node2Vec embeddings
        node_information = None
        if self.use_embedding:
            logger.info("Generating Node2Vec embeddings...")
            node_information = generate_node2vec_embeddings(
                A_train_all, 128, True, train_neg=(train_neg_src, train_neg_tgt))

        # 5. Store in data_dict
        self.data_dict["train_pos"] = (train_pos_src, train_pos_tgt)
        self.data_dict["train_neg"] = (train_neg_src, train_neg_tgt)
        self.data_dict["train_pos_id"] = train_pos_ids
        self.data_dict["train_neg_id"] = train_neg_ids

        # For test, we assume test_mask already contains the anomalies/negatives
        # (usually provided by the dataset labels)
        edge_labels = temporal_graph.edge_labels.cpu().numpy()
        self.data_dict["test_pos"] = (
            src[(test_mask) & (edge_labels == 1)], tgt[(test_mask) & (edge_labels == 1)])
        self.data_dict["test_neg"] = (
            src[(test_mask) & (edge_labels == 0)], tgt[(test_mask) & (edge_labels == 0)])
        self.data_dict["test_pos_id"] = snapshot_ids[(
            test_mask) & (edge_labels == 1)]
        self.data_dict["test_neg_id"] = snapshot_ids[(
            test_mask) & (edge_labels == 0)]

        # 6. Extract Subgraphs
        logger.info("Extracting subgraphs...")
        train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(
            net, self.window_size,
            self.data_dict["train_pos_id"], self.data_dict["train_pos"],
            self.data_dict["train_neg_id"], self.data_dict["train_neg"],
            self.data_dict["test_pos_id"], self.data_dict["test_pos"],
            self.data_dict["test_neg_id"], self.data_dict["test_neg"],
            h=self.hop, node_information=node_information
        )

        # 7. Dynamic K and Model Init (Same as previous fix)
        all_graphs = [g for sublist in (
            train_graphs + test_graphs) for g in sublist]
        num_nodes_list = sorted([g.num_nodes for g in all_graphs])
        k_idx = int(math.ceil(0.6 * len(num_nodes_list))) - 1
        self.sortpooling_k = max(10, num_nodes_list[k_idx])
        self.sortpooling_k = 16

        print("JNDZEILHFEIUBF", self.sortpooling_k)

        # sys.exit(0)

        self.feat_dim = max_n_label + 1
        self.attr_dim = node_information.shape[1] if node_information is not None else 0

        self.data_dict.update(
            {"train_graphs": train_graphs, "test_graphs": test_graphs})

        self.classifier = Classifier(
            gm="DGCNN", latent_dim=[32, 32, 32, 1], out_dim=0,
            feat_dim=self.feat_dim, attr_dim=self.attr_dim, edge_feat_dim=0,
            sortpooling_k=int(self.sortpooling_k), conv1d_activation="relu",
            hidden=128, num_class=2, dropout=0.5,
            mode="gpu" if "cuda" in str(self.device) else "cpu"
        )

        if "cuda" in str(self.device):
            self.classifier = self.classifier.cuda()

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.learning_rate)

    def setup_old(self, temporal_graph: TemporalGraph) -> None:
        logger.info("STRGNN setup started...")

        # Extract basic graph information
        num_nodes = temporal_graph.num_nodes

        # Get edge information
        src = temporal_graph.src.cpu().numpy()
        tgt = temporal_graph.tgt.cpu().numpy()
        timestamps = temporal_graph.t.cpu().numpy()
        edge_labels = temporal_graph.edge_labels.cpu().numpy()

        # Get masks
        train_mask = temporal_graph.train_mask.cpu().numpy()
        test_mask = temporal_graph.test_mask.cpu().numpy()
        val_mask = temporal_graph.val_mask.cpu().numpy(
        ) if temporal_graph.val_mask is not None else None

        # Split edges by mask
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        val_indices = np.where(val_mask)[
            0] if val_mask is not None else np.array([])

        # Create snapshots based on timestamps
        unique_timestamps = np.unique(timestamps)
        snapshot_mapping = {ts: i for i, ts in enumerate(unique_timestamps)}
        snapshot_ids = np.array([snapshot_mapping[ts] for ts in timestamps])
        num_snapshots = len(unique_timestamps)

        logger.info(f"Found {num_snapshots} unique snapshots")

        # Prepare train/test data
        train_src = src[train_indices]
        train_tgt = tgt[train_indices]
        train_snap_ids = snapshot_ids[train_indices]
        train_labels = edge_labels[train_indices]

        test_src = src[test_indices]
        test_tgt = tgt[test_indices]
        test_snap_ids = snapshot_ids[test_indices]
        test_labels = edge_labels[test_indices]

        # Optional validation data
        if len(val_indices) > 0:
            val_src = src[val_indices]
            val_tgt = tgt[val_indices]
            val_snap_ids = snapshot_ids[val_indices]
            val_labels = edge_labels[val_indices]

        # Create adjacency matrices for each snapshot
        net = []
        for snap_id in range(num_snapshots):
            # Get edges for this snapshot
            snap_mask = snapshot_ids == snap_id
            snap_src = src[snap_mask]
            snap_tgt = tgt[snap_mask]

            if len(snap_src) > 0:
                # Create adjacency matrix for this snapshot
                snap_data = np.ones_like(snap_src, dtype=np.int32)
                A_snap = ssp.csr_matrix(
                    (snap_data, (snap_src, snap_tgt)), shape=(num_nodes, num_nodes))
                A_snap = A_snap + A_snap.transpose()  # make symmetric
                A_snap.setdiag(0)  # remove self-loops
            else:
                # Empty snapshot
                A_snap = ssp.csr_matrix((num_nodes, num_nodes))

            net.append(A_snap)

        A = net[0]  # only get embeddings from training graph

        # Node2Vec embeddings (optional)
        if self.use_embedding:
            embeddings = generate_node2vec_embeddings(A, 128)
            self.data_dict["embeddings"] = embeddings
        else:
            embeddings = None

        # Prepare positive/negative samples
        self.data_dict["train_pos"] = (
            train_src[train_labels == 1], train_tgt[train_labels == 1])
        self.data_dict["train_neg"] = (
            train_src[train_labels == 0], train_tgt[train_labels == 0])
        self.data_dict["train_pos_id"] = train_snap_ids[train_labels == 1]
        self.data_dict["train_neg_id"] = train_snap_ids[train_labels == 0]

        self.data_dict["test_pos"] = (
            test_src[test_labels == 1], test_tgt[test_labels == 1])
        self.data_dict["test_neg"] = (
            test_src[test_labels == 0], test_tgt[test_labels == 0])
        self.data_dict["test_pos_id"] = test_snap_ids[test_labels == 1]
        self.data_dict["test_neg_id"] = test_snap_ids[test_labels == 0]

        # Add validation data if available
        if len(val_indices) > 0:
            self.data_dict["val_pos"] = (
                val_src[val_labels == 1], val_tgt[val_labels == 1])
            self.data_dict["val_neg"] = (
                val_src[val_labels == 0], val_tgt[val_labels == 0])
            self.data_dict["val_pos_id"] = val_snap_ids[val_labels == 1]
            self.data_dict["val_neg_id"] = val_snap_ids[val_labels == 0]

        # Subgraph extraction
        logger.info("Extracting subgraphs for STRGNN...")
        # Check if validation data exists
        has_val_data = len(val_indices) > 0

        if has_val_data:
            train_graphs, test_graphs, val_graphs, max_n_label = self.dyn_links2subgraphs_with_val(
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
                self.data_dict["val_pos_id"],
                self.data_dict["val_pos"],
                self.data_dict["val_neg_id"],
                self.data_dict["val_neg"],
                h=self.hop,
                max_nodes_per_hop=None,
                node_information=self.data_dict.get("embeddings")
                if self.data_dict.get("embeddings") is not None
                else self.data_dict.get("attributes"),
            )
        else:
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
            val_graphs = None
        update_dict = {
            "train_graphs": train_graphs,
            "test_graphs": test_graphs,
            "max_n_label": max_n_label,
        }

        if val_graphs is not None:
            update_dict["val_graphs"] = val_graphs

        self.data_dict.update(update_dict)

        self.feat_dim = max_n_label + 1  # Structural labels dimension
        self.attr_dim = 0
        if embeddings is not None:
            self.attr_dim = embeddings.shape[1]  # node2vec dimension (128)

        logger.debug(
            f"feat_dim={self.feat_dim}, attr_dim={self.attr_dim}, total={self.feat_dim + self.attr_dim}")

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
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.learning_rate)

    def dyn_links2subgraphs_with_val(self, net, window_size, train_pos_id, train_pos, train_neg_id, train_neg,
                                     test_pos_id, test_pos, test_neg_id, test_neg,
                                     val_pos_id, val_pos, val_neg_id, val_neg, **kwargs):

        train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(
            net, window_size, train_pos_id, train_pos, train_neg_id, train_neg,
            test_pos_id, test_pos, test_neg_id, test_neg, **kwargs
        )

        _, val_graphs, _ = dyn_links2subgraphs(
            net, window_size, train_pos_id, train_pos, train_neg_id, train_neg,
            val_pos_id, val_pos, val_neg_id, val_neg, **kwargs
        )

        return train_graphs, test_graphs, val_graphs, max_n_label

    def train(self, runnable=None) -> None:
        logger.info(f"Starting STRGNN training for {self.num_epoch} epochs...")
        train_graphs = self.data_dict["train_graphs"]
        val_graphs = self.data_dict.get("val_graphs", None)
        train_idxes = list(range(len(train_graphs)))

        for epoch in range(self.num_epoch):
            np.random.shuffle(train_idxes)
            self.classifier.train()
            avg_loss, labels, preds = loop_dataset(
                train_graphs, self.classifier, train_idxes, optimizer=self.optimizer, bsize=self.batch_size
            )

            # Compute training score
            train_score = self.epoch_evaluation_metric(labels, preds)
            logger.info(
                f"Epoch {epoch}: train loss={avg_loss[0]:.5f}, train score={train_score:.5f}")

            val_preds, val_labels = self.inference("test")
            val_score = self.epoch_evaluation_metric(val_labels, val_preds)
            logger.info(f"Epoch {epoch}: test score={val_score:.5f}")
            if runnable is not None:
                runnable(val_score, self, epoch)

            # if val_graphs is not None:
            #     val_preds, val_labels = self.inference("val")
            #     val_score = self.epoch_evaluation_metric(val_labels, val_preds)
            #     logger.info(f"Epoch {epoch}: val score={val_score:.5f}")

            #     if runnable is not None:
            #         runnable(val_score, self, epoch)
            # elif runnable is not None:
            #     runnable(train_score, self, epoch)

    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        self.classifier.eval()
        # if split == "train":
        #     graphs = self.data_dict["train_graphs"]
        # elif split == "val" and "val_graphs" in self.data_dict:
        #     graphs = self.data_dict["val_graphs"]
        # else:
        #     graphs = self.data_dict["test_graphs"]
        graphs = self.data_dict["test_graphs"]
        avg_loss, labels, preds = loop_dataset(
            graphs, self.classifier, list(range(len(graphs))), bsize=self.batch_size)
        return preds, labels
