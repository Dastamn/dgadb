# UNDIRECTED EDGES
# ASSUMES DATA HAS A LABEL COLUMN
# ASSUMES SORTED BY TIMESTAMP WITH EDGE INDICES IN THAT ORDER (contiguous order)
import os
import pickle
import numpy as np
import pandas as pd
from src.dgadb.storage.temporal_graph import TemporalGraph
from typing import Optional
import logging
import torch
import torch.utils.data as tud
import torch.nn.functional as F
import time
from src.dgadb.models.GeneralDYG.generate_datasets import BatchGraphSample
from src.dgadb.models.GeneralDYG.model.Combine import CombinedModel
from src.dgadb.models.GeneralDYG.model.CensNet import CensNet
from src.dgadb.models.GeneralDYG.model.Transformer import TransformerBinaryClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _fmt_ratio(x: float) -> str:
    # filename formatting (0.5 becomes 0.50)
    return f"{x:.2f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))


class _SimpleCollate:
    """Clone of Collate.dyg_collate_fn from the original implemenmtation"""

    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        input_nodes_feature = [b["input_nodes_feature"] for b in batch]
        input_edges_feature = [b["input_edges_feature"] for b in batch]
        input_edges_pad = torch.stack(
            [b["input_edges_pad"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        Tmats = [b["Tmats"] for b in batch]
        adjs = [b["adjs"] for b in batch]
        eadjs = [b["eadjs"] for b in batch]
        mask_edge = torch.stack([b["mask_edge"] for b in batch], dim=0)
        return {
            "input_nodes_feature": input_nodes_feature,
            "input_edges_feature": input_edges_feature,
            "input_edges_pad": input_edges_pad,
            "labels": labels,
            "Tmats": Tmats,
            "adjs": adjs,
            "eadjs": eadjs,
            "mask_edge": mask_edge,
        }


class _DygDatasetCompat(tud.Dataset):
    """
    Replacement for DygDataset that:
      - loads a specific pickle path
      - uses a provided split index (train_count, val_count)
    """

    def __init__(self, pkl_path: str, split: str, train_count: int, val_count: int, input_dim: int) -> None:
        import pickle

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        node_features = data["nodefeatures"]
        edge_features = data["edgefeatures"]
        labels = data["labels"]
        Tmats = data["Tmats"]
        adjs = data["adjs"]
        eadjs = data["eadjs"]

        total_ns = len(labels)
        train_end = int(train_count)
        val_end = int(train_count + val_count)

        if split == "train":
            idx_slice = slice(0, train_end)
        elif split == "val":
            idx_slice = slice(train_end, val_end)
        elif split == "test":
            idx_slice = slice(val_end, total_ns)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        node_features = node_features[idx_slice]
        edge_features = edge_features[idx_slice]
        labels = labels[idx_slice]
        Tmats = Tmats[idx_slice]
        adjs = adjs[idx_slice]
        eadjs = eadjs[idx_slice]

        # global vocab sizes
        flattened_node = (
            np.concatenate([arr for arr in node_features]) if len(
                node_features) else np.array([], dtype=int)
        )
        flattened_edge = (
            np.concatenate([arr for arr in edge_features]) if len(
                edge_features) else np.array([], dtype=int)
        )
        num_nodes = int(np.max(flattened_node) +
                        1) if flattened_node.size else 0
        num_edges = int(np.max(flattened_edge) +
                        1) if flattened_edge.size else 0

        # random embeddings
        Nfeatures = np.random.uniform(low=0.0, high=1.0, size=(
            max(num_nodes, 1), input_dim)).astype(np.float32)
        Efeatures = np.random.uniform(low=0.0, high=1.0, size=(
            max(num_edges, 1), input_dim)).astype(np.float32)

        # build masks + padded tensors like paper code
        max_mask_edge = max((len(arr) for arr in edge_features), default=0)
        NS = len(edge_features)
        mask_edge = np.ones((NS, max_mask_edge), dtype=np.float32)
        edge_lengths = [len(arr) for arr in edge_features]
        for i, L in enumerate(edge_lengths):
            mask_edge[i, :L] = 0.0

        # edges pad
        input_edges_pad = np.zeros(
            (NS, max_mask_edge, input_dim), dtype=np.float32)
        input_edges_feature = []
        for i, indices in enumerate(edge_features):
            indices = np.asarray(indices, dtype=int)
            if indices.size:
                input_edges_pad[i, : len(indices), :] = Efeatures[indices]
                input_edges_feature.append(torch.tensor(
                    Efeatures[indices], dtype=torch.float32))
            else:
                input_edges_feature.append(torch.empty(
                    (0, input_dim), dtype=torch.float32))

        # nodes
        input_nodes_feature = []
        for indices in node_features:
            indices = np.asarray(indices, dtype=int)
            if indices.size:
                input_nodes_feature.append(torch.tensor(
                    Nfeatures[indices], dtype=torch.float32))
            else:
                input_nodes_feature.append(torch.empty(
                    (0, input_dim), dtype=torch.float32))

        # convert arrays
        self.input_nodes_feature = input_nodes_feature
        self.input_edges_feature = input_edges_feature
        self.input_edges_pad = torch.tensor(
            input_edges_pad, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.Tmats = Tmats
        self.adjs = adjs
        self.eadjs = eadjs
        self.mask_edge = torch.tensor(mask_edge, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, any]:
        return {
            "input_nodes_feature": self.input_nodes_feature[i],
            "input_edges_feature": self.input_edges_feature[i],
            "input_edges_pad": self.input_edges_pad[i],
            "labels": self.labels[i],
            "Tmats": self.Tmats[i],
            "adjs": self.adjs[i],
            "eadjs": self.eadjs[i],
            "mask_edge": self.mask_edge[i],
        }


def _build_generaldyg_pkl_from_temporal_graph(
    temporal_graph: TemporalGraph,
    dataset_name: str,
    dir_data: str,
    train_ratio: float,
    val_ratio: float,
    anomaly_ratio_train: float,
    anomaly_ratio_val: float,
    anomaly_ratio_test: float,
    seed: Optional[int] = 42,
) -> str:
    os.makedirs(dir_data, exist_ok=True)

    # Extract data from tg
    src = temporal_graph.src.cpu().numpy()
    tgt = temporal_graph.tgt.cpu().numpy()
    edge_labels = temporal_graph.edge_labels.cpu().numpy()

    # Create edge ids
    edge_ids = np.arange(len(src))

    # Convert to pandas df for BatchGraphSample
    graph_df = pd.DataFrame({
        'u': src.astype(np.int64),
        'i': tgt.astype(np.int64),
        'id': edge_ids.astype(np.int64),
        'label': edge_labels.astype(np.int64)
    })

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.dir_data = dir_data
    cfg.data_set = dataset_name
    # cfg.neg = int(round(anomaly_ratio * 10))

    sampler = BatchGraphSample(cfg, graph_df)
    idx_column = graph_df["id"].to_numpy()
    label_column = graph_df["label"].to_numpy()

    (
        all_node_features,
        all_edge_features,
        all_Tmat,
        all_adj,
        all_eadj,
        all_mask,
    ) = sampler.get_batch_data(idx_column)

    # pack exactly like original implementation
    all_node_features = np.array(all_node_features, dtype=object)
    all_edge_features = np.array(all_edge_features, dtype=object)
    mask_arr = np.array(all_mask)

    data = {
        "nodefeatures": all_node_features,
        "edgefeatures": all_edge_features,
        "labels": label_column,
        "Tmats": all_Tmat,
        "adjs": all_adj,
        "eadjs": all_eadj,
        "masks": mask_arr,
    }

    # filename
    t_str = _fmt_ratio(train_ratio)
    v_str = _fmt_ratio(val_ratio)
    atr_str = _fmt_ratio(anomaly_ratio_train)
    av_str = _fmt_ratio(anomaly_ratio_val)
    ate_str = _fmt_ratio(anomaly_ratio_test)
    pkl_name = f"{dataset_name}_t{t_str}_v{v_str}_atr{atr_str}_av{av_str}_ate{ate_str}.pkl"
    pkl_path = os.path.join(dir_data, pkl_name)

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return pkl_path


class GeneralDYGModel:

    def __init__(
        self,
        device: torch.device,
        meta_dict: dict[str, any],
        hyperparams: dict[str, any],
        epoch_evaluation_metric,
    ) -> None:
        # meta
        self.dataset_name: str = meta_dict["dataset_name"]
        self.train_ratio: float = float(meta_dict["train_ratio"])
        self.val_ratio: float = float(meta_dict["val_ratio"])
        self.anomaly_ratio_train: float = float(meta_dict["anom_train_ratio"])
        self.anomaly_ratio_val: float = float(meta_dict["anom_val_ratio"])
        self.anomaly_ratio_test: float = float(meta_dict["anom_test_ratio"])
        self.has_val: bool = self.val_ratio > 0.0

        self.print_freq: int = int(hyperparams.get("print_freq", 1))

        # runtime
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        base_path = os.environ["BASE_PATH"]
        dataset_dir = os.path.dirname(
            f"{base_path}/src/dgadb/models/GeneralDYG/data/{self.dataset_name}/")
        self.dir_data: str = os.path.join(base_path, dataset_dir)

        # hyperparameters
        self.batch_size: int = int(hyperparams.get("batch_size", 32))
        self.learning_rate: float = float(
            hyperparams.get("learning_rate", 1e-3))
        self.n_epochs: int = int(hyperparams.get("n_epochs", 30))

        # model architecture
        self.input_dim: int = int(hyperparams.get("input_dim", 64))
        self.hidden_dim: int = int(hyperparams.get("hidden_dim", 128))
        self.drop_out: float = float(hyperparams.get("drop_out", 0.3))
        self.n_heads: int = int(hyperparams.get("n_heads", 4))
        self.n_layer: int = int(hyperparams.get("n_layer", 6))

        # dataloader/system
        self.num_data_workers: int = int(
            hyperparams.get("num_data_workers", 0))
        self.gpus: int | None = hyperparams.get("gpus", None)
        self.seed: int | None = hyperparams.get("seed", None)

        # objects populated in setup()
        self.model = None
        self.optimizer = None
        self.dataset_train = None
        self.dataset_test = None
        self.loader_train = None
        self.loader_test = None
        self.pkl_path: str | None = None

        logger.info(
            "Initialized GeneralDYGModel with "
            f"device={self.device}, dataset={self.dataset_name}, "
            f"train={self.train_ratio}, val={self.val_ratio}, "
            f"batch_size={self.batch_size}, lr={self.learning_rate}, "
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"heads={self.n_heads}, layers={self.n_layer}, dropout={self.drop_out}"
        )

    def setup(self, temporal_graph: TemporalGraph) -> None:
        # resolve pickle name
        t_str = _fmt_ratio(self.train_ratio)
        v_str = _fmt_ratio(self.val_ratio)
        atr_str = _fmt_ratio(self.anomaly_ratio_train)
        av_str = _fmt_ratio(self.anomaly_ratio_val)
        ate_str = _fmt_ratio(self.anomaly_ratio_test)
        pkl_name = f"{self.dataset_name}_t{t_str}_v{v_str}_atr{atr_str}_av{av_str}_ate{ate_str}.pkl"
        self.pkl_path = os.path.join(self.dir_data, pkl_name)

        if not os.path.exists(self.pkl_path):
            logger.info("Building GeneralDYG pickle since it was not found.")
            self.pkl_path = _build_generaldyg_pkl_from_temporal_graph(
                temporal_graph=temporal_graph,
                dataset_name=self.dataset_name,
                dir_data=self.dir_data,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                anomaly_ratio_train=self.anomaly_ratio_train,
                anomaly_ratio_val=self.anomaly_ratio_val,
                anomaly_ratio_test=self.anomaly_ratio_test,
            )

        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Pickle not found: {self.pkl_path}")

        # Get counts from TemporalGraph masks
        N = len(temporal_graph.src)

        train_mask = temporal_graph.train_mask.cpu().numpy()
        test_mask = temporal_graph.test_mask.cpu().numpy()
        val_mask = temporal_graph.val_mask.cpu().numpy(
        ) if temporal_graph.val_mask is not None else None

        self.train_count = int(np.sum(train_mask))
        self.val_count = int(np.sum(val_mask)) if val_mask is not None else 0

        logger.info(
            f"Data splits: train={self.train_count}, val={self.val_count}, test={int(np.sum(test_mask))}")

        # datasets & loaders
        train_ds = _DygDatasetCompat(
            self.pkl_path,
            split="train",
            train_count=self.train_count,
            val_count=self.val_count,
            input_dim=self.input_dim,
        )
        val_ds = _DygDatasetCompat(
            self.pkl_path, split="val", train_count=self.train_count, val_count=self.val_count, input_dim=self.input_dim
        )
        test_ds = _DygDatasetCompat(
            self.pkl_path,
            split="test",
            train_count=self.train_count,
            val_count=self.val_count,
            input_dim=self.input_dim,
        )

        collate = _SimpleCollate()
        self.loader_train = tud.DataLoader(
            dataset=train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_workers,
            pin_memory=True,
            collate_fn=collate,
        )

        self.loader_val = (
            tud.DataLoader(
                dataset=val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_data_workers,
                collate_fn=collate,
            )
            if self.val_count > 0
            else None
        )

        self.loader_test = tud.DataLoader(
            dataset=test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            collate_fn=collate,
        )

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.input_dim = self.input_dim
        cfg.drop_out = self.drop_out
        cfg.hidden_dim = self.hidden_dim
        cfg.learning_rate = self.learning_rate
        cfg.n_heads = self.n_heads
        cfg.n_layer = self.n_layer

        gnn = CensNet(cfg.input_dim, cfg.drop_out)
        transformer = TransformerBinaryClassifier(
            cfg, self.device, hidden_size=cfg.hidden_dim)
        self.model = CombinedModel(
            gnn, transformer, device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        logger.info("GeneralDYG setup complete.")

    def _forward_batch(self, batch):
        input_nodes_feature = [t.to(self.device)
                               for t in batch["input_nodes_feature"]]
        input_edges_feature = [t.to(self.device)
                               for t in batch["input_edges_feature"]]
        input_edges_pad = batch["input_edges_pad"].to(self.device)
        mask_edge = batch["mask_edge"].to(self.device)
        Tmats = [t.to(self.device) for t in batch["Tmats"]]
        adjs = [t.to(self.device) for t in batch["adjs"]]
        eadjs = [t.to(self.device) for t in batch["eadjs"]]
        y = batch["labels"].to(self.device).to(torch.float32)

        logits = self.model(
            input_nodes_feature,
            input_edges_feature,
            input_edges_pad,
            eadjs,
            adjs,
            Tmats,
            mask_edge,
        )
        return logits, y

    def train(self, runnable=None) -> None:
        if any(x is None for x in [self.model, self.optimizer, self.loader_train, self.loader_test]):
            raise RuntimeError("Call setup() before train().")

        def criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        logger.info(f"Starting training for {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            self.model.train()
            t0 = time.time()
            running_loss = 0.0

            for batch in self.loader_train:
                logits, y = self._forward_batch(batch)
                loss = criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / max(1, len(self.loader_train))
            logger.info(
                f"Epoch {epoch + 1:03d} | train_loss={train_loss:.4f} | time={time.time() - t0:.2f}s")

            if ((epoch + 1) % self.print_freq) == 0 or runnable is not None:
                split = "val" if (self.loader_val is not None) else "train"
                preds, labels = self.inference(split=split)
                auc = float(self.epoch_evaluation_metric(labels, preds))
                logger.info(
                    f"[Eval @ epoch {epoch + 1:03d}] {split} AUC = {auc:.4f}")
                if runnable is not None:
                    runnable(auc, self, epoch)

    def inference(self, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Call setup() first.")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train','val','test'}")

        # pick loader. if val requested but not present, fall back to train
        if split == "train":
            loader = self.loader_train
        elif split == "val":
            loader = self.loader_val if self.loader_val is not None else self.loader_train
        else:  # "test"
            loader = self.loader_test

        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits, y = self._forward_batch(batch)
                preds.append(torch.sigmoid(logits).cpu().numpy().ravel())
                labels.append(y.cpu().numpy().ravel())
        return np.concatenate(preds), np.concatenate(labels)
