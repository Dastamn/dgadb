import logging
from collections.abc import Callable


from torch_geometric.nn import Node2Vec

import torch
from torch.nn.functional import sigmoid

from src.dgadb.storage import TemporalGraph
from src.dgadb.storage import TemporalGraphSnapshotLoader
from sklearn.metrics import roc_auc_score
logger = logging.getLogger(__name__)


class n2vModel:

    def __init__(
        self,
        device: torch.DeviceObjType,
        meta_dict: dict[str, str | int | float],
        hyperparams: dict[str, int | float],
        epoch_evaluation_metric: Callable[[
            torch.FloatTensor, torch.FloatTensor], float] = roc_auc_score,
    ) -> None:

        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        logger.info(
            f"Initializing n2vModel with device={self.device} and hyperparams={hyperparams}")

        self.batch_size = hyperparams.get("batch_size", 128)
        self.num_epochs = hyperparams.get("epochs", 20)
        self.num_heads = hyperparams.get("num_heads", 2)
        self.drop_out = hyperparams.get("drop_out", 0.1)
        self.learning_rate = hyperparams.get("learning_rate", 0.001)
        self.emb_dim = hyperparams.get("emb_dim", 128)
        self.lr_decay = hyperparams.get("lr_decay", 0.8)
        self.weight_decay = hyperparams.get("weight_decay", 5e-4)
        self.print_freq = hyperparams.get("print_freq", 10)

        # model specific
        self.walk_length = hyperparams.get("walk_length", 20)
        self.context_size = hyperparams.get("context_size", 10)
        self.walks_per_node = hyperparams.get("walks_per_node", 10)
        self.p = hyperparams.get("p", 1)
        self.q = hyperparams.get("q", 1)
        self.num_negative_samples = hyperparams.get("num_negative_samples", 1)

    def setup(self, temporal_graph: TemporalGraph) -> None:
        logger.info("Setup started...")

        self.train_mask = temporal_graph.train_mask
        self.val_mask = temporal_graph.val_mask
        self.test_mask = temporal_graph.test_mask

        self.edge_index = temporal_graph.edge_index
        self.labels = temporal_graph.edge_labels

        self.num_nodes = temporal_graph.num_nodes

        self.encoder = Node2Vec(
            edge_index=self.edge_index[:, self.train_mask],
            embedding_dim=self.emb_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            num_nodes=self.num_nodes,
            sparse=True
        ).to(self.device)

        self.optimizer = torch.optim.SparseAdam(
            params=self.encoder.parameters(),
            lr=self.learning_rate,
        )

        self.loader = self.encoder.loader(
            batch_size=self.batch_size, shuffle=False, num_workers=0)

    def _ensure_setup(self) -> None:
        """Ensure setup() has been called."""
        if any(x is None for x in [self.optimizer, self.encoder]):
            raise RuntimeError(
                "Model not properly initialized. Call setup() before train() or inference().")

    def train(self, runnable=None) -> None:

        self._ensure_setup()
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            self.encoder.train()
            total_loss = 0.0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.encoder.loss(
                    pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(self.loader)
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - mean loss: {epoch_loss:.4f}")
            probs, labels = self.inference(split="val")
            auc = self.epoch_evaluation_metric(
                labels.cpu(), probs.detach().cpu())
            if runnable is not None:
                runnable(auc, self, epoch, save=False)

    def inference(self, split: str = "test"):
        split_mask_map = {
            "train": self.train_mask,
            "val": self.val_mask,
            "test": self.test_mask
        }
        split_edge_index = self.edge_index[:, split_mask_map[split]]
        self._ensure_setup()
        self.encoder.eval()
        z = self.encoder()
        ei = split_edge_index.to(self.device).long()
        src, dst = ei
        scores = (z[src] * z[dst]).sum(dim=-1)
        probs = sigmoid(scores)

        split_labels = self.labels[split_mask_map[split]].to(
            probs.device).float()
        return probs.detach(), split_labels
