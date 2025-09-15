import logging

from typing import Callable, Optional, Type, cast

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.types import Device
from torch.utils.data import DataLoader
from torch_geometric.nn import GAT, GCN, GraphSAGE
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv, GATConv, InnerProductDecoder
from torch_geometric.loader import DataLoader, LinkNeighborLoader, NeighborLoader, LinkLoader, NodeLoader
from torch_geometric.data import Data

from ..base import BaseModel
from src.dgadb.storage import TemporalGraph, TemporalGraphSnapshotLoader


GNN_TYPE = Type[GCN] | Type[GAT] | Type[GraphSAGE]
GNN_BASELINES_DICT: dict[str, GNN_TYPE] = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE
}


class GNNBaseline(BaseModel):
    def __init__(
        self,
        device: Device,
        meta_dict: dict[str, str | int | float],
        hyperparams: dict[str, int | float],
        epoch_evaluation_metric: Callable[[torch.FloatTensor, torch.FloatTensor], float],
    ) -> None:
        super().__init__()
        self.device = device
        self.hidden_channels = hyperparams.get("hidden_channels", 128)
        self.out_channels = hyperparams.get("out_channels", 128)
        self.num_layers = hyperparams.get("num_layers", 2)
        self.act = hyperparams.get("act", "ReLU")
        self.num_neighbors = hyperparams.get("num_neighbors", 10)
        self.num_epoch = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        self.model_name = hyperparams.get("model", "GCN")
        self.snap_size = hyperparams.get("snap_size", 2000)
        self.print_freq = hyperparams.get("print_freq", 10)

        self.epoch_evaluation_metric = epoch_evaluation_metric

        self.decoder = InnerProductDecoder()

        self.logger.info(f"Initialized GNNBaseline: {self.model_name}.")

    def setup(self, temporal_graph: TemporalGraph):
        n = temporal_graph.num_nodes
        self.temporal_graph = temporal_graph

        # TODO @Dastamn: to be updated
        if temporal_graph.node_attr is None:
            temporal_graph.node_attr = torch.eye(n)

        if self.model_name not in GNN_BASELINES_DICT:
            raise RuntimeError(
                f"Unknown GNN Baseline '{self.model_name}', expected: {list(GNN_BASELINES_DICT.keys())}")

        model_name = GNN_BASELINES_DICT[self.model_name]
        self.model = model_name(
            in_channels=temporal_graph.node_attr.size(1),
            hidden_channels=int(self.hidden_channels),
            num_layers=int(self.num_layers),
            out_channels=int(self.out_channels),
            act=str(self.act)
        )

        if str(self.device).startswith("cuda") and torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

    def train(self, runnable=None):
        self._ensure_setup()
        assert self.temporal_graph is not None
        assert self.model is not None
        assert self.optimizer is not None

        train_mask = self.temporal_graph.train_mask
        train_edge_index = self.temporal_graph.edge_index[:, train_mask]

        val_mask = self.temporal_graph.val_mask
        val_edge_index = self.temporal_graph.edge_index[:, val_mask]

        data = Data(
            x=self.temporal_graph.node_attr,
            y=self.temporal_graph.node_labels,
            edge_index=self.temporal_graph.edge_index,
            edge_attr=self.temporal_graph.msg,
            time=self.temporal_graph.t,
            edge_label=self.temporal_graph.edge_labels,
            edge_weight=self.temporal_graph.w
        )

        train_loader = LinkNeighborLoader(
            data,
            batch_size=self.snap_size,
            num_neighbors=[int(self.num_neighbors)] * int(self.num_layers),
            edge_label_index=train_edge_index,
            edge_label=None,  # All positives
            neg_sampling_ratio=1.0,
            shuffle=False
        )

        val_loader = LinkNeighborLoader(
            data,
            num_neighbors=[int(self.num_neighbors)] * int(self.num_layers),
            edge_label_index=val_edge_index,
            edge_label=self.temporal_graph.edge_labels[val_mask],
            shuffle=False
        )

        for epoch in tqdm(range(int(self.num_epoch))):
            self.model.train()
            self.optimizer.zero_grad()

            for batch in train_loader:
                z = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    edge_weight=batch.edge_weight
                )

                src, tgt = batch.edge_label_index
                train_preds = F.sigmoid((z[src] * z[tgt]).sum(dim=-1))
                # train_preds = self.decoder(
                #     z, batch.edge_label_index, sigmoid=True)
                loss = F.binary_cross_entropy(train_preds, batch.edge_label)

                loss.backward()
                self.optimizer.step()

            if ((epoch + 1) % self.print_freq == 0) or runnable is not None:
                val_preds, val_labels = self.test(val_loader)
                auc = self.epoch_evaluation_metric(
                    val_labels, val_preds.detach())

                self.logger.info("VAL AUC: " + str(auc))

                if runnable is not None:
                    runnable(auc, self, epoch)

    def test(self, loader: Optional[TemporalGraphSnapshotLoader | LinkLoader | NodeLoader] = None) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_setup()
        assert self.temporal_graph is not None
        assert self.model is not None

        self.model.eval()
        all_preds, all_labels = [], []

        if loader is None:
            # Do 'test' inference
            test_mask = self.temporal_graph.test_mask
            test_edge_index = self.temporal_graph.edge_index[:, test_mask]

            data = Data(
                x=self.temporal_graph.node_attr,
                y=self.temporal_graph.node_labels,
                edge_index=self.temporal_graph.edge_index,
                edge_attr=self.temporal_graph.msg,
                time=self.temporal_graph.t,
                edge_label=self.temporal_graph.edge_labels,
                edge_weight=self.temporal_graph.w
            )

            loader = LinkNeighborLoader(
                data,
                num_neighbors=[int(self.num_neighbors)] * int(self.num_layers),
                edge_label_index=test_edge_index,
                edge_label=self.temporal_graph.edge_labels[test_mask],
                shuffle=False
            )

        elif not isinstance(loader, LinkNeighborLoader):
            raise RuntimeError(
                f"'loader' should be of type LinkNeighborLoader, got: {type(loader).__name__}.")

        loader = cast(LinkNeighborLoader, loader)

        for batch in loader:
            z = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_weight=batch.edge_weight
            )
            src, tgt = batch.edge_label_index
            preds = F.sigmoid((z[src] * z[tgt]).sum(dim=-1))
            # preds = self.decoder(z, batch.edge_label_index, sigmoid=True)
            all_preds.append(preds.cpu())
            all_labels.append(batch.edge_label.cpu())

        p, l = torch.cat(all_preds), torch.cat(all_labels)
        # acc = ((p > 0.5) == l.bool()).float().mean().item()
        # print(acc)
        return p, l
        # return torch.cat(all_preds), torch.cat(all_labels)

    def inference(self, split, loader=None):
        self._ensure_setup()
        return self.test(loader=loader if split != "test" and loader is not None else None)
