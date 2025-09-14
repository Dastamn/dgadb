import logging

from typing import Callable, Optional, Type

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.types import Device
from torch_geometric.nn import GAT, GCN, GraphSAGE
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv, GATConv, InnerProductDecoder

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
        self.num_epoch = hyperparams.get("num_epoch", 100)
        self.learning_rate = hyperparams.get("learning_rate", 0.01)
        self.model_name = hyperparams.get("model", "GCN")
        self.snap_size = hyperparams.get("snap_size", 2000)
        self.print_freq = hyperparams.get("print_freq", 10)

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

        # Move to device
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

        train_snaps = TemporalGraphSnapshotLoader(
            self.temporal_graph, window_size=self.snap_size, split="train")
        val_snaps = TemporalGraphSnapshotLoader(
            self.temporal_graph, window_size=self.snap_size, split="val")

        train_src = self.temporal_graph.src[self.temporal_graph.train_mask]
        train_tgt = self.temporal_graph.tgt[self.temporal_graph.train_mask]
        train_edge_index = torch.stack([train_src, train_tgt], dim=1).T

        for epoch in tqdm(range(int(self.num_epoch))):
            self.model.train()
            self.optimizer.zero_grad()

            for train_snap in train_snaps:
                curr = train_snap.current
                assert curr.node_attr is not None
                cum = train_snap.cumulative
                z = self.model(
                    x=cum.node_attr,
                    edge_index=cum.edge_index,
                    edge_attr=curr.msg,
                    edge_weight=curr.w
                )
                z_src = z[curr.src]
                z_tgt = z[curr.tgt]
                pos_logits = (z_src * z_tgt).sum(dim=1)

                neg_edge_index = negative_sampling(
                    edge_index=train_edge_index,
                    num_nodes=self.temporal_graph.num_nodes,
                    num_neg_samples=curr.num_edges,
                )
                neg_z_src = z[neg_edge_index[0]]
                neg_z_tgt = z[neg_edge_index[1]]
                neg_logits = (neg_z_src * neg_z_tgt).sum(dim=1)

                loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos_logits, neg_logits]),
                    torch.cat([torch.zeros_like(pos_logits),
                              torch.ones_like(neg_logits)])
                )
                loss.backward()
                self.optimizer.step()

                self.inference()
                break

            # if (epoch + 1) % self.print_freq == 0:
            #     self.inference()

            break

    def inference(self):
        self._ensure_setup()
        print("inference")
