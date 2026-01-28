import torch
import itertools
import math
from dataclasses import dataclass
from typing import Self

from .model import GCN, HCA, GRU, Score
from .negative_sample import negative_sample, update_adj

from dgadb.models.base import BaseADModel, BaseADModelComponents
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot

@dataclass
class AddGraphComponents(BaseADModelComponents):
    net1_gcn: GCN
    net2_hca: HCA
    net3_gru: GRU
    net4_score: Score
    negative_sampler: negative_sample
    optimizer: torch.optim.Optimizer
    h_list: torch.Tensor 
    global_adj: torch.Tensor

class AddGraphAD(BaseADModel[AddGraphComponents]):
    def __init__(
        self,
        hidden: int = 100,
        nmid1: int = 100,
        nmid2: int = 100,
        window_size: int = 1, # Corresponds to args.w
        lr: float = 0.001,
        weight_decay: float = 5e-7,
        beta: float = 1.0,
        mui: float = 0.3,
        gama: float = 0.6,
        dropout: float = 0.2,
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)
        self.hidden = hidden
        self.nmid1 = nmid1
        self.nmid2 = nmid2
        self.w = window_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.mui = mui
        self.gama = gama
        self.dropout = dropout

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        nodes = data.num_nodes
        net1 = GCN(nfeat=self.hidden, nmid1=self.nmid1, nmid2=self.nmid2, nhid=self.hidden, dropout=self.dropout)
        net2 = HCA(hidden=self.hidden, dropout=self.dropout)
        net3 = GRU(hidden=self.hidden, dropout=self.dropout)
        net4 = Score(beta=self.beta, mui=self.mui, hidden=self.hidden, dropout=self.dropout)
        n_s = negative_sample()

        optimizer = torch.optim.Adam(
            itertools.chain(net1.parameters(), net2.parameters(), net3.parameters(), net4.parameters()),
            lr=self.lr
        )

        h_list = torch.zeros((1, nodes, self.hidden))
        global_adj = torch.zeros((nodes, nodes))

        self._components = AddGraphComponents(
            net1_gcn=net1, net2_hca=net2, net3_gru=net3, net4_score=net4,
            negative_sampler=n_s, optimizer=optimizer,
            h_list=h_list, global_adj=global_adj
        ).to(self.device)

    def _normalize_adj(self, adj):
        D = adj.sum(1)
        r_inv_sqrt = D.pow(-0.5)
        r_inv_sqrt[torch.eq(r_inv_sqrt, float('inf'))] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(adj, r_mat_inv_sqrt).t(), r_mat_inv_sqrt)

    def _reset_epoch_state(self):
        c = self.components
        nodes = c.global_adj.shape[0]
        
        c.global_adj = torch.zeros((nodes, nodes), device=self.device)
        
        h_list = torch.zeros((self.w, nodes, self.hidden), device=self.device)
        stdv = 1. / math.sqrt(self.hidden)
        h_list[-1].data.uniform_(-stdv, stdv)
        c.h_list = h_list

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        if snapshot.snapshot_id == 0:
            self._reset_epoch_state()
            
        c = self.components
        c.optimizer.zero_grad()
        
        nodes = c.global_adj.shape[0]

        # 1-indexed for the original negative_sample logic
        snapshot_edges_1indexed = snapshot.current.edge_index.t() + 1
        
        c.global_adj, Adj = update_adj(adj=c.global_adj, snapshot=snapshot_edges_1indexed, nodes=nodes)
        
        Adjn = self._normalize_adj(Adj + torch.eye(nodes, device=self.device))
        
        H = c.h_list[-1]
        H_ = c.h_list[-self.w:]

        current = c.net1_gcn(x=H, adj=Adjn, Adj=Adj)
        short = c.net2_hca(C=H_)
        Hn = c.net3_gru(current=current, short=short)

        n_loss = c.negative_sampler(
            adj=c.global_adj, 
            Adj=Adj, 
            snapshot=snapshot_edges_1indexed, 
            H=Hn, 
            f=c.net4_score, 
            arg=(str(self.device) != "cpu")
        )

        loss1 = self.weight_decay * (c.net1_gcn.loss() + c.net2_hca.loss() + c.net3_gru.loss() + c.net4_score.loss())
        
        zero = torch.zeros(1, device=self.device)
        loss2 = torch.zeros(1, device=self.device)
        lens = n_loss.shape[0]
        for m in range(lens):
            loss2 += torch.where((self.gama + n_loss[m]) >= 0, (self.gama + n_loss[m]), zero)
        
        loss = loss1 + (loss2 / lens)
        
        loss.backward()
        c.optimizer.step()

        c.h_list = torch.cat([c.h_list, Hn.unsqueeze(0).detach()], dim=0)

        return loss.item()

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        c = self.components
        self.set_training_mode(False)
        
        with torch.no_grad():
            nodes = c.global_adj.shape[0]
            src, tgt = snapshot.current.edge_index[0], snapshot.current.edge_index[1]
            
            snap_adj = torch.zeros((nodes, nodes), device=self.device)
            snap_adj[src, tgt] = 1
            snap_adj[tgt, src] = 1
            Adjn = self._normalize_adj(snap_adj + torch.eye(nodes, device=self.device))
            
            H = c.h_list[-1]
            H_ = c.h_list[-self.w:]

            current = c.net1_gcn(x=H, adj=Adjn, Adj=snap_adj)
            short = c.net2_hca(C=H_)
            Hn = c.net3_gru(current=current, short=short)

            hi = Hn[src] # Shape: [Num_Edges, Hidden]
            hj = Hn[tgt] # Shape: [Num_Edges, Hidden]
            
            # s = self.a * hi + self.b * hj
            s = c.net4_score.a * hi + c.net4_score.b * hj
            
            # Original: torch.norm(s, 2).pow(2) on a vector.
            # Vectorized: torch.norm(s, p=2, dim=1).pow(2)
            s_per_edge = torch.norm(s, p=2, dim=1).pow(2)
            
            # x = beta * s_ - mui
            x = c.net4_score.beta * s_per_edge - c.net4_score.mui
            scores = torch.sigmoid(x)

            c.h_list = torch.cat([c.h_list, Hn.unsqueeze(0)], dim=0)
            
            return scores

    def save(self, save_dir: str) -> None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'net1': self.components.net1_gcn.state_dict(),
            'net2': self.components.net2_hca.state_dict(),
            'net3': self.components.net3_gru.state_dict(),
            'net4': self.components.net4_score.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, "model.pt"))

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        pass