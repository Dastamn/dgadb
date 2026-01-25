import json
import os
import pickle
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm
from dataclasses import dataclass

from .model.CensNet import CensNet
from .model.Transformer import TransformerBinaryClassifier
from .model.Combine import CombinedModel

from src.dgadb.models.base import BaseADModel, BaseADModelComponents
from src.dgadb.storage.temporal_graph import TemporalGraph, TemporalGraphView
from src.dgadb.storage.temporal_snapshot import TemporalGraphSnapshot
import random


@dataclass
class GeneralDyGComponents(BaseADModelComponents):
    model: CombinedModel
    optimizer: torch.optim.Optimizer
    node_features_static: torch.Tensor
    edge_features_static: torch.Tensor


class GeneralDyGAD(BaseADModel[GeneralDyGComponents]):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 258,
        n_heads: int = 4,
        n_layer: int = 6,
        drop_out: float = 0.4,
        learning_rate: float = 0.0001,
        max_mask_len: int = 26,
        k_hop: int = 1,
        device: torch.device | str = "cpu",
        cache_dir: str = "cache/generaldyg"
    ) -> None:
        super().__init__(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.max_mask_len = max_mask_len
        self.k_hop = k_hop
        self.cache_dir = cache_dir
        self.precomputed_data = None

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        dataset_name = data.metadata.get("dataset_name", "default")
        cache_path = os.path.join(
            self.cache_dir, f"{dataset_name}_subgraphs.pkl")

        # 1. Replicate Feature Initialization (datasets.py lines 34-35)
        # These are generated ONCE and never updated.
        self.logger.info("Initializing static random features...")
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        node_feats_np = np.random.uniform(
            low=0.0, high=1.0, size=(num_nodes, self.input_dim))
        edge_feats_np = np.random.uniform(
            low=0.0, high=1.0, size=(num_edges, self.input_dim))

        if os.path.exists(cache_path):
            self.logger.info(f"Loading subgraphs from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.precomputed_data = pickle.load(f)
        else:
            self.logger.info("Preprocessing subgraphs (one-time)...")
            self.precomputed_data = self._global_preprocessing(data)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.precomputed_data, f)

        # Find the maximum number of edges across all subgraphs
        self.max_edges_all = max(len(item['sub_edge_ids'])
                                 for item in self.precomputed_data)
        self.logger.info(
            f"Subgraphs precomputed. Node limit: {self.max_mask_len}, Max edges found: {self.max_edges_all}")

        self.logger.info("Converting precomputed subgraphs to Tensors...")
        self.processed_n_feats = []
        self.processed_e_feats = []
        self.processed_Tmats = []
        self.processed_adjs = []
        self.processed_eadjs = []

        for item in tqdm(self.precomputed_data, desc="Tensor-ifying"):
            n_feat = node_feats_np[item['sub_node_ids']]
            e_feat = edge_feats_np[item['sub_edge_ids']]

            self.processed_n_feats.append(torch.from_numpy(n_feat).float())
            self.processed_e_feats.append(torch.from_numpy(e_feat).float())
            self.processed_Tmats.append(torch.from_numpy(item['T']).float())
            self.processed_adjs.append(torch.from_numpy(item['adj']).float())
            self.processed_eadjs.append(torch.from_numpy(item['eadj']).float())

        gnn = CensNet(self.input_dim, self.drop_out)
        transformer = TransformerBinaryClassifier(
            self, self.device, hidden_size=self.hidden_dim)
        model = CombinedModel(gnn, transformer)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        self._components = GeneralDyGComponents(
            model=model,
            optimizer=optimizer,
            node_features_static=torch.from_numpy(node_feats_np).float(),
            edge_features_static=torch.from_numpy(edge_feats_np).float()
        ).to(self.device)

    def _global_preprocessing(self, data: TemporalGraph):
        import random
        G = nx.Graph()
        src_np = data.src.cpu().numpy().astype(int)
        tgt_np = data.tgt.cpu().numpy().astype(int)
        for i in range(len(src_np)):
            u, v = src_np[i], tgt_np[i]
            if G.has_edge(u, v):
                G[u][v]['weight'].append(i)
            else:
                G.add_edge(u, v, weight=[i])

        all_subgraphs = []
        for i in tqdm(range(len(src_np)), desc="Subgraphs"):
            u, v = src_np[i], tgt_np[i]

            # K-hop extraction
            nodes_u = self._extract_k_hop(G, u, v)
            nodes_v = self._extract_k_hop(G, v, u)
            sub_nodes = nodes_u.union(nodes_v)
            sub_g = G.subgraph(sub_nodes).copy()

            # Random weight resolution
            for eu, ev, edata in sub_g.edges(data=True):
                sub_g[eu][ev]['weight'] = random.choice(edata['weight'])
            sub_g[u][v]['weight'] = i  # Target edge weight

            # Reorder Nodes by Min Weight
            node_weights = {}
            for node in sub_g.nodes:
                edges = sub_g.edges(node, data='weight')
                node_weights[node] = min(
                    w for _, _, w in edges) if edges else float('inf')
            sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1])
            mapping = {old_id: new_id for new_id,
                       (old_id, _) in enumerate(sorted_nodes)}

            # Create Adjacency
            new_sub = nx.relabel_nodes(sub_g, mapping)
            nodelist = sorted(new_sub.nodes())
            adj_raw = nx.adjacency_matrix(
                new_sub, nodelist=nodelist, weight='weight')

            # Create Transition Matrix
            T = self._create_transition_matrix(adj_raw)
            # Create Edge Adj
            eadj, edge_name = self._create_edge_adj(adj_raw)

            # Map Edge Features to Matrix
            sub_edge_ids = []
            edge_indices = np.nonzero(sp.triu(adj_raw, k=1))
            for r, c in zip(edge_indices[0], edge_indices[1]):
                sub_edge_ids.append(int(adj_raw[r, c]))

            all_subgraphs.append({
                'adj': self._normalize(adj_raw + sp.eye(adj_raw.shape[0])).toarray(),
                'T': T.toarray(),
                'eadj': self._normalize(eadj).toarray(),
                'sub_node_ids': np.array([old for old, _ in sorted_nodes]),
                'sub_edge_ids': np.array(sub_edge_ids)
            })
        return all_subgraphs

    def _get_batch_tensors(self, global_indices: range | list):
        # Collate equivalent
        batch_n = [self.processed_n_feats[i].to(
            self.device) for i in global_indices]
        batch_e = [self.processed_e_feats[i].to(
            self.device) for i in global_indices]
        batch_T = [self.processed_Tmats[i].to(
            self.device) for i in global_indices]
        batch_adj = [self.processed_adjs[i].to(
            self.device) for i in global_indices]
        batch_eadj = [self.processed_eadjs[i].to(
            self.device) for i in global_indices]

        # Padding
        batch_size = len(global_indices)
        e_pad = torch.zeros(batch_size, self.max_edges_all,
                            self.input_dim).to(self.device)
        mask_edge = torch.ones(batch_size, self.max_edges_all).to(self.device)

        for i, idx in enumerate(global_indices):
            feat = self.processed_e_feats[idx]
            seq_len = feat.size(0)
            e_pad[i, :seq_len, :] = feat.to(self.device)
            mask_edge[i, :seq_len] = 0

        return batch_n, batch_e, e_pad, batch_eadj, batch_adj, batch_T, mask_edge

    def _get_batch_tensors_old(self, global_indices: range):
        items = [self.precomputed_data[i] for i in global_indices]
        batch_size = len(items)

        input_nodes_feature = []
        input_edges_feature = []
        Tmats, adjs, eadjs = [], [], []

        input_edges_pad = torch.zeros(
            batch_size, self.max_edges_all, self.input_dim).to(self.device)
        mask_edge = torch.ones(batch_size, self.max_edges_all).to(self.device)

        for i, item in enumerate(items):
            n_feat = self.components.node_features_static[item['sub_node_ids']]
            e_feat = self.components.edge_features_static[item['sub_edge_ids']]

            input_nodes_feature.append(n_feat.to(self.device))
            input_edges_feature.append(e_feat.to(self.device))
            Tmats.append(torch.from_numpy(item['T']).float().to(self.device))
            adjs.append(torch.from_numpy(item['adj']).float().to(self.device))
            eadjs.append(torch.from_numpy(
                item['eadj']).float().to(self.device))

            seq_len = len(e_feat)
            input_edges_pad[i, :seq_len, :] = e_feat
            mask_edge[i, :seq_len] = 0

        return input_nodes_feature, input_edges_feature, input_edges_pad, eadjs, adjs, Tmats, mask_edge

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        self.set_training_mode(True)
        view: TemporalGraphView = snapshot.current
        start, stop = view._slice.start, view._slice.stop
        # global_indices = range(view._slice.start, view._slice.stop)

        indices = list(range(start, stop))
        random.shuffle(indices)

        n_feat, e_feat, e_pad, eadjs, adjs, Tmats, mask = self._get_batch_tensors(
            indices)

        # n_feat, e_feat, e_pad, eadjs, adjs, Tmats, mask = self._get_batch_tensors(
        #     global_indices)

        self.components.optimizer.zero_grad()
        # forward (train.py lines 152-160)
        logits = self.components.model(
            n_feat, e_feat, e_pad, eadjs, adjs, Tmats, mask)

        labels = view.edge_labels.to(self.device).float()

        relative_indices = [i - start for i in indices]
        shuffled_labels = labels[relative_indices]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, shuffled_labels)

        loss.backward()
        self.components.optimizer.step()
        return loss.item()

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        self.set_training_mode(False)
        view: TemporalGraphView = snapshot.current
        global_indices = range(view._slice.start, view._slice.stop)
        with torch.no_grad():
            n_feat, e_feat, e_pad, eadjs, adjs, Tmats, mask = self._get_batch_tensors(
                global_indices)
            logits = self.components.model(
                n_feat, e_feat, e_pad, eadjs, adjs, Tmats, mask)
            return torch.sigmoid(logits)

    def _extract_k_hop(self, G, src, dest):
        nodes = {src}
        visited = {src}
        for _ in range(self.k_hop):
            new_nodes = set()
            for n in nodes:
                neighbors = set(G.neighbors(n))
                new_nodes.update(neighbors - visited)
            visited.update(new_nodes)
            nodes.update(new_nodes)

        import random
        nodes_list = list(nodes)
        while len(nodes_list) > self.max_mask_len:
            to_remove = random.choice(nodes_list)
            if to_remove != src and to_remove != dest:
                nodes_list.remove(to_remove)
        return set(nodes_list)

    def _create_transition_matrix(self, adj):
        adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(adj, k=1))
        num_edge = len(edge_index[0])
        row_index = [i for pair in zip(
            edge_index[0], edge_index[1]) for i in pair]
        col_index = np.repeat(np.arange(num_edge), 2)
        return sp.csr_matrix((np.ones(len(row_index)), (row_index, col_index)), shape=(adj.shape[0], num_edge))

    def _create_edge_adj(self, adj):
        adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(adj, k=1))
        num_edge = len(edge_index[0])
        edge_name = list(zip(edge_index[0], edge_index[1]))
        eadj = np.zeros((num_edge, num_edge))
        for i in range(num_edge):
            for j in range(i, num_edge):
                if len(set(edge_name[i]) & set(edge_name[j])) != 0:
                    eadj[i, j] = 1
        adj_out = eadj + eadj.T
        np.fill_diagonal(adj_out, 1)
        return sp.csr_matrix(adj_out), edge_name

    def _normalize(self, mx):
        rowsum = np.array(mx.sum(1)).flatten()
        r_inv = np.power(rowsum, -1, where=rowsum != 0)
        return sp.diags(r_inv).dot(mx)

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        comp = self.components

        checkpoint = {
            'model_state_dict': comp.model.state_dict(),
            'optimizer_state_dict': comp.optimizer.state_dict(),
            'node_features_static': comp.node_features_static.cpu(),
            'edge_features_static': comp.edge_features_static.cpu(),
            'max_edges_all': self.max_edges_all
        }

        torch.save(checkpoint, os.path.join(save_dir, "model.pt"))

        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_layer": self.n_layer,
            "drop_out": self.drop_out,
            "learning_rate": self.learning_rate,
            "max_mask_len": self.max_mask_len,
            "k_hop": self.k_hop,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        self.logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> "GeneralDyGAD":
        config_path = os.path.join(load_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        instance = cls(**config, device=device)

        model_path = os.path.join(load_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        gnn = CensNet(instance.input_dim, instance.drop_out)
        transformer = TransformerBinaryClassifier(
            instance, device, hidden_size=instance.hidden_dim)
        model = CombinedModel(gnn, transformer)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=instance.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        instance.max_edges_all = checkpoint['max_edges_all']

        instance._components = GeneralDyGComponents(
            model=model,
            optimizer=optimizer,
            node_features_static=checkpoint['node_features_static'],
            edge_features_static=checkpoint['edge_features_static']
        ).to(device)

        instance.logger.info(f"Model loaded successfully from {load_dir}")
        return instance
