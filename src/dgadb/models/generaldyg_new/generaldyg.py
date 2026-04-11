import json
import os
import pickle
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm
import random

from .model.CensNet import CensNet
from .model.Transformer import TransformerBinaryClassifier
from .model.Combine import CombinedModel

from dgadb.models.base import BaseADModel, BaseADModelComponents, TrainingState
from dgadb.scalability.streaming_profiler import StreamingProfiler
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.experiment.callbacks import ExperimentCallbackHandler
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from pathlib import Path

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
        batch_size: int = 128,
        k_hop: int = 1,
        device: torch.device | str = "cpu",
        cache_dir: str = "cache",
        seed: int = 1234
    ) -> None:
        super().__init__(device)
        self.seed = seed
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.max_mask_len = max_mask_len
        self.batch_size = batch_size
        self.k_hop = k_hop
        self.cache_dir = cache_dir

        self.cache_dir = Path(cache_dir)

        # Internal storage for precomputed subgraph tensors
        self.precomputed_tensors = {}
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None

    def __get_cache_identifier(self, data: TemporalGraph):
        base_name = data.metadata.get("variant_name", "clean")
        
        tr = data.metadata.get('train_ratio', 0)
        vr = data.metadata.get('val_ratio', 0)
        te = data.metadata.get('test_ratio', 0)
        split_str = f"split_{tr}_{vr}_{te}"
        
        params_str = f"hop{self.k_hop}_mask{self.max_mask_len}"
        
        return f"{base_name}_{split_str}_{params_str}"

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        dataset_name = data.dataset_name
        current_cache_dir = self.cache_dir / dataset_name
        current_cache_dir.mkdir(parents=True, exist_ok=True)

        cache_id = self.__get_cache_identifier(data)
        cache_file = current_cache_dir / f"{cache_id}.pkl"

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.val_mask = data.val_mask
        self.edge_labels = data.edge_labels if data.edge_labels is not None else torch.zeros(
            data.num_edges)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # 1. Feature Initialization (Static/Random as per original)
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        node_feats_np = np.random.uniform(
            0.0, 1.0, size=(num_nodes, self.input_dim))
        edge_feats_np = np.random.uniform(
            0.0, 1.0, size=(num_edges, self.input_dim))
        
        if cache_file.exists():
            self.logger.info(f"CACHE HIT: Loading subgraphs from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_payload = pickle.load(f)
            
            subgraph_data = cached_payload['subgraph_data']
            self.max_edges_all = cached_payload['max_edges_all']
        else:
            self.logger.info(f"CACHE MISS: Preprocessing subgraphs for {cache_id}...")
            subgraph_data = self._global_preprocessing(data)
            self.max_edges_all = max(len(item['sub_edge_ids']) for item in subgraph_data)
            
            payload = {
                'subgraph_data': subgraph_data,
                'max_edges_all': self.max_edges_all
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(payload, f)

        self.precomputed_tensors = {
            'n_feat': [torch.from_numpy(node_feats_np[item['sub_node_ids']]).float() for item in subgraph_data],
            'e_feat': [torch.from_numpy(edge_feats_np[item['sub_edge_ids']]).float() for item in subgraph_data],
            'Tmats': [torch.from_numpy(item['T']).float() for item in subgraph_data],
            'adjs': [torch.from_numpy(item['adj']).float() for item in subgraph_data],
            'eadjs': [torch.from_numpy(item['eadj']).float() for item in subgraph_data],
            'edge_ids': [item['sub_edge_ids'] for item in subgraph_data]
        }

        # 4. Model Initialization
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
        G = nx.Graph()
        src_np = data.src.cpu().numpy().astype(int)
        tgt_np = data.tgt.cpu().numpy().astype(int)

        # Use the full graph as the context (matching original leakage/context)
        for i in range(len(src_np)):
            u, v = src_np[i], tgt_np[i]
            if G.has_edge(u, v):
                G[u][v]['edge_ids'].append(i)
            else:
                G.add_edge(u, v, edge_ids=[i])

        all_subgraphs = []
        for i in tqdm(range(len(src_np)), desc="Subgraphs"):
            u, v = src_np[i], tgt_np[i]

            # K-hop extraction
            nodes_u = self._extract_k_hop(G, u, v)
            nodes_v = self._extract_k_hop(G, v, u)
            sub_nodes = nodes_u.union(nodes_v)
            sub_g = G.subgraph(sub_nodes).copy()

            # Resolution: Store the structural adj separately from the lookup
            # Original logic: Use a random edge if multiple exist, EXCEPT for the target edge
            edge_lookup = {}
            for eu, ev, edata in sub_g.edges(data=True):
                chosen_id = random.choice(edata['edge_ids'])
                edge_lookup[tuple(sorted((eu, ev)))] = chosen_id

            # Ensure the target edge uses the CURRENT index 'i'
            edge_lookup[tuple(sorted((u, v)))] = i

            # Reorder Nodes by Min Weight (Earliest interaction)
            node_weights = {}
            for node in sub_g.nodes:
                # Look up the min edge ID this node participates in within the SUBGRAPH
                incident_edges = sub_g.edges(node)
                node_weights[node] = min(
                    edge_lookup[tuple(sorted(e))] for e in incident_edges)

            sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1])
            mapping = {old_id: new_id for new_id,
                       (old_id, _) in enumerate(sorted_nodes)}

            # Create STRUCTURAL Adjacency (Binary 0/1)
            new_sub = nx.relabel_nodes(sub_g, mapping)
            adj_binary = nx.adjacency_matrix(
                new_sub, nodelist=sorted(new_sub.nodes()), weight=None)

            # Create Transition Matrix from Binary Adj
            T = self._create_transition_matrix(adj_binary)

            # Create Edge Adj from Binary Adj
            eadj, edge_name = self._create_edge_adj(adj_binary)

            # Map Edge IDs to match the columns of T
            # T columns are built from the order in sp.triu(adj_binary)
            sub_edge_ids = []
            edge_indices = np.nonzero(sp.triu(adj_binary, k=1))

            # We need the original IDs for these edges
            inv_mapping = {v: k for k, v in mapping.items()}
            for r, c in zip(edge_indices[0], edge_indices[1]):
                u_old, v_old = inv_mapping[r], inv_mapping[c]
                sub_edge_ids.append(edge_lookup[tuple(sorted((u_old, v_old)))])

            all_subgraphs.append({
                'adj': self._normalize(adj_binary + sp.eye(adj_binary.shape[0])).toarray(),
                'T': T.toarray(),
                'eadj': self._normalize(eadj).toarray(),
                'sub_node_ids': np.array([old for old, _ in sorted_nodes]),
                'sub_edge_ids': np.array(sub_edge_ids)
            })
        return all_subgraphs

    def _global_preprocessing_old(self, data: TemporalGraph):
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

    def train(self, epochs: int, train_loader=None, val_loader=None, callbacks=None):
        assert self.train_mask is not None
        train_indices = torch.where(self.train_mask)[0].tolist()
        val_indices = torch.where(self.val_mask)[
            0].tolist() if self.val_mask is not None else []

        handler = ExperimentCallbackHandler(callbacks)
        state = TrainingState(model=self)
        handler.on_train_begin(state)

        for epoch in range(epochs):
            self.set_training_mode(True)
            state.epoch = epoch
            handler.on_train_epoch_begin(state)

            # Shuffle just like original code
            random.shuffle(train_indices)

            # Manual batching
            for i in tqdm(range(0, len(train_indices), self.batch_size), desc=f"TRAIN - Epoch {epoch}"):
                batch_idx = train_indices[i: i + self.batch_size]

                self.components.optimizer.zero_grad()

                # Forward Pass
                logits = self._forward_batch(batch_idx)
                labels = self.edge_labels[batch_idx].to(
                    self.device).float()

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels)
                loss.backward()
                self.components.optimizer.step()

                state.loss = loss.item()
                state.step_in_epoch = i // self.batch_size
                state.total_steps += 1

            # Validation step
            if val_indices:
                val_auc = self.evaluate_indices(val_indices)
                print(f"Epoch {epoch} Val AUC: {val_auc}")
                state.val_metrics = {'roc_auc': val_auc}

            handler.on_train_epoch_end(state)
        handler.on_train_end(state)

    def run_inference(
        self,
        loader=None,
        profiler: StreamingProfiler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Overrides base.py inference to use precomputed subgraphs for the test mask.
        """
        import time as _time

        assert self.test_mask is not None
        test_indices = torch.where(self.test_mask)[0].tolist()
        all_probs = []

        self.set_training_mode(False)
        with torch.no_grad():
            for i in range(0, len(test_indices), self.batch_size):
                batch_idx = test_indices[i: i + self.batch_size]

                if profiler is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _t0 = _time.perf_counter()

                logits = self._forward_batch(batch_idx)
                probs = torch.sigmoid(logits)

                if profiler is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _elapsed = _time.perf_counter() - _t0
                    profiler.record(
                        num_edges=len(batch_idx),
                        mean_degree=0.0,
                        elapsed_sec=_elapsed,
                    )

                all_probs.append(probs)

        return self.edge_labels[test_indices], torch.cat(all_probs)

    def _forward_batch(self, batch_indices):
        batch_size = len(batch_indices)

        # Prepare lists for GNN (lists of tensors)
        n_feat = [self.precomputed_tensors['n_feat']
                  [i].to(self.device) for i in batch_indices]
        e_feat_list = [self.precomputed_tensors['e_feat']
                       [i].to(self.device) for i in batch_indices]
        Tmats = [self.precomputed_tensors['Tmats']
                 [i].to(self.device) for i in batch_indices]
        adjs = [self.precomputed_tensors['adjs']
                [i].to(self.device) for i in batch_indices]
        eadjs = [self.precomputed_tensors['eadjs']
                 [i].to(self.device) for i in batch_indices]

        # Prepare padded tensor for Transformer
        e_pad = torch.zeros(batch_size, self.max_edges_all,
                            self.input_dim, device=self.device)
        mask_edge = torch.ones(batch_size, self.max_edges_all, device=self.device)

        for i, idx in enumerate(batch_indices):
            feat = self.precomputed_tensors['e_feat'][idx]
            seq_len = feat.size(0)
            e_pad[i, :seq_len, :] = feat
            mask_edge[i, :seq_len] = 0

        return self.components.model(n_feat, e_feat_list, e_pad, eadjs, adjs, Tmats, mask_edge)

    def evaluate_indices(self, indices):
        self.set_training_mode(False)
        all_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(indices), self.batch_size), desc=f"TEST"):
                batch = indices[i: i + self.batch_size]
                logits = self._forward_batch(batch)
                all_probs.append(torch.sigmoid(logits))

        y_true = self.edge_labels[indices].cpu().numpy()
        y_pred = torch.cat(all_probs).cpu().numpy()
        return roc_auc_score(y_true, y_pred)

    def _train_step(self, snapshot, **kwargs): return 0.0
    def _predict(self, snapshot, **kwargs): return torch.zeros(1)

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
