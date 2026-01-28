from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Any
from operator import itemgetter

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..base import BaseADModel, BaseADModelComponents, TrainingState
from ..SAD.model.tgat import TGAT
from ..SAD import datasets as ds
from ..SAD.utils import get_neighbor_finder
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot

logger = logging.getLogger(__name__)

@dataclass
class SADComponents(BaseADModelComponents):
    model: TGAT
    optimizer: torch.optim.Optimizer
    ngh_finder: Any
    edge_features: np.ndarray
    node_features: np.ndarray
    persistent_labels: torch.Tensor 

class SADAD(BaseADModel[SADComponents]):
    def __init__(
        self,
        bipartite: bool = True,
        mode: str = "sad",
        add_scl: bool = False,
        module_type: str = "graph_attention",
        mask_label: bool = False,
        mask_ratio: float = 0.5,
        dev_alpha: float = 1.0,
        dev_beta: float = 1.0,
        anomaly_alpha: float = 1e-1,
        supc_alpha: float = 5e-3,
        memory_size: int = 5000,
        sample_size: int = 2000,
        n_neighbors: int = 20,
        batch_size: int = 256,
        num_data_workers: int = 1,
        input_dim: int = 172,
        hidden_dim: int = 128,
        n_heads: int = 2,
        drop_out: float = 0.2,
        n_layer: int = 2,
        learning_rate: float = 5e-4,
        device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device)
        self.mode = mode
        self.mask_label = mask_label
        self.mask_ratio = mask_ratio
        self.anomaly_alpha = anomaly_alpha
        self.supc_alpha = supc_alpha
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.n_layer = n_layer
        self.learning_rate = learning_rate
        self.module_type = module_type
        self.memory_size = memory_size
        self.sample_size = sample_size

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        print("[SAD] Setting up model logic to match original...", flush=True)

        # 1. Prepare Neighbor Finder
        src = data.src.detach().cpu().numpy()
        tgt = data.tgt.detach().cpu().numpy()
        ts = data.t.detach().cpu().numpy()
        labels = data.edge_labels.cpu() if data.edge_labels is not None else torch.zeros(src.shape[0])
        edge_ids = np.arange(len(src))
        full_data_obj = ds.SADData(src, tgt, ts, edge_ids, labels)
        ngh_finder = get_neighbor_finder(full_data_obj, uniform=False)

        if data.edge_attr is not None:
            edge_features = data.edge_attr.detach().cpu().numpy().astype(np.float32)
            self.input_dim = edge_features.shape[1]
        else:
            edge_features = np.zeros((len(src), self.input_dim), dtype=np.float32)
            print(f"[SAD] No edge features found. Created zeros of shape {edge_features.shape}")
        
        if data.node_attr is not None:
            node_features = data.node_attr.detach().cpu().numpy().astype(np.float32)
        else:
            num_nodes = data.num_nodes
            feat_dim = self.input_dim 
            node_features = np.zeros((num_nodes, feat_dim), dtype=np.float32)
            print(f"[SAD] No node features found. Created zero-matrix of shape ({num_nodes}, {feat_dim})")

        persistent_labels = labels.clone()
        if self.mask_label:
            train_idx = torch.where(data.train_mask)[0].cpu().numpy()
            num_to_mask = int(len(train_idx) * self.mask_ratio)
            mask_idx = np.random.choice(train_idx, num_to_mask, replace=False)
            persistent_labels[mask_idx] = -1
            print(f"[SAD] Masked {num_to_mask} training labels.")

        arg_dict = {
            "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads, "drop_out": self.drop_out,
            "n_layer": self.n_layer, "module_type": self.module_type,
            "mode": self.mode, "memory_size": self.memory_size, "sample_size": self.sample_size,
        }
        model = TGAT(arg_dict, self.device).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        self._components = SADComponents(
            model=model, optimizer=optimizer, ngh_finder=ngh_finder,
            edge_features=edge_features, node_features=node_features,
            persistent_labels=persistent_labels
        ).to(self.device)

    def _criterion(self, prediction_dict, labels):
        filtered_pred = {}
        valid_mask = labels > -1
        
        for key, value in prediction_dict.items():
            if key not in ['root_embedding', 'group', 'dev']:
                filtered_pred[key] = value[valid_mask]
            else:
                filtered_pred[key] = value

        valid_labels = labels[valid_mask]
        if len(valid_labels) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        logits = filtered_pred['logits']
        loss_classify = F.binary_cross_entropy_with_logits(logits, valid_labels.float())

        loss = loss_classify.clone()
        loss_anomaly = torch.tensor(0.0, device=self.device)
        loss_supc = torch.tensor(0.0, device=self.device)

        if self.mode == 'sad':
            loss_anomaly = self.components.model.gdn.dev_loss(
                torch.squeeze(valid_labels),
                torch.squeeze(filtered_pred['anom_score']),
                torch.squeeze(filtered_pred['time'])
            )
            loss_supc = self.components.model.suploss(
                filtered_pred['root_embedding'], filtered_pred['group'], filtered_pred['dev']
            )
            loss += self.anomaly_alpha * loss_anomaly + self.supc_alpha * loss_supc

        return loss, loss_classify, loss_anomaly, loss_supc

    def _process_batch_data(self, sources, timestamps, labels):
        batch_items = []
        for i in range(len(sources)):
            src_node = sources[i]
            ts = timestamps[i]
            
            neigh_edge, neigh_time, neigh_idx = self.components.ngh_finder.get_temporal_neighbor_all(
                src_node, ts, self.n_layer, self.n_neighbors
            )
            
            edge_feat = self.components.edge_features[neigh_idx].astype(np.float32)
            if neigh_edge.shape[0] == 0: # Padding
                neigh_edge = np.array([[src_node, src_node]])
                neigh_time = np.array([0.0])
                edge_feat = np.zeros([1, edge_feat.shape[1]], dtype=np.float32)
                neigh_idx = np.array([0])

            batch_items.append({
                "src_center_node": src_node,
                "src_neigh_edge": neigh_edge,
                "src_edge_feat": edge_feat,
                "src_edge_to_ts": ts - neigh_time,
                "current_time": ts,
            })

        src_neigh_edge_all = np.concatenate([b['src_neigh_edge'] for b in batch_items], axis=0)
        centers = np.array([b['src_center_node'] for b in batch_items])
        
        # Create unique mappings per batch to isolate subgraphs
        batch_idx = []
        for i, b in enumerate(batch_items):
            batch_idx.extend([i + 1] * len(b['src_neigh_edge']))
        
        org_node_ids = []
        for i, edge in enumerate(src_neigh_edge_all):
            b_id = batch_idx[i]
            org_node_ids.append(f"{b_id}_{edge[0]}")
            org_node_ids.append(f"{b_id}_{edge[1]}")
        for i, c in enumerate(centers):
            org_node_ids.append(f"{i+1}_{c}")
            
        unique_nodes = list(set(org_node_ids))
        reid_map = {node: i for i, node in enumerate(unique_nodes)}
        
        # Reconstruct batch tensors
        new_edges = []
        for i, edge in enumerate(src_neigh_edge_all):
            b_id = batch_idx[i]
            new_edges.append([reid_map[f"{b_id}_{edge[0]}"], reid_map[f"{b_id}_{edge[1]}"]])
            
        new_centers = [reid_map[f"{i+1}_{c}"] for i, c in enumerate(centers)]
        
        # Map back to original node features
        true_node_ids = [int(x.split('_')[1]) for x in unique_nodes]
        batch_node_features = self.components.node_features[true_node_ids]

        return {
            "src_edge_feat": torch.from_numpy(np.concatenate([b['src_edge_feat'] for b in batch_items])).to(self.device),
            "src_edge_to_time": torch.from_numpy(np.concatenate([b['src_edge_to_ts'] for b in batch_items]).astype(np.float32)).to(self.device),
            "src_center_node_idx": torch.tensor(new_centers).to(self.device),
            "src_neigh_edge": torch.tensor(new_edges).to(self.device),
            "src_node_features": torch.from_numpy(batch_node_features).to(self.device),
            "current_time": torch.tensor([b['current_time'] for b in batch_items]).float().to(self.device),
            "labels": torch.from_numpy(labels).to(self.device)
        }

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        self.set_training_mode(True)
        model = self.components.model
        optimizer = self.components.optimizer

        # Get edges and use PERSISTENT labels (which contain the static mask)
        indices = snapshot.current._indices # Get global indices from the view
        src = snapshot.current.src.cpu().numpy()
        ts = snapshot.current.t.cpu().numpy()
        labels = self.components.persistent_labels[indices].cpu().numpy()

        total_loss = 0
        num_batches = 0
        for i in range(0, len(src), self.batch_size):
            end = i + self.batch_size
            batch_data = self._process_batch_data(src[i:end], ts[i:end], labels[i:end])
            
            optimizer.zero_grad()
            output = model(
                batch_data["src_edge_feat"], batch_data["src_edge_to_time"],
                batch_data["src_center_node_idx"], batch_data["src_neigh_edge"],
                batch_data["src_node_features"], batch_data["current_time"],
                batch_data["labels"]
            )
            
            loss, _, _, _ = self._criterion(output, batch_data["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        self.set_training_mode(False)
        src = snapshot.current.src.cpu().numpy()
        ts = snapshot.current.t.cpu().numpy()
        assert  snapshot.current.edge_labels is not None
        labels = snapshot.current.edge_labels.cpu().numpy()

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(src), self.batch_size):
                end = i + self.batch_size
                batch_data = self._process_batch_data(src[i:end], ts[i:end], labels[i:end])
                output = self.components.model(
                    batch_data["src_edge_feat"], batch_data["src_edge_to_time"],
                    batch_data["src_center_node_idx"], batch_data["src_neigh_edge"],
                    batch_data["src_node_features"], batch_data["current_time"],
                    batch_data["labels"]
                )
                # Matches train.py: pred_score = x['logits'].sigmoid()
                probs = output["logits"].sigmoid().view(-1)
                all_probs.append(probs)

        return torch.cat(all_probs) if all_probs else torch.empty(0, device=self.device)

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model': self.components.model.state_dict(),
            'opt': self.components.optimizer.state_dict(),
            'labels': self.components.persistent_labels
        }, os.path.join(save_dir, "model.pt"))

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs):
        pass