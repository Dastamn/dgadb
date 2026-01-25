from src.dgadb.storage import TemporalGraph

import os
import json
import torch
import random
import numpy as np
import logging
import sys
from typing import Literal
from datetime import datetime, timezone
from tqdm.auto import tqdm

from sklearn.cluster import SpectralClustering
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import shortest_path, laplacian


class AnomalyInjector:
    def __init__(self, tg: TemporalGraph, max_k: int = 50, cache_dir: str = "processed"):
        self.tg = tg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.current_group_id = 1

        self.dataset_name = self.tg.metadata.get("dataset_name")
        if not self.dataset_name:
            raise ValueError(
                "TemporalGraph metadata must contain 'dataset_name' for caching logic.")

        self.cache_path = os.path.join(
            cache_dir, self.dataset_name, "analysis")
        os.makedirs(self.cache_path, exist_ok=True)

        self.stats_file = os.path.join(self.cache_path, "graph_stats.json")
        self.comm_file = os.path.join(self.cache_path, "communities.npy")

        if self.tg.edge_labels is not None and self.tg.edge_labels.sum() > 0:
            raise ValueError("Graph already contains anomalies.")

        self.seen_edges: set[tuple[int, int]] = set()
        self._build_registry_and_cache()

        self._load_or_compute_analysis(max_k)

        self.dur_map = {"small": 0.001, "medium": 0.01, "large": 0.05}

    def _load_or_compute_analysis(self, max_k: int):
        if os.path.exists(self.stats_file) and os.path.exists(self.comm_file):
            self.logger.info(
                f"Loading cached graph analysis from {self.cache_path}...")

            with open(self.stats_file, "r") as f:
                self.stats = json.load(f)

            labels = np.load(self.comm_file)
            self.cluster_map = {}
            for idx, cid in enumerate(labels):
                self.cluster_map.setdefault(int(cid), []).append(idx)
            self.optimal_k = len(self.cluster_map)
        else:
            self.logger.info(
                "No cache found. Performing graph analysis (this may take a while)...")
            adj_train = self._get_train_adj()

            self.optimal_k = self._estimate_optimal_k(adj_train, max_k)
            self.logger.info(f"Optimal numnber of clusters: {self.optimal_k}.")
            labels = self._compute_communities_labels(
                adj_train, self.optimal_k)

            self.cluster_map = {}
            for idx, cid in enumerate(labels):
                self.cluster_map.setdefault(int(cid), []).append(idx)

            self.stats = self._compute_graph_stats(adj_train)

            self.logger.info(f"Saving analysis cache to {self.cache_path}...")
            np.save(self.comm_file, labels)
            with open(self.stats_file, "w") as f:
                json.dump(self.stats, f, indent=4)

    def _build_registry_and_cache(self):
        self.node_to_last_msg = {}
        src_np = self.tg.src.cpu().numpy()
        tgt_np = self.tg.tgt.cpu().numpy()
        times = self.tg.t.cpu().numpy()
        msgs = self.tg.msg

        sorted_indices = np.argsort(times)
        for idx in sorted_indices:
            u, v = int(src_np[idx]), int(tgt_np[idx])
            self.seen_edges.add((u, v))
            self.seen_edges.add((v, u))
            self.node_to_last_msg[u] = msgs[idx]

    def _compute_graph_stats(self, adj: csr_matrix) -> dict:
        degrees = np.array(adj.sum(axis=1)).flatten()
        sampled_distances = []
        num_nodes = adj.shape[0]

        for _ in range(500):
            start_node = random.randint(0, num_nodes - 1)
            dist_matrix = shortest_path(
                adj, directed=False, indices=start_node, unweighted=True)
            reachable = dist_matrix[np.isfinite(
                dist_matrix) & (dist_matrix > 0)]
            if len(reachable) > 0:
                sampled_distances.append(float(np.max(reachable)))

        avg_cluster_size = num_nodes / self.optimal_k

        return {
            "mean_degree": float(np.mean(degrees)),
            "std_degree": float(np.std(degrees)),
            "est_diameter": float(np.percentile(sampled_distances, 90)) if sampled_distances else 5.0,
            "avg_cluster_size": float(avg_cluster_size)
        }

    def _compute_communities_labels(self, adj, n_clusters) -> np.ndarray:
        model = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize', random_state=42)
        return model.fit_predict(adj)

    def _inject_burst(self, split: str, duration: float) -> int:
        center = random.randint(0, self.tg.num_nodes - 1)
        k = int(self.stats["mean_degree"] + 3 * self.stats["std_degree"])
        k = max(10, min(k, 100))
        others = [random.randint(0, self.tg.num_nodes - 1) for _ in range(k)]
        return self._append_and_register([center]*k, others, split, duration, self.current_group_id)

    def _inject_clique(self, split: str, duration: float) -> int:
        k = int(np.sqrt(self.stats["avg_cluster_size"]) * 2)
        k = max(4, min(k, 15))
        nodes = random.sample(range(self.tg.num_nodes), k)
        src, tgt = [], []
        for i in range(k):
            for j in range(i + 1, k):
                src.append(nodes[i])
                tgt.append(nodes[j])
        return self._append_and_register(src, tgt, split, duration, self.current_group_id)

    def _inject_path(self, split: str, duration: float) -> int:
        length = int(self.stats["est_diameter"] + 1)
        length = max(5, min(length, 20))
        nodes = random.sample(range(self.tg.num_nodes), length + 1)
        return self._append_and_register(nodes[:-1], nodes[1:], split, duration, self.current_group_id)

    def _inject_bridge(self, split: str, duration: float) -> int:
        c1, c2 = random.sample(list(self.cluster_map.keys()), 2)
        u, v = random.choice(self.cluster_map[c1]), random.choice(
            self.cluster_map[c2])
        return self._append_and_register([u], [v], split, duration, self.current_group_id)

    def _inject_random(self, split: str, duration: float) -> int:
        n = self.tg.num_nodes
        batch_size = 10
        us = [random.randint(0, n-1) for _ in range(batch_size)]
        vs = [random.randint(0, n-1) for _ in range(batch_size)]
        return self._append_and_register(us, vs, split, duration, self.current_group_id)

    def generate_anomalous_samples(
        self,
        anom_type: Literal["random", "burst", "clique", "path", "bridge"],
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=0.0,
        duration_type: Literal["small", "medium", "large"] = "medium"
    ):
        meta = {
            "is_injected": True,
            "last_injection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "type": anom_type,
            "duration_category": duration_type,
            "duration_rates": self.dur_map,
            "graph_stats": self.stats,
            "splits": {
                "train": {"ratio": train_ratio, "requested": 0, "generated": 0},
                "val": {"ratio": val_ratio, "requested": 0, "generated": 0},
                "test": {"ratio": test_ratio, "requested": 0, "generated": 0}
            },
            "total_generated": 0
        }

        for split, ratio in [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]:
            if ratio <= 0:
                continue
            req, gen = self._inject_split(
                anom_type, ratio, split, duration_type)
            meta["splits"][split]["requested"], meta["splits"][split]["generated"] = req, gen
            meta["total_generated"] += gen

        self._sort_by_time()
        self.tg.metadata["anomaly_injection"] = meta
        return self.tg

    def _inject_split(self, anom_type: str, ratio: float, split: str, duration_type: str) -> tuple[int, int]:
        mask = getattr(self.tg, f"{split}_mask", None)
        if mask is None:
            return 0, 0
        target_count = int(mask.sum().item() * ratio)
        if target_count == 0:
            return 0, 0
        t_split = self.tg.t[mask]
        duration = float(t_split.max() - t_split.min()) * \
            self.dur_map.get(duration_type, 0.01)

        pbar = tqdm(total=target_count,
                    desc=f"Injecting {anom_type} -> {split}", file=sys.stdout, mininterval=0)

        edges_added, attempts = 0, 0
        max_attempts = target_count * 20

        while edges_added < target_count and attempts < max_attempts:
            added = 0
            if anom_type == "random":
                added = self._inject_random(split, duration)
            elif anom_type == "bridge":
                added = self._inject_bridge(split, duration)
            elif anom_type == "burst":
                added = self._inject_burst(split, duration)
            elif anom_type == "clique":
                added = self._inject_clique(split, duration)
            elif anom_type == "path":
                added = self._inject_path(split, duration)
            if added > 0:
                pbar.update(min(added, target_count - edges_added))
                edges_added += added
                self.current_group_id += 1
            else:
                attempts += 1
        pbar.close()
        return target_count, edges_added

    def _sort_by_time(self):
        perm = torch.argsort(self.tg.t)
        self.tg.src, self.tg.tgt, self.tg.t, self.tg.msg = self.tg.src[
            perm], self.tg.tgt[perm], self.tg.t[perm], self.tg.msg[perm]
        if self.tg.edge_labels is not None:
            self.tg.edge_labels = self.tg.edge_labels[perm]
        self.tg.train_mask = self.tg.train_mask[perm]
        self.tg.test_mask = self.tg.test_mask[perm]
        if self.tg.val_mask is not None:
            self.tg.val_mask = self.tg.val_mask[perm]
        if "anomaly_group_ids" in self.tg.metadata:
            self.tg.metadata["anomaly_group_ids"] = self.tg.metadata["anomaly_group_ids"][perm]

    def _append_and_register(self, src_list, tgt_list, split, duration, group_id):
        valid_src, valid_tgt = [], []
        for u, v in zip(src_list, tgt_list):
            if (u != v) and (u, v) not in self.seen_edges and (v, u) not in self.seen_edges:
                valid_src.append(u)
                valid_tgt.append(v)
                self.seen_edges.add((u, v))
                self.seen_edges.add((v, u))

        if not valid_src:
            return 0

        num_new = len(valid_src)
        device = self.tg.device
        default_msg = torch.zeros(self.tg.msg.size(1), device=device)
        new_msgs = torch.stack([self.node_to_last_msg.get(
            u, default_msg) for u in valid_src]).to(device)

        mask = getattr(self.tg, f"{split}_mask")
        t_min, t_max = float(self.tg.t[mask].min()), float(
            self.tg.t[mask].max())
        t_start = random.uniform(t_min, max(t_min, t_max - duration))
        new_t = torch.empty(num_new, device=device).uniform_(
            t_start, t_start + duration).sort()[0]

        self.tg.src = torch.cat([self.tg.src, torch.tensor(
            valid_src, dtype=torch.long, device=device)])
        self.tg.tgt = torch.cat([self.tg.tgt, torch.tensor(
            valid_tgt, dtype=torch.long, device=device)])
        self.tg.t = torch.cat([self.tg.t, new_t])
        self.tg.msg = torch.cat([self.tg.msg, new_msgs])

        if self.tg.edge_labels is None:
            self.tg.edge_labels = torch.zeros(
                self.tg.num_edges - num_new, dtype=torch.long, device=device)

        self.tg.edge_labels = torch.cat(
            [self.tg.edge_labels, torch.ones(num_new, dtype=torch.long, device=device)])

        for m_name in ["train_mask", "val_mask", "test_mask"]:
            m_attr = getattr(self.tg, m_name)
            if m_attr is not None:
                setattr(self.tg, m_name, torch.cat([m_attr, torch.full(
                    (num_new,), (m_name.startswith(split)), dtype=torch.bool, device=device)]))

        if "anomaly_group_ids" not in self.tg.metadata:
            self.tg.metadata["anomaly_group_ids"] = torch.zeros(
                self.tg.num_edges - num_new, dtype=torch.long, device=device)
        self.tg.metadata["anomaly_group_ids"] = torch.cat([self.tg.metadata["anomaly_group_ids"], torch.full(
            (num_new,), group_id, dtype=torch.long, device=device)])
        return num_new

    def _get_train_adj(self) -> csr_matrix:
        m = self.tg.train_mask
        idx = self.tg.edge_index[:, m].cpu()
        v = torch.ones(idx.size(1))
        N = self.tg.num_nodes
        coo = coo_matrix(
            (v.numpy(), (idx[0].numpy(), idx[1].numpy())), shape=(N, N))
        return (coo + coo.T).tocsr()

    def _estimate_optimal_k(self, adj, max_k):
        try:
            L = laplacian(adj, normed=True)
            e = eigsh(
                L, k=min(max_k + 1, adj.shape[0]-1), which='SM', return_eigenvectors=False)
            return int(np.argmax(np.diff(e)[1:]) + 2)
        except:
            return 10


if __name__ == "__main__":
    import logging
    from src.dgadb.storage.temporal_graph import TemporalGraphLoader, TemporalGraphLoaderNew

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # loader = TemporalGraphLoader()
    # tg = loader.load("bitcoin-otc", create_if_not_found=True)

    # injector = AnomalyInjector(tg)

    # tg_anom = injector.generate_anomalous_samples(
    #     anom_type="path",
    #     val_ratio=0.1,
    #     test_ratio=0.1,
    #     duration_type="medium"
    # )

    loader = TemporalGraphLoaderNew()
    tg = loader.load("bitcoin-alpha", "bridge",
                     anom_test_ratio=.1, create_if_not_found=True)

    from pprint import pprint
    pprint(tg.metadata["anomaly_injection"])
    tg.describe()
