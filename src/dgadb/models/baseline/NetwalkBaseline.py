import logging
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from src.dgadb.storage import TemporalGraph

logger = logging.getLogger(__name__)


class CliqueAutoencoder(nn.Module):
    def __init__(self, num_nodes: int, hidden: int, walk_len: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden = hidden
        self.walk_len = walk_len

        # Use the same architecture as paper
        self.enc = nn.Linear(num_nodes, hidden, bias=True)
        self.dec = nn.Linear(hidden, num_nodes, bias=True)
        self.act = nn.Sigmoid()

        # Laplacian matrix
        phi = np.ones((walk_len, walk_len), dtype=np.float32) - \
            np.eye(walk_len, dtype=np.float32)
        D = np.diag(np.sum(phi, axis=1))
        L = D - phi
        self.register_buffer("L", torch.tensor(L, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for module in [self.enc, self.dec]:
            # Uniform unit scaling initializer
            fan_in = module.in_features
            limit = np.sqrt(3.0 / max(1, fan_in))
            nn.init.uniform_(module.weight, -limit, limit)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, corrupt_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and corrupt_prob > 0.0:
            # Add uniform noise
            noise = torch.rand_like(x) * 0.1
            x_corrupted = x * (1.0 - corrupt_prob) + noise * corrupt_prob
        else:
            x_corrupted = x

        h = self.act(self.enc(x_corrupted.T))
        recon = self.act(self.dec(h))

        return recon.T.contiguous(), h.T.contiguous()


class NetWalkBaseline:

    def __init__(
        self,
        device: torch.DeviceObjType,
        meta_dict: dict[str, str | int | float],
        hyperparams: dict[str, int | float | str],
        epoch_evaluation_metric: Callable[[torch.FloatTensor, torch.FloatTensor], float],
    ) -> None:
        super().__init__()
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric

        # Walk pipeline
        self.walk_len: int = int(hyperparams.get("walk_length", 3))
        self.walks_per_node: int = int(hyperparams.get("walks_per_node", 20))
        # fraction of train edges for the initial graph
        self.init_percent: float = float(hyperparams.get("init_percent", 0.8))
        self.snap_size: int = int(hyperparams.get("snap_size", 1000))
        self.reservoir_dim: int = int(
            hyperparams.get("reservoir_dim", self.snap_size))
        self.rand_seed: int = int(hyperparams.get("random_state", 24))

        self.print_freq: int = int(hyperparams.get("print_freq", 1))

        # AE and loss hyperparams
        self.hidden: int = int(hyperparams.get("representation_size", 128))
        self.epochs: int = int(hyperparams.get("epochs", 10))
        # batches are over walk columns batch_size * walk_len
        self.batch_size_walks: int = int(hyperparams.get("batch_size", 40))
        self.learning_rate: float = float(
            hyperparams.get("learning_rate", 0.01))
        self.gamma: float = float(hyperparams.get("gama", 340.0))
        self.lamb: float = float(hyperparams.get("lamb", 0.0017))
        self.beta: float = float(hyperparams.get("beta", 1.0))
        self.rho: float = float(hyperparams.get("rho", 0.5))
        self.corrupt_prob: float = float(hyperparams.get("corrupt_prob", 0.0))

        # clustering
        self.kmeans_k: int = int(hyperparams.get("kmeans_k", 8))

        # runtime
        self.num_nodes: Optional[int] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_labels: Optional[torch.Tensor] = None
        self.train_mask: Optional[torch.Tensor] = None
        self.val_mask: Optional[torch.Tensor] = None
        self.test_mask: Optional[torch.Tensor] = None

        # Model & optimizer
        self.ae: Optional[CliqueAutoencoder] = None
        self.opt: Optional[optim.Optimizer] = None

        # Reservoir state
        self._reservoir: dict[int, np.ndarray] = {}
        self._degree: dict[int, int] = {}

        # Walk buffers
        self._prev_walks: list[list[int]] = []
        self._rng = np.random.RandomState(self.rand_seed)

        # Embeddings & clustering
        self._embeddings: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._kmeans: Optional[KMeans] = None

        # paper style anom injection
        # self.inject_test_ratio: float = float(hyperparams.get("inject_test_ratio", 0.10))
        # self.spectral_k: int = int(hyperparams.get("spectral_k", 10))
        # self.injection_seed: int = int(hyperparams.get("injection_seed", 1))

        # logger.info(
        #    f"NetWalkBaseline init: walk_len={self.walk_len}, wpn={self.walks_per_node}, "
        #    f"init_percent={self.init_percent}, snap_size={self.snap_size}, res_dim={self.reservoir_dim}, "
        #   f"hidden={self.hidden}, epochs={self.epochs},   k={self.kmeans_k}"
        # )

    def setup(self, temporal_graph: TemporalGraph) -> None:
        self.num_nodes = int(temporal_graph.num_nodes)
        self.edge_index = temporal_graph.edge_index
        self.edge_labels = temporal_graph.edge_labels
        self.train_mask = temporal_graph.train_mask
        self.val_mask = temporal_graph.val_mask
        self.test_mask = temporal_graph.test_mask

        ei_train = self.edge_index[:, self.train_mask].cpu().numpy().T
        Etr = ei_train.shape[0]
        init_E = max(1, int(self.init_percent * Etr))
        init_edges = ei_train[:init_E, :]
        train_snapshots = []
        cur = init_E
        while cur < Etr:
            train_snapshots.append(
                ei_train[cur: min(cur + self.snap_size, Etr), :])
            cur += self.snap_size

        # Initialize reservoir from the initial graph
        self._build_reservoir(init_edges)

        # Initial walks from the initial train graph
        self._prev_walks = self._initial_random_walks(init_edges)

        # Build autoencoder
        self.ae = CliqueAutoencoder(
            num_nodes=self.num_nodes, hidden=self.hidden, walk_len=self.walk_len).to(self.device)

        self.opt = optim.Adam(self.ae.parameters(), lr=self.learning_rate)

        # store snapshots
        self._train_snapshots = train_snapshots

    def train(self, runnable=None) -> None:
        self._ensure_setup()
        assert self.ae is not None and self.opt is not None and self.num_nodes is not None

        # train on initial walks
        x0 = self._walks_to_onehot(self._prev_walks)
        self._fit_autoencoder(x0)

        # Iterate train snapshots
        # update reservoir
        # generate new walks
        # then train
        for snap_i, edges in enumerate(self._train_snapshots, start=1):
            self._reservoir_update(edges)
            new_walks = self._generate_snapshot_walks(edges)

            # keep walks that don't start from affected nodes
            affected_nodes = set(edges.reshape(-1))

            # keep old walks NOT starting from affected nodes
            old_walks = [w for w in self._prev_walks
                         if w and len(w) > 0 and w[0] not in affected_nodes]

            # combine new and filtered old walks
            combined_walks = new_walks + old_walks
            self._prev_walks = combined_walks

            # train on combined walks
            x = self._walks_to_onehot(combined_walks)
            self._fit_autoencoder(x)

            if (snap_i % 10) == 0:
                logger.info(
                    f"NetWalk train: processed snapshot {snap_i}/{len(self._train_snapshots)}")
            # if runnable is not None:
                # print("runnable not none")
                # print(self._embeddings is None)
            # compute current state for evaluation
            # current_embeddings = self._compute_node_embeddings()
            self._embeddings = self._compute_node_embeddings()

            # fit temporary KMeans
            ei = self.edge_index[:, self.train_mask]
            train_codes = self._edge_codes_from_embeddings(
                self._embeddings, ei)
            self._kmeans = KMeans(
                n_clusters=self.kmeans_k, random_state=self.rand_seed, n_init=10)
            self._kmeans.fit(train_codes)
            self._centroids = self._kmeans.cluster_centers_

            # evaluate on validation
            val_ei = self.edge_index[:, self.val_mask]
            val_codes = self._edge_codes_from_embeddings(
                self._embeddings, val_ei)
            d = cdist(val_codes, self._centroids)
            min_d = d.min(axis=1).astype(np.float32)
            probs = self._minmax(min_d)

            val_labels = self.edge_labels[self.val_mask].cpu().numpy()
            auc = roc_auc_score(val_labels, probs)
            print("val AUC:", auc)
            # report to ray tune
            if runnable is not None:
                runnable(auc, self, snap_i)

        # Final node embedding
        self._embeddings = self._compute_node_embeddings()

        # UNCOMMENT BELOW TO USE NETWALK PAPER'S ANOMALY INJECTION INSTEAD. Expects graph with no anomalies before
        # self._inject_intercommunity_anomalies_spectral(
        #    test_ratio=self.inject_test_ratio,
        #    k=self.spectral_k,
        #    seed=self.injection_seed,
        # )

        # Fit KMeans on train edge encodings
        ei = self.edge_index[:, self.train_mask]
        train_codes = self._edge_codes_from_embeddings(self._embeddings, ei)
        self._kmeans = KMeans(n_clusters=self.kmeans_k,
                              random_state=self.rand_seed, n_init=10)
        self._kmeans.fit(train_codes)
        self._centroids = self._kmeans.cluster_centers_
        logger.info(
            "NetWalk: embeddings learned and k-means fitted on TRAIN embs.")

    def inference(self, split: str = "test"):

        mask = {"train": self.train_mask, "val": self.val_mask,
                "test": self.test_mask}[split]
        ei = self.edge_index[:, mask]
        labels = self.edge_labels[mask].to(self.device).float()

        codes = self._edge_codes_from_embeddings(self._embeddings, ei)
        d = cdist(codes, self._centroids)
        min_d = d.min(axis=1).astype(np.float32)

        probs = self._minmax(min_d)
        return torch.from_numpy(probs).to(self.device), labels

    def _ensure_setup(self):
        if any(x is None for x in [self.num_nodes, self.edge_index, self.edge_labels, self.train_mask]):
            raise RuntimeError("Call setup(TemporalGraph) before train().")

    def _build_reservoir(self, init_edges: np.ndarray) -> None:

        assert self.num_nodes is not None
        adj = {i: [] for i in range(self.num_nodes)}
        for u, v in init_edges:
            if u == v:
                continue
            adj[u].append(v)
            adj[v].append(u)

        self._reservoir = {}
        self._degree = {}
        for v in range(self.num_nodes):
            nbrs = list(set(adj.get(v, [])))
            deg = len(nbrs)
            self._degree[v] = deg
            if deg > 0:
                idx = self._rng.randint(deg, size=self.reservoir_dim)
                self._reservoir[v] = np.array(
                    [nbrs[i] for i in idx], dtype=np.int64)
            else:
                self._reservoir[v] = np.full(
                    self.reservoir_dim, -1, dtype=np.int64)
        logger.info("Reservoir initialized from initial graph.")

    def _reservoir_update(self, edges: np.ndarray) -> None:

        for u, v in edges:
            # u side
            self._degree[u] = self._degree.get(u, 0) + 1
            idx = self._rng.randint(self._degree[u], size=self.reservoir_dim)
            replace = (idx == (self._degree[u] - 1))
            if replace.any():
                self._reservoir[u][replace] = v

            # v side
            self._degree[v] = self._degree.get(v, 0) + 1
            idx = self._rng.randint(self._degree[v], size=self.reservoir_dim)
            replace = (idx == (self._degree[v] - 1))
            if replace.any():
                self._reservoir[v][replace] = u

    def _initial_random_walks(self, init_edges: np.ndarray) -> list[list[int]]:
        assert self.num_nodes is not None
        # Build adjacency from init graph
        adj = {i: [] for i in range(self.num_nodes)}
        for u, v in init_edges:
            if u == v:
                continue
            adj[u].append(v)
            adj[v].append(u)
        for k in adj.keys():
            adj[k] = list(set(adj[k]))

        # nodes = list(range(self.num_nodes))
        nodes = [n for n, nbrs in adj.items() if len(nbrs) > 0]
        walks: list[list[int]] = []
        for _ in range(self.walks_per_node):
            self._rng.shuffle(nodes)
            for start in nodes:
                walk = [start]
                while len(walk) < self.walk_len:
                    cur = walk[-1]
                    nbrs = adj.get(cur, [])
                    if nbrs:
                        x = self._rng.choice(nbrs)
                        if len(nbrs) >= self.walk_len and x in walk:
                            # try another neighbor if possible
                            cand = [nb for nb in nbrs if nb not in walk]
                            x = self._rng.choice(cand) if cand else x
                        walk.append(int(x))
                    else:
                        break
                # pad if short
                while len(walk) < self.walk_len:
                    walk.append(walk[-1])
                walks.append(walk[: self.walk_len])
        return walks

    def _generate_snapshot_walks(self, snapshot_edges: np.ndarray) -> list[list[int]]:
        start_nodes = set(snapshot_edges.reshape(-1))
        walks: list[list[int]] = []

        # For each affected node, generate multiple walks
        for n in start_nodes:
            for _ in range(self.walks_per_node):
                walk = [int(n)]
                cur = int(n)

                # Generate walk by sampling from reservoir
                for step in range(self.walk_len - 1):
                    next_node = self._sample_from_reservoir(cur)
                    walk.append(int(next_node))
                    cur = int(next_node)

                walks.append(walk)

        return walks

    def _sample_from_reservoir(self, node: int) -> int:
        arr = self._reservoir.get(node, None)
        if arr is None:
            return node

        # Filter out invalid nodes
        valid = arr[arr >= 0]
        if valid.size == 0:
            return node

        return int(self._rng.choice(valid))

    def _inject_intercommunity_anomalies_spectral(self, test_ratio: float, k: int, seed: int) -> None:
        import numpy as np
        from sklearn.cluster import SpectralClustering

        assert self.edge_index is not None and self.edge_labels is not None
        assert self.train_mask is not None and self.test_mask is not None

        rng = np.random.RandomState(seed)
        device = self.edge_index.device
        N = int(self.num_nodes)

        A = np.zeros((N, N), dtype=np.float32)
        ei = self.edge_index.detach().cpu().numpy()
        u = ei[0]
        v = ei[1]
        A[u, v] = 1.0
        A[v, u] = 1.0
        np.fill_diagonal(A, 0.0)

        sc = SpectralClustering(
            n_clusters=k, affinity='precomputed',
            n_init=100, assign_labels='discretize',
            random_state=seed
        )
        node_labels = sc.fit_predict(A)

        uu = np.minimum(u, v)
        vv = np.maximum(u, v)
        observed = set(zip(uu.tolist(), vv.tolist()))

        def _inject_for_test():
            idx = self.test_mask.nonzero(
                as_tuple=False).squeeze(1).cpu().numpy()
            m = int(idx.size)
            t = int(np.floor(test_ratio * m))
            if t <= 0:
                return []

            fake_pairs = []
            tries = 0
            while len(fake_pairs) < t and tries < 50:
                need = t - len(fake_pairs)
                ktry = max(4 * need, need)
                a = rng.randint(0, N, size=ktry)
                b = rng.randint(0, N, size=ktry)
                keep = (a != b)
                a = a[keep]
                b = b[keep]
                a2 = np.minimum(a, b)
                b2 = np.maximum(a, b)
                for (aa, bb, aaa, bbb) in zip(a, b, a2, b2):
                    if (aaa, bbb) in observed:
                        continue
                    # inter-community condition
                    if node_labels[aa] == node_labels[bb]:
                        continue
                    observed.add((aaa, bbb))
                    fake_pairs.append((aaa, bbb))
                    if len(fake_pairs) == t:
                        break
                tries += 1
            return fake_pairs

        test_fakes = _inject_for_test()
        if not test_fakes:
            return

        add_ei = torch.tensor(
            test_fakes, dtype=self.edge_index.dtype, device=device).T
        add_labels = torch.ones(
            len(test_fakes), dtype=self.edge_labels.dtype, device=device)  # 1 = anomaly

        start = self.edge_index.size(1)
        self.edge_index = torch.cat([self.edge_index, add_ei], dim=1)
        self.edge_labels = torch.cat([self.edge_labels, add_labels], dim=0)

        add_false = torch.zeros(
            len(test_fakes), dtype=torch.bool, device=device)
        self.train_mask = torch.cat([self.train_mask, add_false], dim=0)
        if self.val_mask is not None:
            self.val_mask = torch.cat(
                [self.val_mask, add_false.clone()], dim=0)
        self.test_mask = torch.cat([self.test_mask, add_false.clone()], dim=0)

        # mark the appended positions as test
        pos = torch.arange(start, start + len(test_fakes), device=device)
        self.test_mask[pos] = True

    def _walks_to_onehot(self, walks: list[list[int]]) -> torch.Tensor:
        assert self.num_nodes is not None
        if not walks:
            return torch.zeros((self.num_nodes, 0), dtype=torch.float32, device=self.device)

        cols = len(walks) * self.walk_len
        X = torch.zeros((self.num_nodes, cols),
                        dtype=torch.float32, device=self.device)
        c = 0
        for w in walks:
            for node in w:
                if 0 <= node < self.num_nodes:
                    X[node, c] = 1.0
                c += 1
        return X

    def _fit_autoencoder(self, X: torch.Tensor) -> None:
        assert self.ae is not None and self.opt is not None

        B = max(1, self.batch_size_walks * self.walk_len)
        num_cols = X.size(1)
        num_batches = max(1, num_cols // B)

        self.ae.train()
        for epoch in range(self.epochs):
            total_loss = 0.0

            for i in range(num_batches):
                s, e = i * B, min((i + 1) * B, X.size(1))
                xb = X[:, s:e]

                # Forward pass
                recon, codes = self.ae(xb, corrupt_prob=self.corrupt_prob)

                # Autoencoder reconstruction loss
                ae_loss = (self.gamma / 2.0) * torch.mean((recon - xb) ** 2)

                # KL divergence loss
                rho_hat = torch.clamp(codes.mean(dim=1), 1e-8, 1 - 1e-8)
                kl_loss = self.beta * torch.mean(
                    self.rho * torch.log(self.rho / rho_hat) +
                    (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat))
                )

                clique_loss = self._compute_clique_loss(codes)

                # maunual weight decay
                weight_decay = 0.0
                for name, param in self.ae.named_parameters():
                    if 'weight' in name:
                        weight_decay += (self.lamb / 2.0) * \
                            torch.sum(param ** 2)

                # total loss
                total_loss_batch = clique_loss + ae_loss + kl_loss + weight_decay

                self.opt.zero_grad()
                total_loss_batch.backward()
                self.opt.step()

                total_loss += total_loss_batch.item()

            if ((epoch + 1) % self.print_freq) == 0:
                avg_loss = total_loss / max(1, num_batches)
                logger.info(
                    f"[AE] epoch {epoch+1}/{self.epochs} | loss={avg_loss:.4f}")

    def _compute_clique_loss(self, codes: torch.Tensor) -> torch.Tensor:
        """Compute clique loss exactly as in the paper"""
        H, total_cols = codes.shape

        # Reshape codes into walk groups
        num_walks = total_cols // self.walk_len
        if num_walks == 0:
            return torch.tensor(0.0, device=codes.device)

        usable_cols = num_walks * self.walk_len
        codes_reshaped = codes[:, :usable_cols].T.view(
            num_walks, self.walk_len, H)

        # clique constraint
        codes_t = codes_reshaped.transpose(1, 2)

        # Batch matrix multiplication
        left = torch.einsum('nhi,ij->nhj', codes_t, self.ae.L)
        result = torch.einsum('nhi,nik->nhk', left, codes_reshaped)

        traces = torch.diagonal(result, dim1=-2, dim2=-1).sum(-1)
        return traces.mean()

    def _compute_node_embeddings(self) -> np.ndarray:
        assert self.ae is not None and self.num_nodes is not None
        self.ae.eval()
        with torch.no_grad():
            I = torch.eye(self.num_nodes, dtype=torch.float32,
                          device=self.device)
            _, codes = self.ae(I, corrupt_prob=0.0)
        return codes.T.detach().cpu().numpy()

    def _edge_codes_from_embeddings(self, emb: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
        u = edge_index[0].cpu().numpy()
        v = edge_index[1].cpu().numpy()
        su = emb[u, :]  # TODO: nonetype not subscriptable
        sv = emb[v, :]
        # hadamard emb
        # codes = su * sv
        # return codes.astype(np.float32)
        return ((su + sv) * 0.5).astype(np.float32)

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32)
        xmin, xmax = float(x.min()), float(x.max())
        if xmax - xmin < 1e-12:
            return np.full_like(x, 0.5, dtype=np.float32)
        return ((x - xmin) / (xmax - xmin + 1e-12)).astype(np.float32)
