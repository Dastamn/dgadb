import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Self

from scipy.sparse.linalg import eigsh
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling, get_laplacian, to_scipy_sparse_matrix
from torch_geometric.nn import Node2Vec

from dgadb.models.base import BaseADModel, BaseADModelComponents
from dgadb.storage.temporal_graph import TemporalGraph
from dgadb.storage.temporal_snapshot import TemporalGraphSnapshot

from .model import Generative, Contrastive, FCC

from tqdm import tqdm

@dataclass
class RustGraphComponents(BaseADModelComponents):
    encoder: Generative
    contrastive: Contrastive
    fcc: FCC
    decoder: InnerProductDecoder
    linear: nn.Sequential
    optimizer: torch.optim.Optimizer
    # If no node_attr provided, we use an embedding layer
    # embedding: Optional[nn.Embedding] = None


class RustGraphAD(BaseADModel[RustGraphComponents]):
    def __init__(
        self,
        x_dim: int = 256,
        h_dim: int = 256,
        z_dim: int = 256,
        layer_num: int = 2,
        window: int = 1,
        learning_rate: float = 0.001,
        bce_weight: float = 1.0,
        reg_weight: float = 1.0,
        gen_weight: float = 1.0,
        con_weight: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device)
        self.config = {"x_dim": x_dim, "h_dim": h_dim, "z_dim": z_dim, "layer_num": layer_num, "window": window}
        self.lr = learning_rate
        self.weights = {"bce": bce_weight, "reg": reg_weight, "gen": gen_weight, "con": con_weight}

        # Temporal state variables
        self.h_t = None
        self.pre_ev = None
        self.all_z = []
        self.all_node_idx = []
        self.y_rect_buffer = {}  # Maps snapshot_id to rectified labels

        self.node_attr = None

    def setup(self, data: TemporalGraph, **kwargs) -> None:
        num_nodes = data.num_nodes
        x_dim = self.config["x_dim"]

        embedding = None
        if data.node_attr is not None:
            self.logger.info("Using provided node attributes.")
            self.node_attr = data.node_attr
        else:
            # We train on the full edge list to get global structural context
            self.logger.info(
                f"No node_attr found. Training Node2Vec ({x_dim}) on full graph structure (transductive)..."
            )
            # We train on the full edge list to get global structural context
            edge_index = data.edge_index.to(self.device)

            n2v = Node2Vec(
                edge_index, embedding_dim=x_dim, walk_length=20, context_size=10, walks_per_node=10, num_nodes=num_nodes
            ).to(self.device)

            loader = n2v.loader(batch_size=128, shuffle=True)
            optimizer = torch.optim.Adam(n2v.parameters(), lr=0.01)

            n2v.train()
            for _ in tqdm(range(10), desc="Node2Vec"):
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = n2v.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                # self.logger.info(f"Node2Vec Epoch {epoch}: Loss {total_loss/len(loader):.4f}")

            self.static_x = n2v().detach()

        # if data.node_attr is None:
        #     embedding = nn.Embedding(num_nodes, self.config["x_dim"])

        encoder = Generative(
            self.config["x_dim"], self.config["h_dim"], self.config["z_dim"], self.config["layer_num"], self.device
        )
        contrastive = Contrastive(self.device, self.config["z_dim"], self.config["window"])
        fcc = FCC(self.config["z_dim"], 1, self.device)
        decoder = InnerProductDecoder()
        linear = nn.Sequential(nn.Linear(self.config["z_dim"], self.config["x_dim"]), nn.ReLU())

        params = list(encoder.parameters()) + list(fcc.parameters()) + list(linear.parameters())
        if embedding:
            params += list(embedding.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0.01)

        self._components = RustGraphComponents(
            encoder=encoder,
            contrastive=contrastive,
            fcc=fcc,
            decoder=decoder,
            linear=linear,
            optimizer=optimizer,
            # embedding=embedding
        ).to(self.device)

    def _get_node_features(self, graph_view):
        return self.static_x
        # if self.components.embedding:
        #     indices = torch.arange(
        #         self.components.embedding.num_embeddings, device=self.device)
        #     return self.components.embedding(indices)
        # return graph_view.node_attr.to(self.device)

    def _compute_snapshot_ev(self, graph_view):
        edge_index = graph_view.edge_index
        num_nodes = graph_view.num_nodes

        # Calculate Laplacian on CPU
        edge_index_cpu, edge_weight_cpu = get_laplacian(edge_index.cpu(), num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(edge_index_cpu, edge_weight_cpu, num_nodes)

        # Compute largest eigenvalue/vector
        _, ev = eigsh(L, k=1, which="LM", return_eigenvectors=True)
        return torch.from_numpy(ev).to(self.device)

    def _train_step(self, snapshot: TemporalGraphSnapshot, **kwargs) -> float:
        c = self.components
        curr = snapshot.current
        snap_id = snapshot.snapshot_id

        # Handle reset at start of epoch (step_in_epoch is passed by BaseADModel.train)
        if snap_id == 0:
            self.h_t = None
            self.pre_ev = None
            self.all_z = []
            self.all_node_idx = []
        else:
            if self.h_t is not None:
                self.h_t = self.h_t.detach()

            self.all_z = [z.detach() for z in self.all_z]

        # Structural Change (Spectral Diff)
        ev = self._compute_snapshot_ev(curr)
        if self.pre_ev is None:
            diff = torch.zeros(ev.size(0), 1, device=self.device)
        else:
            # Handle potential node size changes if graph grows
            min_size = min(ev.size(0), self.pre_ev.size(0))
            diff = torch.zeros(ev.size(0), 1, device=self.device)
            diff[:min_size] = torch.abs(ev[:min_size] - self.pre_ev[:min_size])
        self.pre_ev = ev

        # Setup Features and Hidden State
        x = self._get_node_features(curr)
        if self.h_t is None:
            self.h_t = torch.zeros(self.config["layer_num"], x.size(0), self.config["h_dim"], device=self.device)

        # Forward VGRNN
        # Original: (prior_params), (enc_params), z_t, h_t
        (p_m, p_s), (e_m, e_s), z_t, self.h_t = c.encoder(x, self.h_t, diff, curr.edge_index.to(self.device))

        # Anomaly Prediction and Label Rectification
        edge_index = curr.edge_index.to(self.device)
        edge_emb = z_t[edge_index[0]] + z_t[edge_index[1]]
        edge_score = c.fcc(edge_emb)

        # Use rectified labels from buffer if available, else ground truth
        y_raw = curr.edge_labels.unsqueeze(1).float().to(self.device)
        y_true = 1.0 - y_raw
        y_actual = self.y_rect_buffer.get(snap_id, y_true)

        # Calculate Losses
        bce_loss = F.binary_cross_entropy(edge_score, y_actual)

        # Update Rectifier Buffer
        label_rect = c.decoder(z_t, edge_index, sigmoid=True).unsqueeze(1)
        self.y_rect_buffer[snap_id] = (0.9 * y_actual + 0.1 * label_rect).detach()
        reg_loss = torch.norm(label_rect - edge_score, dim=1, p=2).mean()

        # Generative Losses (KLD + Reconstruction)
        # Extract indices of nodes present in this snapshot
        node_idx = torch.unique(edge_index)
        kld_loss = self._kld_gauss(e_m[node_idx], e_s[node_idx], p_m[node_idx], p_s[node_idx])
        recon_loss = self._recon_loss(z_t, x, edge_index)

        # Temporal Contrastive Loss
        self.all_z.append(z_t)
        self.all_node_idx.append(node_idx)
        con_loss = torch.tensor(0.0, device=self.device)
        if len(self.all_z) > self.config["window"]:
            con_loss = c.contrastive(
                self.all_z[-self.config["window"] - 1 :], self.all_node_idx[-self.config["window"] - 1 :]
            )

        total_loss = (
            self.weights["bce"] * bce_loss
            + self.weights["reg"] * reg_loss
            + self.weights["gen"] * (recon_loss + kld_loss)
            + self.weights["con"] * con_loss
        )

        c.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(c.encoder.parameters(), 10)
        c.optimizer.step()

        return total_loss.item()

    def _predict(self, snapshot: TemporalGraphSnapshot, **kwargs) -> torch.Tensor:
        c = self.components
        curr = snapshot.current

        # Ensure hidden state exists (important for inference-only runs)
        x = self._get_node_features(curr)
        if self.h_t is None:
            self.h_t = torch.zeros(self.config["layer_num"], x.size(0), self.config["h_dim"], device=self.device)
            diff = torch.zeros(x.size(0), 1, device=self.device)
        else:
            ev = self._compute_snapshot_ev(curr)
            min_size = min(ev.size(0), self.pre_ev.size(0))
            diff = torch.zeros(ev.size(0), 1, device=self.device)
            diff[:min_size] = torch.abs(ev[:min_size] - self.pre_ev[:min_size])
            self.pre_ev = ev

        with torch.no_grad():
            _, _, z_t, self.h_t = c.encoder(x, self.h_t, diff, curr.edge_index.to(self.device))
            edge_index = curr.edge_index.to(self.device)
            edge_emb = z_t[edge_index[0]] + z_t[edge_index[1]]

            # scores closer to 0 are anomalies in this model's logic
            scores = c.fcc(edge_emb).squeeze()
            return 1.0 - scores

    def _kld_gauss(self, mu1, std1, mu2, std2):
        eps = 1e-8
        kld = (2 * torch.log(std2 + eps) - 2 * torch.log(std1 + eps) + (std1 + eps) ** 2 + (mu1 - mu2) ** 2) / (
            std2 + eps
        ) ** 2 - 1
        return (0.5 / mu1.size(0)) * torch.mean(torch.sum(kld, dim=1))

    def _recon_loss(self, z, x, pos_edge_index):
        x_hat = self.components.linear(z)
        feat_loss = F.mse_loss(x, x_hat)

        pos_probs = self.components.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(pos_probs + 1e-15).mean()

        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=z.size(0))
        neg_probs = self.components.decoder(z, neg_edge_index, sigmoid=True)
        neg_loss = -torch.log(1 - neg_probs + 1e-15).mean()

        return pos_loss + neg_loss + feat_loss

    def save(self, save_dir: str) -> None:
        c = self.components
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "encoder_state_dict": c.encoder.state_dict(),
            "fcc_state_dict": c.fcc.state_dict(),
            "linear_state_dict": c.linear.state_dict(),
            "optimizer_state_dict": c.optimizer.state_dict(),
            "static_x": self.static_x,
        }

        model_path = os.path.join(save_dir, "model.pt")
        torch.save(checkpoint, model_path)

        config_path = os.path.join(save_dir, "config.json")
        config_data = {"params": self.config, "learning_rate": self.lr, "weights": self.weights}
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        self.logger.info(f"Model and Node2Vec features saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: torch.device | str = "cpu", **kwargs) -> Self:
        config_path = os.path.join(load_dir, "config.json")
        model_path = os.path.join(load_dir, "model.pt")

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint files not found in {load_dir}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        instance = cls(
            **config_data["params"],
            learning_rate=config_data["learning_rate"],
            bce_weight=config_data["weights"]["bce"],
            reg_weight=config_data["weights"]["reg"],
            gen_weight=config_data["weights"]["gen"],
            con_weight=config_data["weights"]["con"],
            device=device,
        )

        c_params = config_data["params"]
        encoder = Generative(c_params["x_dim"], c_params["h_dim"], c_params["z_dim"], c_params["layer_num"], device)
        fcc = FCC(c_params["z_dim"], 1, device)
        linear = nn.Sequential(nn.Linear(c_params["z_dim"], c_params["x_dim"]), nn.ReLU())

        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        fcc.load_state_dict(checkpoint["fcc_state_dict"])
        linear.load_state_dict(checkpoint["linear_state_dict"])

        instance.static_x = checkpoint["static_x"].to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(fcc.parameters()) + list(linear.parameters()),
            lr=config_data["learning_rate"],
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        instance._components = RustGraphComponents(
            encoder=encoder,
            contrastive=Contrastive(device, c_params["z_dim"], c_params["window"]),
            fcc=fcc,
            decoder=InnerProductDecoder(),
            linear=linear,
            optimizer=optimizer,
        ).to(device)

        return instance
