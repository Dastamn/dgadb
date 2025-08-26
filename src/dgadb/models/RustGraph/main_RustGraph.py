# expects Canonical edges, sorted by timestamps
# only edges, timestamps are dropped
# in the original they do anomaly injection like add graph, so the model wants a fully labelled dataset.

# creates a node2vec embedding matrix for nodes
# after that they do splitting and anom inj, but that means the embeddings would include test info
# canonicalizes anomalies and drops self-loops


# in the original code they do node2vec embeddings on the entire edgeset which is data leakage between train and test
from src.dgadb.storage.graph import Graph
from src.dgadb.models.RustGraph.model import Model
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
import time
from src.dgadb.models.RustGraph.data import n2v_train
from src.dgadb.models.utils import time_func


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RustGraphModel:
    def __init__(
        self,
        device: torch.device,
        meta_dict: dict[str, any],
        hyperparams: dict[str, any],
        epoch_evaluation_metric,
    ) -> None:
        # runtime
        self.device = device
        self.epoch_evaluation_metric = epoch_evaluation_metric
        self.base_path = os.environ["BASE_PATH"]

        # meta
        self.dataset_name: str = meta_dict["dataset_name"]
        self.train_ratio: float = float(meta_dict["train_ratio"])
        self.val_ratio: float = float(meta_dict["val_ratio"])
        self.anomaly_ratio: float = float(meta_dict["anomaly_ratio"])
        self.has_val: bool = self.val_ratio > 0.0

        self.print_freq: int = int(hyperparams.get("print_freq", 10))

        # hyperparameters
        self.device: str = str(hyperparams.get("device", "cpu"))

        # dataset parameters
        self.snap_size: int = int(hyperparams.get("snap_size", 500))
        self.noise_ratio: float = float(hyperparams.get("noise_ratio", 0.0))

        # training parameters
        self.epochs: int = int(hyperparams.get("num_epochs", 250))
        self.lr: float = float(hyperparams.get("lr", 0.001))
        self.weight_decay: float = float(hyperparams.get("weight_decay", 0.01))

        # hyper-parameters
        self.window: int = int(hyperparams.get("window", 1))
        self.eps: float = float(hyperparams.get("eps", 0.2))
        self.bce_weight: float = float(hyperparams.get("bce_weight", 1))
        self.gen_weight: float = float(hyperparams.get("gen_weight", 1))
        self.con_weight: float = float(hyperparams.get("con_weight", 1))
        self.reg_weight: float = float(hyperparams.get("reg_weight", 1))

        # model parameters
        self.layer_num: int = int(hyperparams.get("layer_num", 2))
        self.x_dim: int = int(hyperparams.get("x_dim", 256))
        self.h_dim: int = int(hyperparams.get("h_dim", 256))
        self.z_dim: int = int(hyperparams.get("z_dim", 256))

        self.h_t = None

    @time_func
    def setup(self, graph: Graph) -> None:

        # node2vec embeddings (should only contain train edges)
        edge_index_train = graph._edges["e_pairs"][:, graph._edges["e_train_mask"]]

        n = graph.num_nodes

        edges_np = edge_index_train.t().cpu().numpy()
        epoch_num = 75

        base_path = os.environ["BASE_PATH"]
        dataset_dir = os.path.dirname(f"{base_path}/src/dgadb/models/RustGraph/n2v_data/")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        n2v_filename = os.path.join(dataset_dir, f"n2v_{self.dataset_name}_{self.x_dim}_{epoch_num}_{self.train_ratio}_{self.val_ratio}")
        if os.path.exists(n2v_filename):
            logger.info(f"Loading n2v features from: {n2v_filename}")
            graph._nodes["n_feat"] = torch.load(n2v_filename, weights_only=True)
        else:
            x = n2v_train(edges_np, self.x_dim, self.device, n, epoch_num)
            graph._nodes["n_feat"] = x
            logger.info(f"Saving n2v features at: {n2v_filename}")
            torch.save(x, n2v_filename)
        

        self.model = Model(
            x_dim=self.x_dim,
            h_dim=self.h_dim,
            z_dim=self.z_dim,
            layer_num=self.layer_num,
            window=self.window,
            eps=self.eps,
            device=self.device,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.graph = graph

    @time_func
    def train(self) -> None:
        y_new = None
        max_auc, max_epoch = -float("inf"), -1
        self.model.train()
        y_new = None
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            with torch.autograd.set_detect_anomaly(True):
                bce_loss, reg_loss, gen_loss, con_loss, y_new, h_t, _, _ = self.model(
                    self.graph, split="train", accumulate=True, y_rect=y_new
                )
                self.h_t = h_t
                loss = bce_loss.mean()
                loss = (
                    self.bce_weight * loss
                    + self.reg_weight * reg_loss
                    + self.gen_weight * gen_loss
                    + self.con_weight * con_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()

            if (epoch + 1) % self.print_freq == 0 or epoch == self.epochs - 1:
                split = "val" if self.has_val else "train"
                preds_per_snap, labels_per_snap = self.inference(split=split)
                preds = np.hstack(preds_per_snap)
                labels = np.hstack(labels_per_snap)
                auc_all = self.epoch_evaluation_metric(labels, preds)
                if auc_all >= max_auc:
                    max_auc, max_epoch = auc_all, epoch
                logger.info(f"Loss: {loss:.4f} in epoch: {epoch},\t")
                logger.info(f"AUC on {split} set: {auc_all:.4f} in epoch: {epoch},\t")

        logger.info(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch},\t")

    @time_func
    def inference(self, split="test"):
        self.model.eval()
        with torch.no_grad():
            _, _, _, _, _, _, pred_list, y_list = self.model(self.graph, split=split, accumulate=True, h_t=self.h_t)

        preds_per_snap = [s.detach().cpu().squeeze() for s in pred_list]
        labels_per_snap = [y.detach().cpu().squeeze() for y in y_list]
        # per snap scores
        # per_snapshot_score = [
        #    self.epoch_evaluation_metric(y_true, s) for y_true, s in zip(labels_per_snap, preds_per_snap)
        # ]
        #preds = np.hstack(preds_per_snap)
        #labels = np.hstack(labels_per_snap)

        #return preds, labels
        return preds_per_snap, labels_per_snap
