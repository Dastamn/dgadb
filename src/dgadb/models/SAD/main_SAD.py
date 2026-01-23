# can handle edge features and node features
# does not use snapshots
# OG implementation uses zeros as node features, of size num_nodes x num_edge_features ??

from dgadb.models.SAD.model.tgat import TGAT
import torch
import torch.nn.functional as F
import torch.utils.data
import dgadb.models.SAD.datasets as ds
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from dgadb.models.utils import time_func
import os
from dgadb.storage.temporal_graph import TemporalGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SADModel:
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
        # self.dataset_name: str = meta_dict["dataset_name"]
        # self.train_ratio: float = float(meta_dict["train_ratio"])
        # self.val_ratio: float = float(meta_dict["val_ratio"])
        # self.anom_train_ratio: float = float(meta_dict["anom_train_ratio"])
        # self.anom_val_ratio: float = float(meta_dict["anom_val_ratio"])
        # self.anom_test_ratio: float = float(meta_dict["anom_test_ratio"])

        # self.has_val: bool = self.val_ratio > 0.0
        # self.dataset_name: str = meta_dict["dataset_name"]
        self.has_val: bool = True

        self.print_freq: int = int(hyperparams.get("print_freq", 1))

        # dataset parameters
        # self.snap_size: int = int(hyperparams.get("snap_size", 500))

        self.bipartite = hyperparams.get("bipartite", True)
        # can be "origin","gdn","sad"
        self.mode = hyperparams.get("mode", "sad")
        self.add_scl = hyperparams.get("add_scl", False)
        # can be "graph_attention","graph_sum"
        self.module_type = hyperparams.get("module_type", "graph_attention")
        self.mask_label = hyperparams.get("mask_label", False)
        self.mask_ratio = hyperparams.get("mask_ratio", 0.5)

        # loss weights
        self.dev_alpha = hyperparams.get("dev_alpha", 1.0)
        self.dev_beta = hyperparams.get("dev_beta", 1.0)
        self.anomaly_alpha = hyperparams.get("anomaly_alpha", 1e-1)
        self.supc_alpha = hyperparams.get("supc_alpha", 5e-3)

        # memory / sampling
        self.memory_size = hyperparams.get("memory_size", 5000)
        self.sample_size = hyperparams.get("sample_size", 2000)

        # data loading
        self.n_neighbors = hyperparams.get("n_neighbors", 20)
        self.batch_size = hyperparams.get("batch_size", 256)
        self.n_epochs = hyperparams.get("epochs", 10)
        self.num_data_workers = hyperparams.get("num_data_workers", 1)
        self.gpus = hyperparams.get("gpus", 1)
        self.accelerator = hyperparams.get("accelerator", "ddp")

        # model
        self.ckpt_file = hyperparams.get("ckpt_file", "./")
        self.input_dim = hyperparams.get("input_dim", 1)
        self.hidden_dim = hyperparams.get("hidden_dim", 128)
        self.n_heads = hyperparams.get("n_heads", 2)
        self.drop_out = hyperparams.get("drop_out", 0.2)
        self.n_layer = hyperparams.get("n_layer", 2)
        self.learning_rate = hyperparams.get("learning_rate", 5e-4)

    @time_func
    def setup(self, graph: TemporalGraph) -> None:
        dataset_train = ds.DygDataset(
            graph, "train", self.n_layer, self.n_neighbors, self.mask_label, self.mask_ratio)
        dataset_val = ds.DygDataset(
            graph, "val", self.n_layer, self.n_neighbors, self.mask_label, self.mask_ratio)
        dataset_test = ds.DygDataset(
            graph, "test", self.n_layer, self.n_neighbors, self.mask_label, self.mask_ratio)

        collate_fn = ds.Collate(dataset_test.node_features)
        print(dataset_test.node_features.shape)

        arg_dict = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "drop_out": self.drop_out,
            "n_layer": self.n_layer,
            "module_type": self.module_type,
            "mode": self.mode,
            "memory_size": self.memory_size,
            "sample_size": self.sample_size,
        }

        backbone = TGAT(arg_dict, self.device)
        self.model = backbone.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            # shuffle=True,
            num_workers=self.num_data_workers,
            pin_memory=True,
            # sampler=dataset.RandomDropSampler(dataset_train, 0.75),   #for reddit
            collate_fn=collate_fn.dyg_collate_fn,
        )

        self.loader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            # shuffle=True,
            num_workers=self.num_data_workers,
            collate_fn=collate_fn.dyg_collate_fn,
        )

        self.loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            # shuffle=True,
            num_workers=self.num_data_workers,
            collate_fn=collate_fn.dyg_collate_fn,
        )

    def _criterion(self, prediction_dict, labels):
        for key, value in prediction_dict.items():
            if key != "root_embedding" and key != "group" and key != "dev":
                prediction_dict[key] = value[labels > -1]

        labels = labels[labels > -1]
        logits = prediction_dict["logits"]

        loss_classify = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none")
        loss_classify = torch.mean(loss_classify)

        loss = loss_classify.clone()
        loss_anomaly = torch.Tensor(0).to(self.device)
        loss_supc = torch.Tensor(0).to(self.device)
        alpha = self.anomaly_alpha  # 1e-1
        beta = self.supc_alpha  # 1e-3
        if self.mode == "sad":
            loss_anomaly = self.model.gdn.dev_loss(
                torch.squeeze(labels),
                torch.squeeze(prediction_dict["anom_score"]),
                torch.squeeze(prediction_dict["time"]),
            )
            loss_supc = self.model.suploss(
                prediction_dict["root_embedding"], prediction_dict["group"], prediction_dict["dev"]
            )
            loss += alpha * loss_anomaly + beta * loss_supc

        return loss, loss_classify, loss_anomaly, loss_supc

    def _eval_epoch(self, dataset):
        m_loss, m_pred, m_label = [], [], []
        m_dev = []
        with torch.no_grad():
            self.model.eval()
            for batch_sample in dataset:
                x = self.model(
                    batch_sample["src_edge_feat"].to(self.device),
                    batch_sample["src_edge_to_time"].to(self.device),
                    batch_sample["src_center_node_idx"].to(self.device),
                    batch_sample["src_neigh_edge"].to(self.device),
                    batch_sample["src_node_features"].to(self.device),
                    batch_sample["current_time"].to(self.device),
                    batch_sample["labels"].to(self.device),
                )
                y = batch_sample["labels"].to(self.device)
                dev_score = x["dev"].cpu().numpy().flatten()
                m_loss = np.concatenate(
                    (m_loss, self._criterion(x, y)[1].cpu().numpy().flatten()))

                pred_score = x["logits"].sigmoid().cpu().numpy().flatten()
                y = y.cpu().numpy().flatten()
                m_pred = np.concatenate((m_pred, pred_score))
                m_label = np.concatenate((m_label, y))
                m_dev = np.concatenate((m_dev, dev_score))

        score = self.epoch_evaluation_metric(m_label, m_pred)
        return score, np.mean(m_loss), m_dev, m_label

    @time_func
    def train(self, runnable=None) -> None:
        for epoch in range(self.n_epochs):
            ave_loss = 0
            count_flag = 0
            m_loss, auc = [], []
            loss_anomaly_list = []
            loss_class_list = []
            loss_supc_list = []
            dev_score_list = np.array([])
            dev_label_list = np.array([])
            with tqdm(total=len(self.loader_train)) as t:
                for batch_sample in self.loader_train:
                    count_flag += 1
                    t.set_description("Epoch %i" % epoch)
                    self.optimizer.zero_grad()
                    self.model.train()
                    x = self.model(
                        batch_sample["src_edge_feat"].to(self.device),
                        batch_sample["src_edge_to_time"].to(self.device),
                        batch_sample["src_center_node_idx"].to(self.device),
                        batch_sample["src_neigh_edge"].to(self.device),
                        batch_sample["src_node_features"].to(self.device),
                        batch_sample["current_time"].to(self.device),
                        batch_sample["labels"].to(self.device),
                    )
                    y = batch_sample["labels"].to(self.device)
                    dev_score = x["dev"]
                    loss, loss_classify, loss_anomaly, loss_supc = self._criterion(
                        x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1, norm_type=2)
                    self.optimizer.step()

                    # get training results
                    with torch.no_grad():
                        self.model.eval()
                        m_loss.append(loss.item())
                        pred_score = x["logits"].sigmoid()

                        dev_score_list = np.concatenate(
                            (dev_score_list, dev_score.cpu().numpy().flatten()))
                        dev_label_list = np.concatenate(
                            (dev_label_list,
                             batch_sample["labels"].cpu().numpy().flatten())
                        )

                    loss_class_list.append(
                        loss_classify.detach().clone().cpu().numpy().flatten())
                    if self.mode == "gdn":
                        loss_anomaly_list.append(
                            loss_anomaly.detach().clone().cpu().numpy().flatten())
                        t.set_postfix(loss=np.mean(loss_class_list),
                                      loss_anomaly=np.mean(loss_anomaly_list))
                    elif self.mode == "sad":
                        loss_anomaly_list.append(
                            loss_anomaly.detach().clone().cpu().numpy().flatten())
                        loss_supc_list.append(
                            loss_supc.detach().clone().cpu().numpy().flatten())
                        t.set_postfix(
                            loss=np.mean(loss_class_list),
                            loss_anomaly=np.mean(loss_anomaly_list),
                            loss_sup=np.mean(loss_supc_list),
                        )
                    else:
                        t.set_postfix(loss=np.mean(loss_class_list))
                    t.update(1)
            logger.info(
                f"Epoch {epoch} - train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, anomaly loss: {np.mean(loss_anomaly_list)}, sup loss: {np.mean(loss_supc_list)}"
            )
            split = "val" if self.has_val else "train"
            m_pred, m_label = self.inference(split=split)
            score = self.epoch_evaluation_metric(m_label, m_pred)
            logger.info(f"Epoch score on {split} data: {score:4f}")
            runnable(score, self, epoch)

    @time_func
    def inference(self, split="test"):
        split_dataset_map = {"train": self.loader_train,
                             "val": self.loader_val, "test": self.loader_test}
        m_loss, m_pred, m_label = [], [], []
        m_dev = []
        with torch.no_grad():
            self.model.eval()
            for batch_sample in split_dataset_map[split]:
                x = self.model(
                    batch_sample["src_edge_feat"].to(self.device),
                    batch_sample["src_edge_to_time"].to(self.device),
                    batch_sample["src_center_node_idx"].to(self.device),
                    batch_sample["src_neigh_edge"].to(self.device),
                    batch_sample["src_node_features"].to(self.device),
                    batch_sample["current_time"].to(self.device),
                    batch_sample["labels"].to(self.device),
                )
                y = batch_sample["labels"].to(self.device)
                dev_score = x["dev"].cpu().numpy().flatten()
                m_loss = np.concatenate(
                    (m_loss, self._criterion(x, y)[1].cpu().numpy().flatten()))

                pred_score = x["logits"].sigmoid().cpu().numpy().flatten()
                y = y.cpu().numpy().flatten()
                m_pred = np.concatenate((m_pred, pred_score))
                m_label = np.concatenate((m_label, y))
                m_dev = np.concatenate((m_dev, dev_score))

        return m_pred, m_label
