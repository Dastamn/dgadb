import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from dgadb.models.TADDY.codes.BaseModel import BaseModel

import time
import numpy as np

from sklearn import metrics
from dgadb.models.TADDY.codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL


class DynADModel(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ""
    save_pretrained_path = ""

    def __init__(self, config, args):
        super(DynADModel, self).__init__(config, args)
        self.all_tied_weights_keys = {}  # Fix for transformers compatibility

        self.args = args
        self.config = config
        self.transformer = BaseModel(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, 1)
        self.weight_decay = config.weight_decay
        self.init_weights()

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):
        outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)

        sequence_output = 0
        for i in range(self.config.k + 1):
            sequence_output += outputs[0][:, i, :]
        sequence_output /= float(self.config.k + 1)

        output = self.cls_y(sequence_output)

        return output

    def batch_cut(self, idx_list):
        batch_list = []
        for i in range(0, len(idx_list), self.config.batch_size):
            batch_list.append(idx_list[i: i + self.config.batch_size])
        return batch_list

    def generate_embedding(self, edges):
        num_snap = len(edges)
        # WL_dict = compute_WL(self.data['idx'], np.vstack(edges[:7]))
        WL_dict = compute_zero_WL(self.data["idx"], None)
        batch_hop_dicts = compute_batch_hop(
            self.data["idx"], edges, num_snap, self.data["S"], self.config.k, self.config.window_size
        )
        # print("batch hop dicts", len(batch_hop_dicts))

        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = dicts_to_embeddings(
            self.data["X"], batch_hop_dicts, WL_dict, num_snap
        )
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

    def negative_sampling(self, edges):
        negative_edges = []
        node_list = self.data["idx"]
        num_node = node_list.shape[0]
        for snap_edge in edges:
            num_edge = snap_edge.shape[0]

            negative_edge = snap_edge.copy()
            fake_idx = np.random.choice(num_node, num_edge)
            fake_position = np.random.choice(2, num_edge).tolist()
            fake_idx = node_list[fake_idx]
            negative_edge[np.arange(num_edge), fake_position] = fake_idx

            negative_edges.append(negative_edge)
        return negative_edges
