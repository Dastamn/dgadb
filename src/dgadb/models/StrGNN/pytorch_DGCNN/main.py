import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from dgadb.models.StrGNN.pytorch_DGCNN.DGCNN_embedding import DGCNN
from dgadb.models.StrGNN.pytorch_DGCNN.mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from dgadb.models.StrGNN.pytorch_DGCNN.util import cmd_args, load_data
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


class Classifier(nn.Module):
    def __init__(
        self,
        gm="DGCNN",
        latent_dim=32,
        out_dim=0,
        feat_dim=0,
        attr_dim=0,
        edge_feat_dim=0,
        sortpooling_k=0.6,
        conv1d_activation="ReLU",
        hidden=50,
        num_class=2,
        dropout=0.5,
        mode="cpu",
        regression=False,
    ):
        super(Classifier, self).__init__()
        # Store parameters as instance variables
        self.gm = gm
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.attr_dim = attr_dim
        self.edge_feat_dim = edge_feat_dim
        self.sortpooling_k = sortpooling_k
        self.conv1d_activation = conv1d_activation
        self.hidden = hidden
        self.num_class = num_class
        self.dropout = dropout
        self.mode = mode
        self.device = torch.device("cuda") if mode == "gpu" else torch.device("cpu")
        self.regression = regression

        if self.gm == "DGCNN":
            model = DGCNN
        else:
            print(("unknown gm %s" % self.gm))
            sys.exit()

        if self.gm == "DGCNN":
            self.gnn = model(
                latent_dim=self.latent_dim,
                output_dim=self.out_dim,
                num_node_feats=self.feat_dim + self.attr_dim,
                num_edge_feats=self.edge_feat_dim,
                k=self.sortpooling_k,
                conv1d_activation=self.conv1d_activation,
            )
        out_dim = self.out_dim
        if out_dim == 0:
            if self.gm == "DGCNN":
                out_dim = self.gnn.dense_dim
            else:
                out_dim = self.latent_dim
        self.mlp = MLPClassifier(
            input_size=out_dim, hidden_size=self.hidden, num_class=self.num_class, with_dropout=self.dropout
        )
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=self.hidden, with_dropout=self.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if self.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type("torch.FloatTensor")
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type("torch.FloatTensor")
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, self.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if self.mode == "gpu":
            node_feat = node_feat.to(self.device)
            labels = labels.to(self.device)
            if edge_feat_flag == True:
                edge_feat = edge_feat.to(self.device)

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph):
        batch = []
        for g in batch_graph:
            batch += g
        feature_label = self.PrepareFeatureLabel(batch)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch, node_feat, edge_feat)
        labels = [labels[i] for i in range(0, labels.shape[0], 5)]
        labels = torch.LongTensor(labels).to(self.device)

        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=32, state=None, handler=None):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(list(range(total_iters)), unit="batch")
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        if state is not None:
            state.step_in_epoch = pos
            state.total_steps += 1
            if handler is not None:
                handler.on_train_step_begin(state)

        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx][-1].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        
        if state is not None:
            state.loss = loss
            if handler is not None:
                handler.on_train_step_end(state)

        if classifier.regression:
            pbar.set_description("MSE_loss: %0.5f MAE_loss: %0.5f" % (loss, mae))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description("loss: %0.5f acc: %0.5f" % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores)

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    if not classifier.regression:
        all_targets = torch.tensor(all_targets)
    return avg_loss, all_targets, all_scores
