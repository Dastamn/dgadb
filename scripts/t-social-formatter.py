from dgl.data.utils import load_graphs
import torch
import pandas as pd

graph_social, label_dict_social = load_graphs('./data/t-datasets/tsocial')
graph_social = graph_social[0]

graph_social.ndata['label'] = graph_social.ndata['label'].long().squeeze(-1)
graph_social.ndata['feature'] = graph_social.ndata['feature'].float()

graph_social.ndata['label'] = graph_social.ndata['label'].long().squeeze(-1)
graph_social.ndata['feature'] = graph_social.ndata['feature'].float()

src_social, tgt_social, edge_id_social = graph_social.edges(form='all')
edges_social = torch.cat((edge_id_social.reshape(-1,1), src_social.reshape(-1,1), tgt_social.reshape(-1,1)), 1)
np_edges_social = edges_social.detach().numpy()
df_edges_social = pd.DataFrame(np_edges_social, columns=['edge_id', 'src', 'tgt'])
df_edges_social.to_csv("data/t-datasets/t-social/t-social-edges.csv", index=False)

np_features_social = graph_social.ndata['feature'].detach().numpy()
df_features_social = pd.DataFrame(np_features_social)
df_features_social.index.names = ['node_id']
df_features_social.to_csv("./data/t-datasets/t-social/t-social-features.csv")

np_label_social = graph_social.ndata['label'].detach().numpy()
df_label_social = pd.DataFrame(np_label_social, columns=['label'])
df_label_social.index.names = ['node_id']
df_label_social.to_csv("./data/t-datasets/t-social/t-social-labels.csv")
