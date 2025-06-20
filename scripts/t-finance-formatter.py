from dgl.data.utils import load_graphs
import torch
import pandas as pd

graph_finance, label_dict_finance = load_graphs('./data/t-datasets/tfinance')
graph_finance = graph_finance[0]
graph_finance.ndata['label'] = graph_finance.ndata['label'].argmax(1)

graph_finance.ndata['label'] = graph_finance.ndata['label'].long().squeeze(-1)
graph_finance.ndata['feature'] = graph_finance.ndata['feature'].float()

src_finance, tgt_finance, edge_id_finance = graph_finance.edges(form='all')
edges_finance = torch.cat((edge_id_finance.reshape(-1,1), src_finance.reshape(-1,1), tgt_finance.reshape(-1,1)), 1)
np_edges_finance = edges_finance.detach().numpy()
df_edges_finance = pd.DataFrame(np_edges_finance, columns=['edge_id', 'src', 'tgt'])
df_edges_finance.to_csv("data/t-datasets/t-finance/t-finance-edges.csv", index=False)

np_features_finance = graph_finance.ndata['feature'].detach().numpy()
df_features_finance = pd.DataFrame(np_features_finance)
df_features_finance.index.names = ['node_id']
df_features_finance.to_csv("./data/t-datasets/t-finance/t-finance-features.csv")

np_label_finance = graph_finance.ndata['label'].detach().numpy()
df_label_finance = pd.DataFrame(np_label_finance, columns=['label'])
df_label_finance.index.names = ['node_id']
df_label_finance.to_csv("./data/t-datasets/t-finance/t-finance-labels.csv")
