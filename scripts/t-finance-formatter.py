from dgl.data.utils import load_graphs
import torch
import pandas as pd

graph, _ = load_graphs('./data/t-datasets/tfinance')
graph = graph[0]
graph.ndata['label'] = graph.ndata['label'].argmax(1)

graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
graph.ndata['feature'] = graph.ndata['feature'].float()

nodes = torch.cat((graph.ndata['feature'], graph.ndata['label'].reshape(-1,1)), 1)
nodes_np = nodes.detach().numpy()
nodes_columns = ['f'+str(i) for i in range(10)] + ["label"]
df_nodes = pd.DataFrame(nodes_np, columns=nodes_columns)
df_nodes.index.names = ['node_id']
df_nodes.to_csv("./data/t-datasets/t-finance/t-finance-nodes.csv")

src, tgt, edge_id = graph.edges(form='all')
edges = torch.cat((edge_id.reshape(-1,1), src.reshape(-1,1), tgt.reshape(-1,1)), 1)
edges_np = edges.detach().numpy()
df_edges = pd.DataFrame(edges_np, columns=['edge_id', 'src', 'tgt'])
df_edges.to_csv("data/t-datasets/t-finance/t-finance-edges.csv", index=False)