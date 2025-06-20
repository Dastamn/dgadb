import numpy as np
import pandas as pd

graph = np.load("./data/dgraph/dgraphfin.npz")
node_time = np.load("./data/dgraph/dgraphfinv2_node_timestamp.npy")
edge_time = np.load("./data/dgraph/dgraphfinv2_edge_timestamp.npy")

#not sure which of these to use? the second is in the newer version...
#print(len(np.unique(graph["edge_timestamp"])))
#print(len(np.unique(edge_time)))

nodes = np.concatenate((graph["x"], node_time.reshape(-1, 1), graph["y"].reshape(-1, 1)), axis=1)
edges = np.concatenate((graph["edge_index"], graph["edge_type"].reshape(-1, 1), edge_time.reshape(-1, 1)), axis=1)

node_cols = ['f'+str(i) for i in range(17)] + ["node_timestamp", "label"]
df_nodes = pd.DataFrame(nodes, columns=node_cols)
df_nodes.index.names = ['node_id']
df_nodes.to_csv("./data/dgraph/dgraph_nodes.csv")

edge_cols = ["src", "tgt", "edge_type", "edge_timestamp"]
df_edges = pd.DataFrame(edges, columns=edge_cols)
df_edges.index.names = ["edge_id"]
df_edges.to_csv("./data/dgraph/dgraph_edges.csv")