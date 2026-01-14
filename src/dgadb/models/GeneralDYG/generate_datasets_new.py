from tqdm import tqdm
import pickle
# from option import args
import pandas as pd
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import sys
import os
import random
import copy
import matplotlib.pyplot as plt

# dataset_name = '{}/ml_{}'.format(config.dir_data, config.data_set)
# graph_df = pd.read_csv('{}.csv'.format(dataset_name))
# edge_features = np.load('{}.npy'.format(dataset_name))
# node_features = np.load('{}_node.npy'.format(dataset_name))


class BatchGraphSample:
    def __init__(self, config, graph_df):
        self.config = config
        self.graph_df = graph_df
        # 生成图 - vectorized construction
        self.G = nx.Graph()
        # Group by edges and collect all weights
        edge_groups = graph_df.groupby(
            ['u', 'i'])['id'].apply(list).reset_index()
        # Vectorized edge addition
        for _, row in edge_groups.iterrows():
            u, v, weights = row['u'], row['i'], row['id']
            self.G.add_edge(u, v, weight=weights)
        # # k = 1
        # if self.config.data_set == 'btc_otc':
        #     self.max_mask_len = 14
        # elif self.config.data_set == 'btc_alpha':
        #     self.max_mask_len = 16

        # k = 1
        if self.config.data_set == 'btc_otc':
            self.max_mask_len = 26
        elif self.config.data_set == 'btc_alpha':
            self.max_mask_len = 26

        # k = 2
        # if self.config.data_set == 'btc_otc':
        #     self.max_mask_len = 20
        # elif self.config.data_set == 'btc_alpha':
        #     self.max_mask_len = 20

    def remove_random_nodes(self, nodes, max_mask_len, src_node, dest_node):
        nodes = np.array(list(nodes))
        # 确保node在nodes中
        if src_node not in nodes:
            raise ValueError("The required 'node' is not in the list 'nodes'")

        # 当nodes的长度超过max_mask_len时，开始随机删除
        if len(nodes) > max_mask_len:
            # Vectorized: filter out src and dest nodes, then randomly sample
            removable = nodes[(nodes != src_node) & (nodes != dest_node)]
            if len(removable) > 0:
                num_to_remove = len(nodes) - max_mask_len
                to_remove = np.random.choice(removable, size=min(
                    num_to_remove, len(removable)), replace=False)
                nodes = np.setdiff1d(nodes, to_remove)
        return set(nodes)

        # 提取k-hop子图的函数 - optimized with vectorized BFS

    def extract_k_hop_subgraph(self, graph, src_node, dest_node, k, max_mask_len):
        # Use adjacency matrix for faster neighbor lookup
        if src_node not in graph:
            return {src_node}

        nodes_list = sorted(graph.nodes())
        adj = nx.adjacency_matrix(graph, nodelist=nodes_list)
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        src_idx = node_to_idx[src_node]
        visited = np.zeros(adj.shape[0], dtype=bool)
        current_frontier = np.zeros(adj.shape[0], dtype=bool)
        current_frontier[src_idx] = True
        visited[src_idx] = True

        # Vectorized BFS for k hops
        for _ in range(k):
            # Get all neighbors of current frontier in one operation
            # Use sparse matrix multiplication for efficiency
            if current_frontier.sum() == 0:
                break
            neighbors = adj[current_frontier].sum(axis=0)
            if isinstance(neighbors, np.matrix):
                new_frontier = neighbors.A1 > 0
            else:
                new_frontier = neighbors > 0
            new_frontier = new_frontier & (~visited)
            if new_frontier.sum() == 0:
                break
            visited = visited | new_frontier
            current_frontier = new_frontier

        # Convert back to node IDs
        nodes = set(idx_to_node[idx] for idx in np.where(visited)[0])
        nodes = self.remove_random_nodes(
            nodes, max_mask_len, src_node, dest_node)
        return nodes

        # 根据权重和时间戳重新排序节点 - vectorized

    def reorder_nodes(self, subgraph):
        # Vectorized: get all edges and compute min weight per node
        nodes = np.array(list(subgraph.nodes()))
        if len(nodes) == 0:
            return {}

        # Get adjacency matrix with weights
        adj = nx.adjacency_matrix(subgraph, nodelist=nodes, weight='weight')
        adj_dense = adj.toarray()

        # Replace zeros with inf to find minimum weights
        adj_dense[adj_dense == 0] = np.inf
        # Get minimum weight per node (row-wise min)
        node_weights = np.min(adj_dense, axis=1)
        # Handle isolated nodes
        node_weights[node_weights == np.inf] = np.inf

        # Sort nodes by weight
        sorted_indices = np.argsort(node_weights)
        sorted_nodes = nodes[sorted_indices]
        node_mapping = {old_id: new_id for new_id,
                        old_id in enumerate(sorted_nodes)}
        return node_mapping

        # 替换子图中的节点和边 - vectorized

    def replace_subgraph(self, subgraph, node_mapping):
        nodes = list(subgraph.nodes())
        num_nodes = len(nodes)
        new_node_features = np.zeros(num_nodes, dtype=np.int64)

        # Vectorized node feature assignment
        for old_id, new_id in node_mapping.items():
            new_node_features[new_id] = int(old_id)

        # Build new subgraph and edge features vectorized
        new_subgraph = nx.Graph()
        new_subgraph.add_nodes_from(range(num_nodes))
        new_edge_features = {}

        # Get all edges at once
        edges = list(subgraph.edges(data=True))
        if edges:
            # Vectorized edge processing
            edge_data = np.array(
                [(edge[0], edge[1], edge[2].get('weight', 1)) for edge in edges])
            for u, v, weight in edge_data:
                new_source = node_mapping[u]
                new_target = node_mapping[v]
                new_subgraph.add_edge(new_source, new_target, weight=1)
                new_edge_features[(new_source, new_target)] = int(weight)

        mask = new_node_features.shape[0]
        return new_subgraph, new_node_features, new_edge_features, mask

    def create_transition_matrix(self, vertex_adj):
        '''create N_v * N_e transition matrix - vectorized'''
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))

        if num_edge == 0:
            return sp.csr_matrix((vertex_adj.shape[0], 0))

        # Vectorized: create row and column indices
        row_indices = np.array(edge_index[0])
        col_indices = np.array(edge_index[1])

        # Flatten edge pairs: [u1, v1, u2, v2, ...] for row_index
        row_index = np.column_stack([row_indices, col_indices]).flatten()
        col_index = np.repeat(np.arange(num_edge), 2)

        data = np.ones(num_edge * 2)
        T = sp.csr_matrix((data, (row_index, col_index)),
                          shape=(vertex_adj.shape[0], num_edge))

        return T

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor. (now dense tensor)"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to_dense()

    def create_edge_adj(self, vertex_adj):
        '''
        create an edge adjacency matrix from vertex adjacency matrix
        Vectorized version - much faster for large graphs
        '''
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = np.array(list(zip(edge_index[0], edge_index[1])))

        if num_edge == 0:
            return sp.csr_matrix((0, 0)), []

        # Vectorized: create edge-node incidence matrix
        # Each edge connects two nodes, so we create a matrix where
        # edge_incidence[i, node] = 1 if edge i is incident to node
        num_nodes = vertex_adj.shape[0]
        edge_incidence = sp.lil_matrix((num_edge, num_nodes))
        edge_incidence[np.arange(num_edge), edge_name[:, 0]] = 1
        edge_incidence[np.arange(num_edge), edge_name[:, 1]] = 1
        edge_incidence = edge_incidence.tocsr()

        # Two edges are adjacent if they share at least one node
        # This is computed as: edge_adj = (edge_incidence @ edge_incidence.T) > 0
        edge_adj = (edge_incidence @
                    edge_incidence.T).astype(bool).astype(float)
        edge_adj = edge_adj.toarray()

        # Make symmetric and set diagonal to 1
        edge_adj = edge_adj + edge_adj.T
        np.fill_diagonal(edge_adj, 1)

        return sp.csr_matrix(edge_adj), edge_name.tolist()

        # 假设 new_subgraph 是一个 networkx.Graph 对象

    def get_adjacency_matrix(self, subgraph):
        # 获取邻接矩阵，返回的是 SciPy 稀疏矩阵
        adj_matrix_sparse = nx.adjacency_matrix(subgraph)
        # 将稀疏矩阵转换为 NumPy 数组
        adj_matrix = adj_matrix_sparse.toarray()
        return adj_matrix

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1)).astype("float")
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def pad_ndarray_list(self, ndarray_list, max_mask_len):
        # 获取列表的大小
        N = len(ndarray_list)
        if N == 0:
            return np.zeros((0, max_mask_len, 0)), np.ones((0, max_mask_len), dtype=bool), torch.tensor([])

        hidden = ndarray_list[0].shape[1]  # 假设hidden维度是固定的

        # 初始化填充后的三维ndarray和掩码矩阵
        padded_ndarray = np.zeros((N, max_mask_len, hidden))
        mask = np.ones((N, max_mask_len), dtype=bool)  # 填充部分为True，原始部分为False

        # Vectorized padding
        seq_lens = np.array([array.shape[0] for array in ndarray_list])
        for i, array in enumerate(ndarray_list):
            seq_len = seq_lens[i]
            padded_ndarray[i, :seq_len, :] = array
            mask[i, :seq_len] = False  # 原始部分为False

        unpadded_indices = torch.tensor(seq_lens)  # 转换为tensor

        return padded_ndarray, mask, unpadded_indices

    def get_batch_data(self, idx_batch):
        k = 1
        all_node_features = []
        all_edge_features = []
        all_Tmat = []
        all_adj = []
        all_eadj = []
        all_mask = []

        edges = self.graph_df[self.graph_df['id'].isin(idx_batch)]
        # Convert to numpy for faster iteration
        edges_array = edges[['u', 'i', 'id']].values

        # for index, edge in edges.iterrows():
        for idx in tqdm(range(len(edges_array)), total=len(edges_array)):
            u, v, edge_id = edges_array[idx]
            src_nodes = self.extract_k_hop_subgraph(
                self.G, u, v, k, self.max_mask_len)
            dest_nodes = self.extract_k_hop_subgraph(
                self.G, v, u, k, self.max_mask_len)
            subgraph_nodes = src_nodes.union(dest_nodes)
            batch_subgraph = self.G.subgraph(subgraph_nodes).copy()

            # Vectorized: update edge weights (avoid deepcopy overhead)
            for edge_u, edge_v in batch_subgraph.edges():
                edge_data = batch_subgraph[edge_u][edge_v]
                weights = edge_data.get('weight', [1])
                if isinstance(weights, list):
                    random_weight = random.choice(weights)
                else:
                    random_weight = weights
                edge_data['weight'] = random_weight
            batch_subgraph[u][v]['weight'] = edge_id

            node_mapping = self.reorder_nodes(batch_subgraph)
            new_subgraph, new_node_features, new_edge_features, mask = self.replace_subgraph(batch_subgraph,
                                                                                             node_mapping)
            adj = nx.adjacency_matrix(new_subgraph)
            T = self.create_transition_matrix(adj)
            tensor_T = self.sparse_mx_to_torch_sparse_tensor(T)
            eadj, edge_name = self.create_edge_adj(adj)

            num_edges = eadj.shape[0]  # 非零元素的数量（即边的数量）
            if num_edges == 0:
                edge_feature_matrix = np.zeros(0)
            else:
                edge_feature_matrix = np.zeros(num_edges)
                temp_T = T.copy().toarray()

                # Vectorized: extract edge pairs directly from transition matrix
                # Each column in T has exactly 2 ones, representing an edge (i, j)
                edge_pairs = np.zeros((num_edges, 2), dtype=int)
                for col in range(num_edges):
                    row_indices = np.where(temp_T[:, col] == 1)[0]
                    if len(row_indices) == 2:
                        edge_pairs[col] = row_indices

                # Vectorized feature extraction using dictionary lookup
                for edge_idx in range(num_edges):
                    i, j = edge_pairs[edge_idx]
                    if (i, j) in new_edge_features:
                        edge_feature_matrix[edge_idx] = new_edge_features[(
                            i, j)]
                    elif (j, i) in new_edge_features:
                        edge_feature_matrix[edge_idx] = new_edge_features[(
                            j, i)]
            eadj = self.sparse_mx_to_torch_sparse_tensor(self.normalize(eadj))

            adj = self.sparse_mx_to_torch_sparse_tensor(
                self.normalize(adj + sp.eye(adj.shape[0])))

            all_node_features.append(new_node_features)
            all_edge_features.append(edge_feature_matrix)
            all_Tmat.append(tensor_T)
            all_adj.append(adj)
            all_eadj.append(eadj)
            all_mask.append(mask)
        return all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask


# if __name__ == '__main__':
#     # import pickle
#     # # def torch_sparse_tensor_to_scipy(torch_sparse_tensor):
#     # #     """Convert a torch sparse tensor to a scipy sparse matrix."""
#     # #     indices = torch_sparse_tensor._indices().numpy()
#     # #     values = torch_sparse_tensor._values().numpy()
#     # #     shape = torch_sparse_tensor.shape
#     # #     return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
#     #
#     config = args
#     #
#     # dataset_name = '{}/{}.pkl'.format(config.dir_data, config.data_set)
#     #
#     # with open(dataset_name, 'rb') as file:
#     #     data = pickle.load(file)
#     #
#     # node_features = data['nodefeatures']
#     # edge_features = data['edgefeatures']
#     # labels = data['labels']
#     # Tmats = data['Tmats']
#     # adjs = data['adjs']
#     # eadjs = data['eadjs']
#     #
#     # # dense_Tmats = [
#     # #     torch.tensor(torch_sparse_tensor_to_scipy(sparse_tensor).toarray())
#     # #     for sparse_tensor in Tmats
#     # # ]
#     # # dense_adjs = [
#     # #     torch.tensor(torch_sparse_tensor_to_scipy(sparse_tensor).toarray())
#     # #     for sparse_tensor in adjs
#     # # ]
#     # # dense_eadjs = [
#     # #     torch.tensor(torch_sparse_tensor_to_scipy(sparse_tensor).toarray())
#     # #     for sparse_tensor in eadjs
#     # # ]
#     #
#     #
#     # new_data = {
#     #     'nodefeatures': node_features,
#     #     'edgefeatures': edge_features,
#     #     'labels': labels,
#     #     'Tmats': Tmats,
#     #     'adjs': adjs,
#     #     'eadjs': eadjs
#     # }
#     # with open(config.data_set + '.pkl', 'wb') as f:
#     #     pickle.dump(new_data, f)
#     # print('装载新数据完成')

#     # sys.exit()
#     #
#     dataset_name = '{}/{}_0.5_0.{}'.format(config.dir_data,
#                                            config.data_set, config.neg)
#     graph_df = pd.read_csv('{}.csv'.format(dataset_name))
#     label_column = graph_df['label'].to_numpy()
#     idx_column = graph_df['id'].to_numpy()
#     a = BatchGraphSample(config, graph_df)
#     all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask = a.get_batch_data(
#         idx_column)
#     all_node_features = np.array(all_node_features)
#     all_edge_features = np.array(all_edge_features)
#     all_mask = np.array(all_mask)

#     data = {
#         'nodefeatures': all_node_features,
#         'edgefeatures': all_edge_features,
#         'masks': all_mask,
#         'labels': label_column,
#         'Tmats': all_Tmat,
#         'adjs': all_adj,
#         'eadjs': all_eadj
#     }
#     with open(config.data_set + '.pkl', 'wb') as f:
#         pickle.dump(data, f)

#     print('-----------------------------------')
