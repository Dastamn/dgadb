# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: negative_sample.py
# @time: 2020/03/11
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn


# def update_adj(adj,snapshot):
#     data = tuple(map(tuple, snapshot))
#     for i, j in data:
#         adj[i - 1][j - 1] = adj[i - 1][j - 1] + 1  # Convert to 0-based index.
#         adj[i - 1][j - 1] = adj[i - 1][j - 1] + 1  # Convert to 0-based index.
#     return adj
def update_adj(adj, snapshot, nodes):
    Adj = torch.zeros((nodes, nodes))
    for edge in snapshot:
        adj[edge[0].item() - 1][edge[1].item() - 1] = adj[edge[0].item() - 1][
                                                          edge[1].item() - 1] + 1  # Convert to 0-based index.
        adj[edge[1].item() - 1][edge[0].item() - 1] = adj[edge[1].item() - 1][
                                                          edge[0].item() - 1] + 1  # Convert to 0-based index.
        Adj[edge[0].item() - 1][edge[1].item() - 1] = Adj[edge[0].item() - 1][edge[1].item() - 1] + 1
        Adj[edge[1].item() - 1][edge[0].item() - 1] = Adj[edge[1].item() - 1][edge[0].item() - 1] + 1
    return adj, Adj


class negative_sample(nn.Module):
    def __init__(self):
        super(negative_sample, self).__init__()

    def forward(self, adj, Adj, snapshot, H, f, arg):
        # 1. Convert to numpy for sampling logic
        if arg: # if cuda
            data = tuple(map(tuple, snapshot.data.cpu().numpy()))
            D = adj.sum(1).data.cpu().numpy()
            D_ = Adj.sum(1).data.cpu().numpy()
            adj_np = adj.data.cpu().numpy()
        else:
            data = tuple(map(tuple, np.array(snapshot)))
            D = np.array(adj.sum(1))
            D_ = np.array(Adj.sum(1))
            adj_np = np.array(adj)

        nodes = adj_np.shape[0]
        
        # Calculate Threshold (first node with 0 degree in history)
        zeros = np.argwhere(D == 0)
        if zeros.size > 0:
            th = (zeros + 1)[0].item()
        else:
            th = nodes + 1 # All nodes have been seen

        n_loss = torch.zeros(len(data))
        index = 0
        
        for i, j in data:
            # Convert to int for indexing
            u_idx, v_idx = int(i) - 1, int(j) - 1
            
            di = D_[u_idx]
            dj = D_[v_idx]
            
            # Probability precision fix
            total_d = float(di + dj)
            if total_d == 0:
                pi, pj = 0.5, 0.5
            else:
                pi = float(di) / total_d
                pj = 1.0 - pi # Ensure sum is exactly 1.0

            # Initial Negative Sample
            d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj]).item()
            
            def get_negative_node(anchor_node, history_adj, threshold, total_nodes):
                # Find nodes not connected to anchor in history
                candidates = np.argwhere(history_adj[int(anchor_node) - 1] == 0) + 1
                candidates = np.squeeze(candidates)
                
                # Filter by threshold (AddGraph sequential logic)
                pool = candidates[candidates < threshold]
                # Remove self
                pool = pool[pool != anchor_node]
                
                # If pool is empty, use any non-neighbor
                if pool.size == 0:
                    pool = candidates[candidates != anchor_node]
                
                # If still empty (node connected to everyone), pick random
                if pool.size == 0:
                    res = np.random.randint(1, total_nodes + 1)
                    while res == anchor_node:
                        res = np.random.randint(1, total_nodes + 1)
                    return res
                
                return np.random.choice(a=pool, size=1, replace=False).item()

            dn = get_negative_node(d, adj_np, th, nodes)
            
            # Order nodes for Score function consistent with original
            a, b = (dn, d) if d > dn else (d, dn)
            
            loss = f(hi=H[int(i)-1], hj=H[int(j)-1]) - f(hi=H[int(a)-1], hj=H[int(b)-1])
            
            # Hard Negative Search Loop
            count = 0
            while (loss > 0):
                count += 1
                if count > nodes or count > 1899: # Safety break
                    loss = torch.zeros(1, device=H.device)
                    break
                
                # Re-sample anchor
                d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj]).item()
                # Re-sample negative neighbor
                dn = get_negative_node(d, adj_np, th, nodes)
                
                a, b = (dn, d) if d > dn else (d, dn)
                loss = f(hi=H[int(i)-1], hj=H[int(j)-1]) - f(hi=H[int(a)-1], hj=H[int(b)-1])

            n_loss[index] = loss
            index += 1
            
        return n_loss
    
class negative_sample_old(nn.Module):
    def __init__(self):
        super(negative_sample_old, self).__init__()

    def forward(self, adj, Adj, snapshot, H, f, arg):
        if arg:
            data = tuple(map(tuple, snapshot.data.cpu().numpy()))
            D = adj.sum(1)
            D_ = Adj.sum(1)
            D = D.data.cpu().numpy()
            D_ = D_.data.cpu().numpy()
            adj = adj.data.cpu().numpy()
        else:
            data = tuple(map(tuple, np.array(snapshot)))
            D = adj.sum(1)
            D_ = Adj.sum(1)
            D = np.array(D)
            D_ = np.array(D_)
            adj = np.array(adj)
        # data = tuple(map(tuple, snapshot.data.cpu().numpy()))
        len = snapshot.shape[0]
        # for i,j in data:
        #     adj[i - 1][j - 1] = adj[i - 1][j - 1]+1  # Convert to 0-based index.
        #     adj[i - 1][j - 1] = adj[i - 1][j - 1]+1  # Convert to 0-based index.
        # D = adj.sum(1)
        # D = D.data.cpu().numpy()
        # adj = adj.data.cpu().numpy()
        th = (np.argwhere(D == 0) + 1)[0].item()
        # n_data=[]
        n_loss = torch.zeros(len)
        index = 0
        # print('----------Begin----------')
        for i, j in data:
            di = D_[i - 1]
            dj = D_[j - 1]

            # pi = float(di) / (di + dj)
            # pj = float(dj) / (di + dj)

            pi = float(di) / float(di + dj)
            pj = 1.0 - pi 

            d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj])
            d = d.item()
            d_ = np.argwhere(adj[d - 1] == 0) + 1
            d_ = (np.squeeze(d_))
            id1 = d_ < th
            d_ = d_[id1]
            id2 = d_ != d
            d_ = d_[id2]

            if d_.size == 0:
                # Use any node not connected to d, ignoring the 'th' constraint
                d_ = d_[d_ != d]

            if d_.size == 0:
                # Pick a random node ID in the graph that isn't d
                dn = np.random.randint(1, nodes + 1)
                while dn == d:
                    dn = np.random.randint(1, nodes + 1)
            else:
                dn = np.random.choice(a=d_pool, size=1, replace=False).item()

            dn = np.random.choice(a=d_, size=1, replace=False)
            dn = dn.item()
            if d > dn:
                a = dn
                b = d
            else:
                a = d
                b = dn
            loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[a - 1], hj=H[b - 1])
            count = 0
            while (loss > 0):  #
                # f function
                count=count+1
                d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj])
                d = d.item()
                d_ = np.argwhere(adj[d - 1] == 0) + 1
                d_ = (np.squeeze(d_))
                id1 = d_ < th
                d_ = d_[id1]
                id2 = d_ != d
                d_ = d_[id2]
                dn = np.random.choice(a=d_, size=1, replace=False)
                dn = dn.item()
                if d > dn:
                    a = dn
                    b = d
                else:
                    a = d
                    b = dn
                loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[a - 1], hj=H[b - 1])
                if (count>1899):
                    loss = torch.zeros(1)
                    break
                # # if d > dn:
                #     t = d
                #     d = dn
                #     dn = t
            # n_data.append([a, b])
            n_loss[index] = loss
            index = index + 1
        # print('----------End----------')
        # return np.array(n_data),n_loss
        return n_loss
