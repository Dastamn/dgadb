import torch


def generate_anomalies_no_timestamp(data: torch.Tensor, train_percent: float, anomaly_percent: float):
    """
    Generates a synthetic graph anomaly detection dataset from a list of edges. 
    Implements the method described in [Li Zheng et al., 2019], with reference to its public implementation.

    This function takes a complete set of graph edges and splits it into a
    training set and a testing set. Any columns beyond the first two in the input data
    (e.g., timestamps) are ignored.

    The final testing set is a shuffled mixture of:
    1.  Normal Edges: Edges from the original graph that were not included in
        the training set (label 0).
    2.  Anomalous Edges: Synthetically generated edges that connect nodes from
        the training graph but did not exist in the original graph (label 1).

    The number of anomalies is calculated to ensure that they constitute the
    specified `anomaly_percent` of the final, combined test set.

    Args:
        data (torch.Tensor): A 2D tensor of shape [n_edges, 2] representing
            the graph's edge list.
        train_percent (float): The percentage of edges (from the beginning of
            the `data` tensor) to be used for the training set. Must be
            between 0.0 and 1.0.
        anomaly_percent (float): The desired percentage of anomalous edges in the
            final testing set. Must be between 0.0 and 1.0.

    Returns:
        tuple:
            - train_edges (torch.Tensor): A tensor of edges for the training graph.
            - synthetic_test_edges (torch.Tensor): A 2D tensor of shape
              [n_test_edges, 3] representing the testing set. Each row is
              (source_node, target_node, label), where label is 0 for normal
              and 1 for anomalous.

    References:
        - Original paper: "AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based
Temporal GCN", Li Zheng et al., IJCAI, 2019.
        - Public implementation: https://github.com/Ljiajie/Addgraph/blob/master/UCI_D_Addgraph/framwork/anomaly_generation.py
    """
    if not 0.0 <= train_percent <= 1.0:
        raise ValueError("train_percent must be between 0.0 and 1.0")
    if not 0.0 <= anomaly_percent <= 1.0:
        raise ValueError("anomaly_percent must be between 0.0 and 1.0")

    all_edges = data[:, :2]
    n_edges = len(all_edges)
    n_train_edges = int(n_edges * train_percent)
    train_edges = all_edges[:n_train_edges]
    normal_test_edges = all_edges[n_train_edges:]

    train_nodes = torch.unique(train_edges).cpu()
    n_train = len(train_nodes)

    # Create a set of all existing undirected edges
    # An edge (u, v) is stored as (min(u,v), max(u,v)) to treat it as undirected
    canonical_edges, _ = torch.sort(all_edges, dim=1)
    unique_canonical_edges = torch.unique(canonical_edges, dim=0)
    canonical_edge_set = set(tuple(edge) for edge in unique_canonical_edges.cpu().tolist())

    # Calculate the number of anomalies to generate
    n_normal_test_edges = len(normal_test_edges)
    n_anomalies = max(0, int((n_normal_test_edges * anomaly_percent) / (1 - anomaly_percent + 1e-10)))

    print(f"Training edges: {len(train_edges)}")
    print(f"Normal test edges: {n_normal_test_edges}")
    print(f"Anomalies to generate: {n_anomalies}\n")

    anomalies = []
    attempts = 0
    max_attempts = n_anomalies * 100 + 100 # Prevent infinite loops

    while len(anomalies) < n_anomalies and attempts < max_attempts:
        # Sample two distinct random training nodes
        sample_indices = torch.randperm(n_train)[:2]
        u, v = train_nodes[sample_indices].tolist()
        edge_candidate = (u, v)
        canonical_edge_candidate = tuple(sorted((u, v)))

        # Add the edge if it's not a self-loop and doesn't already exist
        if u != v and canonical_edge_candidate not in canonical_edge_set:
            anomalies.append(list(edge_candidate))
            canonical_edge_set.add(canonical_edge_candidate) # Add to set to prevent re-generating it
        else:
            attempts += 1
    
    if attempts >= max_attempts and len(anomalies) < n_anomalies:
        print(f"Warning: Could only generate {len(anomalies)} / {n_anomalies} requested anomalies.")

    anomaly_edges = torch.tensor(anomalies, dtype=torch.long)

     # Assemble the final test set
    if len(anomalies) > 0:
        test_edges = torch.cat([normal_test_edges, anomaly_edges], dim=0)
        # Create corresponding labels (0 for normal, 1 for anomaly)
        normal_labels = torch.zeros(n_normal_test_edges, dtype=torch.long)
        anomaly_labels = torch.ones(len(anomalies), dtype=torch.long)
        test_labels = torch.cat([normal_labels, anomaly_labels], dim=0)
        # Shuffle combined test set
        test_perm = torch.randperm(len(test_edges))
        test_edges = test_edges[test_perm]
        test_labels = test_labels[test_perm]
    else: # Case with no anomalies
        test_edges = normal_test_edges
        test_labels = torch.zeros(n_normal_test_edges, dtype=torch.long)

    # Combine edges and labels into (source, target, label) format
    synthetic_test_edges = torch.cat([test_edges, test_labels.unsqueeze(1)], dim=1)

    return train_edges, synthetic_test_edges
