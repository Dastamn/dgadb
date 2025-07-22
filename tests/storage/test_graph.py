import pytest
import torch
import polars as pl
from src.dgadb.storage import Graph


def make_example_graph():
    nodes = {
        "n_feat": torch.randn(3, 4),
        "n_type": torch.tensor([0, 1, 0], dtype=torch.long)
    }
    edges = {
        "e_pairs": torch.tensor([[0, 1], [1, 2]], dtype=torch.long).T,
        "e_type": torch.tensor([0, 1], dtype=torch.long),
        "e_snapshot_id": torch.tensor([0, 1], dtype=torch.long),
    }
    return Graph(nodes=nodes, edges=edges, timestamps={})


def test_graph_construction():
    g = make_example_graph()
    assert g.num_nodes == 3
    assert g.num_edges == 2
    assert g.node_feature_dim == 4
    assert g.edge_types.tolist() == [0, 1]
    assert g.e_id.tolist() == [0, 1]
    assert g.n_id.tolist() == [0, 1, 2]


def test_graph_invalid_node_key():
    nodes = {"bad_key": torch.tensor([1, 2, 3])}
    with pytest.raises(ValueError):
        Graph(nodes=nodes, edges={}, timestamps={})


def test_graph_invalid_edge_key():
    edges = {"bad_key": torch.tensor([1, 2])}
    with pytest.raises(ValueError):
        Graph(nodes={}, edges=edges, timestamps={})


def test_get_snapshot():
    g = make_example_graph()
    snap = g.get_snapshot(snapshot_id=0)
    assert isinstance(snap, Graph)
    assert snap.num_edges == 1


def test_empty_graph():
    g = Graph(nodes={}, edges={}, timestamps={})
    assert g.num_nodes == 0
    assert g.num_edges == 0
