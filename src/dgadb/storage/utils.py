import json
import hashlib
import torch

from .graph import Graph
from .temporal_graph import TemporalGraph


def generate_temporal_graph_filename(temporal_graph: TemporalGraph):
    meta = temporal_graph.metadata
    dataset_name = meta.get("dataset_name", "unknown-dataset")

    if meta.get("anomaly_injection", {}).get("is_injected", False):
        anom_meta = meta["anomaly_injection"]
        type_info = anom_meta.get("type", "anom-type")

        splits_meta = anom_meta.get("splits", {})
        train_ratio = splits_meta.get("train", {}).get("ratio", 0.0)
        val_ratio = splits_meta.get("val", {}).get("ratio", 0.0)
        test_ratio = splits_meta.get("test", {}).get("ratio", 0.0)
        ratio_str = f"ratios-{train_ratio}-{val_ratio}-{test_ratio}"

        gen_params = anom_meta.get("generation_parameters", {})
        if not gen_params:
            params_hash = ""
        else:
            params_str = json.dumps(gen_params, sort_keys=True)
            hasher = hashlib.md5(params_str.encode())
            params_hash = f"_h-{hasher.hexdigest()[:8]}"

        return f"{dataset_name}_{type_info}_{ratio_str}{params_hash}"

    return dataset_name


def convert_temporal_graph_to_legacy_graph(temporal_graph: TemporalGraph) -> Graph:
    device = temporal_graph.device
    temporal_graph = temporal_graph.to("cpu")

    edges = {
        "e_pairs": torch.stack([temporal_graph.src, temporal_graph.tgt]),
        "e_label": temporal_graph.edge_labels,
        "e_train_mask": temporal_graph.train_mask,
        "e_val_mask": temporal_graph.val_mask,
        "e_test_mask": temporal_graph.test_mask,
        "e_timestamp": temporal_graph.t,
    }

    if temporal_graph.msg.numel() > 0:
        edges["e_feat"] = temporal_graph.msg

    nodes = {}
    if temporal_graph.node_attr is not None:
        nodes["n_feat"] = temporal_graph.node_attr

    if temporal_graph.node_labels is not None:
        nodes["n_label"] = temporal_graph.node_labels

    return Graph(nodes=nodes, edges=edges).to(device)
