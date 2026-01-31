import json
import hashlib
import torch

from .graph import Graph
from .temporal_graph import TemporalGraph


def generate_temporal_graph_filename(temporal_graph: TemporalGraph):
    meta = temporal_graph.metadata
    dataset_name = meta.get("dataset_name", "unknown-dataset")

    train_splits = meta.get("splits", {})
    train_ratio = train_splits.get("train_ratio", 0.0)
    val_ratio = train_splits.get("val_ratio", 0.0)
    test_ratio = train_splits.get("test_ratio", 0.0)

    fn = f"{dataset_name}_tr{train_ratio}_val{val_ratio}_te{test_ratio}"

    if "variant_name" in meta:
        return f"{fn}_{meta['variant_name']}"

    if meta.get("anomaly_injection", {}).get("is_injected", False):
        anom_meta = meta["anomaly_injection"]
        type_info = anom_meta.get("type", "unknown-anom-type")

        anom_ratios = anom_meta.get("splits", {})
        anom_train_ratio = anom_ratios.get("train", {}).get("ratio", 0.0)
        anom_val_ratio = anom_ratios.get("val", {}).get("ratio", 0.0)
        anom_test_ratio = anom_ratios.get("test", {}).get("ratio", 0.0)

        anom_dur = anom_meta.get("duration_rate", 0.0)

        fn += f"_{type_info}_tr{anom_train_ratio}-v{anom_val_ratio}-te{anom_test_ratio}_{anom_dur}"

        gen_params = anom_meta.get("generation_parameters", {})
        if gen_params:
            params_str = json.dumps(gen_params, sort_keys=True)
            hasher = hashlib.md5(params_str.encode())
            fn += f"_h-{hasher.hexdigest()[:8]}"

        return fn

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
