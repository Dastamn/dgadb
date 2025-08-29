import logging
from src.dgadb.preprocessing.pipeline import Pipeline
from src.dgadb.preprocessing.anomaly_injector import AnomalyInjector
from src.dgadb.data.builder import build_graph_from_temporal
from src.dgadb.models.TADDY.TADDY_main import TADDYModel
from src.dgadb.evaluation.evaluator import Evaluator
from sklearn.metrics import roc_auc_score
import numpy as np
logger = logging.getLogger(__name__)

from src.dgadb.storage.graph import Graph
import polars as pl
import torch

def graph_edges_to_df(g: Graph) -> pl.DataFrame:
    """
    Convert a Graph's edge data into a pandas DataFrame.

    Columns:
      - tgt, src: taken from g.e_pairs (row 1 = tgt, row 0 = src)
      - timestamp: from g.e_timestamp (if missing -> pd.NA)
      - label: from g.e_label (if missing -> pd.NA)
      - snapshot_id: from g.e_snapshot_id (if missing -> pd.NA)
      - train_mask: from g.e_train_mask (if missing -> pd.NA)
      - test_mask: from g.e_test_mask (if missing -> pd.NA)
      - val_mask: from g.e_val_mask (included only if present)
    """
    e = g.edge_dict()
    if "e_pairs" not in e:
        raise ValueError("Graph.edges must contain 'e_pairs'")

    pairs = e["e_pairs"].detach().cpu()
    if pairs.dim() != 2 or pairs.size(0) != 2:
        raise ValueError("e_pairs must be a 2 x num_edges tensor")

    num_edges = pairs.size(1)

    # Helper to safely pull a column from edge dict, defaulting to pd.NA
    def col_from_edges(key: str):
        if key in e:
            return e[key].detach().cpu().numpy()
        else:
            return pl.Series([None] * num_edges)

    data = {
        # NOTE: convention here: row 0 = src, row 1 = tgt
        "tgt": pairs[1].to(torch.long).numpy(),
        "src": pairs[0].to(torch.long).numpy(),
        "timestamp": col_from_edges("e_timestamp"),
        "label": col_from_edges("e_label"),
        "snapshot_id": col_from_edges("e_snapshot_id"),
        "train_mask": col_from_edges("e_train_mask"),
        "test_mask": col_from_edges("e_test_mask"),
    }

    if "e_val_mask" in e:
        data["val_mask"] = e["e_val_mask"].detach().cpu().numpy()

    return pl.DataFrame(data)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":
    # config_name = "yelp-zip-example"
    config_name = "bitcoin-alpha-example"
    #config_name = "uc-social-example"

    pipeline = Pipeline.from_config(config_name, force_rerun=False)
    evaluator = Evaluator(dataset_name="bitcoin-alpha", method_name="TADDY", output_dir="eval-data")


    g = pipeline.run()
    g.describe()

    temporal_graph = g.to_temporal_graph()
    temporal_graph.describe()

    # TODO @Dastamn: Test on GPU
    anom_injector = AnomalyInjector(temporal_graph)

    anomalous_temporal_graph = anom_injector.generate_anomalous_samples(
        "s", anom_train_ratio=0.05, anom_test_ratio=0.05, anom_val_ratio=0.05, reset_labels=False)

    graph = build_graph_from_temporal(anomalous_temporal_graph)
    # use node2vec embs 
    del graph._nodes["n_feat"]
    graph.generate_snapshots(snapshot_size=1000, temporal_snapshots=False)
    meta = anomalous_temporal_graph.metadata

    meta_dict = {
        "dataset_name":meta["dataset_name"],
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "anomaly_ratio": 0.05,
    }
    # TODO: meta_dict should be produced from the pipeline

    device = "cpu"
    hyperparams = {"num_epochs":250, "device" : device}
    model = TADDYModel(device=device, meta_dict=meta_dict, hyperparams=hyperparams, epoch_evaluation_metric=roc_auc_score)
    model.setup(graph_edges_to_df(graph))
    model.train()
    preds_per_snap, labels_per_snap = model.inference(split="test")
    for snap in range(len(preds_per_snap)):
        y, pred = labels_per_snap[snap], preds_per_snap[snap]
        evaluator.eval_snapshot(y, pred, snapshot_id=snap)
    print(evaluator.get_summary())


    total_auc_score = roc_auc_score(np.hstack(labels_per_snap), np.hstack(preds_per_snap))
    logger.info(f"Total test score: {total_auc_score:.4f}")

