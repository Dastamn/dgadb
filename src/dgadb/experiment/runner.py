import logging
import os
from enum import Enum
from typing import Annotated

import typer
from dgadb.evaluation import ADEvaluator
from dgadb.models.base import BaseADModel, BaseADModelComponentsType
from dgadb.storage import (
    TemporalGraph,
    TemporalGraphLoader,
    TemporalGraphSnapshotLoader,
    generate_temporal_graph_filename,
)

from .callbacks import AimCallback, ExperimentCallback, ResourceMonitor


class Method(str, Enum):
    """Available anomaly detection methods."""

    sad = "sad"
    taddy = "taddy"
    slade = "slade"
    strgnn = "strgnn"
    gcn = "gcn"
    gat = "gat"
    graphsage = "graphsage"
    rustgraph = "rustgraph"
    generaldyg = "generaldyg"
    addgraph = "addgraph"


app = typer.Typer()

class ExperimentRunner:
    def __init__(
        self,
        model: BaseADModel[BaseADModelComponentsType],
        data: TemporalGraph,
        dataset_name: str | None = None,
        output_dir: str = "experiment-results",
    ) -> None:
        """Orchestrates a single training and evaluation experiment.

        This class handles the end-to-end process of training a specific anomaly
        detection model on a temporal graph dataset. It manages data loading
        (splitting into train/val/test snapshots), model initialization,
        the training loop with callbacks, and final evaluation on the test set.

        Args:
            model: An instance of a model inheriting from `BaseADModel`.
            data: The `TemporalGraph` object containing the dataset.
            dataset_name: An optional name for the dataset. If None, a name is
                generated based on the graph's properties.
            output_dir: The directory where logs, plots, and evaluation results
                will be saved.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.data = data
        self.dataset_name = (
            dataset_name
            if dataset_name is not None
            else generate_temporal_graph_filename(self.data)
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # TODO: change snapshot_config to args
    def run(
        self,
        epochs: int,
        snapshot_config: dict,
        callbacks: list[ExperimentCallback] | None = None,
        evaluate: bool = True,
    ):
        self.logger.info(
            f"Starting Experiment: {self.model.__class__.__name__}/{self.dataset_name}"
        )
        self.logger.info(f"Results will be saved to: {self.output_dir}")
        self.callbacks = callbacks or []

        train_loader = TemporalGraphSnapshotLoader(
            self.data, split="train", **snapshot_config
        )
        val_loader = TemporalGraphSnapshotLoader(
            self.data, split="val", **snapshot_config
        )

        self.model.setup(self.data)
        self.model.train(epochs, train_loader, val_loader, callbacks)

        if evaluate:
            self.evaluate(snapshot_config)

        self.logger.info("Done.")

    def evaluate(self, snapshot_config: dict):
        test_loader = TemporalGraphSnapshotLoader(
            self.data, split="test", **snapshot_config
        )
        all_labels, all_scores = self.model.run_inference(test_loader)

        test_group_ids = None
        anomaly_group_ids = self.data.metadata.get("anomaly_group_ids", None)
        anom_injection_meta = self.data.metadata.get("anomaly_injection", None)

        if anomaly_group_ids is not None and anom_injection_meta is not None and anom_injection_meta["type"] not in ["random", "bridge"]:
            test_group_ids = anomaly_group_ids[self.data.test_mask]

        evaluator = ADEvaluator(self.output_dir)
        metrics = evaluator.evaluate(all_labels, all_scores, test_group_ids)
        self.logger.info(
            f"Evaluation result: AUC {metrics['roc_auc']}, AP {metrics['average_precision']}"
        )
        print(
            f"Evaluation result: AUC {metrics['roc_auc']}, AP {metrics['average_precision']}"
        )
        evaluator.save_results()

        # Log evaluation metrics to AimCallback if present
        for callback in getattr(self, "callbacks", []):
            if isinstance(callback, AimCallback):
                callback.log_evaluation_metrics(metrics, context="test")


@app.command()
def run_experiment(
    method: Annotated[Method, typer.Option(help="Method name")],
    dataset: Annotated[str, typer.Option(
        help="Dataset name")] = "bitcoin-alpha",
    experiment_name: Annotated[str, typer.Option(
        help="Aim experiment name")] = "dgadb",
    anom_types: Annotated[list[str], typer.Option(
        help="Types of anomalies to inject")] = ["random", "burst", "bridge", "clique", "path"],
    anom_ratios: Annotated[list[float], typer.Option(
        help="Anomaly ratios to test")] = [0.1, 0.01, 0.05],
    anom_durations: Annotated[list[float], typer.Option(
        help="Anomaly durations to test")] = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
    sad_anom_train_ratio: Annotated[float, typer.Option(
        help="The anomaly ratio used during SAD training phase")] = 0.01,
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 10,
):
    """Run an anomaly detection experiment with the specified method and dataset."""

    window_size = 6000
    if dataset in ["bitcoin-alpha", "bitcoin-otc", "uc-social"]:
        window_size = 2000

    snapshot_config = {
        "strategy": "window",
        "window_size": window_size,
        "include_cumulative": True,
    }

    from dgadb.storage.temporal_graph import TemporalGraphLoaderNew

    loader = TemporalGraphLoaderNew()

    for ar in anom_ratios:
        for at in anom_types:
            for ad in anom_durations:
                print(
                    f"STARTING: anom_type={at}, anom_ratio={ar}, anom_duration={ad}")
                
                match method:
                    case Method.taddy:
                        from dgadb.models.taddy_new.taddy import TADDYAD

                        model = TADDYAD(snap_size=window_size)
                    case Method.slade:
                        from dgadb.models.slade_new.slade import SLADEAD

                        model = SLADEAD()
                    case Method.strgnn:
                        from dgadb.models.StrGNN.strgnn import StrGNNAD

                        model = StrGNNAD(snap_size=window_size)
                    case Method.rustgraph:
                        from dgadb.models.rustgraph_new.rustgraph import RustGraphAD

                        model = RustGraphAD()
                    case Method.generaldyg:
                        from dgadb.models.generaldyg_new.generaldyg import GeneralDyGAD

                        model = GeneralDyGAD()
                    case Method.gcn:
                        from dgadb.models.baseline.gnn import GNNAD

                        model = GNNAD("GCN")
                    case Method.gat:
                        from dgadb.models.baseline.gnn import GNNAD

                        model = GNNAD("GAT")
                    case Method.graphsage:
                        from dgadb.models.baseline.gnn import GNNAD

                        model = GNNAD("GraphSAGE")
                    case "addgraph":
                        from dgadb.models.addgraph.addgraph import AddGraphAD

                        model = AddGraphAD()

                if method == Method.sad:
                    from dgadb.preprocessing.anomaly_injection import AnomalyInjector

                    data = loader.load(dataset, create_if_not_found=True)
                    ai = AnomalyInjector(data)
                    ai.generate_anomalous_samples("random", train_ratio=sad_anom_train_ratio, duration=1.0)
                    ai.generate_anomalous_samples(at, val_ratio=ar, test_ratio=ar, duration=ad)
                else:
                    data = loader.load(dataset, at, anom_val_ratio=ar,
                                   anom_test_ratio=ar, duration=ad, create_if_not_found=True)
                
                # Configure Aim tracking with comprehensive logging
                anom_config = {
                    "anom_type": at,
                    "anom_ratio": ar,
                    "anom_duration": ad
                }

                aim_callback = AimCallback(
                    experiment_name=experiment_name,
                    run_name=f"{method.value}_{dataset}",
                    hparams={
                        "method": method.value,
                        "dataset": dataset,
                        "variant": data.variant_name,
                        "epochs": epochs,
                        "anom_duration": ad,
                        **snapshot_config,
                    },
                    tags=[method.value, dataset, at],
                )
                aim_callback.log_config(anom_config, name="anom_config")
                aim_callback.log_config(
                    snapshot_config, name="snapshot_config")

                output_dir = f"experiment-results/{experiment_name}/{method.value}/{data.variant_name}"

                resource_monitor = ResourceMonitor(output_dir)

                runner = ExperimentRunner(model, data, output_dir=output_dir)
                
                runner.run(epochs, snapshot_config, [
                        aim_callback, resource_monitor])

                print("DONE.")
                print("========================")


if __name__ == "__main__":
    app()
