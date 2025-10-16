import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Optional

import torch
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score

from . import metrics
from .utils import check_matching_device


VALID_METRICS = [
    "best_f1",
    "best_threshold",
    "roc_auc",
    # Ranking-based metrics
    "average_precision",
    "mrr",
    # Classification-based metrics at best_thr
    "accuracy",
    "precision",
    "recall"
]


def compute_metrics(y_true: torch.Tensor, y_scores: torch.Tensor) -> dict:
    y_true = y_true.cpu()
    y_scores = y_scores.cpu()

    max_f1, best_threshold, max_precision, max_recall = metrics.max_f1_score(
        y_true, y_scores)
    y_pred = (y_scores >= best_threshold).to(y_true.dtype)

    return {
        "max_f1": max_f1,
        "best_threshold": best_threshold,
        "roc_auc": roc_auc_score(y_true, y_scores),
        "average_precision": average_precision_score(y_true, y_scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "max_precision": max_precision,
        "max_recall": max_recall
    }


class ADEvaluator:
    """Evaluates anomaly detection models and summarizes results.

    This class provides a framework for running multiple evaluations,
    accumulating the results, computing summary statistics (min, max, mean, std) 
    across runs, and saving the results to a JSON file.

    Args:
        output_dir: The directory where the final evaluation report
            ("evaluation.json") will be saved.
        round_digits: The number of decimal places to round the summary
            statistics to. If None, no rounding is performed.

    Attributes:
        results: A list of dictionaries, where each dictionary holds the
            computed metrics from a single call to `evaluate`.
    """

    def __init__(self, output_dir: str, round_digits: int | None = 2) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        self.round_digits = round_digits
        self.results: list[dict[str, float]] = []

    def _get_summary(self, statistics: list[str] = ["mean", "std", "min", "max"]) -> dict:
        """Calculates descriptive statistics over all collected results.

        Args:
            statistics: A list of statistics to compute. Defaults to
                ["mean", "std", "min", "max"]. Common options from polars'
                `describe` method can be used.

        Returns:
            A nested dictionary summarizing the results. For example:
            {
                "roc_auc": {"mean": 0.95, "std": 0.02}
            }
        """
        if not self.results:
            self.logger.warning("No evaluation results to summarize.")
            return {}

        df = pl.DataFrame(self.results)
        summary_df = df.describe()

        filtered_stats = summary_df.filter(
            pl.col("statistic").is_in(statistics))
        long_format = filtered_stats.unpivot(
            index="statistic", variable_name="metric")

        summary_dict = {}
        for (metric, *_), group_df in long_format.group_by("metric"):
            statistic_dict = dict(
                zip(group_df['statistic'], group_df['value']))
            summary_dict[metric] = {k: (round(v, self.round_digits) if v is not None else v)
                                    for k, v in statistic_dict.items()}

        return summary_dict

    def evaluate(self, y_true: torch.Tensor, y_scores: torch.Tensor) -> dict[str, float]:
        """Computes metrics for a single set of predictions and stores them.

        Args:
            y_true: A tensor of ground truth binary labels (0 or 1).
            y_scores: A tensor of predicted anomaly scores, where higher
                values indicate a higher likelihood of being an anomaly.

        Returns:
            A dictionary containing the computed metrics for this evaluation run.
        """
        metrics = compute_metrics(y_true, y_scores)
        self.results.append(metrics)
        return metrics

    def save_results(self) -> None:
        """Saves all results and a summary to a JSON file.

        This method compiles all individual evaluation results and the summary
        statistics into a single dictionary and saves it as a formatted JSON
        file named "evaluation.json" in the specified `output_dir`.
        """
        results = {
            "results": self.results,
            "summary": self._get_summary()
        }
        save_path = os.path.join(self.output_dir, "evaluation.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4, default=lambda x: float(x))

        self.logger.info(f"Saved evaluation to: {save_path}")


@check_matching_device
def evaluate(y_true: torch.Tensor, y_scores: torch.Tensor, anomaly_as_0: bool = False):
    if anomaly_as_0:
        y_true = 1 - y_true
        y_scores = 1 - y_scores

    best_f1, best_thr = metrics.best_f1_score(y_true, y_scores)
    y_pred = (y_scores >= best_thr).to(y_true.dtype)

    results = {
        "best_f1": best_f1,
        "best_threshold": best_thr,
        "roc_auc": roc_auc_score(y_true.cpu(), y_scores.cpu()),
        "average_precision": metrics.average_precision(y_true, y_scores),
        "mrr": metrics.mean_reciprocal_rank(y_true, y_scores),
        "accuracy": metrics.accuracy(y_true, y_pred),
        "precision": metrics.precision(y_true, y_pred),
        "recall": metrics.recall(y_true, y_pred)
    }

    for k in [10, 50, 100]:
        if len(y_scores) >= k:
            results[f'precision_at_{k}'] = metrics.precision_at_k(
                y_true, y_scores, k=k)
            results[f'recall_at_{k}'] = metrics.recall_at_k(
                y_true, y_scores, k=k)

    return results


class Evaluator:
    def __init__(
        self,
        dataset_name: str,
        method_name: str,
        anomaly_as_0: bool = False,
        output_dir: str = "results/",
        round_digits: Optional[int] = 2
    ):
        if not dataset_name or not method_name:
            raise ValueError(
                "'dataset_name' and 'method_name' must be provided.")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.round_digits = round_digits
        self.anomaly_as_0 = anomaly_as_0
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S/")
        self.output_dir = output_dir

    def evaluate(self, y_true: torch.Tensor, y_scores: torch.Tensor) -> None:
        self.results = evaluate(y_true, y_scores)

    def save(self, save_dir: str, as_dataframe: bool = False) -> None:
        pass

    def eval_preds(self, y_true: torch.Tensor, y_scores: torch.Tensor):
        self.results = evaluate(y_true, y_scores, self.anomaly_as_0)
        return self.results

    def log_roc(self) -> None:
        if not self.results:
            self.logger.warning("No results to print!")
            return
        else:
            auc = self.results["roc_auc"]
            self.logger.info(
                f"AUC on test set with best hyperparams: {auc:4f}")
            print(
                f"AUC on test set with best hyperparams: {auc:4f}")

    def save_results(self, **kwargs):
        if not self.results:
            self.logger.warning("No results to save!")
            return

        save_dir = os.path.join(
            self.output_dir, self.method_name, self.dataset_name, self.time)
        os.makedirs(save_dir, exist_ok=True)

        # dataset_config_path, experiment_config_path, method_config_path, results_path = self._get_output_paths()
        # path_file_map = {
        #     dataset_config_path: self.dataset_config,
        #     experiment_config_path: self.experiment_config,
        #     method_config_path: self.method_config,
        #     results_path: self.results
        # }

        # for p, c in path_file_map.items():
        #     with open(p, 'w') as f:
        #         json.dump({**c, **kwargs}, f, indent=4)
        # self.logger.info(f"Saved results and configs.")

        with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
            json.dump(self.results, f, indent=4)
        self.logger.info(f"Saved metrics.")
