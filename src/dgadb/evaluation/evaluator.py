import os
import json
import logging
from typing import Optional

import torch
import polars as pl
from sklearn.metrics import roc_auc_score

from . import metrics


class Evaluator:
    def __init__(self, dataset_name: str, method_name: str, output_dir: str = "results/", round_digits: Optional[int] = 2):
        if not dataset_name or not method_name:
            raise ValueError(
                "'dataset_name' and 'method_name' must be provided.")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.output_dir = output_dir
        self.round_digits = round_digits

        self.results_per_snapshot: list[dict[str, float]] = []

    def _get_output_paths(self) -> tuple[str, str]:
        target_dir = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(target_dir, exist_ok=True)

        base_filename = os.path.join(target_dir, self.method_name)
        csv_path = f"{base_filename}_snapshots.csv"
        json_path = f"{base_filename}_summary.json"

        return csv_path, json_path

    def eval_snapshot(self, y_true: torch.Tensor, y_scores: torch.Tensor, snapshot_id: int):
        best_f1, best_thr = metrics.best_f1_score(y_true, y_scores)
        y_pred = (y_scores >= best_thr).to(y_true.dtype)

        snapshot_results = {
            'snapshot_id': snapshot_id,
            'best_f1': best_f1,
            'best_threshold': best_thr,
            "roc_auc": roc_auc_score(y_true.cpu(), y_scores.cpu()),
            # Ranking-based metrics
            'average_precision': metrics.average_precision(y_true, y_scores),
            'mrr': metrics.mean_reciprocal_rank(y_true, y_scores),
            # Classification-based metrics at best_thr
            'accuracy': metrics.accuracy(y_true, y_pred),
            'precision': metrics.precision(y_true, y_pred),
            'recall': metrics.recall(y_true, y_pred)
        }

        for k in [10, 50, 100]:
            if len(y_scores) >= k:
                snapshot_results[f'precision_at_{k}'] = metrics.precision_at_k(
                    y_true, y_scores, k=k)
                snapshot_results[f'recall_at_{k}'] = metrics.recall_at_k(
                    y_true, y_scores, k=k)

        self.results_per_snapshot.append(snapshot_results)

    def get_summary(self) -> dict[str, dict[str, float]]:
        if not self.results_per_snapshot:
            return {}

        df = pl.DataFrame(self.results_per_snapshot)

        summary_stats = df.describe()

        filtered_stats = (
            summary_stats
            .filter(pl.col("statistic").is_in(["mean", "std"]))
            .drop("snapshot_id")
        )

        long_format = filtered_stats.unpivot(
            index="statistic", variable_name="metric")

        summary = {}
        for (metric, *_), group_df in long_format.group_by("metric"):
            stats_dict = dict(zip(group_df['statistic'], group_df['value']))

            mean_val = stats_dict.get('mean', 0.0)
            std_val = stats_dict.get('std', 0.0)

            if self.round_digits is not None:
                mean_val = round(mean_val, self.round_digits)
                std_val = round(std_val, self.round_digits)

            summary[metric] = {'mean': mean_val, 'std': std_val}

        return summary

    def save_results(self):
        if not self.results_per_snapshot:
            self.logger.warning("No results to save!")
            return

        summary = self.get_summary()
        csv_path, json_path = self._get_output_paths()

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Saved summary results to: {json_path}")

        df = pl.DataFrame(self.results_per_snapshot)

        if self.round_digits is not None:
            float_cols = [col for col, dtype in df.schema.items()
                          if dtype in [pl.Float32, pl.Float64]]
            df = df.with_columns(
                pl.col(float_cols).round(self.round_digits)
            )

        df.write_csv(csv_path)
        self.logger.info(f"Saved per-snapshot results to: {csv_path}")
