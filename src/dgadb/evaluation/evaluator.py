import os
import json
import logging
from datetime import datetime
from typing import Optional

import torch
import polars as pl
from sklearn.metrics import roc_auc_score

from . import metrics


class Evaluator:
    def __init__(self, dataset_name: str, method_name: str, method_config, dataset_config, experiment_config, anomaly_as_0: bool = False, output_dir: str = "results/", round_digits: Optional[int] = 2):
        if not dataset_name or not method_name:
            raise ValueError(
                "'dataset_name' and 'method_name' must be provided.")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        self.output_dir = output_dir
        self.round_digits = round_digits
        self.anomaly_as_0 = anomaly_as_0

    def _get_output_paths(self) -> tuple[str, str]:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S/")
        target_dir = os.path.join(
            self.output_dir, self.dataset_name, self.method_name, run_name)
        os.makedirs(target_dir, exist_ok=True)

        base_filename = os.path.join(target_dir, self.method_name)
        dataset_config_path = f"{base_filename}_dataset_config.json"
        experiment_config_path = f"{base_filename}_experiment_config.json"
        method_config_path = f"{base_filename}_model_config.json"
        results_path = f"{base_filename}_results.json"

        return dataset_config_path, experiment_config_path, method_config_path, results_path

    def eval_preds(self, y_true: torch.Tensor, y_scores: torch.Tensor):
        if self.anomaly_as_0:
            y_pred = y_pred.copy()
            y_pred = 1 - y_pred
            y_true = y_true.copy()
            y_pred = 1 - y_pred

        best_f1, best_thr = metrics.best_f1_score(y_true, y_scores)
        y_pred = (y_scores >= best_thr).to(y_true.dtype)

        self.results = {
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
                self.results[f'precision_at_{k}'] = metrics.precision_at_k(
                    y_true, y_scores, k=k)
                self.results[f'recall_at_{k}'] = metrics.recall_at_k(
                    y_true, y_scores, k=k)

    def log_roc(self) -> None:
        if not self.results:
            self.logger.warning("No results to print!")
            return
        else:
            auc = self.results["roc_auc"]
            self.logger.info(
                f"AUC on test set with best hyperparams: {auc:4f}")

    def save_results(self, **kwargs):
        if not self.results:
            self.logger.warning("No results to save!")
            return

        dataset_config_path, experiment_config_path, method_config_path, results_path = self._get_output_paths()
        path_file_map = {
            dataset_config_path: self.dataset_config,
            experiment_config_path: self.experiment_config,
            method_config_path: self.method_config,
            results_path: self.results
        }

        for p, c in path_file_map.items():
            with open(p, 'w') as f:
                json.dump({**c, **kwargs}, f, indent=4)
        self.logger.info(f"Saved results and configs.")
