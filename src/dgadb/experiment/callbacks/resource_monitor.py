from __future__ import annotations

import os
import time
import psutil
from typing import TYPE_CHECKING

import torch
import polars as pl

from .base import ExperimentCallback

if TYPE_CHECKING:
    from src.dgadb.models.base import TrainingState


class ResourceMonitor(ExperimentCallback):
    def __init__(self, output_dir: str, interval_steps: int = 10) -> None:
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.interval_steps = interval_steps
        self.process = psutil.Process(os.getpid())
        self.resource_logs = []

    def on_train_step_end(self, state: TrainingState):
        if state.total_steps % self.interval_steps == 0:
            log_entry = {
                "timestamp": time.time(),
                "epoch": state.epoch,
                "step_in_epoch": state.step_in_epoch,
                'total_steps': state.total_steps,
                'cpu_percent': self.process.cpu_percent() / float(psutil.cpu_count() or 1),
                'ram_mb': self.process.memory_info().rss / (1024 * 1024),
                'gpu_mem_mb': 0,
                'gpu_mem_allocated_mb': 0
            }
            if state.model is not None and (device := torch.device(state.model.device)).type == 'cuda':
                torch.cuda.synchronize()
                log_entry['gpu_mem_mb'] = torch.cuda.memory_reserved(
                    device) / (1024 * 1024)
                log_entry['gpu_mem_allocated_mb'] = torch.cuda.memory_allocated(
                    device) / (1024 * 1024)

            self.resource_logs.append(log_entry)

    def on_train_end(self, state: TrainingState):
        df = pl.DataFrame(self.resource_logs)
        save_path = os.path.join(self.output_dir, "resource_usage.csv")
        df.write_csv(save_path)
        self.logger.info(f"Resource logs saved to: {save_path}")
