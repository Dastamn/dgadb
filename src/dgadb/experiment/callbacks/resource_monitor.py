from __future__ import annotations
from .base import ExperimentCallback

import os
import time
import psutil
from typing import Literal, TYPE_CHECKING

import torch
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


if TYPE_CHECKING:
    from dgadb.models.base import TrainingState


class ResourceMonitor(ExperimentCallback):
    """A callback to monitor and log system resource usage during training.

    This callback tracks CPU usage, RAM usage, and (if available) GPU memory
    usage at specified intervals (both step-wise and epoch-wise). The collected
    data is saved to CSV files and visualized in plots, which are stored in a
    specified output directory.

    Args:
        output_dir: The directory where logs and plots will be saved.
        epoch_interval: The frequency (in epochs) for logging and plotting
            epoch-level resource usage.
        step_interval: The frequency (in steps) for logging and plotting
            step-level resource usage.
        enable_memory_history: If True and a GPU is used, enables PyTorch's
            detailed CUDA memory history recording for debugging memory issues.
    """

    def __init__(self, output_dir: str, epoch_interval: int = 1, step_interval: int = 10, enable_memory_history: bool = False, verbose: bool = False) -> None:
        super().__init__(verbose)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval
        self.enable_memory_history = enable_memory_history
        self.process = psutil.Process(os.getpid())
        self.step_resource_logs = []
        self.epoch_resource_logs = []
        self.uses_gpu = False

    def _is_using_gpu(self, state: TrainingState):
        return (state.model is not None
                and torch.device(state.model.device).type == 'cuda')

    def _log(self, state: TrainingState, interval_type: Literal["step", "epoch"]) -> None:
        log_entry = {
            "timestamp": time.time(),
            "epoch": state.epoch,
            "step_in_epoch": state.step_in_epoch,
            "step": state.total_steps,
            "ram_mb": self.process.memory_info().rss / (1024 * 1024),
            "gpu_mem_mb": 0,
            "gpu_mem_allocated_mb": 0
        }

        cpu_count = psutil.cpu_count()
        if cpu_count is None:
            self.logger.warning(
                "Cannot determine CPU count. Defaulting to 1.")
            cpu_count = 1

        log_entry["cpu_percent"] = self.process.cpu_percent() / \
            float(cpu_count)

        if self.uses_gpu:
            torch.cuda.synchronize()
            log_entry['gpu_mem_mb'] = torch.cuda.memory_reserved() / \
                (1024 ** 2)
            log_entry['gpu_mem_allocated_mb'] = torch.cuda.memory_allocated() / \
                (1024 ** 2)

        if interval_type == "step":
            self.step_resource_logs.append(log_entry)
        else:
            self.epoch_resource_logs.append(log_entry)

    def _plot(self, interval_type: Literal["step", "epoch"]) -> None:
        resource_logs = self.epoch_resource_logs if interval_type == "epoch" else self.step_resource_logs
        df = pl.DataFrame(resource_logs)
        df = df.with_columns([
            (df["timestamp"] - df["timestamp"][0]).alias("time_elapsed_sec"),
            df["timestamp"].diff().alias(f"{interval_type}_duration_sec")
        ])

        save_path = os.path.join(
            self.output_dir, f"resource_usage_{interval_type}.csv")
        df.write_csv(save_path)

        if self.verbose:
            self.logger.info(f"Resource logs saved to: {save_path}")

        plt.figure(figsize=(10, 6))
        plt.plot(df["time_elapsed_sec"], df["cpu_percent"], color="blue")
        plt.title("CPU Usage Over Time")
        plt.xlabel("Time Elapsed (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    f"cpu_usage_{interval_type}.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df["time_elapsed_sec"], df["ram_mb"], color="green")
        plt.title("RAM Usage Over Time")
        plt.xlabel("Time Elapsed (seconds)")
        plt.ylabel("RAM (MB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    f"ram_usage_{interval_type}.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df[interval_type],
                 df[f"{interval_type}_duration_sec"], color="purple")
        cap_interval_type = interval_type[0].upper() + interval_type[1:]
        plt.title(f"{cap_interval_type} Duration Over Time")
        plt.xlabel(cap_interval_type)
        plt.ylabel(f"{cap_interval_type} Duration (seconds)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    f"{interval_type}_duration.png"))
        plt.close()

        if "gpu_mem_mb" in df.columns and df["gpu_mem_mb"].sum() > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(df["time_elapsed_sec"], df["gpu_mem_mb"],
                     label="Reserved", color="red", linestyle="--")
            plt.plot(df["time_elapsed_sec"], df["gpu_mem_allocated_mb"],
                     label="Allocated", color="blue", linestyle="-")
            plt.title("GPU Memory Usage Over Time")
            plt.xlabel("Time Elapsed (seconds)")
            plt.ylabel("GPU Memory (MB)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,
                        f"gpu_usage_{interval_type}.png"))
            plt.close()

    def on_train_begin(self, state: TrainingState) -> None:
        self.uses_gpu = self._is_using_gpu(state)
        if not self.uses_gpu and self.enable_memory_history:
            self.logger.warning(
                "Model training not using GPU but memory history is enabled, disabling memory history.")
            self.enable_memory_history = False

        if self.enable_memory_history:
            try:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
            except RuntimeError as e:
                self.logger.error(e)
                self.enable_memory_history = False

    def on_train_step_begin(self, state: TrainingState) -> None:
        if state.total_steps == 0 or state.total_steps % self.step_interval == 0:
            self._log(state, "step")
            self._plot("step")

    def on_train_epoch_begin(self, state: TrainingState) -> None:
        if state.epoch == 0 or state.epoch % self.epoch_interval == 0:
            self._log(state, "epoch")
            self._plot("epoch")

    def on_train_epoch_end(self, state: TrainingState) -> None:
        if self.enable_memory_history:
            torch.cuda.memory._dump_snapshot("snapshot.pickle")
