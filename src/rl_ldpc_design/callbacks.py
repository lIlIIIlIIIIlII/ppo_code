from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .config import normalize_metric_name


class BestMetricCallback(BaseCallback):
    """Log episode BER and save a checkpoint at every rollout boundary."""

    def __init__(
        self,
        log_metric: str,
        episode_log_path: str | Path | None = None,
        rollout_model_path: str | Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_metric = normalize_metric_name(log_metric, "log_metric")
        self.metric_upper = self.log_metric.upper()
        self.best_metric = float("inf")
        self.metric_sum = 0.0
        self.metric_count = 0
        self.episode_idx = 0
        self.best_ber = float("inf")
        self.episode_log_path = (
            Path(episode_log_path) if episode_log_path is not None else None
        )
        self.rollout_model_path = (
            Path(rollout_model_path) if rollout_model_path is not None else None
        )
        if self.episode_log_path is not None:
            self.episode_log_path.parent.mkdir(parents=True, exist_ok=True)
            if (
                not self.episode_log_path.exists()
                or self.episode_log_path.stat().st_size == 0
            ):
                self.episode_log_path.write_text(
                    "episode_idx,global_step,ber,best_ber_so_far\n",
                    encoding="utf-8",
                )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            metric = info.get(self.log_metric)
            if metric is None:
                continue
            metric_value = float(metric)
            self.metric_sum += metric_value
            self.metric_count += 1
            self.best_metric = min(self.best_metric, metric_value)

            ber = info.get("ber")
            if ber is not None:
                ber_value = float(ber)
                self.episode_idx += 1
                self.best_ber = min(self.best_ber, ber_value)
                if self.episode_log_path is not None:
                    with self.episode_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"{self.episode_idx},{int(self.num_timesteps)},"
                            f"{ber_value:.10e},{self.best_ber:.10e}\n"
                        )
        return True

    def _on_rollout_end(self) -> None:
        try:
            env_best = self.training_env.envs[0].unwrapped.best_metric_value
            self.best_metric = float(env_best)
        except (AttributeError, IndexError, TypeError):
            pass

        best_metric_value = (
            f"{self.best_metric:.2e}"
            if np.isfinite(self.best_metric)
            else str(self.best_metric)
        )
        self.logger.record(f"train/best_{self.metric_upper}", best_metric_value)

        best_ber_value = (
            f"{self.best_ber:.2e}"
            if np.isfinite(self.best_ber)
            else str(self.best_ber)
        )
        self.logger.record("train/best_BER", best_ber_value)

        if self.metric_count > 0:
            average = self.metric_sum / self.metric_count
            self.logger.record(f"train/avg_{self.metric_upper}", f"{average:.2e}")
            self.metric_sum = 0.0
            self.metric_count = 0

        if self.rollout_model_path is not None:
            self.rollout_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.rollout_model_path))
