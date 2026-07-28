from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .config import ExperimentConfig
from .decoder import FloodingDecoder
from .gf2 import creates_4cycle_if_add, gf2_rank_from_bitrows, positions_creating_4cycle


class FloodingEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment that incrementally constructs an LDPC H matrix."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device,
        save_path: str | Path,
        best_step_history_path: str | Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        code = config.code
        decoder = config.decoder
        objective = config.objective
        self.library_mode = config.library_mode

        self.core = FloodingDecoder(
            rows=code.rows,
            cols=code.cols,
            iterations=decoder.iterations,
            device=device,
        )
        self.m, self.n = self.core.m, self.core.n
        self.structure_mode = code.structure_mode
        self.use_4cycle_mask = bool(code.use_4cycle_mask)
        self.max_col_degree = code.max_col_degree
        self.systematic = self.structure_mode == "systematic"
        self.qc = self.structure_mode == "qc"
        self.k = self.n - self.m
        self.action_cols = self.k if self.systematic else self.n
        self.num_edit_actions = self.m * self.action_cols
        self.max_episode_steps = self.num_edit_actions
        self.stop_action = (
            self.num_edit_actions
            if self.systematic and not self.library_mode
            else None
        )
        action_count = self.num_edit_actions + int(self.stop_action is not None)
        self.action_space = gym.spaces.Discrete(action_count)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.m, self.action_cols),
            dtype=np.float32,
        )

        self.fixed_block: torch.Tensor | None = None
        if self.systematic:
            self.fixed_block = torch.eye(self.m, dtype=torch.float32, device=device)

        self.lifting_factor: int | None = None
        self._locked_block_mask: np.ndarray | None = None
        if self.qc:
            self.lifting_factor = int(code.lifting_factor)
            self._locked_block_mask = np.zeros((self.m, self.n), dtype=np.bool_)
            self.max_episode_steps = (
                (self.m // self.lifting_factor) * (self.n // self.lifting_factor)
            )

        self.reward_metric = objective.reward_metric
        self.log_metric = objective.log_metric
        self.best_metric_value = float("inf")
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_step_history_path = Path(best_step_history_path)
        self.best_step_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = 3
        self.top_h_entries: list[tuple[float, np.ndarray]] = []
        self.top_h_paths = [
            self.save_path.parent
            / f"H_{self.m}_{self.n}_{self.structure_mode}_top{index + 1}.txt"
            for index in range(self.top_k)
        ]

        self.s: torch.Tensor | None = None
        self._mask: np.ndarray | None = None
        self.prev_r = 0.0
        self._full_mask_warned = False
        self._episode_steps = 0
        self._row_bits = [0] * self.m
        self._col_bits = [0] * self.n
        self._row_weight = np.zeros(self.m, dtype=np.int32)
        self._col_weight = np.zeros(self.n, dtype=np.int32)
        self._nnz = 0
        self._step_metric_history: list[dict[str, Any]] = []
        self._episode_best_ber = float("inf")
        self._episode_best_fer = float("inf")
        self._episode_best_matrix: np.ndarray | None = None
        self._episode_best_step: int | None = None
        self.step_seed = config.runtime.seed

    def _apply_structure(self, matrix: torch.Tensor) -> torch.Tensor:
        if not self.systematic:
            return matrix
        assert self.fixed_block is not None
        matrix = matrix.clone()
        matrix[:, self.k :] = self.fixed_block
        return matrix

    def _policy_obs(self) -> np.ndarray:
        assert self.s is not None
        visible = self.s[:, : self.action_cols]
        return visible.detach().cpu().numpy().astype(np.float32)

    def _action_to_flat_index(self, action: int) -> int:
        if action < 0 or action >= self.num_edit_actions:
            raise ValueError("Action index out of range")
        row = action // self.action_cols
        col = action % self.action_cols
        return int(row * self.n + col)

    def _qc_block_bounds(self, row: int, col: int) -> tuple[int, int, int, int]:
        assert self.lifting_factor is not None
        z = self.lifting_factor
        row_start = (int(row) // z) * z
        col_start = (int(col) // z) * z
        return row_start, row_start + z, col_start, col_start + z

    def _qc_lifted_coords(self, row: int, col: int) -> list[tuple[int, int]]:
        row_start, _, col_start, _ = self._qc_block_bounds(row, col)
        assert self.lifting_factor is not None
        z = self.lifting_factor
        shift = (int(col) - col_start - (int(row) - row_start)) % z
        return [
            (row_start + index, col_start + ((index + shift) % z))
            for index in range(z)
        ]

    def _qc_candidate_valid(self, coords: list[tuple[int, int]]) -> bool:
        temp_rows = list(self._row_bits)
        temp_cols = list(self._col_bits)
        for row, col in coords:
            if (
                self.max_col_degree is not None
                and int(self._col_weight[col]) >= self.max_col_degree
            ):
                return False
            if self.use_4cycle_mask and creates_4cycle_if_add(
                temp_rows, temp_cols, row, col
            ):
                return False
            temp_rows[row] = int(temp_rows[row]) | (1 << col)
            temp_cols[col] = int(temp_cols[col]) | (1 << row)
        return True

    def _add_edge(self, row: int, col: int) -> bool:
        bit = 1 << int(col)
        if self._row_bits[int(row)] & bit:
            return False
        assert self.s is not None
        flat_action = int(row) * self.n + int(col)
        self.s = self.core.toggle_entry(self.s, flat_action)
        self.core.add_edge(int(row), int(col))
        self._row_bits[int(row)] = int(self._row_bits[int(row)]) | bit
        self._col_bits[int(col)] = int(self._col_bits[int(col)]) | (1 << int(row))
        self._row_weight[int(row)] += 1
        self._col_weight[int(col)] += 1
        self._nnz += 1
        return True

    def _apply_qc_action(self, row: int, col: int) -> list[tuple[int, int]]:
        coords = self._qc_lifted_coords(row, col)
        added = [coord for coord in coords if self._add_edge(*coord)]
        row_start, row_end, col_start, col_end = self._qc_block_bounds(row, col)
        assert self._locked_block_mask is not None
        self._locked_block_mask[row_start:row_end, col_start:col_end] = True
        return added

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.s = self._apply_structure(self.core.reset_matrix())
        self.core.set_graph(self.s)
        self._mask = np.ones(self.action_space.n, dtype=np.bool_)
        self.prev_r = 0.0
        self._full_mask_warned = False
        self._episode_steps = 0
        self._row_bits = [0] * self.m
        self._col_bits = [0] * self.n
        self._row_weight.fill(0)
        self._col_weight.fill(0)
        self._nnz = 0
        self._step_metric_history = []
        self._episode_best_ber = float("inf")
        self._episode_best_fer = float("inf")
        self._episode_best_matrix = None
        self._episode_best_step = None
        if self.qc:
            assert self._locked_block_mask is not None
            self._locked_block_mask.fill(False)
        self._sync_rank_state_from_matrix()
        self.step_seed = int(torch.randint(1, 11, (1,)).item())
        return self._policy_obs(), {}

    def _update_top_h(self, metric_value: float, matrix: np.ndarray) -> bool:
        metric_value = float(metric_value)
        updated = False
        if (
            len(self.top_h_entries) < self.top_k
            or metric_value < self.top_h_entries[-1][0]
        ):
            self.top_h_entries.append((metric_value, np.array(matrix, copy=True)))
            self.top_h_entries.sort(key=lambda item: item[0])
            self.top_h_entries = self.top_h_entries[: self.top_k]
            updated = True
            np.savetxt(self.save_path, self.top_h_entries[0][1], fmt="%d")
            for index, (_, top_matrix) in enumerate(self.top_h_entries):
                np.savetxt(self.top_h_paths[index], top_matrix, fmt="%d")
            for index in range(len(self.top_h_entries), self.top_k):
                if self.top_h_paths[index].exists():
                    self.top_h_paths[index].unlink()
        return updated

    def _save_best_step_history(self, saved_ber: float, saved_fer: float) -> None:
        lines = [
            "episode_step,nnz,action,row,col,ber_step,fer_step,"
            "step_seed,saved_ber,saved_fer\n"
        ]
        for entry in self._step_metric_history:
            lines.append(
                f"{entry['episode_step']},{entry['nnz']},{entry['action']},"
                f"{entry['row']},{entry['col']},{entry['ber_step']:.10e},"
                f"{entry['fer_step']:.10e},{entry['step_seed']},"
                f"{float(saved_ber):.10e},{float(saved_fer):.10e}\n"
            )
        self.best_step_history_path.write_text("".join(lines), encoding="utf-8")

    def _sync_rank_state_from_matrix(self) -> None:
        if self.s is None:
            self._row_bits = [0] * self.m
            self._col_bits = [0] * self.n
            self._row_weight.fill(0)
            self._col_weight.fill(0)
            self._nnz = 0
            return
        matrix = self.s.detach().cpu().numpy()
        nonzero_rows, nonzero_cols = np.nonzero(matrix)
        self._row_bits = [0] * self.m
        self._col_bits = [0] * self.n
        for row, col in zip(
            nonzero_rows.tolist(), nonzero_cols.tolist(), strict=True
        ):
            self._row_bits[int(row)] = int(self._row_bits[int(row)]) | (1 << int(col))
            self._col_bits[int(col)] = int(self._col_bits[int(col)]) | (1 << int(row))
        self._row_weight[:] = np.count_nonzero(matrix, axis=1).astype(np.int32)
        self._col_weight[:] = np.count_nonzero(matrix, axis=0).astype(np.int32)
        self._nnz = int(nonzero_rows.size)

    def _is_terminal_matrix(self) -> bool:
        if np.any(self._row_weight == 0) or np.any(self._col_weight == 0):
            return False
        return bool(gf2_rank_from_bitrows(self._row_bits) == self.m)

    def action_mask(self) -> np.ndarray:
        if self.s is None:
            return np.ones(self.action_space.n, dtype=np.bool_)
        matrix = self.s.detach().cpu().numpy()
        editable = matrix[:, : self.action_cols]
        edit_mask = editable == 0

        if self.max_col_degree is not None:
            allowed_cols = (
                self._col_weight[: self.action_cols] < self.max_col_degree
            ).reshape(1, -1)
            edit_mask &= allowed_cols

        if self.use_4cycle_mask and not self.qc:
            cycle_mask = positions_creating_4cycle(matrix)[:, : self.action_cols]
            edit_mask &= ~cycle_mask.astype(np.bool_)

        if self.qc:
            assert self._locked_block_mask is not None
            assert self.lifting_factor is not None
            edit_mask &= ~self._locked_block_mask
            z = self.lifting_factor
            qc_mask = np.zeros_like(edit_mask)
            for row_start in range(0, self.m, z):
                for col_start in range(0, self.n, z):
                    if self._locked_block_mask[row_start, col_start]:
                        continue
                    for shift in range(z):
                        coords = [
                            (
                                row_start + index,
                                col_start + ((index + shift) % z),
                            )
                            for index in range(z)
                        ]
                        if self._qc_candidate_valid(coords):
                            for row, col in coords:
                                qc_mask[row, col] = True
            edit_mask &= qc_mask

        flattened = edit_mask.reshape(-1).astype(np.bool_)
        if self.stop_action is not None:
            self._mask = np.concatenate(
                (flattened, np.array([True], dtype=np.bool_))
            )
        else:
            self._mask = flattened

        if (
            flattened.sum() == 0
            and not self._full_mask_warned
            and not self.library_mode
        ):
            suffix = (
                "; only stop action is valid"
                if self.stop_action is not None
                else ""
            )
            warnings.warn(
                f"Action mask is saturated: no valid edge actions left{suffix}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._full_mask_warned = True
        return self._mask

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            raise ValueError("Action index out of range")
        mask = self.action_mask()
        if not mask[action]:
            raise ValueError("Invalid action selected (masked out)")

        reward = 0.0
        info: dict[str, Any] = {}
        terminated = bool(self.stop_action is not None and action == self.stop_action)
        row = col = None
        if not terminated:
            row = int(action // self.action_cols)
            col = int(action % self.action_cols)
            if self.qc:
                added_coords = self._apply_qc_action(row, col)
                if not added_coords:
                    raise ValueError("QC action did not add any edge")
                info["added_edges"] = added_coords
                info["added_edges_count"] = len(added_coords)
            elif not self._add_edge(row, col):
                raise ValueError("Invalid action selected (edge already exists)")
            self._mask = None

            step_result = self.core.evaluate(
                systematic=self.systematic,
                snr_db=self.config.decoder.train_snr_db,
                num_llrs=self.config.decoder.train_num_llrs,
                max_frame_errors=self.config.decoder.step_max_frame_errors,
                seed=self.step_seed,
            )
            step_metric = (
                step_result.ber
                if self.library_mode or self.reward_metric == "ber"
                else step_result.fer
            )
            score = float(-np.log(max(float(step_metric), 1e-12)))
            if self.library_mode:
                # Preserve the absolute BER reward used by the original ppo.py.
                reward = score * 0.1
            else:
                reward = score - self.prev_r
                self.prev_r = score
            info.update(
                {
                    "ber_step": step_result.ber,
                    "fer_step": step_result.fer,
                    "reward_metric_step": float(step_metric),
                    "reward_mode": (
                        "absolute_log_ber" if self.library_mode else "differential"
                    ),
                    "step_seed": self.step_seed,
                }
            )
            if (
                self.library_mode
                and float(step_result.ber) < self._episode_best_ber
            ):
                assert self.s is not None
                self._episode_best_ber = float(step_result.ber)
                self._episode_best_fer = float(step_result.fer)
                self._episode_best_matrix = np.array(
                    self.s.detach().cpu().numpy(),
                    copy=True,
                )
                self._episode_best_step = self._episode_steps + 1
            self._step_metric_history.append(
                {
                    "episode_step": self._episode_steps + 1,
                    "nnz": self._nnz,
                    "action": action,
                    "row": row,
                    "col": col,
                    "ber_step": float(step_result.ber),
                    "fer_step": float(step_result.fer),
                    "step_seed": self.step_seed,
                }
            )

        if not self.systematic and not terminated:
            terminated = self._is_terminal_matrix()
        valid_edit_actions = int(self.action_mask()[: self.num_edit_actions].sum())
        if not terminated and not self.systematic and valid_edit_actions == 0:
            terminated = True
            info["terminated_by_mask"] = True

        self._episode_steps += 1
        truncated = bool(
            self._episode_steps >= self.max_episode_steps and not terminated
        )
        episode_done = bool(terminated or truncated)

        if episode_done:
            final_result = self.core.evaluate(
                systematic=self.systematic,
                snr_db=self.config.decoder.test_snr_db,
                num_llrs=self.config.decoder.test_num_llrs,
                max_frame_errors=self.config.decoder.test_max_frame_errors,
                seed=self.config.runtime.seed,
            )
            info["final_ber"] = final_result.ber
            info["final_fer"] = final_result.fer

            if self.library_mode:
                if self._episode_best_matrix is None:
                    raise RuntimeError(
                        "library_mode episode ended without an evaluated matrix"
                    )
                saved_ber = self._episode_best_ber
                saved_fer = self._episode_best_fer
                saved_metric = saved_ber
                saved_matrix = self._episode_best_matrix
                info["episode_best_ber"] = saved_ber
                info["episode_best_fer"] = saved_fer
                info["episode_best_step"] = self._episode_best_step
                info["saved_matrix_source"] = "episode_best_ber"
            else:
                saved_ber = float(final_result.ber)
                saved_fer = float(final_result.fer)
                saved_metric = (
                    saved_ber if self.log_metric == "ber" else saved_fer
                )
                assert self.s is not None
                saved_matrix = self.s.detach().cpu().numpy()
                info["saved_matrix_source"] = "episode_final"

            info["ber"] = saved_ber
            info["fer"] = saved_fer
            info["log_metric_value"] = saved_metric
            info["top_h_updated"] = self._update_top_h(
                saved_metric,
                saved_matrix,
            )
            info["top_h_paths"] = [
                str(self.top_h_paths[index])
                for index in range(len(self.top_h_entries))
            ]
            if saved_metric < self.best_metric_value:
                self.best_metric_value = float(saved_metric)
                self._save_best_step_history(saved_ber, saved_fer)
                info["best_step_history_updated"] = True
                info["best_step_history_path"] = str(self.best_step_history_path)

        return self._policy_obs(), reward, terminated, truncated, info
