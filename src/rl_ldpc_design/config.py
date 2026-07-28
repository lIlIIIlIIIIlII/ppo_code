from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_VALID_METRICS = {"ber", "fer"}
_VALID_STRUCTURES = {"none", "systematic", "qc"}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def normalize_metric_name(value: str, name: str) -> str:
    metric = str(value).strip().lower()
    if metric not in _VALID_METRICS:
        raise ValueError(f"{name} must be either 'ber' or 'fer'")
    return metric


def normalize_structure_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in _VALID_STRUCTURES:
        raise ValueError("structure_mode must be one of: 'none', 'systematic', 'qc'")
    return mode


def normalize_max_col_degree(value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_int(value, "max_col_degree")


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    output_root: str = "runs"
    total_timesteps: int = 10_000_000
    seed: int = 4


@dataclass(slots=True)
class PPOConfig:
    learning_rate: float = 1e-5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02


@dataclass(slots=True)
class DecoderConfig:
    iterations: int = 8
    train_num_llrs: int = 10_000
    test_num_llrs: int = 10_000
    train_snr_db: float = 5.0
    test_snr_db: float = 5.0
    step_max_frame_errors: int = 100
    test_max_frame_errors: int = 1_000


@dataclass(slots=True)
class CodeConfig:
    rows: int = 16
    cols: int = 32
    structure_mode: str = "none"
    lifting_factor: int = 4
    use_4cycle_mask: bool = False
    max_col_degree: int | None = None


@dataclass(slots=True)
class ObjectiveConfig:
    reward_metric: str = "ber"
    log_metric: str = "ber"


@dataclass(slots=True)
class ExperimentConfig:
    library_mode: bool = False
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ExperimentConfig":
        data = {} if data is None else dict(data)
        known = {
            "library_mode",
            "runtime",
            "ppo",
            "decoder",
            "code",
            "objective",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown top-level config key(s): {', '.join(unknown)}")
        return cls(
            library_mode=data.get("library_mode", False),
            runtime=RuntimeConfig(**dict(data.get("runtime", {}))),
            ppo=PPOConfig(**dict(data.get("ppo", {}))),
            decoder=DecoderConfig(**dict(data.get("decoder", {}))),
            code=CodeConfig(**dict(data.get("code", {}))),
            objective=ObjectiveConfig(**dict(data.get("objective", {}))),
        )

    def validate(self) -> "ExperimentConfig":
        if not isinstance(self.library_mode, bool):
            raise ValueError("library_mode must be a boolean")

        self.runtime.total_timesteps = _positive_int(
            self.runtime.total_timesteps, "runtime.total_timesteps"
        )
        if isinstance(self.runtime.seed, bool) or int(self.runtime.seed) != self.runtime.seed:
            raise ValueError("runtime.seed must be an integer")
        self.runtime.seed = int(self.runtime.seed)

        self.ppo.n_steps = _positive_int(self.ppo.n_steps, "ppo.n_steps")
        self.ppo.batch_size = _positive_int(self.ppo.batch_size, "ppo.batch_size")
        self.ppo.n_epochs = _positive_int(self.ppo.n_epochs, "ppo.n_epochs")
        if self.ppo.batch_size > self.ppo.n_steps:
            raise ValueError("ppo.batch_size must not exceed ppo.n_steps")
        if self.ppo.n_steps % self.ppo.batch_size:
            raise ValueError("ppo.n_steps must be divisible by ppo.batch_size")
        if self.ppo.learning_rate <= 0:
            raise ValueError("ppo.learning_rate must be positive")
        if not 0 < self.ppo.gamma <= 1:
            raise ValueError("ppo.gamma must satisfy 0 < gamma <= 1")
        if not 0 <= self.ppo.gae_lambda <= 1:
            raise ValueError("ppo.gae_lambda must satisfy 0 <= gae_lambda <= 1")
        if self.ppo.clip_range <= 0:
            raise ValueError("ppo.clip_range must be positive")
        if self.ppo.max_grad_norm <= 0:
            raise ValueError("ppo.max_grad_norm must be positive")
        if self.ppo.target_kl is not None and self.ppo.target_kl <= 0:
            raise ValueError("ppo.target_kl must be positive or null")

        self.decoder.iterations = _positive_int(
            self.decoder.iterations, "decoder.iterations"
        )
        self.decoder.train_num_llrs = _positive_int(
            self.decoder.train_num_llrs, "decoder.train_num_llrs"
        )
        self.decoder.test_num_llrs = _positive_int(
            self.decoder.test_num_llrs, "decoder.test_num_llrs"
        )
        self.decoder.step_max_frame_errors = _positive_int(
            self.decoder.step_max_frame_errors, "decoder.step_max_frame_errors"
        )
        self.decoder.test_max_frame_errors = _positive_int(
            self.decoder.test_max_frame_errors, "decoder.test_max_frame_errors"
        )

        self.code.rows = _positive_int(self.code.rows, "code.rows")
        self.code.cols = _positive_int(self.code.cols, "code.cols")
        if self.code.cols <= self.code.rows:
            raise ValueError("code.cols must be greater than code.rows")
        self.code.structure_mode = normalize_structure_mode(self.code.structure_mode)
        self.code.max_col_degree = normalize_max_col_degree(self.code.max_col_degree)
        if self.library_mode:
            self.code.structure_mode = "systematic"
            self.code.use_4cycle_mask = False
            self.code.max_col_degree = None
        if self.code.structure_mode == "qc":
            self.code.lifting_factor = _positive_int(
                self.code.lifting_factor, "code.lifting_factor"
            )
            if (
                self.code.rows % self.code.lifting_factor
                or self.code.cols % self.code.lifting_factor
            ):
                raise ValueError(
                    "QC mode requires rows and cols to be divisible by lifting_factor"
                )

        self.objective.reward_metric = normalize_metric_name(
            self.objective.reward_metric, "objective.reward_metric"
        )
        self.objective.log_metric = normalize_metric_name(
            self.objective.log_metric, "objective.log_metric"
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("The YAML root must be a mapping")
    return ExperimentConfig.from_mapping(raw).validate()


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
