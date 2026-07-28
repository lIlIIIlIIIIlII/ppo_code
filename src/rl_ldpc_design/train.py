from __future__ import annotations

import argparse
import platform
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .callbacks import BestMetricCallback
from .config import ExperimentConfig, load_config, save_config
from .env import FloodingEnv
from .utils import resolve_device, seed_everything, write_text_lines


def create_run_directory(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    code = config.code
    base_name = f"{timestamp}_{code.rows}_{code.cols}_{code.structure_mode}"
    root = Path(config.runtime.output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
) -> None:
    code = config.code
    observation_cols = code.cols - code.rows if code.structure_mode == "systematic" else code.cols
    policy_hidden = [code.rows * observation_cols * 2, code.rows * observation_cols]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "requested_device": config.runtime.device,
        "resolved_device": str(device),
        "library_mode": config.library_mode,
        "reward_mode": (
            "absolute_log_ber_x_0.1"
            if config.library_mode
            else "differential"
        ),
        "systematic_stop_action": (
            code.structure_mode == "systematic" and not config.library_mode
        ),
        "saved_matrix_source": (
            "episode_best_ber" if config.library_mode else "episode_final"
        ),
        "policy_pi_hidden": policy_hidden,
        "policy_vf_hidden": policy_hidden,
    }
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    write_text_lines(run_dir / "run_metadata.txt", lines)
    save_config(config, run_dir / "config_resolved.yaml")


def build_model(
    config: ExperimentConfig,
    env: ActionMasker,
    device: torch.device,
) -> MaskablePPO:
    observation_dim = int(np.prod(env.observation_space.shape))
    policy_hidden = [observation_dim * 2, observation_dim]
    ppo = config.ppo
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=ppo.learning_rate,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        max_grad_norm=ppo.max_grad_norm,
        target_kl=ppo.target_kl,
        policy_kwargs={
            "net_arch": {
                "pi": policy_hidden,
                "vf": policy_hidden,
            }
        },
        seed=config.runtime.seed,
        device=device,
        verbose=1,
    )


def run_training(config: ExperimentConfig) -> Path:
    config.validate()
    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    run_dir = create_run_directory(config)
    save_run_metadata(config, device, run_dir)

    code = config.code
    print(f"run_dir: {run_dir.resolve()}")
    print(
        f"device={device}, structure={code.structure_mode}, "
        f"library_mode={config.library_mode}, "
        f"4cycle_mask={code.use_4cycle_mask}, "
        f"max_col_degree={code.max_col_degree}"
    )

    raw_env = FloodingEnv(
        config=config,
        device=device,
        save_path=run_dir / f"H_{code.rows}_{code.cols}_{code.structure_mode}.txt",
        best_step_history_path=run_dir / "best_H_step_metrics.csv",
    )
    env = ActionMasker(raw_env, lambda wrapped_env: wrapped_env.action_mask())
    model = build_model(config, env, device)
    callback = BestMetricCallback(
        log_metric="ber" if config.library_mode else config.objective.log_metric,
        episode_log_path=run_dir / "episode_best_ber.csv",
        rollout_model_path=run_dir / "model_rollout_latest",
    )

    try:
        model.learn(
            total_timesteps=config.runtime.total_timesteps,
            callback=callback,
        )
    except KeyboardInterrupt:
        interrupted_path = run_dir / "model_interrupted"
        model.save(str(interrupted_path))
        print(f"Training interrupted; checkpoint saved: {interrupted_path}.zip")
        return run_dir
    finally:
        env.close()

    final_model_path = run_dir / "model_final"
    model.save(str(final_model_path))
    print(f"Training complete; final model saved: {final_model_path}.zip")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO to construct an LDPC parity-check matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument("--device", help="Override runtime.device.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Override runtime.total_timesteps.",
    )
    parser.add_argument("--seed", type=int, help="Override runtime.seed.")
    parser.add_argument("--output-root", help="Override runtime.output_root.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config.runtime.device = args.device
    if args.total_timesteps is not None:
        config.runtime.total_timesteps = args.total_timesteps
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.output_root is not None:
        config.runtime.output_root = args.output_root
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
