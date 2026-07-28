# RL-LDPC Code Design

Reference implementation for constructing LDPC parity-check matrices with MaskablePPO and evaluating them with a flooding Sum-Product Algorithm (SPA) decoder.

The agent adds edges to a parity-check matrix (H) one at a time. Action masking excludes occupied positions and choices that violate structural constraints, while each candidate matrix is scored by its BER or FER over an AWGN channel. The implementation supports unconstrained, systematic, and QC-LDPC structures.

## Layout

```text
src/rl_ldpc_design/
  env.py            Gymnasium environment for matrix construction
  decoder.py        Flooding SPA decoder and BER/FER evaluation
  train.py          MaskablePPO training
  evaluate.py       Evaluation of a saved H matrix
  config.py         YAML configuration loading and validation
  callbacks.py      Metric logging and checkpoint saving
  gf2.py            GF(2) rank and 4-cycle checks
configs/
  default.yaml      Default training configuration
  smoke_test.yaml   Small configuration for a quick sanity check
environment.yml
pyproject.toml
requirements.txt
```

Run all commands from the repository root.

## Install

Requires Python 3.10 or later.

```bash
conda env create -f environment.yml
conda activate rl-ldpc-code-design
```

Alternatively, use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For CUDA, install a PyTorch build compatible with your system before installing the project.

## Quickstart

Check the training pipeline with a small CPU configuration.

```bash
rl-ldpc-train --config configs/smoke_test.yaml
```

Run training with the default configuration.

```bash
rl-ldpc-train --config configs/default.yaml
```

The package can also be executed as a module.

```bash
python -m rl_ldpc_design --config configs/default.yaml
```

The device, training length, seed, and output directory can be overridden from the command line.

```bash
rl-ldpc-train --config configs/default.yaml \
  --device cuda \
  --total-timesteps 100000 \
  --seed 42 \
  --output-root runs_exp
```

## Evaluate

Evaluate a binary parity-check matrix produced during training.

```bash
rl-ldpc-evaluate runs/<run>/H_16_32_systematic.txt \
  --snr-db 5.0 \
  --iterations 8 \
  --num-llrs 10000 \
  --max-frame-errors 1000 \
  --device auto \
  --systematic
```

`--systematic` assumes (H=[P\mid I]) and computes BER only over the first (n-m) information bits. Omit it for a general parity-check matrix.

## Configuration

Experiments are configured with YAML files.

| Key                       | Meaning                            | Default example |
| ------------------------- | ---------------------------------- | --------------- |
| `runtime.device`          | `auto`, `cpu`, or `cuda`           | `auto`          |
| `runtime.total_timesteps` | PPO training length                | `10000000`      |
| `decoder.iterations`      | Number of SPA iterations           | `8`             |
| `decoder.train_snr_db`    | SNR used during training           | `5.0`           |
| `code.rows`, `code.cols`  | Matrix dimensions (m\times n)      | `16`, `32`      |
| `code.structure_mode`     | `none`, `systematic`, or `qc`      | `none`          |
| `code.use_4cycle_mask`    | Mask actions that create 4-cycles  | `false`         |
| `code.max_col_degree`     | Maximum column degree              | `null`          |
| `objective.reward_metric` | Reward metric: `ber` or `fer`      | `ber`           |
| `objective.log_metric`    | Metric used to rank saved matrices | `ber`           |

Structure modes:

* `none`: constructs a general binary matrix without a fixed structure.
* `systematic`: fixes the right block to the identity matrix and constructs (H=[P\mid I]).
* `qc`: adds edges as circulant permutation blocks defined by `lifting_factor`.
* `library_mode`: Set library_mode to true to run in library mode.


## Outputs

Each run creates a separate directory under `runtime.output_root`.

```text
runs/<timestamp>_<rows>_<cols>_<structure>/
  config_resolved.yaml        Resolved experiment configuration
  run_metadata.txt            Runtime and model metadata
  H_*.txt                     Best-performing matrix
  H_*_top1.txt                Up to three top-performing matrices
  best_H_step_metrics.csv     Step history for the best matrix
  episode_best_ber.csv        Episode-level BER history
  model_rollout_latest.zip    Latest rollout checkpoint
  model_final.zip             Final model
```

If training is interrupted, `model_interrupted.zip` is saved.

## Notes

* The default configuration uses a large decoding workload and training horizon. Start with `smoke_test.yaml` to verify the installation.
* Evaluation keeps generating LLR batches until the target number of frame errors is reached, so low-error-rate settings can take longer.
* `device: auto` selects CUDA when available and otherwise falls back to CPU.
* Keep the resolved configuration and `runtime.seed` to reproduce an experiment.
