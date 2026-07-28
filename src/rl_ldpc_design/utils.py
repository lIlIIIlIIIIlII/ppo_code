from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def resolve_device(value: str) -> torch.device:
    name = str(value).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    if resolved.type == "cuda" and resolved.index is not None:
        count = torch.cuda.device_count()
        if resolved.index >= count:
            raise RuntimeError(
                f"CUDA device index {resolved.index} is unavailable; found {count} CUDA device(s)"
            )
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_text_lines(path: str | Path, lines: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
