from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .decoder import FloodingDecoder
from .utils import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a binary parity-check matrix with flooding SPA decoding."
    )
    parser.add_argument("matrix", type=Path, help="Text file containing a binary H matrix.")
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--num-llrs", type=int, default=10_000)
    parser.add_argument("--max-frame-errors", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--systematic",
        action="store_true",
        help="Evaluate only the first n-m bits; assumes H=[P|I].",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = np.loadtxt(args.matrix, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("H matrix must be two-dimensional")
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("H matrix must contain only 0 and 1")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    h = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    rows, cols = matrix.shape
    decoder = FloodingDecoder(rows, cols, args.iterations, device)
    result = decoder.evaluate(
        h,
        systematic=args.systematic,
        snr_db=args.snr_db,
        num_llrs=args.num_llrs,
        max_frame_errors=args.max_frame_errors,
        seed=args.seed,
    )
    print(f"matrix={args.matrix}")
    print(f"shape={rows}x{cols}, rate={decoder.code_rate:.6f}, device={device}")
    print(f"BER={result.ber:.10e}")
    print(f"FER={result.fer:.10e}")
    print(
        f"frames={result.frames}, frame_errors={result.frame_errors}, "
        f"bit_errors={result.bit_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
