from __future__ import annotations

import numpy as np
import torch


def gf2_rank(matrix: torch.Tensor) -> int:
    h = (matrix % 2).clone().to(torch.uint8)
    rows, cols = h.shape
    rank = 0
    pivot_col = 0
    for row in range(rows):
        while pivot_col < cols:
            pivot_row = row
            while pivot_row < rows and h[pivot_row, pivot_col] == 0:
                pivot_row += 1
            if pivot_row == rows:
                pivot_col += 1
                continue
            if pivot_row != row:
                temp = h[row].clone()
                h[row] = h[pivot_row]
                h[pivot_row] = temp
            for other in range(rows):
                if other != row and h[other, pivot_col] == 1:
                    h[other] ^= h[row]
            rank += 1
            pivot_col += 1
            break
        if pivot_col >= cols:
            break
    return rank


def gf2_rank_from_bitrows(bit_rows: list[int]) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for index, row in enumerate(bit_rows):
        value = int(row)
        if value < 0:
            raise ValueError(
                f"bit_rows[{index}] is negative ({value}); expected non-negative GF(2) rows"
            )
        max_reductions = value.bit_length() + 1
        reductions = 0
        while value:
            reductions += 1
            if reductions > max_reductions:
                raise RuntimeError(
                    f"GF(2) elimination did not converge for bit_rows[{index}]={int(row)}"
                )
            pivot = value.bit_length() - 1
            base = basis.get(pivot)
            if base is None:
                basis[pivot] = value
                rank += 1
                break
            reduced = value ^ base
            if reduced >= value:
                raise RuntimeError(
                    f"GF(2) elimination stalled at bit_rows[{index}] "
                    f"(value={value}, base={base}, pivot={pivot})"
                )
            value = reduced
    return rank


def positions_creating_4cycle(matrix: np.ndarray) -> np.ndarray:
    h = np.asarray(matrix, dtype=np.uint8)
    if h.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, cols = h.shape
    result = np.zeros((rows, cols), dtype=np.bool_)
    row_bits = [0] * rows
    for row in range(rows):
        bits = 0
        for col in np.flatnonzero(h[row]):
            bits |= 1 << int(col)
        row_bits[row] = bits

    candidates = [0] * rows
    for first in range(rows):
        for second in range(first + 1, rows):
            if row_bits[first] & row_bits[second]:
                candidates[first] |= row_bits[second]
                candidates[second] |= row_bits[first]

    full_col_mask = (1 << cols) - 1
    for row in range(rows):
        bits = candidates[row] & (~row_bits[row] & full_col_mask)
        while bits:
            least_significant = bits & -bits
            result[row, least_significant.bit_length() - 1] = True
            bits ^= least_significant
    return result


def creates_4cycle_if_add(
    row_bits: list[int],
    col_bits: list[int],
    row_idx: int,
    col_idx: int,
) -> bool:
    existing_rows = int(col_bits[col_idx])
    current_row = int(row_bits[row_idx])
    while existing_rows:
        least_significant = existing_rows & -existing_rows
        other_row = least_significant.bit_length() - 1
        if current_row & int(row_bits[other_row]):
            return True
        existing_rows ^= least_significant
    return False
