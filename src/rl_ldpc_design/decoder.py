from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class DecoderResult:
    ber: float
    fer: float
    evaluated_bits_per_frame: int
    bit_errors: int
    frames: int
    frame_errors: int


def get_node_neighbors(
    h: torch.Tensor,
    edge_count: int,
    check_max_degree: int,
    device: torch.device,
) -> tuple[list[list[int]], list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = h.shape
    edges = torch.nonzero(h == 1, as_tuple=False)
    if edges.shape[0] != edge_count:
        raise ValueError("edge count mismatch")

    edge_to_check = edges[:, 0].to(dtype=torch.int64, device=device)
    edge_to_variable = edges[:, 1].to(dtype=torch.int64, device=device)

    edge_index = np.arange(edge_count, dtype=np.int64)
    row_index = edges[:, 0].detach().cpu().numpy()
    col_index = edges[:, 1].detach().cpu().numpy()

    row_order = np.argsort(row_index, kind="stable")
    row_counts = np.bincount(row_index, minlength=rows)
    row_splits = np.split(edge_index[row_order], np.cumsum(row_counts)[:-1])
    check_to_edge = [array.tolist() for array in row_splits]

    col_order = np.argsort(col_index, kind="stable")
    col_counts = np.bincount(col_index, minlength=cols)
    col_splits = np.split(edge_index[col_order], np.cumsum(col_counts)[:-1])
    variable_to_edge = [array.tolist() for array in col_splits]

    ext_cols = max(check_max_degree - 1, 0)
    edge_to_external = -torch.ones(
        (edge_count, ext_cols), dtype=torch.int64, device=device
    )
    for check in range(rows):
        check_edges = check_to_edge[check]
        degree = len(check_edges)
        if degree <= 1:
            continue
        edges_tensor = torch.tensor(check_edges, dtype=torch.int64, device=device)
        all_neighbors = edges_tensor.repeat(degree, 1)
        mask = ~torch.eye(degree, dtype=torch.bool, device=device)
        external = all_neighbors[mask].view(degree, degree - 1)
        edge_to_external[edges_tensor, : degree - 1] = external
    return (
        check_to_edge,
        variable_to_edge,
        edge_to_variable,
        edge_to_check,
        edge_to_external,
    )


def get_node_info(
    h: torch.Tensor, device: torch.device
) -> tuple[list[list[int]], list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor, int]:
    check_max_degree = int(torch.max(torch.sum(h == 1, dim=1)).item())
    edge_count = int(torch.sum(h == 1).item())
    result = get_node_neighbors(h, edge_count, check_max_degree, device)
    return (*result, edge_count)


def create_llr(
    num_llrs: int,
    code_rate: float,
    snr_db: float,
    code_length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1.0 / (2.0 * code_rate * snr_linear)
    noise_std = np.sqrt(noise_variance)
    if out is None:
        llrs = torch.empty(
            (num_llrs, code_length), dtype=torch.float32, device=device
        )
    else:
        if out.shape != (num_llrs, code_length):
            raise ValueError("out tensor shape mismatch for create_llr")
        llrs = out
    llrs.normal_(mean=0.0, std=float(noise_std), generator=generator)
    llrs.add_(1.0).mul_(2.0 / noise_variance)
    return llrs


class FloodingDecoder:
    """Flooding-schedule sum-product decoder for an incrementally built H matrix."""

    def __init__(
        self,
        rows: int,
        cols: int,
        iterations: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.m = int(rows)
        self.n = int(cols)
        self.iterations = int(iterations)
        self.device = torch.device(device)
        self.code_rate = 1.0 - self.m / self.n
        self._m_c2v_buf: torch.Tensor | None = None
        self._m_v2c_buf: torch.Tensor | None = None
        self._sum_buf: torch.Tensor | None = None
        self._decode_cache_key: tuple[int, int, int] | None = None
        self._graph_ready = False
        self.num_edges = 0

    def _ensure_decode_buffers(self, batch_size: int, llr_dim: int) -> None:
        key = (batch_size, self.num_edges, llr_dim)
        if self._decode_cache_key == key:
            return
        self._m_c2v_buf = torch.empty(
            (batch_size, self.num_edges), dtype=torch.float32, device=self.device
        )
        self._m_v2c_buf = torch.empty(
            (batch_size, self.num_edges), dtype=torch.float32, device=self.device
        )
        self._sum_buf = torch.empty(
            (batch_size, llr_dim), dtype=torch.float32, device=self.device
        )
        self._decode_cache_key = key

    def set_graph(self, h: torch.Tensor) -> None:
        h = h.to(device=self.device)
        (
            self.check_to_edge,
            self.variable_to_edge,
            self.edge_to_variable,
            self.edge_to_check,
            self.edge_to_external,
            self.num_edges,
        ) = get_node_info(h, self.device)
        self._graph_ready = True
        self._decode_cache_key = None

    def add_edge(self, row_idx: int, col_idx: int) -> bool:
        if not self._graph_ready:
            raise RuntimeError("graph is not initialized")
        for edge in self.variable_to_edge[col_idx]:
            if int(self.edge_to_check[edge].item()) == row_idx:
                return False

        new_edge = self.num_edges
        self.num_edges += 1
        self.check_to_edge[row_idx].append(new_edge)
        self.variable_to_edge[col_idx].append(new_edge)

        new_check = torch.tensor([row_idx], dtype=torch.int64, device=self.device)
        new_variable = torch.tensor([col_idx], dtype=torch.int64, device=self.device)
        self.edge_to_check = torch.cat((self.edge_to_check, new_check), dim=0)
        self.edge_to_variable = torch.cat((self.edge_to_variable, new_variable), dim=0)

        current_cols = 0 if self.edge_to_external.numel() == 0 else self.edge_to_external.shape[1]
        new_degree = len(self.check_to_edge[row_idx])
        required_cols = max(new_degree - 1, 0)
        if required_cols > current_cols:
            padding = -torch.ones(
                (self.edge_to_external.shape[0], required_cols - current_cols),
                dtype=torch.int64,
                device=self.device,
            )
            self.edge_to_external = torch.cat((self.edge_to_external, padding), dim=1)

        new_row = -torch.ones(
            (1, self.edge_to_external.shape[1]),
            dtype=torch.int64,
            device=self.device,
        )
        self.edge_to_external = torch.cat((self.edge_to_external, new_row), dim=0)

        if new_degree > 1:
            row_edges = self.check_to_edge[row_idx]
            edges_tensor = torch.tensor(row_edges, dtype=torch.int64, device=self.device)
            all_neighbors = edges_tensor.repeat(new_degree, 1)
            mask = ~torch.eye(new_degree, dtype=torch.bool, device=self.device)
            external = all_neighbors[mask].view(new_degree, new_degree - 1)
            self.edge_to_external[edges_tensor, : new_degree - 1] = external
            if self.edge_to_external.shape[1] > new_degree - 1:
                self.edge_to_external[edges_tensor, new_degree - 1 :] = -1
        self._decode_cache_key = None
        return True

    def _ensure_graph(self, h: torch.Tensor | None = None) -> None:
        if h is not None:
            self.set_graph(h)
        elif not self._graph_ready:
            raise RuntimeError("graph is not initialized")

    def reset_matrix(self) -> torch.Tensor:
        return torch.zeros((self.m, self.n), dtype=torch.float32, device=self.device)

    @staticmethod
    def toggle_entry(state: torch.Tensor, flat_index: int) -> torch.Tensor:
        next_state = state.clone()
        flattened = next_state.view(-1)
        flattened[flat_index] = 1 - flattened[flat_index]
        return flattened.view_as(state)

    def _check_to_variable(self, variable_messages: torch.Tensor) -> torch.Tensor:
        tanh_messages = torch.tanh(0.5 * variable_messages)
        values = torch.where(
            (self.edge_to_external < 0).unsqueeze(0),
            tanh_messages.new_tensor(1.0),
            tanh_messages[:, self.edge_to_external],
        )
        products = values.prod(dim=2).clamp(-0.999999, 0.999999)
        output = 2 * torch.atanh(products)
        return output.clamp(-20, 20)

    def _variable_to_check(
        self,
        check_messages: torch.Tensor,
        llrs: torch.Tensor,
        work_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        totals = torch.zeros_like(llrs) if work_buffer is None else work_buffer.zero_()
        totals.index_add_(1, self.edge_to_variable, check_messages)
        gathered = (llrs + totals).gather(
            1, self.edge_to_variable.unsqueeze(0).expand(llrs.shape[0], -1)
        )
        return (gathered - check_messages).clamp(-20, 20)

    def decode(self, llrs: torch.Tensor) -> torch.Tensor:
        self._ensure_graph()
        batch_size = llrs.shape[0]
        self._ensure_decode_buffers(batch_size, llrs.shape[1])
        assert self._m_c2v_buf is not None
        assert self._m_v2c_buf is not None
        assert self._sum_buf is not None

        check_messages = self._m_c2v_buf
        variable_messages = self._m_v2c_buf
        check_messages.zero_()
        variable_messages.copy_(
            llrs.gather(
                1, self.edge_to_variable.unsqueeze(0).expand(batch_size, -1)
            )
        )
        for _ in range(self.iterations):
            check_messages = self._check_to_variable(variable_messages)
            variable_messages = self._variable_to_check(
                check_messages, llrs, work_buffer=self._sum_buf
            )
        self._sum_buf.zero_().index_add_(1, self.edge_to_variable, check_messages)
        return llrs + self._sum_buf

    def evaluate(
        self,
        h: torch.Tensor | None = None,
        *,
        systematic: bool = False,
        snr_db: float,
        num_llrs: int,
        max_frame_errors: int,
        seed: int,
    ) -> DecoderResult:
        self._ensure_graph(h)
        if num_llrs <= 0:
            raise ValueError("num_llrs must be positive")
        if max_frame_errors <= 0:
            raise ValueError("max_frame_errors must be positive")

        information_bits = self.n - self.m
        evaluated_bits = information_bits if systematic else self.n
        if systematic and information_bits <= 0:
            raise ValueError("systematic mode requires n > m")

        frames = bit_errors = frame_errors = 0
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        llr_buffer = torch.empty(
            (num_llrs, self.n), dtype=torch.float32, device=self.device
        )
        while frame_errors < max_frame_errors:
            llrs = create_llr(
                num_llrs,
                self.code_rate,
                snr_db,
                self.n,
                self.device,
                generator=generator,
                out=llr_buffer,
            )
            estimates = (self.decode(llrs) <= 0).to(torch.int32)
            per_frame = (
                estimates[:, :information_bits].sum(dim=1)
                if systematic
                else estimates.sum(dim=1)
            )
            frame_flags = (per_frame > 0).to(torch.int32)
            bit_errors += int(per_frame.sum().item())
            frame_errors += int(frame_flags.sum().item())
            frames += num_llrs

        return DecoderResult(
            ber=bit_errors / (frames * evaluated_bits),
            fer=frame_errors / frames,
            evaluated_bits_per_frame=evaluated_bits,
            bit_errors=bit_errors,
            frames=frames,
            frame_errors=frame_errors,
        )

    def evaluate_fixed_llrs(
        self,
        llrs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        chunk_size: int | None = None,
        systematic: bool = False,
    ) -> DecoderResult:
        self._ensure_graph(h)
        information_bits = self.n - self.m
        evaluated_bits = information_bits if systematic else self.n
        if systematic and information_bits <= 0:
            raise ValueError("systematic mode requires n > m")

        total = llrs.shape[0]
        if total <= 0:
            raise ValueError("llrs must contain at least one frame")
        if chunk_size is None or chunk_size >= total:
            chunk_size = total
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        frames = bit_errors = frame_errors = 0
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            estimates = (self.decode(llrs[start:end]) <= 0).to(torch.int32)
            per_frame = (
                estimates[:, :information_bits].sum(dim=1)
                if systematic
                else estimates.sum(dim=1)
            )
            frame_flags = (per_frame > 0).to(torch.int32)
            bit_errors += int(per_frame.sum().item())
            frame_errors += int(frame_flags.sum().item())
            frames += end - start

        return DecoderResult(
            ber=bit_errors / (frames * evaluated_bits),
            fer=frame_errors / frames,
            evaluated_bits_per_frame=evaluated_bits,
            bit_errors=bit_errors,
            frames=frames,
            frame_errors=frame_errors,
        )
