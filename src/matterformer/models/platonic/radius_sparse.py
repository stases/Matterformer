from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RadiusBlockSparseLayout:
    """Block-sparse radius-attention layout for flat, varlen token batches.

    `perm` is a gather index such that `x_sorted = x[perm]`.
    `inv_perm` maps original absolute token indices to sorted absolute indices.
    """

    q_block_start: torch.Tensor
    q_block_end: torch.Tensor
    k_block_start: torch.Tensor
    k_block_end: torch.Tensor
    block_ptr: torch.Tensor
    block_col: torch.Tensor
    block_ptr_t: torch.Tensor
    block_col_t: torch.Tensor
    perm: torch.Tensor
    inv_perm: torch.Tensor
    cu_seqlens: torch.Tensor
    block_m: int
    block_n: int
    cutoff: float
    include_self: bool
    dense_pair_count: int
    radius_edge_count: int
    max_block_row_length: int

    @property
    def num_q_blocks(self) -> int:
        return int(self.q_block_start.numel())

    @property
    def num_k_blocks(self) -> int:
        return int(self.k_block_start.numel())

    @property
    def num_live_block_pairs(self) -> int:
        return int(self.block_col.numel())

    @property
    def effective_tile_density(self) -> float:
        if self.dense_pair_count <= 0:
            return 0.0
        tile_pairs = self.num_live_block_pairs * int(self.block_m) * int(self.block_n)
        return float(tile_pairs / self.dense_pair_count)

    @property
    def true_edge_density(self) -> float:
        if self.dense_pair_count <= 0:
            return 0.0
        return float(self.radius_edge_count / self.dense_pair_count)

    @property
    def mean_radius_degree(self) -> float:
        if self.perm.numel() == 0:
            return 0.0
        return float(self.radius_edge_count / int(self.perm.numel()))

    def to(self, device: torch.device | str) -> "RadiusBlockSparseLayout":
        return RadiusBlockSparseLayout(
            q_block_start=self.q_block_start.to(device),
            q_block_end=self.q_block_end.to(device),
            k_block_start=self.k_block_start.to(device),
            k_block_end=self.k_block_end.to(device),
            block_ptr=self.block_ptr.to(device),
            block_col=self.block_col.to(device),
            block_ptr_t=self.block_ptr_t.to(device),
            block_col_t=self.block_col_t.to(device),
            perm=self.perm.to(device),
            inv_perm=self.inv_perm.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            block_m=self.block_m,
            block_n=self.block_n,
            cutoff=self.cutoff,
            include_self=self.include_self,
            dense_pair_count=self.dense_pair_count,
            radius_edge_count=self.radius_edge_count,
            max_block_row_length=self.max_block_row_length,
        )


def _cell_key(coord: torch.Tensor, cutoff: float) -> tuple[int, int, int]:
    cell = torch.floor(coord / float(cutoff)).to(dtype=torch.int64)
    return int(cell[0].item()), int(cell[1].item()), int(cell[2].item())


def _neighbor_cells(cell: tuple[int, int, int]):
    cx, cy, cz = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield cx + dx, cy + dy, cz + dz


def _segment_sort_indices(pos_cpu: torch.Tensor, *, cutoff: float, sort: str) -> list[int]:
    if sort in {"none", "off", "false", "0"}:
        return list(range(int(pos_cpu.shape[0])))
    if sort not in {"cell", "lexicographic_cell"}:
        raise ValueError("radius sparse layout sort must be one of {'cell', 'none'}")
    keyed = []
    for idx in range(int(pos_cpu.shape[0])):
        keyed.append((*_cell_key(pos_cpu[idx], cutoff), idx))
    keyed.sort()
    return [idx for *_cell, idx in keyed]


def _prefix_from_lengths(lengths: list[int]) -> list[int]:
    out = [0]
    total = 0
    for length in lengths:
        total += int(length)
        out.append(total)
    return out


def build_radius_block_sparse_layout(
    pos: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    cutoff: float,
    block_m: int = 16,
    block_n: int = 32,
    sort: str = "cell",
    include_self: bool = True,
) -> RadiusBlockSparseLayout:
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be 1D")
    if float(cutoff) <= 0.0:
        raise ValueError("cutoff must be positive")
    if int(block_m) <= 0 or int(block_n) <= 0:
        raise ValueError("block_m and block_n must be positive")
    total_tokens = int(pos.shape[0])
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError("cu_seqlens[-1] must equal pos.shape[0]")

    device = pos.device
    cu_cpu = cu_seqlens.detach().cpu().to(dtype=torch.int64)
    pos_cpu = pos.detach().cpu().to(dtype=torch.float64)
    starts = cu_cpu[:-1].tolist()
    ends = cu_cpu[1:].tolist()

    perm_list: list[int] = []
    q_block_start: list[int] = []
    q_block_end: list[int] = []
    k_block_start: list[int] = []
    k_block_end: list[int] = []
    q_block_bases: list[int] = []
    k_block_bases: list[int] = []
    dense_pair_count = 0

    for start, end in zip(starts, ends):
        start_i = int(start)
        end_i = int(end)
        seg_len = end_i - start_i
        dense_pair_count += seg_len * seg_len
        local_order = _segment_sort_indices(pos_cpu[start_i:end_i], cutoff=float(cutoff), sort=str(sort))
        perm_list.extend(start_i + idx for idx in local_order)

    perm = torch.tensor(perm_list, dtype=torch.long)
    inv_perm = torch.empty(total_tokens, dtype=torch.long)
    inv_perm[perm] = torch.arange(total_tokens, dtype=torch.long)
    sorted_pos_cpu = pos_cpu[perm]

    for start, end in zip(starts, ends):
        start_i = int(start)
        end_i = int(end)
        q_block_bases.append(len(q_block_start))
        for block_start in range(start_i, end_i, int(block_m)):
            q_block_start.append(block_start)
            q_block_end.append(min(block_start + int(block_m), end_i))
        k_block_bases.append(len(k_block_start))
        for block_start in range(start_i, end_i, int(block_n)):
            k_block_start.append(block_start)
            k_block_end.append(min(block_start + int(block_n), end_i))

    live_pairs: set[tuple[int, int]] = set()
    cutoff2 = float(cutoff) * float(cutoff)
    radius_edge_count = 0
    for seg_idx, (start, end) in enumerate(zip(starts, ends)):
        start_i = int(start)
        end_i = int(end)
        seg_len = end_i - start_i
        if seg_len <= 0:
            continue
        cells: dict[tuple[int, int, int], list[int]] = {}
        for local_idx in range(seg_len):
            key = _cell_key(sorted_pos_cpu[start_i + local_idx], cutoff=float(cutoff))
            cells.setdefault(key, []).append(local_idx)
        for local_i in range(seg_len):
            abs_i = start_i + local_i
            cell = _cell_key(sorted_pos_cpu[abs_i], cutoff=float(cutoff))
            for neighbor_cell in _neighbor_cells(cell):
                for local_j in cells.get(neighbor_cell, ()):
                    if not include_self and local_i == local_j:
                        continue
                    abs_j = start_i + local_j
                    if local_i == local_j:
                        keep = bool(include_self)
                    else:
                        delta = sorted_pos_cpu[abs_i] - sorted_pos_cpu[abs_j]
                        keep = float(delta.dot(delta).item()) < cutoff2
                    if not keep:
                        continue
                    radius_edge_count += 1
                    q_block = q_block_bases[seg_idx] + local_i // int(block_m)
                    k_block = k_block_bases[seg_idx] + local_j // int(block_n)
                    live_pairs.add((q_block, k_block))

    block_rows: list[list[int]] = [[] for _ in range(len(q_block_start))]
    block_rows_t: list[list[int]] = [[] for _ in range(len(k_block_start))]
    for q_block, k_block in sorted(live_pairs):
        block_rows[q_block].append(k_block)
        block_rows_t[k_block].append(q_block)

    block_lengths = [len(row) for row in block_rows]
    block_t_lengths = [len(row) for row in block_rows_t]
    max_block_row_length = max(block_lengths, default=0)
    block_ptr = torch.tensor(_prefix_from_lengths(block_lengths), dtype=torch.int32)
    block_ptr_t = torch.tensor(_prefix_from_lengths(block_t_lengths), dtype=torch.int32)
    block_col = torch.tensor([col for row in block_rows for col in row], dtype=torch.int32)
    block_col_t = torch.tensor([col for row in block_rows_t for col in row], dtype=torch.int32)

    return RadiusBlockSparseLayout(
        q_block_start=torch.tensor(q_block_start, dtype=torch.int32, device=device),
        q_block_end=torch.tensor(q_block_end, dtype=torch.int32, device=device),
        k_block_start=torch.tensor(k_block_start, dtype=torch.int32, device=device),
        k_block_end=torch.tensor(k_block_end, dtype=torch.int32, device=device),
        block_ptr=block_ptr.to(device=device),
        block_col=block_col.to(device=device),
        block_ptr_t=block_ptr_t.to(device=device),
        block_col_t=block_col_t.to(device=device),
        perm=perm.to(device=device),
        inv_perm=inv_perm.to(device=device),
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        block_m=int(block_m),
        block_n=int(block_n),
        cutoff=float(cutoff),
        include_self=bool(include_self),
        dense_pair_count=int(dense_pair_count),
        radius_edge_count=int(radius_edge_count),
        max_block_row_length=int(max_block_row_length),
    )
