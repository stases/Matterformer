from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from matterformer.geometry.adapters import GeometryFeatures, NonPeriodicGeometryAdapter
from matterformer.geometry.cache import GeometryCache

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA nodes with Triton installed.
    triton = None
    tl = None

TRITON_NONPERIODIC_KNN_AVAILABLE = triton is not None and tl is not None


def _next_power_of_2(value: int) -> int:
    return 1 << max(int(value) - 1, 1).bit_length()


def _rbf(dist: torch.Tensor, rbf_dim: int, cutoff: float | None = None) -> torch.Tensor:
    max_dist = float(cutoff) if cutoff is not None else 1.0
    centers = torch.linspace(0.0, max_dist, int(rbf_dim), device=dist.device, dtype=dist.dtype)
    delta = max_dist / max(int(rbf_dim) - 1, 1)
    gamma = 1.0 / max(delta * delta, 1e-6)
    return torch.exp(-gamma * (dist[..., None] - centers.view(*((1,) * dist.ndim), -1)).square())


def _gather_neighbor_values(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_atoms, _, channels = values.shape
    _, _, num_neighbors = neighbor_idx.shape
    idx = neighbor_idx.to(dtype=torch.long)[..., None].expand(batch_size, num_atoms, num_neighbors, channels)
    return torch.gather(values, dim=2, index=idx)


def _dense_nonperiodic_knn_geometry_cache(
    coords: torch.Tensor,
    *,
    pad_mask: torch.Tensor | None,
    k_neighbors: int,
    rbf_dim: int,
    cutoff: float | None,
    seq_len: int,
) -> GeometryCache:
    features = NonPeriodicGeometryAdapter()(coords=coords, pad_mask=pad_mask)
    pair_mask = features.pair_mask.clone()
    coords_len = coords.shape[1]
    eye = torch.eye(coords_len, device=pair_mask.device, dtype=torch.bool).view(1, coords_len, coords_len)
    pair_mask = pair_mask & ~eye
    if pad_mask is not None:
        atom_pad = pad_mask[:, :coords_len].bool()
        pair_mask = pair_mask & ~atom_pad[:, :, None] & ~atom_pad[:, None, :]
    k_eff = min(int(k_neighbors), max(coords_len, 1))
    masked_dist = features.pair_dist.masked_fill(~pair_mask, torch.finfo(features.pair_dist.dtype).max)
    dist, neighbor_idx = torch.topk(masked_dist, k=k_eff, dim=-1, largest=False)
    neighbor_mask = torch.gather(pair_mask, dim=-1, index=neighbor_idx)
    if k_eff < int(k_neighbors):
        pad_k = int(k_neighbors) - k_eff
        neighbor_idx = F.pad(neighbor_idx, (0, pad_k), value=0)
        neighbor_mask = F.pad(neighbor_mask, (0, pad_k), value=False)
        dist = F.pad(dist, (0, pad_k), value=0.0)
    rel = _gather_neighbor_values(-features.pair_delta, neighbor_idx)
    dist = torch.where(neighbor_mask, dist, torch.zeros_like(dist))
    unit = rel / dist.clamp_min(1e-8).unsqueeze(-1)
    unit = torch.where(neighbor_mask[..., None], unit, torch.zeros_like(unit))
    pair_knn_mask = neighbor_mask[:, :, :, None] & neighbor_mask[:, :, None, :]
    return GeometryCache(
        features=features,
        coords_len=coords_len,
        seq_len=seq_len,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        rel=rel,
        dist=dist,
        unit=unit,
        rbf=_rbf(dist, rbf_dim=rbf_dim, cutoff=cutoff),
        pair_mask=pair_knn_mask,
    )


if TRITON_NONPERIODIC_KNN_AVAILABLE:

    @triton.jit
    def _nonperiodic_knn_topk_kernel(
        coords_ptr,
        pad_mask_ptr,
        idx_out_ptr,
        mask_out_ptr,
        dist2_out_ptr,
        num_atoms: tl.constexpr,
        k_neighbors: tl.constexpr,
        HAS_PAD_MASK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_MERGE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // num_atoms
        query_idx = pid - batch_idx * num_atoms

        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        offs_merge = tl.arange(0, BLOCK_MERGE)
        k_mask = offs_k < k_neighbors

        coord_base = (batch_idx * num_atoms + query_idx) * 3
        xi0 = tl.load(coords_ptr + coord_base + 0).to(tl.float32)
        xi1 = tl.load(coords_ptr + coord_base + 1).to(tl.float32)
        xi2 = tl.load(coords_ptr + coord_base + 2).to(tl.float32)

        query_valid = True
        if HAS_PAD_MASK:
            query_valid = tl.load(pad_mask_ptr + batch_idx * num_atoms + query_idx) == 0

        inf = 3.4028234663852886e38
        best_d2 = tl.full((BLOCK_K,), inf, dtype=tl.float32)
        best_idx = tl.full((BLOCK_K,), 0, dtype=tl.int32)

        # Use a Triton loop rather than a Python range.  num_atoms is a constexpr,
        # and Python range can be statically expanded into a huge kernel for large
        # OMol padded batches.  tl.range keeps compile time bounded while still
        # streaming exact all-atom candidates.
        for start in tl.range(0, num_atoms, BLOCK_N):
            cand = start + offs_n
            cand_mask = cand < num_atoms
            valid = cand_mask & (cand != query_idx) & query_valid
            if HAS_PAD_MASK:
                cand_pad = tl.load(pad_mask_ptr + batch_idx * num_atoms + cand, mask=cand_mask, other=1)
                valid = valid & (cand_pad == 0)

            cand_base = (batch_idx * num_atoms + cand) * 3
            xj0 = tl.load(coords_ptr + cand_base + 0, mask=cand_mask, other=0.0).to(tl.float32)
            xj1 = tl.load(coords_ptr + cand_base + 1, mask=cand_mask, other=0.0).to(tl.float32)
            xj2 = tl.load(coords_ptr + cand_base + 2, mask=cand_mask, other=0.0).to(tl.float32)
            d0 = xj0 - xi0
            d1 = xj1 - xi1
            d2v = xj2 - xi2
            block_d2 = tl.where(valid, d0 * d0 + d1 * d1 + d2v * d2v, inf)
            block_idx = cand.to(tl.int32)

            best_match = offs_merge[:, None] == offs_k[None, :]
            best_values = tl.sum(tl.where(best_match, best_d2[None, :], 0.0), axis=1)
            best_indices = tl.sum(tl.where(best_match, best_idx[None, :], 0), axis=1)

            block_pos = BLOCK_K + offs_n
            block_match = offs_merge[:, None] == block_pos[None, :]
            block_values = tl.sum(tl.where(block_match, block_d2[None, :], 0.0), axis=1)
            block_indices = tl.sum(tl.where(block_match, block_idx[None, :], 0), axis=1)

            in_best = offs_merge < BLOCK_K
            in_block = (offs_merge >= BLOCK_K) & (offs_merge < BLOCK_K + BLOCK_N)
            merged_d2 = tl.where(in_best, best_values, tl.where(in_block, block_values, inf))
            merged_idx = tl.where(in_best, best_indices, tl.where(in_block, block_indices, 0))

            new_best_d2 = tl.full((BLOCK_K,), inf, dtype=tl.float32)
            new_best_idx = tl.full((BLOCK_K,), 0, dtype=tl.int32)
            for slot in tl.static_range(0, BLOCK_K):
                selected = tl.min(merged_d2, axis=0)
                selected_pos = tl.min(tl.where(merged_d2 == selected, offs_merge, BLOCK_MERGE), axis=0)
                selected_idx = tl.sum(tl.where(offs_merge == selected_pos, merged_idx, 0), axis=0)
                new_best_d2 = tl.where(offs_k == slot, selected, new_best_d2)
                new_best_idx = tl.where(offs_k == slot, selected_idx, new_best_idx)
                merged_d2 = tl.where(offs_merge == selected_pos, inf, merged_d2)
            best_d2 = new_best_d2
            best_idx = new_best_idx

        out_base = (batch_idx * num_atoms + query_idx) * k_neighbors
        has_neighbor = best_d2 < (inf * 0.5)
        tl.store(idx_out_ptr + out_base + offs_k, tl.where(has_neighbor, best_idx, 0), mask=k_mask)
        tl.store(mask_out_ptr + out_base + offs_k, has_neighbor, mask=k_mask)
        tl.store(dist2_out_ptr + out_base + offs_k, tl.where(has_neighbor, best_d2, 0.0), mask=k_mask)

    @triton.jit
    def _nonperiodic_knn_features_kernel(
        coords_ptr,
        idx_ptr,
        mask_ptr,
        dist2_ptr,
        rel_out_ptr,
        dist_out_ptr,
        unit_out_ptr,
        rbf_out_ptr,
        num_atoms: tl.constexpr,
        k_neighbors: tl.constexpr,
        rbf_dim: tl.constexpr,
        rbf_delta: tl.constexpr,
        rbf_gamma: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // num_atoms
        query_idx = pid - batch_idx * num_atoms

        offs_k = tl.arange(0, BLOCK_K)
        offs_r = tl.arange(0, BLOCK_R)
        k_mask = offs_k < k_neighbors
        r_mask = offs_r < rbf_dim

        n_base = (batch_idx * num_atoms + query_idx) * k_neighbors
        idx = tl.load(idx_ptr + n_base + offs_k, mask=k_mask, other=0)
        valid = (tl.load(mask_ptr + n_base + offs_k, mask=k_mask, other=0) > 0) & k_mask

        qi_base = (batch_idx * num_atoms + query_idx) * 3
        q0 = tl.load(coords_ptr + qi_base + 0).to(tl.float32)
        q1 = tl.load(coords_ptr + qi_base + 1).to(tl.float32)
        q2 = tl.load(coords_ptr + qi_base + 2).to(tl.float32)

        ji_base = (batch_idx * num_atoms + idx) * 3
        j0 = tl.load(coords_ptr + ji_base + 0, mask=valid, other=0.0).to(tl.float32)
        j1 = tl.load(coords_ptr + ji_base + 1, mask=valid, other=0.0).to(tl.float32)
        j2 = tl.load(coords_ptr + ji_base + 2, mask=valid, other=0.0).to(tl.float32)

        rel0 = tl.where(valid, j0 - q0, 0.0)
        rel1 = tl.where(valid, j1 - q1, 0.0)
        rel2 = tl.where(valid, j2 - q2, 0.0)
        dist2 = tl.load(dist2_ptr + n_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        dist = tl.where(valid, tl.sqrt(dist2), 0.0)
        inv_dist = 1.0 / tl.maximum(dist, 1.0e-8)
        unit0 = tl.where(valid, rel0 * inv_dist, 0.0)
        unit1 = tl.where(valid, rel1 * inv_dist, 0.0)
        unit2 = tl.where(valid, rel2 * inv_dist, 0.0)

        rel_base = ((batch_idx * num_atoms + query_idx) * k_neighbors + offs_k) * 3
        tl.store(rel_out_ptr + rel_base + 0, rel0, mask=k_mask)
        tl.store(rel_out_ptr + rel_base + 1, rel1, mask=k_mask)
        tl.store(rel_out_ptr + rel_base + 2, rel2, mask=k_mask)
        tl.store(dist_out_ptr + n_base + offs_k, dist, mask=k_mask)
        tl.store(unit_out_ptr + rel_base + 0, unit0, mask=k_mask)
        tl.store(unit_out_ptr + rel_base + 1, unit1, mask=k_mask)
        tl.store(unit_out_ptr + rel_base + 2, unit2, mask=k_mask)

        centers = offs_r.to(tl.float32) * rbf_delta
        rbf = tl.exp(-rbf_gamma * (dist[:, None] - centers[None, :]) * (dist[:, None] - centers[None, :]))
        rbf = tl.where(r_mask[None, :], rbf, 0.0)
        rbf_base = ((batch_idx * num_atoms + query_idx) * k_neighbors + offs_k[:, None]) * rbf_dim
        tl.store(rbf_out_ptr + rbf_base + offs_r[None, :], rbf, mask=k_mask[:, None] & r_mask[None, :])


def build_triton_nonperiodic_knn_geometry_cache(
    coords: torch.Tensor,
    *,
    pad_mask: torch.Tensor | None,
    k_neighbors: int,
    rbf_dim: int,
    cutoff: float | None,
    seq_len: int,
    strict: bool = False,
) -> GeometryCache:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, N, 3], got {tuple(coords.shape)}")
    if pad_mask is not None and pad_mask.shape != coords.shape[:2]:
        raise ValueError(f"pad_mask must have shape {tuple(coords.shape[:2])}, got {tuple(pad_mask.shape)}")
    if int(k_neighbors) <= 0:
        raise ValueError("k_neighbors must be positive")
    if int(k_neighbors) > 64:
        raise ValueError("triton_nonperiodic kNN supports k_neighbors <= 64")
    if int(rbf_dim) <= 0:
        raise ValueError("rbf_dim must be positive")
    if int(rbf_dim) > 128:
        raise ValueError("triton_nonperiodic kNN supports rbf_dim <= 128")

    fallback_reason = None
    if not TRITON_NONPERIODIC_KNN_AVAILABLE:
        fallback_reason = "triton is not installed"
    elif coords.device.type != "cuda":
        fallback_reason = "coords are not CUDA tensors"
    elif coords.requires_grad:
        fallback_reason = "compact Triton kNN is forward-only for coordinates"

    if fallback_reason is not None:
        if strict:
            raise RuntimeError(f"triton_nonperiodic kNN geometry is unavailable: {fallback_reason}")
        with record_function("triton_knn/fallback_dense_geometry"):
            return _dense_nonperiodic_knn_geometry_cache(
                coords,
                pad_mask=pad_mask,
                k_neighbors=k_neighbors,
                rbf_dim=rbf_dim,
                cutoff=cutoff,
                seq_len=seq_len,
            )

    coords_f = coords.float().contiguous()
    batch_size, num_atoms = coords_f.shape[:2]
    pad = (
        pad_mask.to(device=coords.device, dtype=torch.bool).contiguous()
        if pad_mask is not None
        else torch.empty((0,), device=coords.device, dtype=torch.bool)
    )
    k_neighbors = int(k_neighbors)
    rbf_dim = int(rbf_dim)
    block_k = _next_power_of_2(k_neighbors)
    block_n = 32
    block_merge = _next_power_of_2(block_k + block_n)
    block_r = _next_power_of_2(rbf_dim)

    neighbor_idx_i32 = torch.empty((batch_size, num_atoms, k_neighbors), device=coords.device, dtype=torch.int32)
    neighbor_mask = torch.empty((batch_size, num_atoms, k_neighbors), device=coords.device, dtype=torch.bool)
    dist2 = torch.empty((batch_size, num_atoms, k_neighbors), device=coords.device, dtype=torch.float32)

    grid = (batch_size * num_atoms,)
    with record_function("triton_knn/topk_kernel"):
        _nonperiodic_knn_topk_kernel[grid](
            coords_f,
            pad,
            neighbor_idx_i32,
            neighbor_mask,
            dist2,
            num_atoms=num_atoms,
            k_neighbors=k_neighbors,
            HAS_PAD_MASK=pad_mask is not None,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_MERGE=block_merge,
            num_warps=4,
            num_stages=1,
        )

    rel = torch.empty((batch_size, num_atoms, k_neighbors, 3), device=coords.device, dtype=torch.float32)
    dist = torch.empty((batch_size, num_atoms, k_neighbors), device=coords.device, dtype=torch.float32)
    unit = torch.empty((batch_size, num_atoms, k_neighbors, 3), device=coords.device, dtype=torch.float32)
    rbf = torch.empty((batch_size, num_atoms, k_neighbors, rbf_dim), device=coords.device, dtype=torch.float32)

    rbf_max = float(cutoff) if cutoff is not None else 1.0
    rbf_delta = rbf_max / max(rbf_dim - 1, 1)
    rbf_gamma = 1.0 / max(rbf_delta * rbf_delta, 1e-6)
    with record_function("triton_knn/features_kernel"):
        _nonperiodic_knn_features_kernel[grid](
            coords_f,
            neighbor_idx_i32,
            neighbor_mask,
            dist2,
            rel,
            dist,
            unit,
            rbf,
            num_atoms=num_atoms,
            k_neighbors=k_neighbors,
            rbf_dim=rbf_dim,
            rbf_delta=float(rbf_delta),
            rbf_gamma=float(rbf_gamma),
            BLOCK_K=block_k,
            BLOCK_R=block_r,
            num_warps=4,
            num_stages=1,
        )
    with record_function("triton_knn/pair_mask_pack"):
        pair_mask = neighbor_mask[:, :, :, None] & neighbor_mask[:, :, None, :]
        neighbor_idx = neighbor_idx_i32.to(dtype=torch.long)
    return GeometryCache(
        features=None,
        coords_len=num_atoms,
        seq_len=int(seq_len),
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        rel=rel,
        dist=dist,
        unit=unit,
        rbf=rbf,
        pair_mask=pair_mask,
    )
