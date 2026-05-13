from __future__ import annotations

from dataclasses import dataclass

import torch

from matterformer.geometry.adapters import GeometryFeatures


@dataclass
class GeometryCache:
    features: GeometryFeatures | None
    coords_len: int
    seq_len: int
    neighbor_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    rel: torch.Tensor
    dist: torch.Tensor
    unit: torch.Tensor
    rbf: torch.Tensor
    pair_mask: torch.Tensor


@dataclass
class FlatGeometryCache:
    features: GeometryFeatures | None
    coords_len: int
    seq_len: int
    num_graphs: int
    cu_seqlens: torch.Tensor
    batch_index: torch.Tensor
    neighbor_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    rel: torch.Tensor
    dist: torch.Tensor
    unit: torch.Tensor
    rbf: torch.Tensor

    @property
    def k_neighbors(self) -> int:
        return int(self.neighbor_idx.shape[-1])

    def as_single_batch_geometry_cache(self) -> GeometryCache:
        pair_mask = self.neighbor_mask[:, :, None] & self.neighbor_mask[:, None, :]
        return GeometryCache(
            features=None,
            coords_len=self.coords_len,
            seq_len=self.seq_len,
            neighbor_idx=self.neighbor_idx.unsqueeze(0).contiguous(),
            neighbor_mask=self.neighbor_mask.unsqueeze(0).contiguous(),
            rel=self.rel.unsqueeze(0).contiguous(),
            dist=self.dist.unsqueeze(0).contiguous(),
            unit=self.unit.unsqueeze(0).contiguous(),
            rbf=self.rbf.unsqueeze(0).contiguous(),
            pair_mask=pair_mask.unsqueeze(0).contiguous(),
        )


def flatten_padded_geometry_cache(
    geom: GeometryCache,
    *,
    valid: torch.Tensor,
    batch_index: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> FlatGeometryCache:
    if valid.ndim != 2:
        raise ValueError(f"valid must have shape [B, Nmax], got {tuple(valid.shape)}")
    if geom.neighbor_idx.ndim != 3:
        raise ValueError(f"geom.neighbor_idx must have shape [B, Nmax, K], got {tuple(geom.neighbor_idx.shape)}")
    if geom.neighbor_idx.shape[:2] != valid.shape:
        raise ValueError(
            f"geom.neighbor_idx shape {tuple(geom.neighbor_idx.shape[:2])} does not match valid {tuple(valid.shape)}"
        )

    valid = valid.bool()
    batch_size, num_slots = valid.shape
    total_atoms = int(valid.sum().item())
    device = valid.device

    flat_of_slot = torch.zeros((batch_size, num_slots), device=device, dtype=torch.long)
    flat_of_slot[valid] = torch.arange(total_atoms, device=device, dtype=torch.long)

    local_idx = geom.neighbor_idx.to(device=device, dtype=torch.long)
    flat_neighbor = torch.gather(
        flat_of_slot,
        dim=1,
        index=local_idx.reshape(batch_size, -1),
    ).reshape_as(local_idx)
    flat_mask = geom.neighbor_mask.to(device=device, dtype=torch.bool) & valid[:, :, None]
    flat_neighbor = flat_neighbor.masked_fill(~flat_mask, 0)

    return FlatGeometryCache(
        features=None,
        coords_len=total_atoms,
        seq_len=total_atoms,
        num_graphs=batch_size,
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32).contiguous(),
        batch_index=batch_index.to(device=device).contiguous(),
        neighbor_idx=flat_neighbor[valid].contiguous(),
        neighbor_mask=flat_mask[valid].contiguous(),
        rel=geom.rel[valid].contiguous(),
        dist=geom.dist[valid].contiguous(),
        unit=geom.unit[valid].contiguous(),
        rbf=geom.rbf[valid].contiguous(),
    )
