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
