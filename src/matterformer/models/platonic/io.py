from __future__ import annotations

import torch

from matterformer.models.platonic.groups import PlatonicSolidGroup


def lift_scalars(x: torch.Tensor, group: PlatonicSolidGroup) -> torch.Tensor:
    return x.unsqueeze(-2).expand(*x.shape[:-1], group.G, x.shape[-1])


def readout_scalars(x: torch.Tensor, group: PlatonicSolidGroup | None = None) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("group scalar readout expects at least two dimensions")
    return x.mean(dim=-2)
