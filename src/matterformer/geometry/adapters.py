from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import torch
from torch import nn

from matterformer.geometry.lattice import lattice_latent_to_gram, lattice_latent_to_y1


@dataclass
class GeometryFeatures:
    pair_delta: torch.Tensor
    pair_dist: torch.Tensor
    pair_dist2: torch.Tensor
    pair_dist_norm: torch.Tensor
    pair_mask: torch.Tensor
    global_geom: torch.Tensor
    kind: str
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


class BaseGeometryAdapter(nn.Module):
    geometry_kind: str = "base"

    @staticmethod
    def _valid_atom_mask(
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_atoms = coords.shape[:2]
        if pad_mask is None:
            return torch.ones(
                batch_size,
                num_atoms,
                device=coords.device,
                dtype=torch.bool,
            )
        if pad_mask.shape != (batch_size, num_atoms):
            raise ValueError(
                f"pad_mask must have shape {(batch_size, num_atoms)}, got {tuple(pad_mask.shape)}"
            )
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        return ~pad_mask

    @staticmethod
    def _pair_mask_from_valid(valid_atom_mask: torch.Tensor) -> torch.Tensor:
        return valid_atom_mask[:, :, None] & valid_atom_mask[:, None, :]

    @staticmethod
    def _apply_pair_mask(
        pair_tensor: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        while pair_mask.ndim < pair_tensor.ndim:
            pair_mask = pair_mask.unsqueeze(-1)
        return torch.where(pair_mask, pair_tensor, torch.zeros_like(pair_tensor))

    @staticmethod
    def _safe_mean(
        value: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        masked = value * mask.to(dtype=value.dtype)
        denom = mask.sum(dim=dim).clamp_min(1)
        return masked.sum(dim=dim) / denom

    def forward(
        self,
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
    ) -> GeometryFeatures:
        raise NotImplementedError


class NonPeriodicGeometryAdapter(BaseGeometryAdapter):
    geometry_kind = "nonperiodic"

    def forward(
        self,
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
    ) -> GeometryFeatures:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (B, N, 3), got {tuple(coords.shape)}")
        if lattice is not None:
            raise ValueError("lattice must be None for NonPeriodicGeometryAdapter")

        coords = coords.float()
        valid_atom_mask = self._valid_atom_mask(coords, pad_mask)
        pair_mask = self._pair_mask_from_valid(valid_atom_mask)

        pair_delta = coords[:, :, None, :] - coords[:, None, :, :]
        pair_dist2 = pair_delta.square().sum(dim=-1).clamp_min(0.0)
        pair_dist = torch.sqrt(pair_dist2)

        sq_norm = coords.square().sum(dim=-1)
        rms_radius = torch.sqrt(
            self._safe_mean(sq_norm, valid_atom_mask, dim=1).clamp_min(1e-8)
        )
        pair_dist_norm = pair_dist / rms_radius[:, None, None].clamp_min(1e-8)

        pair_delta = self._apply_pair_mask(pair_delta, pair_mask)
        pair_dist = self._apply_pair_mask(pair_dist, pair_mask)
        pair_dist2 = self._apply_pair_mask(pair_dist2, pair_mask)
        pair_dist_norm = self._apply_pair_mask(pair_dist_norm, pair_mask)

        metric = torch.eye(3, device=coords.device, dtype=coords.dtype).expand(
            coords.shape[0], -1, -1
        )
        return GeometryFeatures(
            pair_delta=pair_delta,
            pair_dist=pair_dist,
            pair_dist2=pair_dist2,
            pair_dist_norm=pair_dist_norm,
            pair_mask=pair_mask,
            global_geom=rms_radius[:, None],
            kind=self.geometry_kind,
            extras={"metric": metric},
        )


class PeriodicGeometryAdapter(BaseGeometryAdapter):
    geometry_kind = "periodic"

    def __init__(
        self,
        pbc_radius: int = 1,
        lattice_repr: str = "y1",
    ) -> None:
        super().__init__()
        if pbc_radius not in {1, 2}:
            raise ValueError(f"pbc_radius must be 1 or 2, got {pbc_radius}")
        self.pbc_radius = int(pbc_radius)
        self.lattice_repr = lattice_repr.lower()
        offsets = torch.tensor(
            list(product(range(-self.pbc_radius, self.pbc_radius + 1), repeat=3)),
            dtype=torch.float32,
        )
        self.register_buffer(
            "_pbc_offsets",
            offsets.view(1, 1, 1, offsets.shape[0], 3),
            persistent=False,
        )

    def forward(
        self,
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
    ) -> GeometryFeatures:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (B, N, 3), got {tuple(coords.shape)}")
        if lattice is None:
            raise ValueError("lattice is required for PeriodicGeometryAdapter")
        if lattice.ndim != 2 or lattice.shape[-1] != 6:
            raise ValueError(f"lattice must have shape (B, 6), got {tuple(lattice.shape)}")
        if lattice.shape[0] != coords.shape[0]:
            raise ValueError(
                f"lattice batch {lattice.shape[0]} does not match coords batch {coords.shape[0]}"
            )

        coords = coords.float()
        lattice = lattice.float()
        valid_atom_mask = self._valid_atom_mask(coords, pad_mask)
        pair_mask = self._pair_mask_from_valid(valid_atom_mask)

        lattice_y1 = lattice_latent_to_y1(lattice, lattice_repr=self.lattice_repr)
        gram = lattice_latent_to_gram(lattice, lattice_repr=self.lattice_repr)

        pair_delta = coords[:, :, None, :] - coords[:, None, :, :]
        offsets = self._pbc_offsets.to(device=coords.device, dtype=coords.dtype)
        pair_delta_images = pair_delta[:, :, :, None, :] + offsets
        pair_dist2_images = torch.einsum(
            "bnmki,bij,bnmkj->bnmk",
            pair_delta_images,
            gram.to(dtype=coords.dtype),
            pair_delta_images,
        )
        min_idx = pair_dist2_images.argmin(dim=-1, keepdim=True)
        pair_dist2 = torch.gather(pair_dist2_images, dim=-1, index=min_idx).squeeze(-1)
        pair_dist2 = pair_dist2.clamp_min(0.0)
        gather_idx = min_idx[..., None].expand(-1, -1, -1, 1, 3)
        pair_delta = torch.gather(pair_delta_images, dim=3, index=gather_idx).squeeze(3)
        pair_dist = torch.sqrt(pair_dist2)

        cell_scale = torch.exp(lattice_y1[:, :3]).mean(dim=-1).clamp_min(1e-8)
        pair_dist_norm = pair_dist / cell_scale[:, None, None]

        pair_delta = self._apply_pair_mask(pair_delta, pair_mask)
        pair_dist = self._apply_pair_mask(pair_dist, pair_mask)
        pair_dist2 = self._apply_pair_mask(pair_dist2, pair_mask)
        pair_dist_norm = self._apply_pair_mask(pair_dist_norm, pair_mask)

        return GeometryFeatures(
            pair_delta=pair_delta,
            pair_dist=pair_dist,
            pair_dist2=pair_dist2,
            pair_dist_norm=pair_dist_norm,
            pair_mask=pair_mask,
            global_geom=lattice_y1,
            kind=self.geometry_kind,
            extras={
                "metric": gram,
                "cell_scale": cell_scale[:, None],
            },
        )
