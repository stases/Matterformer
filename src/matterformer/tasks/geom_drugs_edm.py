from __future__ import annotations

import torch

from matterformer.data.geom_drugs import (
    GEOM_DRUGS_ATOM_PAD_TOKEN,
    GEOM_DRUGS_NUM_ATOM_TYPES,
    GEOM_DRUGS_NUM_CHARGE_TYPES,
    GEOM_DRUGS_INDEX_TO_CHARGE,
)
from matterformer.tasks.edm import (
    EDMPreconditioner,
    _masked_sample_mse,
    recenter_coordinates,
)


def split_geom_drugs_node_features(
    node_features: torch.Tensor,
    *,
    atom_type_channels: int = GEOM_DRUGS_NUM_ATOM_TYPES,
    charge_channels: int = GEOM_DRUGS_NUM_CHARGE_TYPES,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_features.shape[-1] != atom_type_channels + charge_channels:
        raise ValueError(
            "GEOM-Drugs node features must concatenate atom-type and charge channels"
        )
    atom_logits = node_features[..., :atom_type_channels]
    charge_logits = node_features[..., atom_type_channels : atom_type_channels + charge_channels]
    return atom_logits, charge_logits


def decode_geom_drugs_types_and_charges(
    node_features: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    atom_type_channels: int = GEOM_DRUGS_NUM_ATOM_TYPES,
    charge_channels: int = GEOM_DRUGS_NUM_CHARGE_TYPES,
) -> tuple[torch.Tensor, torch.Tensor]:
    atom_logits, charge_logits = split_geom_drugs_node_features(
        node_features,
        atom_type_channels=atom_type_channels,
        charge_channels=charge_channels,
    )
    atom_types = atom_logits.argmax(dim=-1)
    atom_types = atom_types.masked_fill(pad_mask, GEOM_DRUGS_ATOM_PAD_TOKEN)

    charge_indices = charge_logits.argmax(dim=-1)
    charge_indices = charge_indices.masked_fill(pad_mask, 0)
    charge_values = torch.empty_like(charge_indices, dtype=torch.long)
    for idx, charge in GEOM_DRUGS_INDEX_TO_CHARGE.items():
        charge_values[charge_indices == idx] = int(charge)
    charge_values = charge_values.masked_fill(pad_mask, 0)
    return atom_types, charge_values


class GeomDrugsEDMLoss:
    def __init__(
        self,
        *,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        node_feature_scale: float = 4.0,
    ) -> None:
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)
        self.node_feature_scale = float(node_feature_scale)

    def __call__(self, net: EDMPreconditioner, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        node_clean = batch.node_features().float() / self.node_feature_scale
        coords_clean = recenter_coordinates(batch.coords.float(), batch.pad_mask)

        sigma = torch.exp(
            torch.randn(node_clean.shape[0], device=node_clean.device, dtype=node_clean.dtype) * self.p_std
            + self.p_mean
        )
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()

        node_noisy = node_clean + torch.randn_like(node_clean) * sigma[:, None, None]
        coord_noise = torch.randn_like(coords_clean) * sigma[:, None, None]
        coord_noise = recenter_coordinates(coord_noise, batch.pad_mask)
        coords_noisy = recenter_coordinates(coords_clean + coord_noise, batch.pad_mask)

        node_denoised, coords_denoised = net(
            node_noisy,
            coords_noisy,
            batch.pad_mask,
            sigma,
            lattice=batch.lattice,
        )
        node_loss = _masked_sample_mse(node_denoised, node_clean, batch.pad_mask)
        coord_loss = _masked_sample_mse(coords_denoised, coords_clean, batch.pad_mask)
        loss = (weight * node_loss).mean() + (weight * coord_loss).mean()
        log_sigma_over_4 = torch.log(sigma.clamp_min(1e-8)) / 4.0
        return loss, {
            "sigma": sigma,
            "log_sigma_over_4": log_sigma_over_4,
            "node_loss": node_loss,
            "coord_loss": coord_loss,
        }
