from __future__ import annotations

import numpy as np
import torch
from torch import nn

from matterformer.data.qm9 import QM9_ATOM_PAD_TOKEN, QM9_NUM_ATOM_TYPES, build_pad_mask


def recenter_coordinates(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    valid = (~pad_mask).to(dtype=coords.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
    coords = coords - mean
    return coords.masked_fill(pad_mask[..., None], 0.0)


def _masked_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    valid = (~pad_mask)
    while valid.ndim < prediction.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.to(dtype=prediction.dtype)
    squared_error = (prediction - target).square() * valid
    reduce_dims = tuple(range(1, squared_error.ndim))
    denom = valid.sum(dim=reduce_dims).clamp_min(1.0)
    return squared_error.sum(dim=reduce_dims) / denom


class EDMPreconditioner(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)

    def forward(
        self,
        atom_noisy: torch.Tensor,
        coords_noisy: torch.Tensor,
        pad_mask: torch.Tensor,
        sigma: torch.Tensor,
        *,
        lattice: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.ndim == 2 and sigma.shape[-1] == 1:
            sigma = sigma[:, 0]
        sigma = sigma.float()
        c_skip = self.sigma_data**2 / (sigma.square() + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma.square() + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(self.sigma_data**2 + sigma.square())

        atom_in = atom_noisy * c_in[:, None, None]
        coords_in = recenter_coordinates(coords_noisy * c_in[:, None, None], pad_mask)
        atom_delta, coord_delta = self.model(
            atom_in,
            coords_in,
            pad_mask,
            sigma,
            lattice=lattice,
        )

        atom_clean_scaled = atom_in - atom_delta
        coord_clean_scaled = recenter_coordinates(coords_in - coord_delta, pad_mask)
        atom_denoised = c_skip[:, None, None] * atom_noisy + c_out[:, None, None] * atom_clean_scaled
        coord_denoised = c_skip[:, None, None] * coords_noisy + c_out[:, None, None] * coord_clean_scaled
        atom_denoised = atom_denoised.masked_fill(pad_mask[..., None], 0.0)
        coord_denoised = recenter_coordinates(coord_denoised, pad_mask)
        return atom_denoised, coord_denoised


class EDMLoss:
    def __init__(
        self,
        *,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        atom_feature_scale: float = 4.0,
        charge_feature_scale: float = 8.0,
        use_charges: bool = True,
        max_weight: float | None = 1000.0,
    ) -> None:
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)
        self.atom_feature_scale = float(atom_feature_scale)
        self.charge_feature_scale = float(charge_feature_scale)
        self.use_charges = bool(use_charges)
        self.max_weight = None if max_weight is None else float(max_weight)

    def __call__(self, net: EDMPreconditioner, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        atom_clean = batch.atom_onehot().float() / self.atom_feature_scale
        if self.use_charges:
            charges = batch.formal_charges().to(dtype=atom_clean.dtype)
            charge_clean = charges[..., None] / self.charge_feature_scale
            atom_clean = torch.cat([atom_clean, charge_clean], dim=-1)
        model_channels = int(getattr(net.model, "atom_channels", atom_clean.shape[-1]))
        if atom_clean.shape[-1] != model_channels:
            raise ValueError(
                f"EDMLoss produced {atom_clean.shape[-1]} atom channels, "
                f"but model expects {model_channels}. Set --use-charges consistently."
            )
        coords_clean = recenter_coordinates(batch.coords.float(), batch.pad_mask)

        sigma = torch.exp(
            torch.randn(atom_clean.shape[0], device=atom_clean.device, dtype=atom_clean.dtype) * self.p_std
            + self.p_mean
        )
        raw_weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        weight = raw_weight
        if self.max_weight is not None:
            weight = weight.clamp(max=self.max_weight)

        atom_noisy = atom_clean + torch.randn_like(atom_clean) * sigma[:, None, None]
        coord_noise = torch.randn_like(coords_clean) * sigma[:, None, None]
        coord_noise = recenter_coordinates(coord_noise, batch.pad_mask)
        coords_noisy = recenter_coordinates(coords_clean + coord_noise, batch.pad_mask)

        atom_denoised, coords_denoised = net(
            atom_noisy,
            coords_noisy,
            batch.pad_mask,
            sigma,
            lattice=batch.lattice,
        )
        atom_loss = _masked_sample_mse(atom_denoised, atom_clean, batch.pad_mask)
        coord_loss = _masked_sample_mse(coords_denoised, coords_clean, batch.pad_mask)
        loss = (weight * atom_loss).mean() + (weight * coord_loss).mean()
        log_sigma_over_4 = torch.log(sigma.clamp_min(1e-8)) / 4.0
        return loss, {
            "sigma": sigma,
            "log_sigma_over_4": log_sigma_over_4,
            "atom_loss": atom_loss,
            "coord_loss": coord_loss,
            "loss_weight": weight,
            "loss_weight_clamped": (raw_weight > weight).to(dtype=weight.dtype),
        }


def edm_sampler(
    net: EDMPreconditioner,
    num_atoms: torch.Tensor,
    *,
    atom_channels: int | None = None,
    lattice: torch.Tensor | None = None,
    num_steps: int = 100,
    sigma_min: float = 0.002,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    s_churn: float = 30.0,
    s_min: float = 0.0,
    s_max: float = float("inf"),
    s_noise: float = 1.003,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = net.model.atom_proj.weight.device
    num_atoms = num_atoms.to(device=device, dtype=torch.long)
    pad_mask = build_pad_mask(num_atoms).to(device)
    batch_size, max_atoms = pad_mask.shape
    if atom_channels is None:
        atom_channels = int(getattr(net.model, "atom_channels", QM9_NUM_ATOM_TYPES))

    atom_next = torch.randn(batch_size, max_atoms, atom_channels, device=device)
    atom_next = atom_next.masked_fill(pad_mask[..., None], 0.0)
    coords_next = torch.randn(batch_size, max_atoms, 3, device=device)
    coords_next = recenter_coordinates(coords_next, pad_mask)

    sigma_min = max(float(sigma_min), net.sigma_min)
    sigma_max = min(float(sigma_max), net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    atom_next = atom_next * t_steps[0]
    coords_next = coords_next * t_steps[0]
    coords_next = recenter_coordinates(coords_next, pad_mask)

    for step_index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        atom_cur, coords_cur = atom_next, coords_next
        gamma = min(s_churn / max(num_steps, 1), np.sqrt(2.0) - 1.0) if s_min <= t_cur <= s_max else 0.0
        t_hat = t_cur + gamma * t_cur
        atom_hat = atom_cur + torch.sqrt(t_hat.square() - t_cur.square()) * s_noise * torch.randn_like(atom_cur)
        coords_hat = coords_cur + torch.sqrt(t_hat.square() - t_cur.square()) * s_noise * torch.randn_like(coords_cur)
        atom_hat = atom_hat.masked_fill(pad_mask[..., None], 0.0)
        coords_hat = recenter_coordinates(coords_hat, pad_mask)

        sigma_hat = torch.full((batch_size,), float(t_hat.item()), device=device)
        atom_denoised, coords_denoised = net(
            atom_hat,
            coords_hat,
            pad_mask,
            sigma_hat,
            lattice=lattice,
        )
        atom_d = (atom_hat - atom_denoised) / t_hat
        coords_d = (coords_hat - coords_denoised) / t_hat
        atom_next = atom_hat + (t_next - t_hat) * atom_d
        coords_next = coords_hat + (t_next - t_hat) * coords_d
        coords_next = recenter_coordinates(coords_next, pad_mask)

        if step_index < num_steps - 1:
            sigma_next = torch.full((batch_size,), float(t_next.item()), device=device)
            atom_denoised, coords_denoised = net(
                atom_next,
                coords_next,
                pad_mask,
                sigma_next,
                lattice=lattice,
            )
            atom_d_prime = (atom_next - atom_denoised) / t_next.clamp_min(1e-8)
            coords_d_prime = (coords_next - coords_denoised) / t_next.clamp_min(1e-8)
            atom_next = atom_hat + (t_next - t_hat) * 0.5 * (atom_d + atom_d_prime)
            coords_next = coords_hat + (t_next - t_hat) * 0.5 * (coords_d + coords_d_prime)
            coords_next = recenter_coordinates(coords_next, pad_mask)

        atom_next = atom_next.masked_fill(pad_mask[..., None], 0.0)

    return atom_next, coords_next, pad_mask


def decode_atom_types(atom_features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    atom_type_features = atom_features[..., :QM9_NUM_ATOM_TYPES]
    atom_types = atom_type_features.argmax(dim=-1)
    atom_types = atom_types.masked_fill(pad_mask, QM9_ATOM_PAD_TOKEN)
    return atom_types


def decode_qm9_charges(
    atom_features: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    charge_feature_scale: float = 8.0,
) -> torch.Tensor:
    if atom_features.shape[-1] <= QM9_NUM_ATOM_TYPES:
        charges = torch.zeros_like(pad_mask, dtype=torch.long)
    else:
        charges = (atom_features[..., QM9_NUM_ATOM_TYPES] * float(charge_feature_scale)).round().long()
    return charges.masked_fill(pad_mask, 0)
