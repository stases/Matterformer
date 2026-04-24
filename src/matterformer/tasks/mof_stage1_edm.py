from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from matterformer.data.mof_bwdb import build_pad_mask
from matterformer.geometry.lattice import (
    clamp_lattice_latent,
    lattice_latent_to_y1,
    y1_to_lattice_latent,
)


def mod1(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def wrap_frac(delta: torch.Tensor) -> torch.Tensor:
    return delta - torch.round(delta)


def lattice_params_to_y1(lattice_params: torch.Tensor) -> torch.Tensor:
    lattice_params = torch.as_tensor(lattice_params, dtype=torch.float32)
    if lattice_params.ndim != 2 or lattice_params.shape[-1] != 6:
        raise ValueError(f"lattice_params must have shape (B, 6), got {tuple(lattice_params.shape)}")
    lengths = lattice_params[:, :3].clamp_min(1e-8)
    angles_deg = lattice_params[:, 3:].clamp(min=1e-3, max=179.999)
    cosines = torch.cos(torch.deg2rad(angles_deg)).clamp(min=-0.9999, max=0.9999)
    return torch.cat([torch.log(lengths), cosines], dim=-1)


def y1_to_lattice_params(lattice_y1: torch.Tensor) -> torch.Tensor:
    lattice_y1 = torch.as_tensor(lattice_y1, dtype=torch.float32)
    if lattice_y1.ndim != 2 or lattice_y1.shape[-1] != 6:
        raise ValueError(f"lattice_y1 must have shape (B, 6), got {tuple(lattice_y1.shape)}")
    lengths = torch.exp(lattice_y1[:, :3].clamp(min=-5.0, max=5.0))
    cosines = lattice_y1[:, 3:].clamp(min=-0.9999, max=0.9999)
    angles = torch.rad2deg(torch.arccos(cosines))
    return torch.cat([lengths, angles], dim=-1)


def clamp_lattice_y1(lattice_y1: torch.Tensor) -> torch.Tensor:
    lattice_y1 = torch.as_tensor(lattice_y1, dtype=torch.float32)
    log_lengths = lattice_y1[..., :3].clamp(min=-5.0, max=5.0)
    cosines = lattice_y1[..., 3:].clamp(min=-0.9999, max=0.9999)
    return torch.cat([log_lengths, cosines], dim=-1)


def lattice_params_to_lattice_latent(
    lattice_params: torch.Tensor,
    lattice_repr: str = "y1",
) -> torch.Tensor:
    lattice_y1 = lattice_params_to_y1(lattice_params)
    return y1_to_lattice_latent(lattice_y1, lattice_repr=lattice_repr)


def lattice_latent_to_lattice_params(
    lattice_latent: torch.Tensor,
    lattice_repr: str = "y1",
) -> torch.Tensor:
    lattice_y1 = lattice_latent_to_y1(lattice_latent, lattice_repr=lattice_repr)
    return y1_to_lattice_params(lattice_y1)


def _masked_feature_mean(values: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).float()
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    denom = valid.sum().clamp_min(1.0)
    if values.ndim == valid.ndim:
        denom = valid.sum().clamp_min(1.0)
    else:
        denom = valid.sum().clamp_min(1.0) * float(values.shape[-1])
    return (values * valid).sum() / denom


def _masked_frac_rmse(diff: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).float()
    denom = valid.sum().clamp_min(1.0) * float(diff.shape[-1])
    return torch.sqrt(((diff.square()) * valid.unsqueeze(-1)).sum() / denom)


def _remove_global_shift(diff: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_shift = (diff * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
    return wrap_frac(diff - mean_shift)


class MOFStage1EDMPreconditioner(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data_coord: float = 0.5,
        sigma_data_lattice: float = 0.5,
        lattice_repr: str = "ltri",
    ) -> None:
        super().__init__()
        self.model = model
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data_coord = float(sigma_data_coord)
        self.sigma_data_lattice = float(sigma_data_lattice)
        self.lattice_repr = str(lattice_repr).lower()

    def forward(
        self,
        block_features: torch.Tensor,
        block_type_ids: torch.Tensor,
        coords_noisy: torch.Tensor,
        pad_mask: torch.Tensor,
        sigma: torch.Tensor,
        *,
        lattice_noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.ndim == 2 and sigma.shape[-1] == 1:
            sigma = sigma[:, 0]
        sigma = sigma.float()
        sigma_coord = sigma[:, None, None]
        sigma_lattice = sigma[:, None]

        c_skip_f = self.sigma_data_coord**2 / (sigma_coord.square() + self.sigma_data_coord**2)
        c_out_f = sigma_coord * self.sigma_data_coord / torch.sqrt(
            sigma_coord.square() + self.sigma_data_coord**2
        )
        c_skip_y = self.sigma_data_lattice**2 / (sigma_lattice.square() + self.sigma_data_lattice**2)
        c_out_y = sigma_lattice * self.sigma_data_lattice / torch.sqrt(
            sigma_lattice.square() + self.sigma_data_lattice**2
        )
        c_in_y = 1.0 / torch.sqrt(sigma_lattice.square() + self.sigma_data_lattice**2)

        lattice_noisy = clamp_lattice_latent(lattice_noisy, lattice_repr=self.lattice_repr)
        coord_raw, lattice_raw = self.model(
            block_features,
            block_type_ids,
            mod1(coords_noisy),
            pad_mask,
            sigma,
            lattice=c_in_y * lattice_noisy,
            lattice_bias_latent=lattice_noisy,
        )
        coords_denoised = c_skip_f * coords_noisy + c_out_f * coord_raw
        coords_denoised = coords_denoised.masked_fill(pad_mask[..., None], 0.0)
        lattice_denoised = clamp_lattice_latent(
            c_skip_y * lattice_noisy + c_out_y * lattice_raw,
            lattice_repr=self.lattice_repr,
        )
        return coords_denoised, lattice_denoised


class MOFStage1EDMLoss:
    def __init__(
        self,
        *,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data_coord: float = 0.5,
        sigma_data_lattice: float = 0.5,
        coord_weight: float = 1.0,
        lattice_weight: float = 1.0,
        align_global_shift: bool = True,
        lattice_repr: str = "ltri",
    ) -> None:
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data_coord = float(sigma_data_coord)
        self.sigma_data_lattice = float(sigma_data_lattice)
        self.coord_weight = float(coord_weight)
        self.lattice_weight = float(lattice_weight)
        self.align_global_shift = bool(align_global_shift)
        self.lattice_repr = str(lattice_repr).lower()

    def __call__(
        self,
        net: MOFStage1EDMPreconditioner,
        batch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coords_clean = batch.block_com_frac.float()
        lattice_clean = lattice_params_to_lattice_latent(
            batch.lattice.float(),
            lattice_repr=self.lattice_repr,
        )
        sigma = torch.exp(
            torch.randn(coords_clean.shape[0], device=coords_clean.device, dtype=coords_clean.dtype) * self.p_std
            + self.p_mean
        )

        coords_noisy = coords_clean + torch.randn_like(coords_clean) * sigma[:, None, None]
        coords_noisy = coords_noisy.masked_fill(batch.block_pad_mask[..., None], 0.0)
        lattice_noisy = clamp_lattice_latent(
            lattice_clean + torch.randn_like(lattice_clean) * sigma[:, None]
            ,
            lattice_repr=self.lattice_repr,
        )

        coords_denoised, lattice_denoised = net(
            batch.block_features,
            batch.block_type_ids,
            coords_noisy,
            batch.block_pad_mask,
            sigma,
            lattice_noisy=lattice_noisy,
        )
        diff = wrap_frac(coords_denoised - coords_clean)
        if self.align_global_shift:
            diff = _remove_global_shift(diff, batch.block_pad_mask)
        coord_weight = (sigma.square() + self.sigma_data_coord**2) / (
            sigma * self.sigma_data_coord
        ).square()
        lattice_weight = (sigma.square() + self.sigma_data_lattice**2) / (
            sigma * self.sigma_data_lattice
        ).square()

        coord_loss = _masked_feature_mean(
            coord_weight[:, None, None] * diff.square(),
            batch.block_pad_mask,
        )
        lattice_loss = (lattice_weight[:, None] * (lattice_denoised - lattice_clean).square()).mean()
        total = self.coord_weight * coord_loss + self.lattice_weight * lattice_loss
        lattice_params_pred = lattice_latent_to_lattice_params(
            lattice_denoised,
            lattice_repr=self.lattice_repr,
        )
        lattice_params_clean = batch.lattice.float()
        return total, {
            "sigma": sigma,
            "coord_loss": coord_loss.detach(),
            "lattice_loss": lattice_loss.detach(),
            "coord_frac_rmse": _masked_frac_rmse(diff, batch.block_pad_mask).detach(),
            "length_mae": (lattice_params_pred[:, :3] - lattice_params_clean[:, :3]).abs().mean().detach(),
            "angle_mae": (lattice_params_pred[:, 3:] - lattice_params_clean[:, 3:]).abs().mean().detach(),
        }


def _karras_sigma_steps(
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if num_steps == 1:
        steps = torch.tensor([float(sigma_max)], device=device, dtype=torch.float32)
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        steps = (
            sigma_max ** (1.0 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
    return torch.cat([steps, torch.zeros_like(steps[:1])])


@torch.no_grad()
def mof_stage1_edm_sampler(
    net: MOFStage1EDMPreconditioner,
    block_features: torch.Tensor,
    block_type_ids: torch.Tensor,
    num_blocks: torch.Tensor,
    *,
    num_steps: int = 100,
    sigma_min: float = 0.002,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    s_churn: float = 30.0,
    s_min: float = 0.0,
    s_max: float = float("inf"),
    s_noise: float = 1.003,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = block_features.device
    num_blocks = num_blocks.to(device=device, dtype=torch.long)
    pad_mask = build_pad_mask(num_blocks).to(device)
    batch_size, max_blocks = pad_mask.shape

    coords_next = torch.randn(batch_size, max_blocks, 3, device=device)
    coords_next = coords_next.masked_fill(pad_mask[..., None], 0.0)
    lattice_next = torch.randn(batch_size, 6, device=device)

    sigma_min = max(float(sigma_min), net.sigma_min)
    sigma_max = min(float(sigma_max), net.sigma_max)
    t_steps = _karras_sigma_steps(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    coords_next = coords_next * t_steps[0]
    lattice_next = lattice_next * t_steps[0]

    for step_index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        gamma = min(s_churn / max(num_steps, 1), np.sqrt(2.0) - 1.0) if s_min <= t_cur <= s_max else 0.0
        t_hat = t_cur + gamma * t_cur
        coords_hat = coords_next + torch.sqrt(t_hat.square() - t_cur.square()) * s_noise * torch.randn_like(coords_next)
        lattice_hat = lattice_next + torch.sqrt(t_hat.square() - t_cur.square()) * s_noise * torch.randn_like(lattice_next)
        coords_hat = coords_hat.masked_fill(pad_mask[..., None], 0.0)
        lattice_hat = clamp_lattice_latent(lattice_hat, lattice_repr=net.lattice_repr)

        sigma_hat = torch.full((batch_size,), float(t_hat.item()), device=device)
        coords_denoised, lattice_denoised = net(
            block_features,
            block_type_ids,
            coords_hat,
            pad_mask,
            sigma_hat,
            lattice_noisy=lattice_hat,
        )
        coords_d = wrap_frac(coords_hat - coords_denoised) / t_hat
        lattice_d = (lattice_hat - lattice_denoised) / t_hat
        coords_next = coords_hat + (t_next - t_hat) * coords_d
        lattice_next = clamp_lattice_latent(
            lattice_hat + (t_next - t_hat) * lattice_d,
            lattice_repr=net.lattice_repr,
        )
        coords_next = wrap_frac(coords_next)
        coords_next = coords_next.masked_fill(pad_mask[..., None], 0.0)

        if step_index < num_steps - 1:
            sigma_next = torch.full((batch_size,), float(t_next.item()), device=device)
            coords_denoised, lattice_denoised = net(
                block_features,
                block_type_ids,
                coords_next,
                pad_mask,
                sigma_next,
                lattice_noisy=lattice_next,
            )
            coords_d_prime = wrap_frac(coords_next - coords_denoised) / t_next.clamp_min(1e-8)
            lattice_d_prime = (lattice_next - lattice_denoised) / t_next.clamp_min(1e-8)
            coords_next = coords_hat + (t_next - t_hat) * 0.5 * (coords_d + coords_d_prime)
            lattice_next = clamp_lattice_latent(
                lattice_hat + (t_next - t_hat) * 0.5 * (lattice_d + lattice_d_prime),
                lattice_repr=net.lattice_repr,
            )
            coords_next = wrap_frac(coords_next)
            coords_next = coords_next.masked_fill(pad_mask[..., None], 0.0)

    return mod1(coords_next), clamp_lattice_latent(lattice_next, lattice_repr=net.lattice_repr), pad_mask
