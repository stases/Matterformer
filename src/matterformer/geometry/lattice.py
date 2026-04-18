from __future__ import annotations

import torch


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(value):
        value = value.float()
    return value


def lattice_latent_to_lower_triangular(
    lattice_latent: torch.Tensor,
    lattice_repr: str = "y1",
) -> torch.Tensor:
    lattice_latent = _as_float_tensor(lattice_latent)
    if lattice_latent.ndim != 2 or lattice_latent.shape[-1] != 6:
        raise ValueError(
            f"lattice_latent must have shape (B, 6), got {tuple(lattice_latent.shape)}"
        )

    lattice_repr = lattice_repr.lower()
    batch_size = lattice_latent.shape[0]
    cell = torch.zeros(
        batch_size,
        3,
        3,
        device=lattice_latent.device,
        dtype=lattice_latent.dtype,
    )

    if lattice_repr == "y1":
        log_lengths = lattice_latent[:, :3].clamp(min=-5.0, max=5.0)
        cosines = lattice_latent[:, 3:].clamp(min=-0.9999, max=0.9999)
        lengths = torch.exp(log_lengths)
        cos_alpha, cos_beta, cos_gamma = cosines.unbind(dim=-1)

        sin_gamma = torch.sqrt((1.0 - cos_gamma.square()).clamp_min(1e-8))
        cell[:, 0, 0] = lengths[:, 0]
        cell[:, 1, 0] = lengths[:, 1] * cos_gamma
        cell[:, 1, 1] = lengths[:, 1] * sin_gamma
        cell[:, 2, 0] = lengths[:, 2] * cos_beta
        cell[:, 2, 1] = lengths[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cell[:, 2, 2] = torch.sqrt(
            lengths[:, 2].square() - cell[:, 2, 0].square() - cell[:, 2, 1].square()
        ).clamp_min(1e-8)
        return cell

    if lattice_repr == "ltri":
        diag = torch.exp(
            lattice_latent[:, [0, 2, 5]].clamp(min=-5.0, max=5.0)
        )
        off_diag = lattice_latent[:, [1, 3, 4]].clamp(min=-10.0, max=10.0)
        cell[:, 0, 0] = diag[:, 0]
        cell[:, 1, 0] = off_diag[:, 0]
        cell[:, 1, 1] = diag[:, 1]
        cell[:, 2, 0] = off_diag[:, 1]
        cell[:, 2, 1] = off_diag[:, 2]
        cell[:, 2, 2] = diag[:, 2]
        return cell

    raise ValueError(f"Unsupported lattice_repr: {lattice_repr}")


def lattice_latent_to_gram(
    lattice_latent: torch.Tensor,
    lattice_repr: str = "y1",
) -> torch.Tensor:
    cell = lattice_latent_to_lower_triangular(lattice_latent, lattice_repr=lattice_repr)
    gram = torch.matmul(cell, cell.transpose(-1, -2))
    return torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)


def gram_to_y1(gram: torch.Tensor) -> torch.Tensor:
    gram = _as_float_tensor(gram)
    if gram.ndim != 3 or gram.shape[-2:] != (3, 3):
        raise ValueError(f"gram must have shape (B, 3, 3), got {tuple(gram.shape)}")

    lengths = torch.sqrt(torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-8))
    denom_alpha = (lengths[:, 1] * lengths[:, 2]).clamp_min(1e-8)
    denom_beta = (lengths[:, 0] * lengths[:, 2]).clamp_min(1e-8)
    denom_gamma = (lengths[:, 0] * lengths[:, 1]).clamp_min(1e-8)
    cos_alpha = (gram[:, 1, 2] / denom_alpha).clamp(min=-0.9999, max=0.9999)
    cos_beta = (gram[:, 0, 2] / denom_beta).clamp(min=-0.9999, max=0.9999)
    cos_gamma = (gram[:, 0, 1] / denom_gamma).clamp(min=-0.9999, max=0.9999)
    return torch.cat(
        [torch.log(lengths.clamp_min(1e-8)), torch.stack([cos_alpha, cos_beta, cos_gamma], dim=-1)],
        dim=-1,
    )


def lattice_latent_to_y1(
    lattice_latent: torch.Tensor,
    lattice_repr: str = "y1",
) -> torch.Tensor:
    lattice_repr = lattice_repr.lower()
    if lattice_repr == "y1":
        lattice_latent = _as_float_tensor(lattice_latent)
        lattice_y1 = lattice_latent.clone()
        lattice_y1[:, :3] = lattice_y1[:, :3].clamp(min=-5.0, max=5.0)
        lattice_y1[:, 3:] = lattice_y1[:, 3:].clamp(min=-0.9999, max=0.9999)
        return lattice_y1
    if lattice_repr == "ltri":
        gram = lattice_latent_to_gram(lattice_latent, lattice_repr=lattice_repr)
        return gram_to_y1(gram)
    raise ValueError(f"Unsupported lattice_repr: {lattice_repr}")
