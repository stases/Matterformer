from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from matterformer.data import OMolBatch


@dataclass
class OMolLossOutput:
    loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class OMolElementReferences(nn.Module):
    def __init__(self, element_references: torch.Tensor) -> None:
        super().__init__()
        refs = torch.as_tensor(element_references, dtype=torch.float64).view(-1)
        if refs.numel() == 0:
            raise ValueError("element_references must not be empty")
        self.register_buffer("element_references", refs)

    def reference_energy(self, atomic_numbers: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        refs = self.element_references.to(device=atomic_numbers.device)
        z = atomic_numbers.clamp(min=0, max=refs.numel() - 1)
        per_atom = refs[z].masked_fill(pad_mask, 0.0)
        return per_atom.sum(dim=1).to(dtype=torch.float32)

    def subtract_refs(self, batch: OMolBatch) -> torch.Tensor:
        return batch.energy - self.reference_energy(batch.atomic_numbers, batch.pad_mask).to(dtype=batch.energy.dtype)

    def add_refs(self, batch: OMolBatch, residual_energy: torch.Tensor) -> torch.Tensor:
        return residual_energy + self.reference_energy(batch.atomic_numbers, batch.pad_mask).to(dtype=residual_energy.dtype)


def load_omol_element_references(path: str | Path, *, key: str = "omol_elem_refs") -> OMolElementReferences:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    refs = payload[key] if isinstance(payload, Mapping) else payload
    return OMolElementReferences(torch.tensor(refs, dtype=torch.float64))


def _masked_component_mae(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask[..., None].expand_as(error)
    if not expanded_mask.any():
        return error.new_tensor(0.0)
    return error.abs().masked_select(expanded_mask).mean()


class OMolDirectForceLoss(nn.Module):
    def __init__(
        self,
        element_references: OMolElementReferences,
        *,
        normalizer_rmsd: float = 1.433569,
        energy_weight: float = 10.0,
        force_weight: float = 10.0,
        energy_loss: str = "per_atom_mae",
        force_loss: str = "l2norm",
    ) -> None:
        super().__init__()
        self.element_references = element_references
        self.normalizer_rmsd = float(normalizer_rmsd)
        self.energy_weight = float(energy_weight)
        self.force_weight = float(force_weight)
        self.energy_loss = str(energy_loss).lower()
        self.force_loss = str(force_loss).lower()
        if self.normalizer_rmsd <= 0.0:
            raise ValueError("normalizer_rmsd must be positive")
        if self.energy_loss not in {"mae", "per_atom_mae"}:
            raise ValueError("energy_loss must be one of {'mae', 'per_atom_mae'}")
        if self.force_loss not in {"mae", "l2norm"}:
            raise ValueError("force_loss must be one of {'mae', 'l2norm'}")

    def forward(self, predictions: dict[str, torch.Tensor], batch: OMolBatch) -> OMolLossOutput:
        pred_energy = predictions["energy"].view_as(batch.energy)
        pred_forces = predictions["forces"]
        if pred_forces.shape != batch.forces.shape:
            raise ValueError(f"predicted forces {tuple(pred_forces.shape)} do not match target {tuple(batch.forces.shape)}")

        target_residual = self.element_references.subtract_refs(batch).to(dtype=pred_energy.dtype)
        target_energy_norm = target_residual / self.normalizer_rmsd
        target_forces_norm = batch.forces.to(dtype=pred_forces.dtype) / self.normalizer_rmsd

        energy_abs = (pred_energy - target_energy_norm).abs()
        if self.energy_loss == "per_atom_mae":
            energy_loss = (energy_abs / batch.num_atoms.to(device=energy_abs.device, dtype=energy_abs.dtype).clamp_min(1)).mean()
        else:
            energy_loss = energy_abs.mean()

        force_mask = batch.free_atom_mask & ~batch.pad_mask
        force_error = pred_forces - target_forces_norm
        if self.force_loss == "l2norm":
            per_atom_norm = torch.linalg.norm(force_error, dim=-1)
            per_structure = (per_atom_norm * force_mask.to(dtype=per_atom_norm.dtype)).sum(dim=1)
            denom = force_mask.sum(dim=1).to(dtype=per_atom_norm.dtype).clamp_min(1.0)
            force_loss = (per_structure / denom).mean()
        else:
            force_loss = _masked_component_mae(force_error, force_mask)

        loss = self.energy_weight * energy_loss + self.force_weight * force_loss
        energy_error_ev = (pred_energy.detach() * self.normalizer_rmsd) - target_residual.to(dtype=pred_energy.dtype)
        force_error_ev = (pred_forces.detach() * self.normalizer_rmsd) - batch.forces.to(dtype=pred_forces.dtype)
        diagnostics = {
            "loss": loss.detach(),
            "energy_loss": energy_loss.detach(),
            "force_loss": force_loss.detach(),
            "e_mae": energy_error_ev.abs().mean() * 1000.0,
            "e_mae_per_atom": (
                energy_error_ev.abs() / batch.num_atoms.to(device=energy_error_ev.device, dtype=energy_error_ev.dtype).clamp_min(1)
            ).mean()
            * 1000.0,
            "f_mae": _masked_component_mae(force_error_ev, force_mask) * 1000.0,
        }
        extra_diagnostics = predictions.get("diagnostics")
        if isinstance(extra_diagnostics, dict):
            for key, value in extra_diagnostics.items():
                diagnostics[str(key)] = torch.as_tensor(value, device=loss.device).detach()
        return OMolLossOutput(loss=loss, diagnostics=diagnostics)
