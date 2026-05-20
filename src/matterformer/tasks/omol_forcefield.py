from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn

from matterformer.data import OMolBatch


@dataclass
class OMolLossOutput:
    loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]]


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


def _ddp_corrected_mean(local_sum: torch.Tensor, local_count: torch.Tensor) -> torch.Tensor:
    local_count = local_count.to(device=local_sum.device, dtype=local_sum.dtype)
    if not (dist.is_available() and dist.is_initialized()):
        return local_sum / local_count.clamp_min(1.0)

    global_count = local_count.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    return local_sum * float(dist.get_world_size()) / global_count.clamp_min(1.0)


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
        graph_count = torch.as_tensor(pred_energy.numel(), device=pred_energy.device, dtype=pred_energy.dtype)
        if self.energy_loss == "per_atom_mae":
            per_graph_energy_loss = energy_abs / batch.num_atoms.to(device=energy_abs.device, dtype=energy_abs.dtype).clamp_min(1)
            energy_loss = _ddp_corrected_mean(per_graph_energy_loss.sum(), graph_count)
        else:
            per_graph_energy_loss = energy_abs
            energy_loss = _ddp_corrected_mean(per_graph_energy_loss.sum(), graph_count)

        force_mask = batch.free_atom_mask & ~batch.pad_mask
        force_error = pred_forces - target_forces_norm
        if self.force_loss == "l2norm":
            per_atom_norm = torch.linalg.norm(force_error, dim=-1)
            force_mask_float = force_mask.to(dtype=per_atom_norm.dtype)
            force_loss_sum = (per_atom_norm * force_mask_float).sum()
            force_loss_count = force_mask_float.sum()
            force_loss = _ddp_corrected_mean(
                force_loss_sum,
                force_loss_count,
            )
        else:
            expanded_force_mask = force_mask[..., None].expand_as(force_error)
            force_loss_sum = force_error.abs().masked_select(expanded_force_mask).sum()
            force_loss_count = expanded_force_mask.to(dtype=force_error.dtype).sum()
            force_loss = _ddp_corrected_mean(force_loss_sum, force_loss_count)

        loss = self.energy_weight * energy_loss + self.force_weight * force_loss
        energy_error_ev = (pred_energy.detach() * self.normalizer_rmsd) - target_residual.to(dtype=pred_energy.dtype)
        force_error_ev = (pred_forces.detach() * self.normalizer_rmsd) - batch.forces.to(dtype=pred_forces.dtype)
        num_atoms_float = batch.num_atoms.to(device=energy_error_ev.device, dtype=energy_error_ev.dtype).clamp_min(1)
        expanded_force_mask_ev = force_mask[..., None].expand_as(force_error_ev)
        f_mae_sum = force_error_ev.abs().masked_select(expanded_force_mask_ev).sum() * 1000.0
        f_mae_count = expanded_force_mask_ev.to(dtype=force_error_ev.dtype).sum()
        diagnostics = {
            "loss": loss.detach(),
            "energy_loss": energy_loss.detach(),
            "force_loss": force_loss.detach(),
            "e_mae": energy_error_ev.abs().mean() * 1000.0,
            "e_mae_per_atom": (
                energy_error_ev.abs() / num_atoms_float
            ).mean()
            * 1000.0,
            "f_mae": f_mae_sum / f_mae_count.clamp_min(1.0),
        }
        metric_sums = {
            "energy_loss": (per_graph_energy_loss.detach().sum(), graph_count.detach()),
            "force_loss": (force_loss_sum.detach(), force_loss_count.detach()),
            "e_mae": (energy_error_ev.abs().detach().sum() * 1000.0, graph_count.detach()),
            "e_mae_per_atom": ((energy_error_ev.abs() / num_atoms_float).detach().sum() * 1000.0, graph_count.detach()),
            "f_mae": (f_mae_sum.detach(), f_mae_count.detach()),
        }
        extra_diagnostics = predictions.get("diagnostics")
        if isinstance(extra_diagnostics, dict):
            for key, value in extra_diagnostics.items():
                diagnostics[str(key)] = torch.as_tensor(value, device=loss.device).detach()
        return OMolLossOutput(loss=loss, diagnostics=diagnostics, metric_sums=metric_sums)
