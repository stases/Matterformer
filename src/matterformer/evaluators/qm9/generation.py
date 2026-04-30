from __future__ import annotations

import math

import torch

from matterformer.data import QM9_DATASET_INFO
from matterformer.evaluators.qm9.bonds import check_stability
from matterformer.evaluators.qm9.rdkit import BasicMolecularMetrics, Chem
from matterformer.tasks import decode_atom_types, decode_qm9_charges, edm_sampler


def has_rdkit() -> bool:
    return Chem is not None


def build_rdkit_metrics(reference_smiles: list[str] | None = None) -> BasicMolecularMetrics | None:
    if not has_rdkit():
        return None
    return BasicMolecularMetrics(QM9_DATASET_INFO, reference_smiles)


def sample_qm9_molecules(
    net,
    num_atoms_sampler,
    *,
    device: torch.device,
    num_molecules: int,
    sample_batch_size: int,
    sampler_kwargs: dict[str, float | int],
) -> list[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if num_molecules <= 0:
        return []
    sample_batch_size = max(int(sample_batch_size), 1)
    generated: list[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    remaining = int(num_molecules)
    net.eval()
    model_channels = int(getattr(net.model, "atom_channels", 5))
    charge_feature_scale = float(getattr(net.model, "charge_feature_scale", 8.0))
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(sample_batch_size, remaining)
            num_atoms = num_atoms_sampler(batch_size, device=device)
            atom_features, coords, pad_mask = edm_sampler(
                net,
                num_atoms,
                atom_channels=model_channels,
                **sampler_kwargs,
            )
            atom_types = decode_atom_types(atom_features, pad_mask)
            charges = decode_qm9_charges(
                atom_features,
                pad_mask,
                charge_feature_scale=charge_feature_scale,
            )
            for batch_idx in range(batch_size):
                count = int(num_atoms[batch_idx].item())
                molecule = (
                    coords[batch_idx, :count].detach().cpu(),
                    atom_types[batch_idx, :count].detach().cpu(),
                )
                if model_channels > 5:
                    molecule = (
                        molecule[0],
                        molecule[1],
                        charges[batch_idx, :count].detach().cpu(),
                    )
                generated.append(molecule)
            remaining -= batch_size
    return generated


def evaluate_generated_qm9(
    generated: list[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    rdkit_metrics: BasicMolecularMetrics | None = None,
) -> dict[str, float]:
    stable_molecules = 0
    stable_atoms = 0
    total_atoms = 0
    for molecule in generated:
        molecule_stable, num_stable_atoms, num_atoms = check_stability(*molecule)
        stable_molecules += int(molecule_stable)
        stable_atoms += int(num_stable_atoms)
        total_atoms += int(num_atoms)

    metrics = {
        "atom_stability": stable_atoms / max(total_atoms, 1),
        "molecule_stability": stable_molecules / max(len(generated), 1),
        "validity": math.nan,
        "uniqueness": math.nan,
        "rdkit_available": 0.0,
    }
    if rdkit_metrics is None:
        return metrics

    try:
        (validity, uniqueness, _), _ = rdkit_metrics.evaluate(generated)
    except RuntimeError:
        return metrics
    metrics["validity"] = float(validity)
    metrics["uniqueness"] = float(uniqueness)
    metrics["rdkit_available"] = 1.0
    return metrics


def sample_and_evaluate_qm9(
    net,
    num_atoms_sampler,
    *,
    device: torch.device,
    num_molecules: int,
    sample_batch_size: int,
    sampler_kwargs: dict[str, float | int],
    rdkit_metrics: BasicMolecularMetrics | None = None,
) -> dict[str, float]:
    generated = sample_qm9_molecules(
        net,
        num_atoms_sampler,
        device=device,
        num_molecules=num_molecules,
        sample_batch_size=sample_batch_size,
        sampler_kwargs=sampler_kwargs,
    )
    return evaluate_generated_qm9(generated, rdkit_metrics=rdkit_metrics)
