from __future__ import annotations

import torch
from rdkit import Chem
from rdkit import RDLogger

from matterformer.data.geom_drugs import (
    GEOM_DRUGS_DATASET_INFO,
    GEOM_DRUGS_NUM_ATOM_TYPES,
    GEOM_DRUGS_NUM_CHARGE_TYPES,
)
from matterformer.evaluators.geom_drugs.corrected import (
    compute_molecules_stability,
    corrected_is_valid,
)
from matterformer.evaluators.geom_drugs.geometry import (
    LEGACY_GEOM_ALLOWED_FC_BONDS,
    allowed_bonds_match,
    atom_symbol_counts,
    build_inferred_mol_from_geometry,
    build_unbonded_mol,
    mol_to_connected_smiles,
    mol_to_largest_fragment_smiles,
)
from matterformer.evaluators.geom_drugs.io import load_or_build_train_reference_smiles
from matterformer.tasks import decode_geom_drugs_types_and_charges, edm_sampler


RDLogger.DisableLog("rdApp.*")


def fraction(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def evaluate_raw_metrics(mols: list[Chem.Mol], train_reference_smiles: set[str]) -> dict[str, float]:
    inferred_mols: list[Chem.Mol] = []
    stable_molecules = 0
    stable_atoms = 0
    total_atoms = 0
    valid_molecules = 0

    for mol in mols:
        inferred_mol, bond_counts, atom_symbols, charges = build_inferred_mol_from_geometry(
            mol,
            limit_bonds_to_one=False,
        )
        inferred_mols.append(inferred_mol)
        per_atom_ok = [
            allowed_bonds_match(symbol, int(valence), int(charge), LEGACY_GEOM_ALLOWED_FC_BONDS)
            for symbol, valence, charge in zip(atom_symbols, bond_counts, charges)
        ]
        stable_molecules += int(all(per_atom_ok))
        stable_atoms += sum(per_atom_ok)
        total_atoms += len(per_atom_ok)
        valid_molecules += int(corrected_is_valid(inferred_mol))

    valid_smiles = [mol_to_largest_fragment_smiles(mol) for mol in inferred_mols]
    valid_smiles = [smiles for smiles in valid_smiles if smiles is not None]
    unique_smiles = set(valid_smiles)
    return {
        "validity": fraction(valid_molecules, len(mols)),
        "molecule_stability": fraction(stable_molecules, len(mols)),
        "atom_stability": fraction(stable_atoms, total_atoms),
        "uniqueness": fraction(len(unique_smiles), len(valid_smiles)),
        "novelty": fraction(len(unique_smiles - train_reference_smiles), len(unique_smiles)),
    }


def evaluate_corrected_metrics(mols: list[Chem.Mol], train_reference_smiles: set[str]) -> dict[str, float]:
    inferred_mols = [
        build_inferred_mol_from_geometry(mol, limit_bonds_to_one=False)[0]
        for mol in mols
    ]
    aromatic = any(bond.GetIsAromatic() for mol in inferred_mols for bond in mol.GetBonds())
    validity, stable_mask, n_stable_atoms, n_atoms = compute_molecules_stability(
        inferred_mols,
        aromatic=aromatic,
    )
    total_atoms = int(n_atoms.sum().item())
    stable_atoms = int(n_stable_atoms.sum().item())
    valid_smiles = [
        mol_to_connected_smiles(mol)
        for mol, is_valid in zip(inferred_mols, validity.tolist())
        if is_valid
    ]
    valid_smiles = [smiles for smiles in valid_smiles if smiles is not None]
    unique_smiles = set(valid_smiles)
    return {
        "validity": float(validity.mean().item()) if len(inferred_mols) else 0.0,
        "molecule_stability": float(stable_mask.mean().item()) if len(inferred_mols) else 0.0,
        "atom_stability": fraction(stable_atoms, total_atoms),
        "uniqueness": fraction(len(unique_smiles), len(valid_smiles)),
        "novelty": fraction(len(unique_smiles - train_reference_smiles), len(unique_smiles)),
    }


def evaluate_generated_geom_drugs(
    generated: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    train_reference_smiles: set[str] | None = None,
    data_root: str | None = "./data/geom_drugs",
) -> dict[str, object]:
    if train_reference_smiles is None:
        train_reference_smiles = load_or_build_train_reference_smiles(data_root=data_root or "./data/geom_drugs")

    mols = [build_unbonded_mol(positions, atom_types, charges) for positions, atom_types, charges in generated]
    raw_metrics = evaluate_raw_metrics(mols, train_reference_smiles)
    corrected_metrics = evaluate_corrected_metrics(mols, train_reference_smiles)
    delta = {
        key: float(corrected_metrics[key] - raw_metrics[key])
        for key in ("validity", "molecule_stability", "atom_stability", "uniqueness", "novelty")
    }
    return {
        "report_type": "geom_drugs_generated_evaluation",
        "molecule_count": len(mols),
        "sample_elements": atom_symbol_counts(mols),
        "metric_notes": {
            "raw_metrics": "Deterministic geometry bond inference with charge-aware legacy valence checks and RDKit validity on inferred molecules.",
            "corrected_metrics": "Deterministic bond reconstruction from the same outputs followed by corrected Isayevlab validity and aromatic-aware stability.",
        },
        "raw_metrics": raw_metrics,
        "corrected_metrics": corrected_metrics,
        "corrected_minus_raw": delta,
    }


def sample_geom_drugs_molecules(
    net,
    num_atoms_sampler,
    *,
    device: torch.device,
    num_molecules: int,
    sample_batch_size: int,
    sampler_kwargs: dict[str, float | int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if num_molecules <= 0:
        return []
    sample_batch_size = max(int(sample_batch_size), 1)
    generated: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    remaining = int(num_molecules)
    net.eval()
    feature_channels = GEOM_DRUGS_NUM_ATOM_TYPES + GEOM_DRUGS_NUM_CHARGE_TYPES
    model_channels = int(getattr(net.model, "atom_channels", feature_channels))
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(sample_batch_size, remaining)
            num_atoms = num_atoms_sampler(batch_size, device=device)
            node_features, coords, pad_mask = edm_sampler(
                net,
                num_atoms,
                atom_channels=model_channels,
                **sampler_kwargs,
            )
            atom_types, charges = decode_geom_drugs_types_and_charges(node_features, pad_mask)
            for batch_idx in range(batch_size):
                count = int(num_atoms[batch_idx].item())
                generated.append(
                    (
                        coords[batch_idx, :count].detach().cpu(),
                        atom_types[batch_idx, :count].detach().cpu(),
                        charges[batch_idx, :count].detach().cpu(),
                    )
                )
            remaining -= batch_size
    return generated


def sample_and_evaluate_geom_drugs(
    net,
    num_atoms_sampler,
    *,
    device: torch.device,
    num_molecules: int,
    sample_batch_size: int,
    sampler_kwargs: dict[str, float | int],
    train_reference_smiles: set[str] | None = None,
    data_root: str | None = "./data/geom_drugs",
) -> dict[str, object]:
    generated = sample_geom_drugs_molecules(
        net,
        num_atoms_sampler,
        device=device,
        num_molecules=num_molecules,
        sample_batch_size=sample_batch_size,
        sampler_kwargs=sampler_kwargs,
    )
    return evaluate_generated_geom_drugs(
        generated,
        train_reference_smiles=train_reference_smiles,
        data_root=data_root,
    )
