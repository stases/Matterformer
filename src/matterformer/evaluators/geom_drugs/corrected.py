from __future__ import annotations

import torch
from rdkit import Chem


GEOM_DRUGS_TUPLE_VALENCIES = {
    "Br": {0: [(0, 1)], 1: [(0, 2)]},
    "C": {0: [(0, 4), (2, 2), (2, 1), (3, 0)], -1: [(0, 3), (2, 1), (3, 0)], 1: [(0, 3), (2, 1), (3, 0)]},
    "N": {0: [(0, 3), (2, 0), (2, 1), (3, 0)], 1: [(0, 4), (2, 0), (2, 1), (2, 2), (3, 0)], -1: [(0, 2), (2, 0)], -2: [(0, 1)]},
    "H": {0: [(0, 1)]},
    "S": {0: [(0, 2), (0, 3), (0, 6), (2, 0)], 1: [(0, 3), (2, 0), (2, 1), (3, 0)], 2: [(0, 4), (2, 1), (2, 2)], 3: [(0, 2), (0, 5)], -1: [(0, 1)]},
    "O": {0: [(0, 2), (2, 0)], -1: [(0, 1)], 1: [(0, 3)]},
    "F": {0: [(0, 1)]},
    "Cl": {0: [(0, 1)], 1: [(0, 2)]},
    "P": {0: [(0, 3), (0, 5)], 1: [(0, 4)]},
    "I": {0: [(0, 1)], 1: [(0, 2)], 2: [(0, 3)]},
    "Si": {0: [(0, 4)], 1: [(0, 5)]},
    "B": {-1: [(0, 4)], 0: [(0, 3)]},
    "Bi": {0: [(0, 3)], 2: [(0, 5)]},
}


def corrected_is_valid(mol: Chem.Mol, verbose: bool = False) -> bool:
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as exc:
        if verbose:
            print(f"Kekulization failed: {exc}")
        return False
    except ValueError as exc:
        if verbose:
            print(f"Sanitization failed: {exc}")
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        if verbose:
            print("Molecule has multiple fragments.")
        return False
    return True


def _is_valid_valence_tuple(combo, allowed, charge: int) -> bool:
    if isinstance(allowed, tuple):
        return combo == allowed
    if isinstance(allowed, (list, set)):
        return combo in allowed
    if isinstance(allowed, dict):
        if charge not in allowed:
            return False
        return _is_valid_valence_tuple(combo, allowed[charge], charge)
    return False


def compute_molecules_stability_from_graph(
    adjacency_matrices: torch.Tensor,
    numbers: torch.Tensor,
    charges: torch.Tensor,
    *,
    allowed_bonds=None,
    aromatic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if adjacency_matrices.ndim == 2:
        adjacency_matrices = adjacency_matrices.unsqueeze(0)
        numbers = numbers.unsqueeze(0)
        charges = charges.unsqueeze(0)

    if allowed_bonds is None:
        allowed_bonds = GEOM_DRUGS_TUPLE_VALENCIES

    if not aromatic:
        assert (adjacency_matrices == 1.5).sum() == 0 and (adjacency_matrices == 4).sum() == 0

    batch_size = adjacency_matrices.shape[0]
    stable_mask = torch.zeros(batch_size)
    n_stable_atoms = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size)

    for i in range(batch_size):
        adj = adjacency_matrices[i]
        atom_nums = numbers[i]
        atom_charges = charges[i]

        mol_stable = True
        n_atoms_i = 0
        n_stable_i = 0
        for j, (atomic_num, charge) in enumerate(zip(atom_nums, atom_charges)):
            if atomic_num.item() == 0:
                continue
            row = adj[j]
            aromatic_count = int((row == 1.5).sum().item())
            normal_valence = float((row * (row != 1.5)).sum().item())
            combo = (aromatic_count, int(normal_valence))
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num))
            allowed = allowed_bonds.get(symbol, {})
            if _is_valid_valence_tuple(combo, allowed, int(charge)):
                n_stable_i += 1
            else:
                mol_stable = False
            n_atoms_i += 1
        stable_mask[i] = float(mol_stable)
        n_stable_atoms[i] = n_stable_i
        n_atoms[i] = n_atoms_i
    return stable_mask, n_stable_atoms, n_atoms


def compute_molecules_stability(
    rdkit_molecules: list[Chem.Mol],
    *,
    aromatic: bool = True,
    allowed_bonds=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stable_list = []
    stable_atoms_list = []
    atom_counts_list = []
    validity_list = []

    for mol in rdkit_molecules:
        if mol is None:
            continue
        n_atoms = mol.GetNumAtoms()
        adjacency = torch.zeros((1, n_atoms, n_atoms))
        numbers = torch.zeros((1, n_atoms), dtype=torch.long)
        charges = torch.zeros((1, n_atoms), dtype=torch.long)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            numbers[0, idx] = atom.GetAtomicNum()
            charges[0, idx] = atom.GetFormalCharge()

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            adjacency[0, i, j] = adjacency[0, j, i] = bond_type

        stable, stable_atoms, atom_count = compute_molecules_stability_from_graph(
            adjacency,
            numbers,
            charges,
            allowed_bonds=allowed_bonds,
            aromatic=aromatic,
        )
        stable_list.append(stable.item())
        stable_atoms_list.append(stable_atoms.item())
        atom_counts_list.append(atom_count.item())
        validity_list.append(float(corrected_is_valid(mol)))

    return (
        torch.tensor(validity_list),
        torch.tensor(stable_list),
        torch.tensor(stable_atoms_list),
        torch.tensor(atom_counts_list),
    )
