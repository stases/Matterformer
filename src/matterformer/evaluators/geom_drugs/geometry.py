from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from matterformer.data.geom_drugs import (
    GEOM_DRUGS_ATOM_DECODER,
    GEOM_DRUGS_ATOM_ENCODER,
)


LEGACY_GEOM_ALLOWED_FC_BONDS = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

GEOM_BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92, "B": 119, "Si": 148, "P": 144, "As": 152, "S": 134, "Cl": 127, "Br": 141, "I": 161},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135, "Si": 185, "P": 184, "S": 182, "Cl": 177, "Br": 194, "I": 214},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136, "Cl": 175, "Br": 214, "S": 168, "I": 222, "P": 177},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142, "Br": 172, "S": 151, "P": 163, "Si": 163, "Cl": 164, "I": 194},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142, "S": 158, "Si": 160, "Cl": 166, "Br": 178, "P": 156, "I": 187},
    "B": {"H": 119, "Cl": 175},
    "Si": {"Si": 233, "H": 148, "C": 185, "O": 163, "S": 200, "F": 160, "Cl": 202, "Br": 215, "I": 243},
    "Cl": {"Cl": 199, "H": 127, "C": 177, "N": 175, "O": 164, "P": 203, "S": 207, "B": 175, "Si": 202, "F": 166, "Br": 214},
    "S": {"H": 134, "C": 182, "N": 168, "O": 151, "S": 204, "F": 158, "Cl": 207, "Br": 225, "Si": 200, "P": 210, "I": 234},
    "Br": {"Br": 228, "H": 141, "C": 194, "O": 172, "N": 214, "Si": 215, "S": 225, "F": 178, "Cl": 214, "P": 222},
    "P": {"P": 221, "H": 144, "C": 184, "O": 163, "Cl": 203, "S": 210, "F": 156, "N": 177, "Br": 222},
    "I": {"H": 161, "C": 214, "Si": 243, "N": 222, "O": 194, "S": 234, "F": 187, "I": 266},
    "As": {"H": 152},
}
GEOM_BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}
GEOM_BONDS3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

ORDER_TO_RDKIT_BOND = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}


def geometry_bond_order(atom1: str, atom2: str, distance: float) -> int:
    distance_pm = 100.0 * distance
    if atom1 not in GEOM_BONDS1 or atom2 not in GEOM_BONDS1[atom1]:
        return 0
    if distance_pm < GEOM_BONDS1[atom1][atom2] + 10:
        if atom1 in GEOM_BONDS2 and atom2 in GEOM_BONDS2[atom1]:
            if distance_pm < GEOM_BONDS2[atom1][atom2] + 5:
                if atom1 in GEOM_BONDS3 and atom2 in GEOM_BONDS3[atom1]:
                    if distance_pm < GEOM_BONDS3[atom1][atom2] + 3:
                        return 3
                return 2
        return 1
    return 0


def allowed_bonds_match(symbol: str, valence: int, charge: int | None, allowed_table) -> bool:
    allowed = allowed_table.get(symbol)
    if allowed is None:
        return False
    if isinstance(allowed, int):
        return valence == allowed
    if isinstance(allowed, list):
        return valence in allowed
    if charge is None:
        return False
    expected = allowed.get(charge, allowed.get(0))
    if expected is None:
        return False
    if isinstance(expected, int):
        return valence == expected
    return valence in expected


def build_unbonded_mol(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
) -> Chem.Mol:
    rw = Chem.RWMol()
    for atom_idx, charge in zip(atom_types.tolist(), charges.tolist()):
        atom = Chem.Atom(GEOM_DRUGS_ATOM_DECODER[int(atom_idx)])
        if int(charge) != 0:
            atom.SetFormalCharge(int(charge))
        rw.AddAtom(atom)

    mol = rw.GetMol()
    conf = Chem.Conformer(len(atom_types))
    for idx, (x, y, z) in enumerate(positions.tolist()):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def build_inferred_mol_from_geometry(
    mol: Chem.Mol,
    *,
    limit_bonds_to_one: bool = False,
) -> tuple[Chem.Mol, np.ndarray, list[str], list[int]]:
    positions = mol.GetConformer().GetPositions()
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [int(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    n_atoms = len(atom_symbols)
    bond_counts = np.zeros(n_atoms, dtype=float)
    inferred = Chem.RWMol()

    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetSymbol())
        charge = int(atom.GetFormalCharge())
        if charge != 0:
            new_atom.SetFormalCharge(charge)
        inferred.AddAtom(new_atom)

    conf = Chem.Conformer(n_atoms)
    for idx in range(n_atoms):
        x, y, z = positions[idx]
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    inferred.AddConformer(conf)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            order = geometry_bond_order(
                atom_symbols[i],
                atom_symbols[j],
                float(np.linalg.norm(positions[i] - positions[j])),
            )
            if order <= 0:
                continue
            if limit_bonds_to_one:
                order = 1
            bond_counts[i] += order
            bond_counts[j] += order
            inferred.AddBond(i, j, ORDER_TO_RDKIT_BOND[int(order)])

    return inferred.GetMol(), bond_counts, atom_symbols, charges


def atom_symbol_counts(mols: list[Chem.Mol]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def mol_to_largest_fragment_smiles(mol: Chem.Mol) -> str | None:
    mol_copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol_copy)
    except Exception:
        return None
    try:
        fragments = Chem.GetMolFrags(mol_copy, asMols=True)
    except Exception:
        return None
    if not fragments:
        return None
    largest = max(fragments, key=lambda frag: frag.GetNumAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(largest, canonical=True)
    except Exception:
        return None


def mol_to_connected_smiles(mol: Chem.Mol) -> str | None:
    mol_copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol_copy)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol_copy)) != 1:
        return None
    try:
        return Chem.MolToSmiles(mol_copy, canonical=True)
    except Exception:
        return None
