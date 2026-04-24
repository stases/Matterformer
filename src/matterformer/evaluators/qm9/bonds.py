from __future__ import annotations

import numpy as np


bonds1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}

bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

allowed_bonds = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}

margin1, margin2, margin3 = 10, 5, 3


def get_bond_order(atom1: str, atom2: str, distance: float, check_exists: bool = False) -> int:
    distance = 100.0 * float(distance)
    if check_exists and (atom1 not in bonds1 or atom2 not in bonds1[atom1]):
        return 0
    if distance >= bonds1[atom1][atom2] + margin1:
        return 0
    if atom1 in bonds2 and atom2 in bonds2[atom1]:
        if distance < bonds2[atom1][atom2] + margin2:
            if atom1 in bonds3 and atom2 in bonds3[atom1]:
                if distance < bonds3[atom1][atom2] + margin3:
                    return 3
            return 2
    return 1


def check_stability(
    positions,
    atom_types,
    charges=None,
    debug: bool = False,
) -> tuple[bool, int, int]:
    del charges
    atom_decoder = ["H", "C", "N", "O", "F"]

    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    if hasattr(atom_types, "detach"):
        atom_types = atom_types.detach().cpu().numpy()

    positions = np.asarray(positions)
    atom_types = np.asarray(atom_types)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")

    num_atoms = positions.shape[0]
    bond_count = np.zeros(num_atoms, dtype=np.int64)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            atom1 = atom_decoder[int(atom_types[i])]
            atom2 = atom_decoder[int(atom_types[j])]
            order = get_bond_order(atom1, atom2, distance)
            bond_count[i] += order
            bond_count[j] += order

    stable_atoms = 0
    for atom_type, atom_bonds in zip(atom_types, bond_count):
        atom_symbol = atom_decoder[int(atom_type)]
        expected = allowed_bonds[atom_symbol]
        is_stable = atom_bonds == expected
        if debug and not is_stable:
            print(f"Invalid bonds for {atom_symbol}: observed {atom_bonds}, expected {expected}")
        stable_atoms += int(is_stable)

    return stable_atoms == num_atoms, stable_atoms, num_atoms
