from __future__ import annotations

import torch

from matterformer.evaluators.qm9.bonds import get_bond_order

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised only when rdkit is missing.
    Chem = None


bond_dict = [
    None,
    None if Chem is None else Chem.rdchem.BondType.SINGLE,
    None if Chem is None else Chem.rdchem.BondType.DOUBLE,
    None if Chem is None else Chem.rdchem.BondType.TRIPLE,
    None if Chem is None else Chem.rdchem.BondType.AROMATIC,
]


def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is required for QM9 molecular validity metrics")


class BasicMolecularMetrics:
    def __init__(
        self,
        dataset_info: dict[str, object],
        dataset_smiles_list: list[str] | None = None,
    ) -> None:
        self.dataset_info = dataset_info
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_smiles_list = dataset_smiles_list

    def compute_validity(self, generated: list[tuple[torch.Tensor, ...]]):
        _require_rdkit()
        valid_smiles: list[str] = []
        for molecule_data in generated:
            positions, atom_types = molecule_data[:2]
            molecule = build_molecule(positions, atom_types, self.dataset_info)
            smiles = mol_to_smiles(molecule)
            if smiles is None:
                continue
            fragments = Chem.rdmolops.GetMolFrags(molecule, asMols=True)
            largest = max(fragments, default=molecule, key=lambda mol: mol.GetNumAtoms())
            smiles = mol_to_smiles(largest)
            if smiles is not None:
                valid_smiles.append(smiles)
        validity = len(valid_smiles) / max(len(generated), 1)
        return valid_smiles, validity

    @staticmethod
    def compute_uniqueness(valid_smiles: list[str]):
        unique = list(set(valid_smiles))
        uniqueness = len(unique) / max(len(valid_smiles), 1)
        return unique, uniqueness

    def compute_novelty(self, unique_smiles: list[str]):
        if self.dataset_smiles_list is None:
            return [], 0.0
        novel = [smiles for smiles in unique_smiles if smiles not in self.dataset_smiles_list]
        novelty = len(novel) / max(len(unique_smiles), 1)
        return novel, novelty

    def evaluate(self, generated: list[tuple[torch.Tensor, ...]]):
        valid, validity = self.compute_validity(generated)
        if not valid:
            return [0.0, 0.0, 0.0], []
        unique, uniqueness = self.compute_uniqueness(valid)
        _, novelty = self.compute_novelty(unique)
        return [validity, uniqueness, novelty], unique


def mol_to_smiles(molecule):
    _require_rdkit()
    try:
        Chem.SanitizeMol(molecule)
    except ValueError:
        return None
    return Chem.MolToSmiles(molecule)


def build_molecule(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    dataset_info: dict[str, object],
):
    _require_rdkit()
    atom_decoder = dataset_info["atom_decoder"]
    positions = positions.detach().cpu()
    atom_types = atom_types.detach().cpu()
    adjacency = torch.zeros((atom_types.shape[0], atom_types.shape[0]), dtype=torch.bool)
    edge_types = torch.zeros((atom_types.shape[0], atom_types.shape[0]), dtype=torch.long)
    distances = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)

    for i in range(atom_types.shape[0]):
        for j in range(i):
            atom1 = atom_decoder[int(atom_types[i])]
            atom2 = atom_decoder[int(atom_types[j])]
            order = get_bond_order(atom1, atom2, float(distances[i, j]), check_exists=True)
            if order > 0:
                adjacency[i, j] = True
                edge_types[i, j] = order

    molecule = Chem.RWMol()
    for atom_index in atom_types:
        molecule.AddAtom(Chem.Atom(atom_decoder[int(atom_index)]))
    for i in range(atom_types.shape[0]):
        for j in range(atom_types.shape[0]):
            if adjacency[i, j]:
                molecule.AddBond(i, j, bond_dict[int(edge_types[i, j])])
    return molecule
