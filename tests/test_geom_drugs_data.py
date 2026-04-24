import pickle

import pytest
import torch

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from matterformer.data import GeomDrugsDataset, GeomDrugsPaddedBatchSampler, collate_geom_drugs
from matterformer.data.geom_drugs_processing import (
    build_processed_split,
    build_train_reference_smiles,
    prepare_geom_drugs_dataset,
    process_geom_drugs,
)


def _embedded_molecule(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    status = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    assert status == 0
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _write_raw_split(path, entries) -> None:
    with path.open("wb") as handle:
        pickle.dump(entries, handle)


def test_geom_drugs_processing_builds_dataset_and_reference_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    cleaned_dir = tmp_path / "cleaned"
    processed_dir = tmp_path / "processed"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir()

    train_entries = [
        ("C", [_embedded_molecule("C")]),
        ("[NH4+]", [_embedded_molecule("[NH4+]")]),
    ]
    val_entries = [("C", [_embedded_molecule("C")])]
    test_entries = [("O", [_embedded_molecule("O")])]

    _write_raw_split(raw_dir / "train_data.pickle", train_entries)
    _write_raw_split(raw_dir / "val_data.pickle", val_entries)
    _write_raw_split(raw_dir / "test_data.pickle", test_entries)

    stats = process_geom_drugs(raw_dir, cleaned_dir, force=True)
    assert stats["train"]["saved_molecules"] == 2

    build_processed_split(cleaned_dir / "train_data.pickle", processed_dir / "train.pt")
    build_processed_split(cleaned_dir / "val_data.pickle", processed_dir / "val.pt")
    build_processed_split(cleaned_dir / "test_data.pickle", processed_dir / "test.pt")
    train_reference_smiles = build_train_reference_smiles(
        processed_dir / "train.pt",
        cache_dir / "train_reference_smiles.pkl",
    )
    assert len(train_reference_smiles) == 2

    dataset = GeomDrugsDataset(tmp_path, split="train")
    assert len(dataset) == 2
    batch = collate_geom_drugs([dataset[0], dataset[1]])
    assert batch.coords.shape[0] == 2
    assert batch.atom_types.shape == batch.charges.shape
    assert batch.node_features().shape[-1] > 0


def test_build_processed_split_rejects_unsupported_charge(tmp_path):
    cleaned_path = tmp_path / "train_data.pickle"
    rw = Chem.RWMol()
    atom = Chem.Atom("N")
    atom.SetFormalCharge(4)
    rw.AddAtom(atom)
    mol = rw.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conf)
    with cleaned_path.open("wb") as handle:
        pickle.dump([("N", [mol])], handle)

    with pytest.raises(ValueError, match="Unsupported GEOM-Drugs formal charge"):
        build_processed_split(cleaned_path, tmp_path / "train.pt")


def test_prepare_geom_drugs_dataset_builds_processed_files_from_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _write_raw_split(raw_dir / "train_data.pickle", [("C", [_embedded_molecule("C")])])
    _write_raw_split(raw_dir / "val_data.pickle", [("O", [_embedded_molecule("O")])])
    _write_raw_split(raw_dir / "test_data.pickle", [("[NH4+]", [_embedded_molecule("[NH4+]")])])

    report = prepare_geom_drugs_dataset(tmp_path)

    assert report["processed_stats"]["train"]["molecules"] == 1
    assert (tmp_path / "processed" / "train.pt").is_file()
    assert (tmp_path / "processed" / "val.pt").is_file()
    assert (tmp_path / "processed" / "test.pt").is_file()
    assert (tmp_path / "cache" / "train_reference_smiles.pkl").is_file()


def test_geom_drugs_padded_batch_sampler_respects_caps():
    num_atoms = torch.tensor([12, 13, 14, 48, 49, 50], dtype=torch.long)
    sampler = GeomDrugsPaddedBatchSampler(
        num_atoms,
        max_examples_per_batch=4,
        max_padded_atoms_per_batch=100,
        max_attention_cost_per_batch=2500,
        shuffle=False,
    )
    batches = list(sampler)
    assert batches
    for batch in batches:
        batch_atoms = [int(num_atoms[idx]) for idx in batch]
        assert len(batch) <= 4
        assert max(batch_atoms) * len(batch) <= 100
        assert max(batch_atoms) * max(batch_atoms) * len(batch) <= 2500
