import json
import math

import pytest
import torch

pytest.importorskip("rdkit")

from matterformer.evaluators.geom_drugs import evaluate_generated_geom_drugs, load_generated_samples


def _tetrahedral_shell(bond_length: float):
    scale = bond_length / math.sqrt(3.0)
    return [
        (scale, scale, scale),
        (-scale, -scale, scale),
        (-scale, scale, -scale),
        (scale, -scale, -scale),
    ]


def _geom_generated_samples(include_invalid: bool = True):
    methane = (
        torch.tensor([(0.0, 0.0, 0.0), *_tetrahedral_shell(1.09)], dtype=torch.float32),
        torch.tensor([2, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([0, 0, 0, 0, 0], dtype=torch.long),
    )
    ammonium = (
        torch.tensor([(0.0, 0.0, 0.0), *_tetrahedral_shell(1.02)], dtype=torch.float32),
        torch.tensor([3, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([1, 0, 0, 0, 0], dtype=torch.long),
    )
    if not include_invalid:
        return [methane, ammonium]
    disconnected = (
        torch.tensor([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)], dtype=torch.float32),
        torch.tensor([2, 4], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
    )
    return [methane, ammonium, disconnected]


def test_geom_drugs_evaluator_matches_reference_values():
    report = evaluate_generated_geom_drugs(
        _geom_generated_samples(include_invalid=True),
        train_reference_smiles=set(),
    )

    assert report["molecule_count"] == 3
    assert set(report) >= {"raw_metrics", "corrected_metrics", "corrected_minus_raw"}

    assert report["raw_metrics"]["validity"] == pytest.approx(2.0 / 3.0)
    assert report["raw_metrics"]["molecule_stability"] == pytest.approx(2.0 / 3.0)
    assert report["raw_metrics"]["atom_stability"] == pytest.approx(5.0 / 6.0)
    assert report["corrected_metrics"]["validity"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert report["corrected_metrics"]["molecule_stability"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert report["corrected_metrics"]["atom_stability"] == pytest.approx(5.0 / 6.0)
    assert report["raw_metrics"]["uniqueness"] == pytest.approx(1.0)
    assert report["corrected_metrics"]["uniqueness"] == pytest.approx(1.0)
    assert report["raw_metrics"]["novelty"] == pytest.approx(1.0)
    assert report["corrected_metrics"]["novelty"] == pytest.approx(1.0)


def test_geom_drugs_evaluator_reports_valid_vs_invalid_cases():
    valid_report = evaluate_generated_geom_drugs(
        _geom_generated_samples(include_invalid=False),
        train_reference_smiles=set(),
    )
    mixed_report = evaluate_generated_geom_drugs(
        _geom_generated_samples(include_invalid=True),
        train_reference_smiles=set(),
    )

    assert valid_report["raw_metrics"]["validity"] == pytest.approx(1.0)
    assert valid_report["corrected_metrics"]["validity"] == pytest.approx(1.0)
    assert valid_report["raw_metrics"]["molecule_stability"] == pytest.approx(1.0)
    assert valid_report["corrected_metrics"]["molecule_stability"] == pytest.approx(1.0)

    assert mixed_report["raw_metrics"]["validity"] < 1.0
    assert mixed_report["corrected_metrics"]["validity"] < 1.0
    assert mixed_report["raw_metrics"]["molecule_stability"] < 1.0
    assert mixed_report["corrected_metrics"]["molecule_stability"] < 1.0


def test_load_generated_samples_supports_dict_records_with_one_hot(tmp_path):
    samples = [
        {
            "pos": [[0.0, 0.0, 0.0], *_tetrahedral_shell(1.09)],
            "x": [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            "charges": [
                [0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
            ],
        }
    ]
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(samples))

    loaded = load_generated_samples(path)
    assert len(loaded) == 1
    positions, atom_types, charges = loaded[0]
    assert positions.shape == (5, 3)
    assert atom_types.tolist() == [2, 0, 0, 0, 0]
    assert charges.tolist() == [1, 0, 0, 0, 0]
