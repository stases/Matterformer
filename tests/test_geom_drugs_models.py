import pytest
import torch

pytest.importorskip("rdkit")

from matterformer.data import (
    GEOM_DRUGS_ATOM_PAD_TOKEN,
    GEOM_DRUGS_CHARGE_PAD_TOKEN,
    GEOM_DRUGS_CHARGE_TO_INDEX,
    GeomDrugsBatch,
    GeomDrugsNumAtomsSampler,
)
from matterformer.evaluators.geom_drugs import sample_and_evaluate_geom_drugs
from matterformer.metrics import build_rdkit_metrics, check_stability, sample_and_evaluate_qm9
from matterformer.models import GeomDrugsEDMModel
from matterformer.tasks import EDMPreconditioner, GeomDrugsEDMLoss, decode_geom_drugs_types_and_charges, edm_sampler


def _dummy_geom_batch() -> GeomDrugsBatch:
    atom_types = torch.tensor(
        [
            [2, 0, 0, 0, 0, GEOM_DRUGS_ATOM_PAD_TOKEN],
            [3, 0, 0, 0, 0, GEOM_DRUGS_ATOM_PAD_TOKEN],
        ],
        dtype=torch.long,
    )
    charges = torch.tensor(
        [
            [GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_PAD_TOKEN],
            [GEOM_DRUGS_CHARGE_TO_INDEX[1], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_TO_INDEX[0], GEOM_DRUGS_CHARGE_PAD_TOKEN],
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.59, 0.59, 0.59],
                [-0.59, -0.59, 0.59],
                [-0.59, 0.59, -0.59],
                [0.59, -0.59, -0.59],
                [0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    pad_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, False, False, True],
        ]
    )
    return GeomDrugsBatch(
        atom_types=atom_types,
        charges=charges,
        coords=coords,
        pad_mask=pad_mask,
        num_atoms=torch.tensor([5, 5], dtype=torch.long),
        smiles=["C", "[NH4+]"],
        indices=torch.tensor([0, 1], dtype=torch.long),
    )


def test_geom_drugs_edm_loss_and_sampling_smoke():
    torch.manual_seed(0)
    batch = _dummy_geom_batch()
    model = GeomDrugsEDMModel(d_model=32, n_heads=4, n_layers=2)
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = GeomDrugsEDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    assert diagnostics["node_loss"].shape == (2,)
    loss.backward()

    node_features, coords, pad_mask = edm_sampler(
        net,
        num_atoms=torch.tensor([4, 5]),
        atom_channels=model.atom_channels,
        num_steps=3,
        sigma_max=1.0,
    )
    atom_types, charges = decode_geom_drugs_types_and_charges(node_features, pad_mask)
    assert atom_types.shape == (2, 5)
    assert charges.shape == (2, 5)
    assert torch.allclose(coords[0, :4].mean(dim=0), torch.zeros(3), atol=1e-4)

    sampler = GeomDrugsNumAtomsSampler([4, 5, 6])
    report = sample_and_evaluate_geom_drugs(
        net,
        sampler,
        device=torch.device("cpu"),
        num_molecules=4,
        sample_batch_size=2,
        sampler_kwargs={
            "num_steps": 3,
            "sigma_min": 0.002,
            "sigma_max": 1.0,
            "rho": 7.0,
            "s_churn": 0.0,
            "s_min": 0.0,
            "s_max": float("inf"),
            "s_noise": 1.0,
        },
        train_reference_smiles=set(),
        data_root=".",
    )
    assert set(report) >= {"raw_metrics", "corrected_metrics"}
    for prefix in ("raw_metrics", "corrected_metrics"):
        assert 0.0 <= report[prefix]["atom_stability"] <= 1.0
        assert 0.0 <= report[prefix]["molecule_stability"] <= 1.0
        assert 0.0 <= report[prefix]["validity"] <= 1.0


def test_metrics_compatibility_exports_still_resolve():
    assert callable(check_stability)
    assert callable(sample_and_evaluate_qm9)
    metrics = build_rdkit_metrics(reference_smiles=["C"])
    assert metrics is None or hasattr(metrics, "evaluate")
