import math

import pytest
import torch

from matterformer.data import QM9Batch, QM9NumAtomsSampler
from matterformer.metrics import check_stability, evaluate_generated_qm9, sample_and_evaluate_qm9
from matterformer.models import QM9EDMModel, QM9RegressionModel
from matterformer.tasks import EDMLoss, EDMPreconditioner, decode_atom_types, edm_sampler


def _dummy_batch() -> QM9Batch:
    atom_types = torch.tensor(
        [
            [0, 1, 2, 5],
            [1, 3, 5, 5],
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 0.8, -0.1], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    pad_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
        ]
    )
    return QM9Batch(
        atom_types=atom_types,
        coords=coords,
        pad_mask=pad_mask,
        num_atoms=torch.tensor([3, 2], dtype=torch.long),
        targets=torch.tensor([[0.1], [-0.2]], dtype=torch.float32),
        smiles=["a", "b"],
        indices=torch.tensor([0, 1], dtype=torch.long),
    )


@pytest.mark.parametrize("simplicial_geom_mode", ["none", "factorized", "angle_residual"])
def test_qm9_regression_model_forward_backward(simplicial_geom_mode):
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9RegressionModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        simplicial_geom_mode=simplicial_geom_mode,
    )
    prediction = model(batch.atom_types, batch.coords, batch.pad_mask)
    assert prediction.shape == (2,)
    loss = (prediction - batch.targets[:, 0]).pow(2).mean()
    loss.backward()


def test_qm9_edm_loss_and_sampling_smoke():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(d_model=32, n_heads=4, n_layers=2)
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    assert diagnostics["log_sigma_over_4"].shape == (2,)
    loss.backward()

    atom_features, coords, pad_mask = edm_sampler(
        net,
        num_atoms=torch.tensor([3, 4]),
        num_steps=3,
        sigma_max=1.0,
    )
    atom_types = decode_atom_types(atom_features, pad_mask)
    assert atom_types.shape == (2, 4)
    assert torch.allclose(
        coords[0, :3].mean(dim=0),
        torch.zeros(3),
        atol=1e-4,
    )
    molecule_stable, stable_atoms, total_atoms = check_stability(coords[0, :3], atom_types[0, :3])
    assert isinstance(molecule_stable, bool)
    assert stable_atoms <= total_atoms


def test_qm9_generation_metrics_without_rdkit_smoke():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(d_model=32, n_heads=4, n_layers=2)
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    _ = criterion(net, batch)

    num_atoms_sampler = QM9NumAtomsSampler([2, 3, 4])
    metrics = sample_and_evaluate_qm9(
        net,
        num_atoms_sampler,
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
        rdkit_metrics=None,
    )
    assert 0.0 <= metrics["atom_stability"] <= 1.0
    assert 0.0 <= metrics["molecule_stability"] <= 1.0
    assert math.isnan(metrics["validity"])
    assert math.isnan(metrics["uniqueness"])
    assert metrics["rdkit_available"] == 0.0

    atom_features, coords, pad_mask = edm_sampler(
        net,
        num_atoms=torch.tensor([3]),
        num_steps=3,
        sigma_max=1.0,
    )
    atom_types = decode_atom_types(atom_features, pad_mask)
    direct_metrics = evaluate_generated_qm9(
        [(coords[0, :3].detach().cpu(), atom_types[0, :3].detach().cpu())],
        rdkit_metrics=None,
    )
    assert 0.0 <= direct_metrics["atom_stability"] <= 1.0
