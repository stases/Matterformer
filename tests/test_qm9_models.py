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
        charges=torch.zeros_like(atom_types),
        coords=coords,
        pad_mask=pad_mask,
        num_atoms=torch.tensor([3, 2], dtype=torch.long),
        targets=torch.tensor([[0.1], [-0.2]], dtype=torch.float32),
        smiles=["a", "b"],
        indices=torch.tensor([0, 1], dtype=torch.long),
    )


@pytest.mark.parametrize("simplicial_geom_mode", ["none", "factorized", "angle_residual", "angle_low_rank"])
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
    assert model.coord_embedding is None
    assert model.coord_head_mode == "equivariant"
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    assert diagnostics["log_sigma_over_4"].shape == (2,)
    assert diagnostics["loss_weight"].shape == (2,)
    assert diagnostics["loss_weight_clamped"].shape == (2,)
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


def test_qm9_edm_mha_factorized_marginal_loss_smoke():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="mha",
        simplicial_geom_mode="factorized",
        mha_geom_bias_mode="factorized_marginal",
    )
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


@pytest.mark.parametrize("coord_embed_mode", ["fourier", "rff", "learnable_rff"])
def test_qm9_edm_coord_input_embedding_smoke(coord_embed_mode):
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        coord_embed_mode=coord_embed_mode,
        coord_n_freqs=8,
    )
    assert model.coord_embedding is not None
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


def test_qm9_edm_learnable_rff_coord_embedding_has_trainable_projection():
    torch.manual_seed(0)
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        coord_embed_mode="learnable-rff",
        coord_rff_dim=8,
    )
    assert model.coord_embed_mode == "learnable_rff"
    assert model.coord_embedding is not None
    assert model.coord_embedding.proj.requires_grad
    assert tuple(model.coord_embedding.proj.shape) == (3, 8)


def test_qm9_edm_rope_coord_mode_uses_mha_rope_without_token_coord_embedding():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="mha",
        use_geometry_bias=False,
        coord_embed_mode="rope",
    )
    assert model.coord_embed_mode == "none"
    assert model.coord_embedding is None
    assert model.mha_position_mode == "rope"
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


@pytest.mark.parametrize(
    ("noise_conditioning", "expected_concat", "expected_adaln"),
    [
        ("concat", True, False),
        ("adaln", False, True),
        ("concat,adaln", True, True),
    ],
)
def test_qm9_edm_noise_conditioning_modes(noise_conditioning, expected_concat, expected_adaln):
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="mha",
        use_geometry_bias=False,
        noise_conditioning=noise_conditioning,
    )
    assert model.concat_sigma_condition is expected_concat
    assert model.use_adaln_conditioning is expected_adaln
    assert model.trunk.blocks[0].use_adaln_conditioning is expected_adaln

    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


def test_qm9_edm_legacy_concat_flag_maps_to_adaln_only_when_disabled():
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=1,
        concat_sigma_condition=False,
    )
    assert model.noise_conditioning == ("adaln",)
    assert not model.concat_sigma_condition
    assert model.use_adaln_conditioning


def test_qm9_edm_direct_coord_head_smoke():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="mha",
        use_geometry_bias=False,
        coord_embed_mode="rff",
        coord_n_freqs=8,
        coord_head_mode="direct",
    )
    assert model.coord_head_mode == "direct"
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


def test_qm9_edm_factorized_message_low_rank_loss_smoke():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        simplicial_geom_mode="factorized",
        simplicial_message_mode="low_rank",
        simplicial_message_rank=3,
    )
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


def test_qm9_edm_simplicial_closed_rope_loss_smoke():
    torch.manual_seed(0)
    batch = _dummy_batch()
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        simplicial_geom_mode="factorized",
        simplicial_impl="torch",
        simplicial_message_mode="low_rank",
        simplicial_message_rank=3,
        simplicial_position_mode="closed_rope",
        simplicial_rope_n_freqs=2,
        simplicial_rope_gate="none",
    )
    assert model.simplicial_position_mode == "closed_rope"
    assert model.trunk.blocks[0].attn.closed_rope is not None
    net = EDMPreconditioner(model, sigma_data=1.0)
    criterion = EDMLoss(sigma_data=1.0)
    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    loss.backward()


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
