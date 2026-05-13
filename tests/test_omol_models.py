import torch

from matterformer.data import SyntheticOMolDataset, collate_omol
from matterformer.models import MatterformerOMolForceField
from matterformer.tasks import OMolDirectForceLoss, OMolElementReferences
from matterformer.utils import random_rotation_matrices


def _batch():
    dataset = SyntheticOMolDataset(num_samples=3, seed=5, min_atoms=3, max_atoms=5)
    return collate_omol([dataset[0], dataset[1]])


def _scalar_config(d_model: int = 32):
    return {
        "num_blocks": 1,
        "stream_type": "scalar",
        "block_mix": [0, 0, 1],
        "scalar_dim": d_model,
        "trivial": {
            "attention": {
                "kind": "mha",
                "num_heads": 4,
                "position_encoding": "distance_bias",
            }
        },
    }


def _tetra_config():
    return {
        "num_blocks": 1,
        "stream_type": "tetra",
        "block_mix": [0, 1, 0],
        "scalar_dim": 24,
        "tetra_dim_per_frame": 2,
        "tetra": {
            "group": "tetrahedron",
            "group_order": 12,
            "heads_per_frame": 1,
            "rope_sigma": 1.0,
            "learned_freqs": True,
            "use_key": False,
            "rope_on_values": True,
            "ffn_mult": 2,
        },
        "readout": {"kind": "platonic_ffn"},
    }


def test_omol_scalar_model_forward_backward():
    torch.manual_seed(0)
    batch = _batch()
    refs = OMolElementReferences(torch.zeros(119))
    criterion = OMolDirectForceLoss(refs, normalizer_rmsd=1.0)
    model = MatterformerOMolForceField(
        d_model=32,
        n_heads=4,
        n_layers=1,
        hybrid_config=_scalar_config(32),
        chgspin_mode="add",
        pair_hidden_dim=32,
        pair_n_rbf=8,
    )
    outputs = model(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == batch.forces.shape
    loss = criterion(outputs, batch).loss
    loss.backward()


def test_omol_tetra_model_forward_backward():
    torch.manual_seed(0)
    batch = _batch()
    refs = OMolElementReferences(torch.zeros(119))
    criterion = OMolDirectForceLoss(refs, normalizer_rmsd=1.0)
    model = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(),
        chgspin_mode="off",
        pair_hidden_dim=24,
        pair_n_rbf=8,
    )
    outputs = model(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == batch.forces.shape
    criterion(outputs, batch).loss.backward()


def test_omol_internal_flat_tetra_matches_padded_runtime():
    torch.manual_seed(12)
    batch = _batch()
    padded = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        runtime_mode="padded",
    )
    with torch.no_grad():
        padded.group_force_head[-1].weight.normal_(std=1e-3)
        padded.group_force_head[-1].bias.normal_(std=1e-3)
    flat = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        runtime_mode="internal_flat_tetra",
    )
    flat.load_state_dict(padded.state_dict())
    padded.eval()
    flat.eval()

    with torch.no_grad():
        padded_out = padded(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
        flat_out = flat(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)

    valid = ~batch.pad_mask
    torch.testing.assert_close(flat_out["energy"], padded_out["energy"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(flat_out["forces"][valid], padded_out["forces"][valid], atol=1e-5, rtol=1e-5)
    padded_force_values = flat_out["forces"].masked_select(batch.pad_mask[..., None])
    assert torch.allclose(padded_force_values, torch.zeros_like(padded_force_values))


def test_omol_platonic_tetra_readout_flat_matches_padded_runtime():
    torch.manual_seed(13)
    batch = _batch()
    padded = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="padded",
    )
    flat = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="internal_flat_tetra",
    )
    flat.load_state_dict(padded.state_dict())
    padded.eval()
    flat.eval()

    with torch.no_grad():
        padded_out = padded(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
        flat_out = flat(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)

    valid = ~batch.pad_mask
    torch.testing.assert_close(flat_out["energy"], padded_out["energy"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(flat_out["forces"][valid], padded_out["forces"][valid], atol=1e-5, rtol=1e-5)
    padded_force_values = flat_out["forces"].masked_select(batch.pad_mask[..., None])
    assert torch.allclose(padded_force_values, torch.zeros_like(padded_force_values))


def test_omol_loss_normalization_and_metrics():
    batch = _batch()
    refs = torch.zeros(119)
    refs[1] = -1.0
    refs[6] = -6.0
    refs[8] = -8.0
    criterion = OMolDirectForceLoss(OMolElementReferences(refs), normalizer_rmsd=2.0, energy_weight=10.0, force_weight=10.0)
    residual = criterion.element_references.subtract_refs(batch)
    predictions = {
        "energy": residual / 2.0,
        "forces": batch.forces / 2.0,
    }
    output = criterion(predictions, batch)
    assert torch.allclose(output.loss, torch.tensor(0.0))
    assert torch.allclose(output.diagnostics["e_mae_per_atom"], torch.tensor(0.0))
    assert torch.allclose(output.diagnostics["f_mae"], torch.tensor(0.0))


def test_scalar_direct_force_head_rotates_with_coordinates():
    torch.manual_seed(0)
    batch = _batch()
    model = MatterformerOMolForceField(
        d_model=32,
        n_heads=4,
        n_layers=1,
        hybrid_config=_scalar_config(32),
        chgspin_mode="off",
        pair_hidden_dim=32,
        pair_n_rbf=8,
    )
    with torch.no_grad():
        model.scalar_force_head[-1].weight.fill_(0.01)
        model.scalar_force_head[-1].bias.fill_(0.02)
    model.eval()
    rotations = random_rotation_matrices(batch.coords.shape[0], device=batch.coords.device, dtype=batch.coords.dtype)
    rotated_coords = torch.einsum("bij,bnj->bni", rotations, batch.coords).masked_fill(batch.pad_mask[..., None], 0.0)
    out = model(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
    out_rot = model(batch.atomic_numbers, rotated_coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
    expected_forces = torch.einsum("bij,bnj->bni", rotations, out["forces"]).masked_fill(batch.pad_mask[..., None], 0.0)
    assert torch.allclose(out["energy"], out_rot["energy"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(expected_forces, out_rot["forces"], atol=1e-4, rtol=1e-4)


def test_omol_optimizer_step_smoke():
    torch.manual_seed(0)
    batch = _batch()
    model = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_scalar_config(24),
        chgspin_mode="off",
        pair_hidden_dim=24,
        pair_n_rbf=8,
    )
    criterion = OMolDirectForceLoss(OMolElementReferences(torch.zeros(119)), normalizer_rmsd=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = criterion(model(batch.atomic_numbers, batch.coords, batch.pad_mask), batch)
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    optimizer.step()
