import torch

from matterformer.data import SyntheticOMolDataset, collate_omol
from matterformer.geometry.cache import flatten_padded_geometry_cache
from matterformer.models import MatterformerOMolForceField
from matterformer.models.platonic import PLATONIC_GROUPS
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


def _tetra_config(attention_backend: str | None = None, attention_bias: dict | None = None):
    tetra = {
        "group": "tetrahedron",
        "group_order": 12,
        "heads_per_frame": 1,
        "rope_sigma": 1.0,
        "learned_freqs": True,
        "use_key": False,
        "rope_on_values": True,
        "ffn_mult": 2,
    }
    if attention_backend is not None:
        tetra["attention_backend"] = attention_backend
    if attention_bias is not None:
        tetra["attention_bias"] = attention_bias
    return {
        "num_blocks": 1,
        "stream_type": "tetra",
        "block_mix": [0, 1, 0],
        "scalar_dim": 24,
        "tetra_dim_per_frame": 2,
        "tetra": tetra,
        "readout": {"kind": "platonic_ffn"},
    }


def _tetra_plus_sg_config():
    cfg = _tetra_config()
    cfg.update(
        {
            "num_blocks": 2,
            "order_policy": "explicit",
            "explicit_orders": [["tetra"], ["simplicial", "tetra"]],
            "simplicial": {
                "target": "group",
                "k_neighbors": 2,
                "num_heads": 1,
                "head_dim": 2,
                "projection_mode": "shared_frame",
                "layer_scale_init_value": 0.1,
                "bias": {
                    "kind": "spherical_low_rank",
                    "representation": "compact",
                    "angle_rank": 4,
                    "channels_by_l": {"0": 1, "1": 1},
                    "radial_basis_dim": 4,
                    "hidden_dim": 8,
                    "use_radial_uv": True,
                    "use_angle": True,
                },
                "message": {"enabled": False},
                "geometry": {"builder": "dense"},
                "kernel": {"backend": "torch", "strict": False},
            },
        }
    )
    return cfg


def _tetra_plus_shared_s_config():
    cfg = _tetra_plus_sg_config()
    cfg["simplicial"] = {
        **cfg["simplicial"],
        "target": "shared_frame",
        "kernel": {"backend": "torch", "strict": False},
    }
    cfg["simplicial"].pop("projection_mode", None)
    return cfg


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


def test_omol_irrep_tetra_readout_flat_matches_padded_runtime():
    batch = _batch()
    for scalar_input in ("rho1", "invariants"):
        torch.manual_seed(17)
        padded = MatterformerOMolForceField(
            d_model=24,
            n_heads=4,
            n_layers=1,
            hybrid_config=_tetra_config(),
            chgspin_mode="add",
            pair_hidden_dim=24,
            pair_n_rbf=8,
            readout_head_mode="platonic",
            tetra_readout_mode="irrep",
            tetra_irrep_scalar_input=scalar_input,
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
            tetra_readout_mode="irrep",
            tetra_irrep_scalar_input=scalar_input,
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


def test_omol_irrep_tetra_readout_uses_scalar_and_vector_irreps():
    torch.manual_seed(18)
    model = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(),
        chgspin_mode="off",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        tetra_readout_mode="irrep",
    )
    model.eval()
    group = PLATONIC_GROUPS["tetrahedron"]
    group_features = torch.randn(2, 4, group.G, 2)
    pad_mask = torch.zeros(2, 4, dtype=torch.bool)

    energy, forces = model._irrep_tetra_readout(group_features, pad_mask)
    permutation = group.cayley_table[5, :]
    energy_perm, forces_perm = model._irrep_tetra_readout(group_features[:, :, permutation], pad_mask)
    expected_forces = torch.einsum("ji,bnj->bni", group.elements[5], forces)

    torch.testing.assert_close(energy_perm, energy, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(forces_perm, expected_forces, atol=1e-5, rtol=1e-5)


def test_omol_tetra_platonic_and_irrep_readouts_are_rotation_equivariant():
    batch = _batch()
    rotations = PLATONIC_GROUPS["tetrahedron"].elements.to(dtype=batch.coords.dtype)
    valid = ~batch.pad_mask

    for readout_mode in ("platonic", "irrep"):
        scalar_inputs = ("rho1", "invariants") if readout_mode == "irrep" else ("rho1",)
        for scalar_input in scalar_inputs:
            for runtime_mode in ("padded", "internal_flat_tetra"):
                torch.manual_seed(210)
                model = MatterformerOMolForceField(
                    d_model=24,
                    n_heads=4,
                    n_layers=1,
                    hybrid_config=_tetra_config(),
                    chgspin_mode="add",
                    pair_hidden_dim=24,
                    pair_n_rbf=8,
                    readout_head_mode="platonic",
                    tetra_readout_mode=readout_mode,
                    tetra_irrep_scalar_input=scalar_input,
                    readout_activation="sin",
                    runtime_mode=runtime_mode,
                )
                model.eval()

                with torch.no_grad():
                    base = model(
                        batch.atomic_numbers,
                        batch.coords,
                        batch.pad_mask,
                        charge=batch.charge,
                        spin=batch.spin,
                    )
                    base_energy = base["energy"]
                    base_forces = base["forces"]

                    for rotation in rotations:
                        rotated = model(
                            batch.atomic_numbers,
                            batch.coords @ rotation.T,
                            batch.pad_mask,
                            charge=batch.charge,
                            spin=batch.spin,
                        )
                        expected_forces = base_forces @ rotation.T
                        torch.testing.assert_close(rotated["energy"], base_energy, atol=5e-5, rtol=5e-5)
                        torch.testing.assert_close(
                            rotated["forces"][valid],
                            expected_forces[valid],
                            atol=5e-5,
                            rtol=5e-5,
                        )


def test_omol_tetra_pair_force_residual_flat_matches_padded_runtime():
    torch.manual_seed(130)
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
        tetra_pair_force_mode="residual",
        tetra_pair_k_neighbors=2,
        tetra_pair_feature_dim=8,
        tetra_pair_element_dim=4,
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
        tetra_pair_force_mode="residual",
        tetra_pair_k_neighbors=2,
        tetra_pair_feature_dim=8,
        tetra_pair_element_dim=4,
        runtime_mode="internal_flat_tetra",
    )
    with torch.no_grad():
        padded.tetra_pair_force_gate.fill_(1.0)
    flat.load_state_dict(padded.state_dict())
    padded.eval()
    flat.eval()

    with torch.no_grad():
        padded_out = padded(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
        flat_out = flat(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)

    valid = ~batch.pad_mask
    torch.testing.assert_close(flat_out["energy"], padded_out["energy"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(flat_out["forces"][valid], padded_out["forces"][valid], atol=1e-5, rtol=1e-5)
    force_sum = flat_out["forces"].masked_fill(batch.pad_mask[..., None], 0.0).sum(dim=1)
    torch.testing.assert_close(force_sum, torch.zeros_like(force_sum), atol=1e-5, rtol=1e-5)
    for key in (
        "pair_force/gate",
        "pair_force/direct_force_rms",
        "pair_force/residual_force_rms",
        "pair_force/residual_to_direct_rms",
        "pair_force/coeff_rms",
        "pair_force/coeff_abs_max",
        "pair_force/knn_cap_fraction",
        "pair_force/knn_cap_percent",
    ):
        assert key in flat_out["diagnostics"]
        assert torch.isfinite(flat_out["diagnostics"][key])


def test_omol_internal_flat_tetra_triton_zero_init_matches_flash_flat():
    torch.manual_seed(131)
    batch = _batch()
    flash = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config("flash"),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="internal_flat_tetra",
    )
    triton = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config("triton"),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="internal_flat_tetra",
    )
    radial = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=_tetra_config(
            "triton_radial_rbf",
            {"kind": "radial_rbf", "num_rbf": 8, "zero_init": True, "gate_init": 0.0},
        ),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="internal_flat_tetra",
    )
    triton.load_state_dict(flash.state_dict())
    radial.load_state_dict(flash.state_dict(), strict=False)
    flash.eval()
    triton.eval()
    radial.eval()

    with torch.no_grad():
        flash_out = flash(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
        triton_out = triton(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
        radial_out = radial(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)

    valid = ~batch.pad_mask
    torch.testing.assert_close(triton_out["energy"], flash_out["energy"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(triton_out["forces"][valid], flash_out["forces"][valid], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(radial_out["energy"], triton_out["energy"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(radial_out["forces"][valid], triton_out["forces"][valid], atol=1e-6, rtol=1e-6)


def test_flat_geometry_cache_has_global_neighbors():
    torch.manual_seed(14)
    batch = _batch()
    model = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=2,
        hybrid_config=_tetra_plus_sg_config(),
        chgspin_mode="off",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        runtime_mode="padded",
    )
    valid = ~batch.pad_mask
    batch_index = valid.nonzero(as_tuple=False)[:, 0]
    counts_i32 = valid.sum(dim=1).to(dtype=torch.int32)
    cu_seqlens = torch.zeros(valid.shape[0] + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(counts_i32, dim=0)
    centered = batch.coords - batch.coords.masked_fill(batch.pad_mask[..., None], 0.0).sum(dim=1, keepdim=True) / (
        valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=batch.coords.dtype)[..., None]
    )
    centered = centered.masked_fill(batch.pad_mask[..., None], 0.0)
    geom = model.trunk._build_geom_cache(coords=centered, pad_mask=batch.pad_mask, lattice=None, seq_len=batch.coords.shape[1])
    assert geom is not None
    flat_geom = flatten_padded_geometry_cache(
        geom,
        valid=valid,
        batch_index=batch_index,
        cu_seqlens=cu_seqlens,
    )
    assert flat_geom.neighbor_idx.shape[0] == int(valid.sum().item())
    src_batch = flat_geom.batch_index[:, None].expand_as(flat_geom.neighbor_idx)
    dst_batch = flat_geom.batch_index[flat_geom.neighbor_idx.long()]
    assert torch.all(dst_batch[flat_geom.neighbor_mask] == src_batch[flat_geom.neighbor_mask])


def test_omol_internal_flat_hybrid_matches_padded_runtime():
    torch.manual_seed(15)
    batch = _batch()
    padded = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=2,
        hybrid_config=_tetra_plus_sg_config(),
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
        n_layers=2,
        hybrid_config=_tetra_plus_sg_config(),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="internal_flat_hybrid",
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


def test_omol_internal_flat_hybrid_shared_s_matches_padded_runtime():
    torch.manual_seed(16)
    batch = _batch()
    padded = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=2,
        hybrid_config=_tetra_plus_shared_s_config(),
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
        n_layers=2,
        hybrid_config=_tetra_plus_shared_s_config(),
        chgspin_mode="add",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
        readout_activation="sin",
        runtime_mode="internal_flat_hybrid",
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
