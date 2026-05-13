from __future__ import annotations

import importlib

import pytest
import torch

from matterformer.data.omol_fairchem import repad_flat_vectors
from matterformer.models.allscaip import MatterformerAllScAIPDirectForceField, build_allscaip_direct_config


def _allscaip_available() -> bool:
    try:
        importlib.import_module("fairchem.core.models.allscaip.AllScAIP")
    except ImportError:
        return False
    return True


def test_allscaip_config_derives_frequency_list_for_nonstandard_head_dim() -> None:
    config = build_allscaip_direct_config(hidden_size=32, atten_num_heads=2, max_atoms=8, max_batch_size=2)
    assert config["direct_forces"] is True
    assert config["regress_forces"] is True
    assert config["regress_stress"] is False
    assert config["freequency_list"] == [16]


def test_repad_flat_vectors_restores_padded_shape() -> None:
    pad_mask = torch.tensor([[False, False, True], [False, True, True]])
    flat = torch.arange(9, dtype=torch.float32).view(3, 3)
    padded = repad_flat_vectors(flat, pad_mask)
    assert padded.shape == (2, 3, 3)
    assert torch.equal(padded[~pad_mask], flat)
    assert torch.count_nonzero(padded[pad_mask]) == 0


def test_allscaip_backend_has_lazy_dependency_error() -> None:
    if _allscaip_available():
        pytest.skip("FairChem AllScAIP is available in this environment")
    with pytest.raises(RuntimeError, match="AllScAIP OMol backend requires FairChem"):
        MatterformerAllScAIPDirectForceField(hidden_size=8, atten_num_heads=2, num_layers=1, max_atoms=8, max_batch_size=2)


@pytest.mark.skipif(not _allscaip_available(), reason="FairChem AllScAIP is not installed")
def test_allscaip_tiny_forward_if_available() -> None:
    model = MatterformerAllScAIPDirectForceField(
        hidden_size=8,
        atten_num_heads=2,
        num_layers=1,
        freequency_list=[2, 2],
        max_atoms=8,
        max_batch_size=2,
        max_radius=6.0,
        knn_k=4,
        knn_pad_size=4,
        use_compile=False,
        use_padding=True,
    )
    atomic_numbers = torch.tensor([[1, 6, 0], [8, 0, 0]])
    coords = torch.randn(2, 3, 3)
    pad_mask = atomic_numbers == 0
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    output = model(
        atomic_numbers,
        coords,
        pad_mask,
        charge=torch.zeros(2, dtype=torch.long),
        spin=torch.zeros(2, dtype=torch.long),
    )
    assert output["energy"].shape == (2,)
    assert output["forces"].shape == (2, 3, 3)
