from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from matterformer.data.omol_fairchem import omol_tensors_to_atomic_data, repad_flat_vectors


DEFAULT_ALLSCAIP_DIRECT_CONFIG: dict[str, Any] = {
    "regress_forces": True,
    "direct_forces": True,
    "regress_stress": False,
    "hidden_size": 128,
    "num_layers": 5,
    "activation": "gelu",
    "use_compile": False,
    "use_padding": True,
    "use_residual_scaling": True,
    "use_node_path": True,
    "dataset_list": ["omol"],
    "single_system_no_padding": False,
    "max_num_elements": 120,
    "max_atoms": 12000,
    "max_batch_size": 128,
    "max_radius": 6.0,
    "knn_k": 30,
    "knn_pad_size": 30,
    "knn_soft": True,
    "knn_sigmoid_scale": 0.2,
    "knn_lse_scale": 0.1,
    "knn_use_low_mem": True,
    "distance_function": "gaussian",
    "use_envelope": True,
    "use_chunked_graph": False,
    "graph_chunk_size": 512,
    "preprocess_on_cpu": False,
    "atten_name": "memory_efficient",
    "atten_num_heads": 2,
    "node_direction_expansion_size": 10,
    "edge_direction_expansion_size": 6,
    "edge_distance_expansion_size": 512,
    "output_hidden_layer_multiplier": 2,
    "ffn_hidden_layer_multiplier": 2,
    "attn_num_freq": 32,
    "freequency_list": [20, 10, 4, 10, 20],
    "energy_reduce": "sum",
    "use_freq_mask": True,
    "use_sincx_mask": True,
    "normalization": "rmsnorm",
    "mlp_dropout": 0.0,
    "atten_dropout": 0.0,
}


def _frequency_list_for_head_dim(head_dim: int) -> list[int]:
    if int(head_dim) == 64:
        return [20, 10, 4, 10, 20]
    return [int(head_dim)]


def build_allscaip_direct_config(
    config: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_ALLSCAIP_DIRECT_CONFIG)
    if config:
        merged.update(dict(config))
    merged.update({key: value for key, value in overrides.items() if value is not None})
    merged["regress_forces"] = True
    merged["direct_forces"] = True
    merged["regress_stress"] = False
    if bool(merged.get("use_compile", False)):
        merged["use_padding"] = True

    hidden_size = int(merged["hidden_size"])
    num_heads = int(merged["atten_num_heads"])
    if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError("AllScAIP requires hidden_size to be a positive multiple of atten_num_heads")
    head_dim = hidden_size // num_heads
    if "freequency_list" not in (config or {}) and overrides.get("freequency_list") is None:
        merged["freequency_list"] = _frequency_list_for_head_dim(head_dim)
    if sum(int(value) for value in merged["freequency_list"]) != head_dim:
        raise ValueError(
            "AllScAIP freequency_list must sum to hidden_size / atten_num_heads "
            f"({head_dim}), got {merged['freequency_list']}"
        )
    if int(merged["max_atoms"]) <= 0:
        raise ValueError("AllScAIP max_atoms must be positive")
    if int(merged["max_batch_size"]) <= 0:
        raise ValueError("AllScAIP max_batch_size must be positive")
    return merged


def _require_allscaip_classes() -> tuple[Any, Any, Any]:
    try:
        from fairchem.core.models.allscaip.AllScAIP import (
            AllScAIPBackbone,
            AllScAIPDirectForceHead,
            AllScAIPEnergyHead,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The AllScAIP OMol backend requires FairChem with AllScAIP available. "
            "Set FAIRCHEM_SRC=/home/thadziv/GitHub/fairchem or install fairchem-core "
            "with the AllScAIP dependencies in PYTHONPATH."
        ) from exc
    return AllScAIPBackbone, AllScAIPEnergyHead, AllScAIPDirectForceHead


class MatterformerAllScAIPDirectForceField(nn.Module):
    """FairChem AllScAIP direct-force model behind the Matterformer OMol interface."""

    stream_type = "allscaip_direct"

    def __init__(
        self,
        allscaip_config: dict[str, Any] | None = None,
        **config_overrides: Any,
    ) -> None:
        super().__init__()
        self.allscaip_config = build_allscaip_direct_config(allscaip_config, **config_overrides)
        AllScAIPBackbone, AllScAIPEnergyHead, AllScAIPDirectForceHead = _require_allscaip_classes()
        self.backbone = AllScAIPBackbone(**self.allscaip_config)
        self.energy_head = AllScAIPEnergyHead(self.backbone)
        self.force_head = AllScAIPDirectForceHead(self.backbone)

    def collect_sg_diagnostics(self) -> dict[str, float]:
        return {}

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        charge: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        data = omol_tensors_to_atomic_data(
            atomic_numbers,
            coords,
            pad_mask,
            charge=charge,
            spin=spin,
        )
        emb = self.backbone(data)
        energy = self.energy_head(data, emb)["energy"].view(atomic_numbers.shape[0])
        flat_forces = self.force_head(data, emb)["forces"]
        forces = repad_flat_vectors(flat_forces, pad_mask)
        return {"energy": energy.to(dtype=coords.dtype), "forces": forces.to(dtype=coords.dtype)}
