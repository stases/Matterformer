#!/usr/bin/env python
"""Measure isolated numerical deltas for OMol Platonic parity toggles.

The benchmark holds a checkpoint, batch, and weights fixed, then flips only:

1. Matterformer RoPE angle/trig computation from fp32 to fp64.
2. Matterformer Platonic readout matmuls from TF32-allowed to TF32-disabled.

It is meant for one GPU and prints JSON so SLURM logs can be compared directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from matterformer.data.omol import FairChemOMolDataset, SyntheticOMolDataset, collate_omol
from matterformer.models.omol import MatterformerOMolForceField, _segment_mean_flat
from matterformer.models.platonic.rope import PlatonicRoPE
from scripts.train_omol_forcefield import load_hybrid_config


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def summarize_delta(reference: torch.Tensor, candidate: torch.Tensor, *, scale: float = 1.0) -> dict[str, float]:
    ref = reference.detach().float() * float(scale)
    cand = candidate.detach().float() * float(scale)
    diff = cand - ref
    flat_diff = diff.reshape(-1)
    flat_ref = ref.reshape(-1)
    rms_diff = torch.sqrt(torch.mean(flat_diff.square())).item()
    rms_ref = torch.sqrt(torch.mean(flat_ref.square())).item()
    abs_diff = flat_diff.abs()
    return {
        "mean_abs": float(abs_diff.mean().item()),
        "p95_abs": float(torch.quantile(abs_diff, 0.95).item()),
        "p99_abs": float(torch.quantile(abs_diff, 0.99).item()),
        "max_abs": float(abs_diff.max().item()),
        "rms_abs": float(rms_diff),
        "rms_reference": float(rms_ref),
        "rms_relative": float(rms_diff / max(rms_ref, 1.0e-30)),
    }


@contextmanager
def readout_tf32_disabled():
    if not torch.cuda.is_available():
        yield
        return
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def cos_sin_fp64(rope: PlatonicRoPE, pos: torch.Tensor, *, dtype: torch.dtype, device: torch.device | None = None):
    if pos.shape[-1] != 3:
        raise ValueError(f"pos trailing dimension must be 3, got {pos.shape[-1]}")
    target_device = device if device is not None else pos.device
    with torch.amp.autocast(device_type=pos.device.type, enabled=False):
        freqs = rope.freqs.to(device=target_device, dtype=torch.float64)
        frames = rope.group_elements.to(device=target_device, dtype=torch.float64)
        rotated_freqs = torch.einsum("gde,hfe->ghfd", frames, freqs)
        angles = torch.einsum("...d,ghfd->...ghf", pos.to(device=target_device, dtype=torch.float64), rotated_freqs)
        return torch.cos(angles).to(dtype=dtype), torch.sin(angles).to(dtype=dtype)


def patch_rope_fp64(model: torch.nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if isinstance(module, PlatonicRoPE):
            module.cos_sin = lambda pos, *, dtype, device=None, _rope=module: cos_sin_fp64(  # type: ignore[method-assign]
                _rope,
                pos,
                dtype=dtype,
                device=device,
            )
            patched += 1
    return patched


def patch_readout_tf32_disabled(model: MatterformerOMolForceField) -> None:
    orig_flat = model._platonic_tetra_readout_flat
    orig_padded = model._platonic_tetra_readout

    def flat_wrapper(*args, **kwargs):
        with readout_tf32_disabled():
            return orig_flat(*args, **kwargs)

    def padded_wrapper(*args, **kwargs):
        with readout_tf32_disabled():
            return orig_padded(*args, **kwargs)

    model._platonic_tetra_readout_flat = flat_wrapper  # type: ignore[method-assign]
    model._platonic_tetra_readout = padded_wrapper  # type: ignore[method-assign]


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
    *,
    use_ema: bool,
    rope_fp64: bool,
    readout_disable_tf32: bool,
) -> MatterformerOMolForceField:
    args = checkpoint["args"]
    hybrid_config = load_hybrid_config(args.get("hybrid_config_json"))
    model = MatterformerOMolForceField(
        max_atomic_number=int(args.get("max_atomic_number", 99)),
        d_model=int(args.get("d_model", 1920)),
        n_heads=int(args.get("n_heads", 12)),
        n_layers=int(args.get("n_layers", 16)),
        mlp_ratio=float(args.get("mlp_ratio", 4.0)),
        dropout=float(args.get("dropout", 0.0)),
        attn_dropout=float(args.get("attn_dropout", 0.0)),
        hybrid_config=hybrid_config,
        chgspin_mode=str(args.get("chgspin_mode", "add")),
        chgspin_emb_dim=args.get("chgspin_emb_dim"),
        pair_hidden_dim=int(args.get("pair_hidden_dim", 128)),
        pair_n_rbf=int(args.get("pair_n_rbf", 16)),
        pair_rbf_max=float(args.get("pair_rbf_max", 6.0)),
        tetra_pair_force_mode=str(args.get("tetra_pair_force_mode", "off")),
        tetra_pair_k_neighbors=int(args.get("tetra_pair_k_neighbors", 30)),
        tetra_pair_feature_dim=int(args.get("tetra_pair_feature_dim", 128)),
        tetra_pair_element_dim=int(args.get("tetra_pair_element_dim", 32)),
        tetra_pair_gate_init=float(args.get("tetra_pair_gate_init", 0.0)),
        tetra_pair_geometry_strict=bool(args.get("tetra_pair_geometry_strict", False)),
        force_head_mode=str(args.get("force_head_mode", "auto")),
        readout_head_mode=str(args.get("readout_head_mode", "platonic")),
        tetra_readout_mode=str(args.get("tetra_readout_mode", "platonic")),
        tetra_irrep_scalar_input=str(args.get("tetra_irrep_scalar_input", "rho1")),
        readout_activation=args.get("readout_activation"),
        runtime_mode=str(args.get("omol_runtime_mode", "internal_flat_tetra")),
        platonic_input_conditioning=bool(args.get("platonic_input_conditioning", True)),
        force_zero_mean=bool(args.get("force_zero_mean", False)),
        rope_fp64=rope_fp64,
        readout_disable_tf32=readout_disable_tf32,
    ).to(device)
    state = checkpoint["model_state"]
    if use_ema and checkpoint.get("ema_state_dict", {}).get("shadow") is not None:
        state = checkpoint["ema_state_dict"]["shadow"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_batch(args: argparse.Namespace, checkpoint_args: dict[str, Any], device: torch.device):
    if args.synthetic:
        dataset = SyntheticOMolDataset(num_samples=max(args.num_graphs, 2), seed=args.seed, min_atoms=8, max_atoms=64)
    else:
        data_path = args.data_path or checkpoint_args.get("val_data_path")
        if data_path is None:
            raise ValueError("--data-path is required when checkpoint args do not contain val_data_path")
        dataset = FairChemOMolDataset(data_path)
    samples = [dataset[i] for i in range(args.num_graphs)]
    if hasattr(dataset, "close"):
        dataset.close()
    return collate_omol(samples).to(device)


def forward_model(model: MatterformerOMolForceField, batch) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        return model(
            batch.atomic_numbers,
            batch.coords,
            batch.pad_mask,
            charge=batch.charge,
            spin=batch.spin,
        )


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model: MatterformerOMolForceField, batch, device: torch.device) -> tuple[dict[str, torch.Tensor], float]:
    cuda_sync(device)
    start = time.perf_counter()
    out = forward_model(model, batch)
    cuda_sync(device)
    return out, time.perf_counter() - start


def centered_flat_positions(batch) -> torch.Tensor:
    valid = ~batch.pad_mask
    flat_index = valid.nonzero(as_tuple=False)
    batch_index = flat_index[:, 0]
    coords_flat = batch.coords[valid]
    counts = valid.sum(dim=1)
    center = _segment_mean_flat(coords_flat, batch_index, batch.coords.shape[0], counts=counts)
    return coords_flat - center[batch_index]


def rope_factor_probe(model: MatterformerOMolForceField, batch) -> dict[str, dict[str, float]]:
    rope = next(module for module in model.modules() if isinstance(module, PlatonicRoPE))
    pos = centered_flat_positions(batch)
    cos32, sin32 = rope.cos_sin(pos, dtype=torch.float32, device=pos.device)
    cos64, sin64 = cos_sin_fp64(rope, pos, dtype=torch.float32, device=pos.device)
    return {
        "cos_fp64_minus_fp32": summarize_delta(cos32, cos64),
        "sin_fp64_minus_fp32": summarize_delta(sin32, sin64),
    }


def compare_outputs(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    normalizer_rmsd: float,
) -> dict[str, dict[str, float]]:
    scale = float(normalizer_rmsd) * 1000.0
    return {
        "energy_normalized": summarize_delta(reference["energy"], candidate["energy"]),
        "energy_meV": summarize_delta(reference["energy"], candidate["energy"], scale=scale),
        "forces_normalized": summarize_delta(reference["forces"], candidate["forces"]),
        "forces_meV_per_A": summarize_delta(reference["forces"], candidate["forces"], scale=scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--num-graphs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalizer-rmsd", type=float, default=1.433569)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    batch = load_batch(args, checkpoint["args"], device)

    base_model = build_model_from_checkpoint(
        checkpoint,
        device,
        use_ema=args.use_ema,
        rope_fp64=False,
        readout_disable_tf32=False,
    )
    rope_probe = rope_factor_probe(base_model, batch)
    baseline, baseline_s = timed_forward(base_model, batch, device)

    rope_model = build_model_from_checkpoint(
        checkpoint,
        device,
        use_ema=args.use_ema,
        rope_fp64=True,
        readout_disable_tf32=False,
    )
    rope_modules = sum(1 for module in rope_model.modules() if isinstance(module, PlatonicRoPE))
    rope_out, rope_s = timed_forward(rope_model, batch, device)

    readout_model = build_model_from_checkpoint(
        checkpoint,
        device,
        use_ema=args.use_ema,
        rope_fp64=False,
        readout_disable_tf32=True,
    )
    readout_out, readout_s = timed_forward(readout_model, batch, device)

    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint.get("global_step"),
        "use_ema": args.use_ema,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_after_setup": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "batch": {
            "num_graphs": int(batch.atomic_numbers.shape[0]),
            "num_atoms": int((~batch.pad_mask).sum().item()),
            "max_atoms": int(batch.num_atoms.max().item()),
            "mean_atoms": float(batch.num_atoms.float().mean().item()),
        },
        "timing_s": {
            "baseline": baseline_s,
            "rope_fp64": rope_s,
            "readout_tf32_disabled": readout_s,
        },
        "rope_factor_probe": rope_probe,
        "rope_fp64_vs_baseline": {
            "rope_modules": rope_modules,
            **compare_outputs(baseline, rope_out, normalizer_rmsd=args.normalizer_rmsd),
        },
        "readout_no_tf32_vs_baseline": compare_outputs(
            baseline,
            readout_out,
            normalizer_rmsd=args.normalizer_rmsd,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
