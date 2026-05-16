#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn

from matterformer.models import MatterformerOMolForceField
from matterformer.models.platonic.layers import PlatonicAttention
from matterformer.models.platonic.linear import PlatonicLinear


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _variant_config(base: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    tetra = cfg.setdefault("tetra", {})
    if variant == "recompute":
        tetra["linear_backend"] = "spatial"
        tetra["attention_linear_backend"] = "spatial"
        tetra["ffn_linear_backend"] = "spatial"
        tetra["rope_cache"] = False
        tetra["constant_key_fastpath"] = False
        tetra["fused_qv"] = False
    elif variant == "attention_fast":
        tetra["linear_backend"] = "spatial"
        tetra["attention_linear_backend"] = "spatial"
        tetra["ffn_linear_backend"] = "spatial"
        tetra["rope_cache"] = True
        tetra["constant_key_fastpath"] = True
        tetra["fused_qv"] = True
    elif variant == "mixed":
        tetra["linear_backend"] = "spatial"
        tetra["attention_linear_backend"] = "spatial"
        tetra["ffn_linear_backend"] = "fourier_direct"
        tetra["rope_cache"] = True
        tetra["constant_key_fastpath"] = True
        tetra["fused_qv"] = True
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    return cfg


def _make_counts(total_atoms: int, num_graphs: int, max_atoms_per_graph: int, device: torch.device) -> torch.Tensor:
    if total_atoms < num_graphs:
        raise ValueError("total_atoms must be >= num_graphs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    raw = torch.rand(num_graphs, generator=generator).square().clamp_min(1.0e-4)
    raw = raw / raw.sum()
    counts = torch.clamp((raw * float(total_atoms)).round().long(), min=1, max=max_atoms_per_graph)
    diff = int(total_atoms - counts.sum().item())
    while diff != 0:
        if diff > 0:
            room = torch.nonzero(counts < max_atoms_per_graph, as_tuple=False).flatten()
            if room.numel() == 0:
                raise ValueError("Cannot fit total_atoms with requested num_graphs/max_atoms_per_graph")
            take = min(diff, int(room.numel()))
            counts[room[:take]] += 1
            diff -= take
        else:
            removable = torch.nonzero(counts > 1, as_tuple=False).flatten()
            take = min(-diff, int(removable.numel()))
            counts[removable[:take]] -= 1
            diff += take
    return counts.to(device=device, dtype=torch.long)


def _make_batch(
    *,
    total_atoms: int,
    num_graphs: int,
    max_atoms_per_graph: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    counts = _make_counts(total_atoms, num_graphs, max_atoms_per_graph, device)
    batch_size = int(counts.numel())
    max_slots = int(counts.max().item())
    generator = torch.Generator(device=device)
    generator.manual_seed(456)
    atomic_numbers = torch.zeros(batch_size, max_slots, dtype=torch.long, device=device)
    coords = torch.zeros(batch_size, max_slots, 3, dtype=torch.float32, device=device)
    pad_mask = torch.ones(batch_size, max_slots, dtype=torch.bool, device=device)
    for index, count_t in enumerate(counts.tolist()):
        count = int(count_t)
        atomic_numbers[index, :count] = torch.randint(1, 31, (count,), device=device, generator=generator)
        coords[index, :count] = torch.randn(count, 3, device=device, generator=generator)
        pad_mask[index, :count] = False
    charge = torch.zeros(batch_size, dtype=torch.long, device=device)
    spin = torch.zeros(batch_size, dtype=torch.long, device=device)
    stats = {
        "graphs": float(batch_size),
        "atoms": float(int(counts.sum().item())),
        "max_atoms": float(max_slots),
        "sum_n2": float(counts.square().sum().item()),
        "padded_slots": float(batch_size * max_slots),
    }
    return atomic_numbers, coords, pad_mask, charge, spin, stats


def _copy_matching_state(target: nn.Module, source: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            copied[key] = value
    target.load_state_dict(copied, strict=False)


def _copy_equivalent_platonic_modules(target: nn.Module, source: nn.Module) -> None:
    source_modules = dict(source.named_modules())
    for name, target_module in target.named_modules():
        source_module = source_modules.get(name)
        if isinstance(target_module, PlatonicAttention) and isinstance(source_module, PlatonicAttention):
            if target_module.qv_proj is not None and source_module.q_proj is not None and source_module.v_proj is not None:
                target_module.set_fused_qv_from_separate_(source_module.q_proj, source_module.v_proj)
        if isinstance(target_module, PlatonicLinear) and isinstance(source_module, PlatonicLinear):
            if target_module.linear_backend == "fourier_direct" and source_module.kernel is not None:
                target_module.set_spatial_parameters_(source_module.kernel, source_module.bias)


def _make_model(args: argparse.Namespace, cfg: dict[str, Any]) -> MatterformerOMolForceField:
    return MatterformerOMolForceField(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        attn_dropout=0.0,
        hybrid_config=cfg,
        chgspin_mode=args.chgspin_mode,
        chgspin_emb_dim=args.chgspin_emb_dim,
        pair_hidden_dim=args.pair_hidden_dim,
        pair_n_rbf=args.pair_n_rbf,
        pair_rbf_max=args.pair_rbf_max,
        force_head_mode="auto",
        readout_head_mode="platonic",
        readout_activation=args.readout_activation,
        runtime_mode="internal_flat_tetra",
    )


def _compile_model(model: MatterformerOMolForceField, compile_scope: str) -> MatterformerOMolForceField:
    if compile_scope == "none":
        return model
    if compile_scope == "model":
        return torch.compile(model, mode="default")  # type: ignore[return-value]
    if compile_scope == "trunk_flat":
        model.trunk.forward_flat_tetra = torch.compile(model.trunk.forward_flat_tetra, mode="default")  # type: ignore[method-assign]
        return model
    raise ValueError(f"Unknown compile_scope {compile_scope!r}")


def _loss_from_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["energy"].float().square().mean() + outputs["forces"].float().square().mean()


def _forward(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    atomic_numbers, coords, pad_mask, charge, spin = batch
    return model(atomic_numbers, coords, pad_mask, charge=charge, spin=spin)


def _time_forward_backward(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _loss_from_outputs(_forward(model, batch)).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        _loss_from_outputs(_forward(model, batch)).backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    for parameter in params:
        parameter.grad = None
    return 1000.0 * elapsed / float(repeats), torch.cuda.max_memory_allocated() / 1024**3


def _time_forward(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _forward(model, batch)
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / float(repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full OMol tetra model with mixed Platonic backends.")
    parser.add_argument("--hybrid-config-json", default="configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale.json")
    parser.add_argument("--d-model", type=int, default=1920)
    parser.add_argument("--n-heads", type=int, default=60)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--chgspin-mode", default="add", choices=["off", "add", "concat"])
    parser.add_argument("--chgspin-emb-dim", type=int, default=160)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--pair-n-rbf", type=int, default=16)
    parser.add_argument("--pair-rbf-max", type=float, default=6.0)
    parser.add_argument("--readout-activation", default="sin", choices=["gelu", "silu", "relu", "mish", "sin"])
    parser.add_argument("--total-atoms", type=int, default=12000)
    parser.add_argument("--num-graphs", type=int, default=218)
    parser.add_argument("--max-atoms-per-graph", type=int, default=220)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile-scope", choices=["none", "model", "trunk_flat"], default="trunk_flat")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--variants", nargs="+", default=["recompute", "attention_fast", "mixed"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(0)
    device = torch.device("cuda")

    base_cfg = _load_json(args.hybrid_config_json)
    batch_tensors = _make_batch(
        total_atoms=args.total_atoms,
        num_graphs=args.num_graphs,
        max_atoms_per_graph=args.max_atoms_per_graph,
        device=device,
    )
    batch = batch_tensors[:5]
    batch_stats = batch_tensors[5]

    base_model = _make_model(args, _variant_config(base_cfg, "recompute")).to(device)
    base_model.eval()
    with torch.no_grad():
        base_outputs = _forward(base_model, batch)

    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.version.cuda,
        "device",
        torch.cuda.get_device_name(0),
        "compile_scope",
        args.compile_scope,
        "matmul_precision",
        args.matmul_precision,
        "allow_tf32",
        torch.backends.cuda.matmul.allow_tf32,
    )
    print("batch " + json.dumps(batch_stats, sort_keys=True))

    results: dict[str, dict[str, float]] = {}
    for variant in args.variants:
        cfg = _variant_config(base_cfg, variant)
        model = _make_model(args, cfg).to(device)
        model.eval()
        if variant == "recompute":
            model.load_state_dict(base_model.state_dict())
        else:
            _copy_matching_state(model, base_model)
            _copy_equivalent_platonic_modules(model, base_model)
        with torch.no_grad():
            outputs = _forward(model, batch)
        energy_error = (outputs["energy"] - base_outputs["energy"]).abs()
        force_error = (outputs["forces"] - base_outputs["forces"]).abs()
        parity = {
            "energy_max_abs": float(energy_error.max().item()),
            "energy_mean_abs": float(energy_error.mean().item()),
            "force_max_abs": float(force_error.max().item()),
            "force_mean_abs": float(force_error.mean().item()),
        }

        model = _compile_model(model, args.compile_scope)
        forward_ms = _time_forward(model, batch, warmup=max(1, args.warmup), repeats=args.repeats)
        fwd_bwd_ms, mem_gb = _time_forward_backward(model, batch, warmup=args.warmup, repeats=args.repeats)
        result = {
            **parity,
            "forward_ms": forward_ms,
            "fwd_bwd_ms": fwd_bwd_ms,
            "max_mem_gb": mem_gb,
        }
        results[variant] = result
        print("result " + json.dumps({"variant": variant, **result}, sort_keys=True))

    if "recompute" in results:
        base = results["recompute"]["fwd_bwd_ms"]
        for variant, result in results.items():
            result["fwd_bwd_speedup_vs_recompute"] = base / result["fwd_bwd_ms"]
    if "attention_fast" in results and "mixed" in results:
        results["mixed"]["fwd_bwd_speedup_vs_attention_fast"] = (
            results["attention_fast"]["fwd_bwd_ms"] / results["mixed"]["fwd_bwd_ms"]
        )
    print("summary " + json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
