#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from matterformer.models import MatterformerOMolForceField
from matterformer.tasks import OMolDirectForceLoss, load_omol_element_references
from matterformer.utils import default_device, seed_everything
from scripts.train_omol_forcefield import (
    apply_rotation_augmentation,
    build_datasets,
    build_loader,
    load_hybrid_config,
    make_autocast_context,
)


NON_TRITON_NOTES = [
    "scalar MHA/RoPE: PyTorch Linear/trig/einsum plus torch scaled_dot_product_attention, not a custom Matterformer Triton kernel",
    "scalar S bias: radial/angle coefficient MLPs are PyTorch modules",
    "scalar S angle representation: compact coefficients are expanded to [B,H,N,K,R] before the scalar Triton kernel unless the config/kernel path is changed",
    "dense scalar geometry bias: any trivial layer with position_encoding != none requires dense [B,N,N] geometry",
    "direct 3D force head: PyTorch MLP, small and no dense pair materialization",
    "energy head, FFNs, LayerNorms, AdamW optimizer: standard PyTorch/cuBLAS kernels",
]


def _load_json(path_or_json: str | None) -> dict[str, Any] | None:
    if path_or_json is None:
        return None
    return load_hybrid_config(path_or_json)


def _profiler_table(prof: profile, *, sort_by: str, row_limit: int) -> str:
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    except Exception as exc:  # pragma: no cover - profiler table fields vary by PyTorch version.
        return f"Could not render profiler table sorted by {sort_by!r}: {exc}"


def _model_layer_counts(model: MatterformerOMolForceField) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in model.trunk.blocks:
        for layer in getattr(block, "sublayers", []):
            name = type(layer).__name__
            counts[name] = counts.get(name, 0) + 1
    return counts


def _print_run_summary(args: argparse.Namespace, model: MatterformerOMolForceField) -> None:
    cfg = _load_json(args.hybrid_config_json) or {}
    simplicial = dict(cfg.get("simplicial", {}))
    trivial = dict(cfg.get("trivial", {}))
    print("=== OMol profile run ===")
    print(f"device={args.device_resolved} bf16={args.bf16} matmul={args.float32_matmul_precision}")
    print(f"hybrid_config={args.hybrid_config_json}")
    print(f"stream_type={model.stream_type} force_head_mode={model.force_head_mode}")
    print(f"layer_counts={json.dumps(_model_layer_counts(model), sort_keys=True)}")
    print(
        "simplicial="
        + json.dumps(
            {
                "k_neighbors": simplicial.get("k_neighbors"),
                "geometry": simplicial.get("geometry"),
                "kernel": simplicial.get("kernel"),
                "bias": simplicial.get("bias"),
                "message": simplicial.get("message"),
            },
            sort_keys=True,
        )
    )
    print("trivial=" + json.dumps(trivial, sort_keys=True))
    print("=== expected non-Triton / non-fused regions for this profile ===")
    for note in NON_TRITON_NOTES:
        print(f"- {note}")
    print("============================================")


def _next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _train_step(
    *,
    model: MatterformerOMolForceField,
    criterion: OMolDirectForceLoss,
    optimizer: torch.optim.Optimizer,
    batch,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    with record_function("profile/h2d"):
        batch = batch.to(device)
    with record_function("profile/augmentation"):
        batch = apply_rotation_augmentation(batch, args.train_augmentation)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    with record_function("profile/forward_model_and_loss"):
        with make_autocast_context(device, args.bf16):
            predictions = model(
                batch.atomic_numbers,
                batch.coords,
                batch.pad_mask,
                charge=batch.charge,
                spin=batch.spin,
            )
            output = criterion(predictions, batch)
    with record_function("profile/backward"):
        output.loss.backward()
    with record_function("profile/optimizer"):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    valid_atoms = int((~batch.pad_mask).sum().item())
    graphs = int(batch.energy.shape[0])
    metrics = {
        "wall_ms": elapsed * 1000.0,
        "graphs": float(graphs),
        "atoms": float(valid_atoms),
        "graphs_per_sec": graphs / max(elapsed, 1e-12),
        "atoms_per_sec": valid_atoms / max(elapsed, 1e-12),
        "loss": float(output.loss.detach().item()),
        "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
    }
    if torch.cuda.is_available():
        metrics["max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        metrics["max_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    return metrics


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    device = default_device()
    args.device_resolved = str(device)
    args.hybrid_config = _load_json(args.hybrid_config_json)

    train_dataset, _, _ = build_datasets(args)
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_atoms=args.max_atoms_per_batch,
        max_edges=args.max_edges_per_batch,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
    )
    element_refs = load_omol_element_references(args.element_refs_json).to(device)
    criterion = OMolDirectForceLoss(
        element_refs,
        normalizer_rmsd=args.normalizer_rmsd,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        energy_loss=args.energy_loss,
        force_loss=args.force_loss,
    )
    model = MatterformerOMolForceField(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        hybrid_config=args.hybrid_config,
        chgspin_mode=args.chgspin_mode,
        chgspin_emb_dim=args.chgspin_emb_dim,
        pair_hidden_dim=args.pair_hidden_dim,
        pair_n_rbf=args.pair_n_rbf,
        pair_rbf_max=args.pair_rbf_max,
        force_head_mode=args.force_head_mode,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    _print_run_summary(args, model)

    loader_iter = iter(train_loader)
    warmup_metrics: list[dict[str, float]] = []
    for step in range(args.warmup_steps):
        batch, loader_iter = _next_batch(loader_iter, train_loader)
        metrics = _train_step(model=model, criterion=criterion, optimizer=optimizer, batch=batch, device=device, args=args)
        warmup_metrics.append(metrics)
        print(
            f"warmup step={step + 1:03d} wall_ms={metrics['wall_ms']:.2f} "
            f"atoms={metrics['atoms']:.0f} mem_gb={metrics.get('max_mem_allocated_gb', 0.0):.3f}"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    profiled_metrics: list[dict[str, float]] = []
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=True,
        with_stack=args.with_stack,
    ) as prof:
        for step in range(args.profile_steps):
            batch, loader_iter = _next_batch(loader_iter, train_loader)
            metrics = _train_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                batch=batch,
                device=device,
                args=args,
            )
            profiled_metrics.append(metrics)
            prof.step()
            print(
                f"profiled step={step + 1:03d} wall_ms={metrics['wall_ms']:.2f} "
                f"graphs={metrics['graphs']:.0f} atoms={metrics['atoms']:.0f} "
                f"atoms_per_sec={metrics['atoms_per_sec']:.1f} "
                f"mem_gb={metrics.get('max_mem_allocated_gb', 0.0):.3f}"
            )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.export_chrome_trace:
            prof.export_chrome_trace(str(output_dir / "omol_profile_trace.json"))
        with (output_dir / "omol_profile_summary.txt").open("w", encoding="utf-8") as handle:
            handle.write(_profiler_table(prof, sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=args.row_limit))
            handle.write("\n\n")
            handle.write(_profiler_table(prof, sort_by="self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage", row_limit=args.row_limit))

    if profiled_metrics:
        avg = {
            key: sum(metric.get(key, 0.0) for metric in profiled_metrics) / len(profiled_metrics)
            for key in profiled_metrics[0]
        }
        print("=== averaged profiled steps ===")
        for key in sorted(avg):
            print(f"{key}: {avg[key]:.6g}")

    print("=== profiler table: total time ===")
    print(_profiler_table(prof, sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=args.row_limit))
    print("=== profiler table: self memory ===")
    print(_profiler_table(prof, sort_by="self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage", row_limit=args.row_limit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed profiler for a Matterformer OMol training step")
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--validation-mode", type=str, default="heldout", choices=["heldout", "train_split"])
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument("--keep-in-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-subset", type=str, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--synthetic-min-atoms", type=int, default=2)
    parser.add_argument("--synthetic-max-atoms", type=int, default=8)
    parser.add_argument("--element-refs-json", type=str, default="configs/omol/element_refs.json")
    parser.add_argument("--hybrid-config-json", type=str, default="configs/omol/scalar_is_triton_d768_l19.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-atoms-per-batch", type=int, default=None)
    parser.add_argument("--max-edges-per-batch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--normalizer-rmsd", type=float, default=1.433569)
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--energy-loss", type=str, default="per_atom_mae", choices=["mae", "per_atom_mae"])
    parser.add_argument("--force-loss", type=str, default="l2norm", choices=["mae", "l2norm"])
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=19)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--chgspin-mode", type=str, default="add", choices=["off", "add", "concat"])
    parser.add_argument("--chgspin-emb-dim", type=int, default=None)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--pair-n-rbf", type=int, default=16)
    parser.add_argument("--pair-rbf-max", type=float, default=6.0)
    parser.add_argument(
        "--force-head-mode",
        default="direct",
        choices=["auto", "pairwise", "direct", "direct_3d", "non_equivariant", "tetra_vector"],
    )
    parser.add_argument("--train-augmentation", type=str, default="o3", choices=["off", "so3", "o3"])
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--float32-matmul-precision", type=str, default="highest")
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--row-limit", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--export-chrome-trace", action=argparse.BooleanOptionalAction, default=False)
    main(parser.parse_args())
