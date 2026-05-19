#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

import matterformer.models.hybrid as hybrid_mod
import matterformer.models.omol as omol_mod
from matterformer.models import MatterformerOMolForceField
from matterformer.models.hybrid import TetraPlatonicGlobalLayer
from matterformer.tasks import OMolDirectForceLoss, load_omol_element_references
from matterformer.utils import seed_everything
from scripts.train_omol_forcefield import (
    apply_rotation_augmentation,
    build_datasets,
    build_loader,
    load_hybrid_config,
    make_autocast_context,
)


@dataclass
class LayerMeta:
    name: str
    block_idx: int
    sublayer_idx: int
    backend: str
    bias_kind: str | None
    radius_cutoff: float | None
    group_order: int
    dim_per_frame: int
    heads: int
    head_dim: int


class LayoutRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []
        self._orig = hybrid_mod.build_radius_block_sparse_layout

    def install(self) -> None:
        orig = self._orig
        records = self.records

        def wrapped(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = int(torch.cuda.memory_allocated())
            else:
                before = 0
            start = time.perf_counter()
            layout = orig(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            else:
                peak = before
            records.append(
                {
                    "layout_build_ms": 1000.0 * (time.perf_counter() - start),
                    "peak_cuda_delta_gb": max(0, peak - before) / 1024**3,
                    "num_live_block_pairs": float(layout.num_live_block_pairs),
                    "effective_tile_density": float(layout.effective_tile_density),
                    "true_edge_density": float(layout.true_edge_density),
                    "mean_radius_degree": float(layout.mean_radius_degree),
                    "max_block_row_length": float(layout.max_block_row_length),
                    "radius_edge_count": float(layout.radius_edge_count),
                    "dense_pair_count": float(layout.dense_pair_count),
                }
            )
            return layout

        hybrid_mod.build_radius_block_sparse_layout = wrapped

    def restore(self) -> None:
        hybrid_mod.build_radius_block_sparse_layout = self._orig

    def clear(self) -> None:
        self.records.clear()


class FixedKContextRecorder:
    def __init__(self, model: MatterformerOMolForceField) -> None:
        self.model = model
        self.records: list[dict[str, float]] = []
        self._orig = model.trunk.prepare_fixed_k_local_context

    def install(self) -> None:
        orig = self._orig
        records = self.records

        def wrapped(_trunk, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = int(torch.cuda.memory_allocated())
            else:
                before = 0
            start = time.perf_counter()
            ctx = orig(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            else:
                peak = before
            valid_counts = ctx.neighbor_mask.sum(dim=-1).float()
            records.append(
                {
                    "fixed_k_context_ms": 1000.0 * (time.perf_counter() - start),
                    "peak_cuda_delta_gb": max(0, peak - before) / 1024**3,
                    "k_neighbors": float(ctx.neighbor_idx.shape[1]),
                    "valid_degree_mean": float(valid_counts.mean().item()),
                    "valid_degree_max": float(valid_counts.max().item()),
                    "valid_degree_p99": float(torch.quantile(valid_counts, 0.99).item()),
                }
            )
            return ctx

        self.model.trunk.prepare_fixed_k_local_context = types.MethodType(wrapped, self.model.trunk)  # type: ignore[method-assign]

    def restore(self) -> None:
        self.model.trunk.prepare_fixed_k_local_context = self._orig  # type: ignore[method-assign]

    def clear(self) -> None:
        self.records.clear()


class FixedKGeometryRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, float | str]] = []
        self._orig_build = omol_mod.build_triton_nonperiodic_knn_geometry_cache
        self._orig_flatten = omol_mod.flatten_padded_geometry_cache

    def install(self) -> None:
        build_orig = self._orig_build
        flatten_orig = self._orig_flatten
        records = self.records

        def timed_call(kind: str, fn, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = int(torch.cuda.memory_allocated())
            else:
                before = 0
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            else:
                peak = before
            row: dict[str, float | str] = {
                "kind": kind,
                "wall_ms": 1000.0 * (time.perf_counter() - start),
                "peak_cuda_delta_gb": max(0, peak - before) / 1024**3,
            }
            if hasattr(out, "neighbor_mask"):
                valid_counts = out.neighbor_mask.sum(dim=-1).float()
                row.update(
                    {
                        "k_neighbors": float(out.neighbor_idx.shape[-1]),
                        "valid_degree_mean": float(valid_counts.mean().item()),
                        "valid_degree_p99": float(torch.quantile(valid_counts.reshape(-1), 0.99).item()),
                        "valid_degree_max": float(valid_counts.max().item()),
                    }
                )
            records.append(row)
            return out

        def build_wrapped(*args, **kwargs):
            return timed_call("fixed_k_knn_build", build_orig, *args, **kwargs)

        def flatten_wrapped(*args, **kwargs):
            return timed_call("fixed_k_flatten", flatten_orig, *args, **kwargs)

        omol_mod.build_triton_nonperiodic_knn_geometry_cache = build_wrapped
        omol_mod.flatten_padded_geometry_cache = flatten_wrapped

    def restore(self) -> None:
        omol_mod.build_triton_nonperiodic_knn_geometry_cache = self._orig_build
        omol_mod.flatten_padded_geometry_cache = self._orig_flatten

    def clear(self) -> None:
        self.records.clear()


class LayerTimer:
    def __init__(self, model: MatterformerOMolForceField) -> None:
        self.model = model
        self.records: list[dict[str, float | str]] = []
        self._originals: list[tuple[TetraPlatonicGlobalLayer, Any]] = []
        self.metas = _layer_metas(model)

    def install(self) -> None:
        metas_by_name = {meta.name: meta for meta in self.metas}
        for name, module in _iter_tetra_layers(self.model):
            meta = metas_by_name[name]
            original = module.forward_flat
            self._originals.append((module, original))

            def timed_forward_flat(*args, __name=name, __meta=meta, __original=original, **kwargs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = int(torch.cuda.memory_allocated())
                else:
                    before = 0
                start = time.perf_counter()
                out = __original(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak = int(torch.cuda.max_memory_allocated())
                else:
                    peak = before
                self.records.append(
                    {
                        "name": __name,
                        "backend": __meta.backend,
                        "bias_kind": __meta.bias_kind or "",
                        "radius_cutoff": -1.0 if __meta.radius_cutoff is None else float(__meta.radius_cutoff),
                        "block_idx": float(__meta.block_idx),
                        "sublayer_idx": float(__meta.sublayer_idx),
                        "heads": float(__meta.heads),
                        "head_dim": float(__meta.head_dim),
                        "forward_ms": 1000.0 * (time.perf_counter() - start),
                        "peak_cuda_delta_gb": max(0, peak - before) / 1024**3,
                    }
                )
                return out

            module.forward_flat = timed_forward_flat  # type: ignore[method-assign]

    def restore(self) -> None:
        for module, original in self._originals:
            module.forward_flat = original  # type: ignore[method-assign]
        self._originals.clear()

    def clear(self) -> None:
        self.records.clear()


class ComponentTimer:
    def __init__(self, model: MatterformerOMolForceField) -> None:
        self.model = model
        self.records: list[dict[str, float | str]] = []
        self._originals: list[tuple[nn.Module, Any]] = []
        self.metas = _layer_metas(model)

    def install(self) -> None:
        metas_by_name = {meta.name: meta for meta in self.metas}
        for name, layer in _iter_tetra_layers(self.model):
            meta = metas_by_name[name]
            block = layer.block
            original = block.forward_flat
            self._originals.append((block, original))
            records = self.records

            def timed_block_forward_flat(
                block_self,
                x: torch.Tensor,
                *,
                pos: torch.Tensor,
                atom_types: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor,
                max_seqlen: int,
                radius_layout=None,
                fixed_k_context=None,
                __name=name,
                __meta=meta,
            ) -> torch.Tensor:
                def run_component(component: str, fn):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before = int(torch.cuda.memory_allocated())
                    else:
                        before = 0
                    start = time.perf_counter()
                    value = fn()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        peak = int(torch.cuda.max_memory_allocated())
                    else:
                        peak = before
                    records.append(
                        {
                            "name": __name,
                            "component": component,
                            "backend": __meta.backend,
                            "bias_kind": __meta.bias_kind or "",
                            "radius_cutoff": -1.0 if __meta.radius_cutoff is None else float(__meta.radius_cutoff),
                            "block_idx": float(__meta.block_idx),
                            "sublayer_idx": float(__meta.sublayer_idx),
                            "heads": float(__meta.heads),
                            "head_dim": float(__meta.head_dim),
                            "forward_ms": 1000.0 * (time.perf_counter() - start),
                            "peak_cuda_delta_gb": max(0, peak - before) / 1024**3,
                        }
                    )
                    return value

                norm1_x = run_component("norm1", lambda: block_self.norm1(x))
                attn_out = run_component(
                    "attention",
                    lambda: block_self.attn.forward_flat(
                        norm1_x,
                        pos=pos,
                        atom_types=atom_types,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        radius_layout=radius_layout,
                        fixed_k_context=fixed_k_context,
                    ),
                )
                x = run_component(
                    "attention_residual",
                    lambda: x + block_self._apply_layer_scale(block_self.dropout(attn_out), block_self.gamma_1),
                )
                norm2_x = run_component("norm2", lambda: block_self.norm2(x))
                ffn_hidden = run_component("ffn_linear1", lambda: block_self.linear1(norm2_x))
                ffn_hidden = run_component("ffn_activation", lambda: block_self.activation(ffn_hidden))
                ffn_out = run_component("ffn_linear2", lambda: block_self.linear2(ffn_hidden))
                x = run_component(
                    "ffn_residual",
                    lambda: x + block_self._apply_layer_scale(block_self.dropout(ffn_out), block_self.gamma_2),
                )
                return x

            block.forward_flat = types.MethodType(timed_block_forward_flat, block)  # type: ignore[method-assign]

    def restore(self) -> None:
        for block, original in self._originals:
            block.forward_flat = original  # type: ignore[method-assign]
        self._originals.clear()

    def clear(self) -> None:
        self.records.clear()


def _json_load(path: str) -> dict[str, Any]:
    return load_hybrid_config(path)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _iter_tetra_layers(model: MatterformerOMolForceField):
    for block_idx, block in enumerate(model.trunk.blocks):
        for sublayer_idx, layer in enumerate(getattr(block, "sublayers", [])):
            if isinstance(layer, TetraPlatonicGlobalLayer):
                yield f"block{block_idx:02d}.sublayer{sublayer_idx:02d}", layer


def _layer_metas(model: MatterformerOMolForceField) -> list[LayerMeta]:
    metas: list[LayerMeta] = []
    for name, layer in _iter_tetra_layers(model):
        attn = layer.block.attn
        bias_kind = getattr(attn, "radial_bias_kind", None)
        radius_cutoff = None
        if bias_kind in {"radius_rbf_type_enveloped", "fixed_k_esen"}:
            radius_cutoff = float(getattr(attn, "local_cutoff", 0.0))
        metas.append(
            LayerMeta(
                name=name,
                block_idx=int(name[5:7]),
                sublayer_idx=int(name[-2:]),
                backend=str(getattr(attn, "attention_backend", "")),
                bias_kind=bias_kind,
                radius_cutoff=radius_cutoff,
                group_order=int(layer.group_order),
                dim_per_frame=int(layer.dim_per_frame),
                heads=int(attn.num_heads),
                head_dim=int(attn.head_dim),
            )
        )
    return metas


def _make_model(args: argparse.Namespace, config_path: str) -> MatterformerOMolForceField:
    return MatterformerOMolForceField(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        attn_dropout=0.0,
        hybrid_config=_json_load(config_path),
        chgspin_mode=args.chgspin_mode,
        chgspin_emb_dim=args.chgspin_emb_dim,
        pair_hidden_dim=args.pair_hidden_dim,
        pair_n_rbf=args.pair_n_rbf,
        pair_rbf_max=args.pair_rbf_max,
        tetra_pair_force_mode="off",
        force_head_mode="auto",
        readout_head_mode="platonic",
        tetra_readout_mode="platonic",
        tetra_irrep_scalar_input="rho1",
        readout_activation=args.readout_activation,
        runtime_mode="internal_flat_tetra",
    )


def _compile_model(model: MatterformerOMolForceField, *, strategy: str, mode: str) -> dict[str, Any]:
    if strategy == "none":
        return {"strategy": "none"}
    if strategy == "run_matched":
        if model.trunk._radius_sparse_layout_config() is not None or model.trunk._fixed_k_local_config() is not None:
            stats = model.trunk.compile_flat_tetra_layer_forwards(mode=mode)
            return {"strategy": "dynamic_local_layer_selective", **stats}
        model.trunk.forward_flat_tetra = torch.compile(model.trunk.forward_flat_tetra, mode=mode)  # type: ignore[method-assign]
        return {"strategy": "whole_flat_tetra"}
    if strategy == "layer_flat":
        stats = model.trunk.compile_flat_tetra_layer_forwards(mode=mode)
        return {"strategy": "layer_flat", **stats}
    raise ValueError(f"Unknown compile strategy {strategy!r}")


def _batch_to_device(batch, device: torch.device):
    return batch.to(device)


def _load_batch(args: argparse.Namespace, device: torch.device):
    train_dataset, _, _ = build_datasets(args)
    loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        max_graphs_per_batch=args.max_graphs_per_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_atoms=args.max_atoms_per_batch,
        max_edges=args.max_edges_per_batch,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        batching_mode=args.batching_mode,
        bucket_window_size=args.bucket_window_size,
        bucket_shuffle_groups=args.bucket_shuffle_groups,
    )
    batch = next(iter(loader))
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    batch = _batch_to_device(batch, device)
    batch = apply_rotation_augmentation(batch, args.train_augmentation)
    return batch


def _criterion(args: argparse.Namespace, device: torch.device) -> OMolDirectForceLoss:
    refs = load_omol_element_references(args.element_refs_json).to(device)
    return OMolDirectForceLoss(
        refs,
        normalizer_rmsd=args.normalizer_rmsd,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        energy_loss=args.energy_loss,
        force_loss=args.force_loss,
    )


def _batch_stats(batch) -> dict[str, float]:
    num_atoms = batch.num_atoms.float()
    return {
        "graphs": float(batch.energy.shape[0]),
        "atoms": float(num_atoms.sum().item()),
        "mean_atoms": float(num_atoms.mean().item()),
        "max_atoms": float(num_atoms.max().item()),
        "sum_n2": float((batch.num_atoms.to(torch.int64) ** 2).sum().item()),
        "padded_slots": float(batch.atomic_numbers.numel()),
    }


def _forward_loss(model: MatterformerOMolForceField, criterion: OMolDirectForceLoss, batch, args: argparse.Namespace):
    with make_autocast_context(batch.atomic_numbers.device, args.bf16):
        predictions = model(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
        return criterion(predictions, batch)


def _forward_model(model: MatterformerOMolForceField, batch, args: argparse.Namespace):
    with make_autocast_context(batch.atomic_numbers.device, args.bf16):
        return model(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)


def _prep_ms(records: list[dict[str, float]], key: str) -> float:
    return sum(float(row[key]) for row in records)


def _bench_full_forward(
    *,
    name: str,
    model: MatterformerOMolForceField,
    batch,
    args: argparse.Namespace,
    layout_recorder: LayoutRecorder,
    fixed_k_geometry_recorder: FixedKGeometryRecorder,
    fixed_k_recorder: FixedKContextRecorder,
) -> dict[str, Any]:
    metrics: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for step in range(args.full_forward_warmup + args.full_forward_repeats):
            layout_recorder.clear()
            fixed_k_geometry_recorder.clear()
            fixed_k_recorder.clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            _ = _forward_model(model, batch, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            else:
                mem_gb = 0.0
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            if step >= args.full_forward_warmup:
                metrics.append(
                    {
                        "wall_ms": elapsed_ms,
                        "max_mem_allocated_gb": mem_gb,
                        "radius_layout_build_ms": _prep_ms(layout_recorder.records, "layout_build_ms"),
                        "radius_layout_calls": float(len(layout_recorder.records)),
                        "fixed_k_knn_build_ms": sum(
                            float(row["wall_ms"])
                            for row in fixed_k_geometry_recorder.records
                            if row["kind"] == "fixed_k_knn_build"
                        ),
                        "fixed_k_flatten_ms": sum(
                            float(row["wall_ms"])
                            for row in fixed_k_geometry_recorder.records
                            if row["kind"] == "fixed_k_flatten"
                        ),
                        "fixed_k_geometry_calls": float(len(fixed_k_geometry_recorder.records)),
                        "fixed_k_context_ms": _prep_ms(fixed_k_recorder.records, "fixed_k_context_ms"),
                        "fixed_k_context_calls": float(len(fixed_k_recorder.records)),
                    }
                )
    return {
        "name": name,
        "wall_ms": _summary([row["wall_ms"] for row in metrics]),
        "max_mem_allocated_gb": _summary([row["max_mem_allocated_gb"] for row in metrics]),
        "radius_layout_build_ms": _summary([row["radius_layout_build_ms"] for row in metrics]),
        "radius_layout_calls": _summary([row["radius_layout_calls"] for row in metrics]),
        "fixed_k_knn_build_ms": _summary([row["fixed_k_knn_build_ms"] for row in metrics]),
        "fixed_k_flatten_ms": _summary([row["fixed_k_flatten_ms"] for row in metrics]),
        "fixed_k_geometry_calls": _summary([row["fixed_k_geometry_calls"] for row in metrics]),
        "fixed_k_context_ms": _summary([row["fixed_k_context_ms"] for row in metrics]),
        "fixed_k_context_calls": _summary([row["fixed_k_context_calls"] for row in metrics]),
        "raw": metrics,
    }


def _bench_full_step(
    *,
    name: str,
    model: MatterformerOMolForceField,
    criterion: OMolDirectForceLoss,
    batch,
    args: argparse.Namespace,
    layout_recorder: LayoutRecorder,
    fixed_k_geometry_recorder: FixedKGeometryRecorder,
    fixed_k_recorder: FixedKContextRecorder,
) -> dict[str, Any]:
    metrics: list[dict[str, float]] = []
    for step in range(args.full_warmup + args.full_repeats):
        model.zero_grad(set_to_none=True)
        layout_recorder.clear()
        fixed_k_geometry_recorder.clear()
        fixed_k_recorder.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = _forward_loss(model, criterion, batch, args)
        output.loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            mem_gb = 0.0
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        if step >= args.full_warmup:
            metrics.append(
                {
                    "wall_ms": elapsed_ms,
                    "max_mem_allocated_gb": mem_gb,
                    "radius_layout_build_ms": _prep_ms(layout_recorder.records, "layout_build_ms"),
                    "radius_layout_calls": float(len(layout_recorder.records)),
                    "fixed_k_knn_build_ms": sum(
                        float(row["wall_ms"])
                        for row in fixed_k_geometry_recorder.records
                        if row["kind"] == "fixed_k_knn_build"
                    ),
                    "fixed_k_flatten_ms": sum(
                        float(row["wall_ms"])
                        for row in fixed_k_geometry_recorder.records
                        if row["kind"] == "fixed_k_flatten"
                    ),
                    "fixed_k_geometry_calls": float(len(fixed_k_geometry_recorder.records)),
                    "fixed_k_context_ms": _prep_ms(fixed_k_recorder.records, "fixed_k_context_ms"),
                    "fixed_k_context_calls": float(len(fixed_k_recorder.records)),
                }
            )
    return {
        "name": name,
        "wall_ms": _summary([row["wall_ms"] for row in metrics]),
        "max_mem_allocated_gb": _summary([row["max_mem_allocated_gb"] for row in metrics]),
        "radius_layout_build_ms": _summary([row["radius_layout_build_ms"] for row in metrics]),
        "radius_layout_calls": _summary([row["radius_layout_calls"] for row in metrics]),
        "fixed_k_knn_build_ms": _summary([row["fixed_k_knn_build_ms"] for row in metrics]),
        "fixed_k_flatten_ms": _summary([row["fixed_k_flatten_ms"] for row in metrics]),
        "fixed_k_geometry_calls": _summary([row["fixed_k_geometry_calls"] for row in metrics]),
        "fixed_k_context_ms": _summary([row["fixed_k_context_ms"] for row in metrics]),
        "fixed_k_context_calls": _summary([row["fixed_k_context_calls"] for row in metrics]),
        "raw": metrics,
    }


def _bench_layer_forward(
    *,
    name: str,
    model: MatterformerOMolForceField,
    batch,
    args: argparse.Namespace,
    layout_recorder: LayoutRecorder,
    fixed_k_geometry_recorder: FixedKGeometryRecorder,
    fixed_k_recorder: FixedKContextRecorder,
) -> dict[str, Any]:
    timer = LayerTimer(model)
    timer.install()
    try:
        for _ in range(args.layer_warmup):
            timer.clear()
            layout_recorder.clear()
            fixed_k_geometry_recorder.clear()
            fixed_k_recorder.clear()
            with torch.no_grad():
                _forward_model(model, batch, args)
        layer_records: list[dict[str, float | str]] = []
        layout_records: list[dict[str, float]] = []
        fixed_k_geometry_records: list[dict[str, float | str]] = []
        fixed_k_records: list[dict[str, float]] = []
        for _ in range(args.layer_repeats):
            timer.clear()
            layout_recorder.clear()
            fixed_k_geometry_recorder.clear()
            fixed_k_recorder.clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.no_grad():
                _forward_model(model, batch, args)
            layer_records.extend(timer.records)
            layout_records.extend(layout_recorder.records)
            fixed_k_geometry_records.extend(fixed_k_geometry_recorder.records)
            fixed_k_records.extend(fixed_k_recorder.records)
    finally:
        timer.restore()

    grouped: dict[str, dict[str, Any]] = {}
    for row in layer_records:
        key = str(row["name"])
        grouped.setdefault(
            key,
            {
                "name": key,
                "backend": row["backend"],
                "bias_kind": row["bias_kind"],
                "radius_cutoff": row["radius_cutoff"],
                "block_idx": row["block_idx"],
                "sublayer_idx": row["sublayer_idx"],
                "heads": row["heads"],
                "head_dim": row["head_dim"],
                "forward_ms_values": [],
                "peak_cuda_delta_gb_values": [],
            },
        )
        grouped[key]["forward_ms_values"].append(float(row["forward_ms"]))
        grouped[key]["peak_cuda_delta_gb_values"].append(float(row["peak_cuda_delta_gb"]))
    layer_summary = []
    for key in sorted(grouped):
        item = grouped[key]
        layer_summary.append(
            {
                "name": item["name"],
                "backend": item["backend"],
                "bias_kind": item["bias_kind"],
                "radius_cutoff": item["radius_cutoff"],
                "block_idx": item["block_idx"],
                "sublayer_idx": item["sublayer_idx"],
                "heads": item["heads"],
                "head_dim": item["head_dim"],
                "forward_ms": _summary(item["forward_ms_values"]),
                "peak_cuda_delta_gb": _summary(item["peak_cuda_delta_gb_values"]),
            }
        )
    return {
        "name": name,
        "layers": layer_summary,
        "layout": {
            "calls": len(layout_records),
            "layout_build_ms": _summary([float(row["layout_build_ms"]) for row in layout_records]),
            "mean_radius_degree": _summary([float(row["mean_radius_degree"]) for row in layout_records]),
            "effective_tile_density": _summary([float(row["effective_tile_density"]) for row in layout_records]),
            "max_block_row_length": _summary([float(row["max_block_row_length"]) for row in layout_records]),
        },
        "fixed_k_context": {
            "calls": len(fixed_k_records),
            "fixed_k_context_ms": _summary([float(row["fixed_k_context_ms"]) for row in fixed_k_records]),
            "valid_degree_mean": _summary([float(row["valid_degree_mean"]) for row in fixed_k_records]),
            "valid_degree_p99": _summary([float(row["valid_degree_p99"]) for row in fixed_k_records]),
            "valid_degree_max": _summary([float(row["valid_degree_max"]) for row in fixed_k_records]),
        },
        "fixed_k_geometry": {
            "calls": len(fixed_k_geometry_records),
            "knn_build_ms": _summary(
                [float(row["wall_ms"]) for row in fixed_k_geometry_records if row["kind"] == "fixed_k_knn_build"]
            ),
            "flatten_ms": _summary(
                [float(row["wall_ms"]) for row in fixed_k_geometry_records if row["kind"] == "fixed_k_flatten"]
            ),
        },
    }


def _bench_component_forward(
    *,
    name: str,
    model: MatterformerOMolForceField,
    batch,
    args: argparse.Namespace,
    layout_recorder: LayoutRecorder,
    fixed_k_geometry_recorder: FixedKGeometryRecorder,
    fixed_k_recorder: FixedKContextRecorder,
) -> dict[str, Any]:
    layer_timer = LayerTimer(model)
    component_timer = ComponentTimer(model)
    layer_timer.install()
    component_timer.install()
    try:
        for _ in range(args.component_warmup):
            layer_timer.clear()
            component_timer.clear()
            layout_recorder.clear()
            fixed_k_geometry_recorder.clear()
            fixed_k_recorder.clear()
            with torch.no_grad():
                _forward_model(model, batch, args)
        layer_records: list[dict[str, float | str]] = []
        component_records: list[dict[str, float | str]] = []
        fixed_k_geometry_records: list[dict[str, float | str]] = []
        fixed_k_records: list[dict[str, float]] = []
        for _ in range(args.component_repeats):
            layer_timer.clear()
            component_timer.clear()
            layout_recorder.clear()
            fixed_k_geometry_recorder.clear()
            fixed_k_recorder.clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.no_grad():
                _forward_model(model, batch, args)
            layer_records.extend(layer_timer.records)
            component_records.extend(component_timer.records)
            fixed_k_geometry_records.extend(fixed_k_geometry_recorder.records)
            fixed_k_records.extend(fixed_k_recorder.records)
    finally:
        component_timer.restore()
        layer_timer.restore()

    grouped_components: dict[tuple[str, str], dict[str, Any]] = {}
    for row in component_records:
        key = (str(row["name"]), str(row["component"]))
        grouped_components.setdefault(
            key,
            {
                "name": row["name"],
                "component": row["component"],
                "backend": row["backend"],
                "bias_kind": row["bias_kind"],
                "radius_cutoff": row["radius_cutoff"],
                "block_idx": row["block_idx"],
                "sublayer_idx": row["sublayer_idx"],
                "heads": row["heads"],
                "head_dim": row["head_dim"],
                "forward_ms_values": [],
                "peak_cuda_delta_gb_values": [],
            },
        )
        grouped_components[key]["forward_ms_values"].append(float(row["forward_ms"]))
        grouped_components[key]["peak_cuda_delta_gb_values"].append(float(row["peak_cuda_delta_gb"]))
    component_summary = []
    for key in sorted(grouped_components):
        item = grouped_components[key]
        component_summary.append(
            {
                "name": item["name"],
                "component": item["component"],
                "backend": item["backend"],
                "bias_kind": item["bias_kind"],
                "radius_cutoff": item["radius_cutoff"],
                "block_idx": item["block_idx"],
                "sublayer_idx": item["sublayer_idx"],
                "heads": item["heads"],
                "head_dim": item["head_dim"],
                "forward_ms": _summary(item["forward_ms_values"]),
                "peak_cuda_delta_gb": _summary(item["peak_cuda_delta_gb_values"]),
            }
        )

    grouped_layers: dict[str, dict[str, Any]] = {}
    for row in layer_records:
        key = str(row["name"])
        grouped_layers.setdefault(
            key,
            {
                "name": key,
                "backend": row["backend"],
                "bias_kind": row["bias_kind"],
                "forward_ms_values": [],
            },
        )
        grouped_layers[key]["forward_ms_values"].append(float(row["forward_ms"]))
    layer_summary = [
        {
            "name": item["name"],
            "backend": item["backend"],
            "bias_kind": item["bias_kind"],
            "forward_ms": _summary(item["forward_ms_values"]),
        }
        for item in (grouped_layers[key] for key in sorted(grouped_layers))
    ]

    return {
        "name": name,
        "layers": layer_summary,
        "components": component_summary,
        "fixed_k_context": {
            "calls": len(fixed_k_records),
            "fixed_k_context_ms": _summary([float(row["fixed_k_context_ms"]) for row in fixed_k_records]),
            "valid_degree_mean": _summary([float(row["valid_degree_mean"]) for row in fixed_k_records]),
            "valid_degree_p99": _summary([float(row["valid_degree_p99"]) for row in fixed_k_records]),
            "valid_degree_max": _summary([float(row["valid_degree_max"]) for row in fixed_k_records]),
        },
        "fixed_k_geometry": {
            "calls": len(fixed_k_geometry_records),
            "knn_build_ms": _summary(
                [float(row["wall_ms"]) for row in fixed_k_geometry_records if row["kind"] == "fixed_k_knn_build"]
            ),
            "flatten_ms": _summary(
                [float(row["wall_ms"]) for row in fixed_k_geometry_records if row["kind"] == "fixed_k_flatten"]
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "architecture",
        "name",
        "backend",
        "bias_kind",
        "radius_cutoff",
        "forward_ms_mean",
        "forward_ms_median",
        "forward_ms_min",
        "forward_ms_max",
        "peak_cuda_delta_gb_mean",
        "peak_cuda_delta_gb_max",
    ]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("architecture", "")),
                    str(row.get("name", "")),
                    str(row.get("backend", "")),
                    str(row.get("bias_kind", "")),
                    str(row.get("radius_cutoff", "")),
                    f"{row['forward_ms']['mean']:.6f}",
                    f"{row['forward_ms']['median']:.6f}",
                    f"{row['forward_ms']['min']:.6f}",
                    f"{row['forward_ms']['max']:.6f}",
                    f"{row['peak_cuda_delta_gb']['mean']:.6f}",
                    f"{row['peak_cuda_delta_gb']['max']:.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_component_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "architecture",
        "name",
        "component",
        "backend",
        "bias_kind",
        "radius_cutoff",
        "forward_ms_mean",
        "forward_ms_median",
        "forward_ms_min",
        "forward_ms_max",
        "peak_cuda_delta_gb_mean",
        "peak_cuda_delta_gb_max",
    ]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("architecture", "")),
                    str(row.get("name", "")),
                    str(row.get("component", "")),
                    str(row.get("backend", "")),
                    str(row.get("bias_kind", "")),
                    str(row.get("radius_cutoff", "")),
                    f"{row['forward_ms']['mean']:.6f}",
                    f"{row['forward_ms']['median']:.6f}",
                    f"{row['forward_ms']['min']:.6f}",
                    f"{row['forward_ms']['max']:.6f}",
                    f"{row['peak_cuda_delta_gb']['mean']:.6f}",
                    f"{row['peak_cuda_delta_gb']['max']:.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile h1536 OMol global flash vs local/global alternating architectures.")
    parser.add_argument("--baseline-config", default="configs/omol/tetra_t_only_h1536_l16_pt2_goodrun_qkn_uk_csfilm.json")
    parser.add_argument("--sparse-config", default="configs/omol/tetra_t_only_h1536_l16_pt2_goodrun_qkn_uk_csfilm_radius_sparse_r4_every2.json")
    parser.add_argument("--alt-config", default=None)
    parser.add_argument("--alt-name", default="local_alt")
    parser.add_argument("--train-data-path", default="/home/ebekker/data/omol/open_mol/train_4M")
    parser.add_argument("--val-data-path", default="/home/ebekker/data/omol/open_mol/val")
    parser.add_argument("--validation-mode", default="heldout")
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument("--keep-in-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-subset", default=None)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--synthetic-min-atoms", type=int, default=2)
    parser.add_argument("--synthetic-max-atoms", type=int, default=8)
    parser.add_argument("--element-refs-json", default="configs/omol/element_refs.json")
    parser.add_argument("--d-model", type=int, default=1536)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--chgspin-mode", default="add")
    parser.add_argument("--chgspin-emb-dim", type=int, default=128)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--pair-n-rbf", type=int, default=16)
    parser.add_argument("--pair-rbf-max", type=float, default=6.0)
    parser.add_argument("--readout-activation", default="gelu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-graphs-per-batch", type=int, default=999999)
    parser.add_argument("--max-atoms-per-batch", type=int, default=12000)
    parser.add_argument("--max-edges-per-batch", type=int, default=2400000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batching-mode", default="random")
    parser.add_argument("--bucket-window-size", type=int, default=4096)
    parser.add_argument("--bucket-shuffle-groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-augmentation", default="o3")
    parser.add_argument("--normalizer-rmsd", type=float, default=1.433569)
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--force-weight", type=float, default=20.0)
    parser.add_argument("--energy-loss", default="per_atom_mae")
    parser.add_argument("--force-loss", default="l2norm")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--float32-matmul-precision", default="high")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--full-forward-warmup", type=int, default=2)
    parser.add_argument("--full-forward-repeats", type=int, default=5)
    parser.add_argument("--full-warmup", type=int, default=1)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--layer-warmup", type=int, default=1)
    parser.add_argument("--layer-repeats", type=int, default=2)
    parser.add_argument("--component-warmup", type=int, default=1)
    parser.add_argument("--component-repeats", type=int, default=2)
    parser.add_argument("--skip-fwd-bwd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-component", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch = _load_batch(args, device)
    criterion = _criterion(args, device)
    batch_stats = _batch_stats(batch)
    print("batch " + json.dumps(batch_stats, sort_keys=True))

    layout_recorder = LayoutRecorder()
    layout_recorder.install()
    results: dict[str, Any] = {
        "system": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "matmul_precision": args.float32_matmul_precision,
        },
        "batch": batch_stats,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "architectures": {},
    }
    layer_csv_rows: list[dict[str, Any]] = []
    component_csv_rows: list[dict[str, Any]] = []
    alt_config = args.alt_config or args.sparse_config
    arch_specs = [("flash_baseline", args.baseline_config), (args.alt_name, alt_config)]
    fixed_k_geometry_recorder = FixedKGeometryRecorder()
    fixed_k_geometry_recorder.install()
    try:
        for arch_name, config_path in arch_specs:
            print(f"=== {arch_name}: full forward run-matched compile ===")
            model = _make_model(args, config_path).to(device).train()
            compile_stats = _compile_model(model, strategy="run_matched", mode=args.compile_mode)
            fixed_k_recorder = FixedKContextRecorder(model)
            fixed_k_recorder.install()
            try:
                full_forward = _bench_full_forward(
                    name=arch_name,
                    model=model,
                    batch=batch,
                    args=args,
                    layout_recorder=layout_recorder,
                    fixed_k_geometry_recorder=fixed_k_geometry_recorder,
                    fixed_k_recorder=fixed_k_recorder,
                )
                full = None
                if not args.skip_fwd_bwd:
                    model.train()
                    full = _bench_full_step(
                        name=arch_name,
                        model=model,
                        criterion=criterion,
                        batch=batch,
                        args=args,
                        layout_recorder=layout_recorder,
                        fixed_k_geometry_recorder=fixed_k_geometry_recorder,
                        fixed_k_recorder=fixed_k_recorder,
                    )
            finally:
                fixed_k_recorder.restore()
            del model
            torch.cuda.empty_cache()

            print(f"=== {arch_name}: per-layer forward layer-compiled profile ===")
            layer_model = _make_model(args, config_path).to(device).eval()
            layer_compile_stats = _compile_model(layer_model, strategy="layer_flat", mode=args.compile_mode)
            layer_fixed_k_recorder = FixedKContextRecorder(layer_model)
            layer_fixed_k_recorder.install()
            try:
                layer = _bench_layer_forward(
                    name=arch_name,
                    model=layer_model,
                    batch=batch,
                    args=args,
                    layout_recorder=layout_recorder,
                    fixed_k_geometry_recorder=fixed_k_geometry_recorder,
                    fixed_k_recorder=layer_fixed_k_recorder,
                )
            finally:
                layer_fixed_k_recorder.restore()
            del layer_model
            torch.cuda.empty_cache()

            component = None
            if not args.skip_component:
                print(f"=== {arch_name}: per-component eager forward profile ===")
                component_model = _make_model(args, config_path).to(device).eval()
                component_fixed_k_recorder = FixedKContextRecorder(component_model)
                component_fixed_k_recorder.install()
                try:
                    component = _bench_component_forward(
                        name=arch_name,
                        model=component_model,
                        batch=batch,
                        args=args,
                        layout_recorder=layout_recorder,
                        fixed_k_geometry_recorder=fixed_k_geometry_recorder,
                        fixed_k_recorder=component_fixed_k_recorder,
                    )
                finally:
                    component_fixed_k_recorder.restore()
                del component_model
                torch.cuda.empty_cache()

            arch_result = {
                "config": config_path,
                "run_matched_compile": compile_stats,
                "layer_profile_compile": layer_compile_stats,
                "full_forward": full_forward,
                "layer_forward": layer,
            }
            if full is not None:
                arch_result["full_fwd_bwd"] = full
            if component is not None:
                arch_result["component_forward"] = component
            results["architectures"][arch_name] = arch_result
            for row in layer["layers"]:
                row_for_csv = dict(row)
                row_for_csv["architecture"] = arch_name
                layer_csv_rows.append(row_for_csv)
            if component is not None:
                for row in component["components"]:
                    row_for_csv = dict(row)
                    row_for_csv["architecture"] = arch_name
                    component_csv_rows.append(row_for_csv)
            print("full_forward " + json.dumps({arch_name: full_forward}, sort_keys=True))
            if full is not None:
                print("full_fwd_bwd " + json.dumps({arch_name: full}, sort_keys=True))
            print("layout " + json.dumps({arch_name: layer["layout"]}, sort_keys=True))
            print("fixed_k_geometry " + json.dumps({arch_name: layer["fixed_k_geometry"]}, sort_keys=True))
            print("fixed_k_context " + json.dumps({arch_name: layer["fixed_k_context"]}, sort_keys=True))
    finally:
        fixed_k_geometry_recorder.restore()
        layout_recorder.restore()

    baseline_ms = results["architectures"]["flash_baseline"]["full_forward"]["wall_ms"]["mean"]
    alt_ms = results["architectures"][args.alt_name]["full_forward"]["wall_ms"]["mean"]
    results["comparison"] = {
        "alt_vs_flash_forward_time_ratio": float(alt_ms / baseline_ms) if baseline_ms > 0 else None,
        "alt_extra_forward_ms": float(alt_ms - baseline_ms),
    }
    output_json = args.output_dir / "profile_results.json"
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "layer_forward.csv", layer_csv_rows)
    if component_csv_rows:
        _write_component_csv(args.output_dir / "component_forward.csv", component_csv_rows)
    print("results_json:", output_json)
    print("layer_csv:", args.output_dir / "layer_forward.csv")
    if component_csv_rows:
        print("component_csv:", args.output_dir / "component_forward.csv")
    print("comparison " + json.dumps(results["comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
