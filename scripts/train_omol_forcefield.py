#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from matterformer.data import (
    FairChemOMolDataset,
    OMolBatch,
    OMolDynamicBatchSampler,
    SyntheticOMolDataset,
    apply_debug_subset,
    collate_omol,
    split_train_val,
)
from matterformer.models import MatterformerAllScAIPDirectForceField, MatterformerOMolForceField
from matterformer.tasks import OMolDirectForceLoss, load_omol_element_references
from matterformer.utils import EMA, CosineWarmupScheduler, default_device, random_rotation_matrices, seed_everything


def load_json_config(value: str | None) -> dict | None:
    if value is None:
        return None
    stripped = value.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(value)
    candidate = Path(value)
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def load_hybrid_config(value: str | None) -> dict | None:
    return load_json_config(value)


def load_allscaip_config(value: str | None) -> dict | None:
    return load_json_config(value)


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("["):
        return [int(item) for item in json.loads(stripped)]
    return [int(item.strip()) for item in stripped.split(",") if item.strip()]


def _allscaip_dataclass_values(value) -> dict:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def collect_allscaip_runtime_config(model: torch.nn.Module) -> dict:
    base_model = unwrap_model(model)
    config: dict = {}
    if hasattr(base_model, "allscaip_config"):
        config.update(dict(getattr(base_model, "allscaip_config")))
    backbone = getattr(base_model, "backbone", None)
    if backbone is not None:
        for attr in ("global_cfg", "molecular_graph_cfg", "gnn_cfg", "reg_cfg"):
            config.update(_allscaip_dataclass_values(getattr(backbone, attr, None)))
    return config


def _parse_name_fragments(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(fragment.strip().lower() for fragment in value.split(",") if fragment.strip())


def _uses_muon_update(name: str, parameter: torch.nn.Parameter, args: argparse.Namespace) -> bool:
    if parameter.ndim < args.muon_min_ndim:
        return False
    lowered = name.lower()
    if args.muon_hidden_only and not lowered.startswith("trunk.blocks."):
        return False
    if any(fragment in lowered for fragment in _parse_name_fragments(args.muon_exclude_name_fragments)):
        return False
    return True


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "adamw":
        args.optimizer_resolved = {
            "name": "adamw",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if optimizer_name != "muon":
        raise ValueError(f"Unknown optimizer {args.optimizer!r}")

    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError as exc:
        raise RuntimeError(
            "Muon optimizer is not importable. Install it in the training environment with "
            "`python -m pip install git+https://github.com/KellerJordan/Muon` or "
            "`python -m pip install -e /home/thadziv/GitHub/Muon`."
        ) from exc

    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    adam_names: list[str] = []
    muon_view_counts: dict[str, int] = {}
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if _uses_muon_update(name, parameter, args):
            muon_view = "default"
            if (
                args.muon_platonic_kernel_view == "conv"
                and name.endswith(".kernel")
                and parameter.ndim == 3
            ):
                setattr(parameter, "_muon_view", "platonic_conv")
                muon_view = "platonic_conv"
            muon_params.append(parameter)
            muon_names.append(name)
            muon_view_counts[muon_view] = muon_view_counts.get(muon_view, 0) + 1
        else:
            adam_params.append(parameter)
            adam_names.append(name)

    if not muon_params:
        raise RuntimeError(
            "Optimizer muon selected but no parameters matched the Muon group. "
            "Use --no-muon-hidden-only or adjust --muon-exclude-name-fragments."
        )

    muon_adam_lr = args.muon_adam_lr if args.muon_adam_lr is not None else args.lr
    muon_adam_weight_decay = (
        args.muon_adam_weight_decay if args.muon_adam_weight_decay is not None else args.weight_decay
    )
    param_groups: list[dict[str, object]] = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": float(args.muon_lr),
            "momentum": float(args.muon_momentum),
            "weight_decay": float(args.muon_weight_decay),
        },
    ]
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "use_muon": False,
                "lr": float(muon_adam_lr),
                "betas": (float(args.muon_adam_beta1), float(args.muon_adam_beta2)),
                "eps": float(args.muon_adam_eps),
                "weight_decay": float(muon_adam_weight_decay),
            }
        )

    args.optimizer_resolved = {
        "name": "muon",
        "muon_lr": float(args.muon_lr),
        "muon_momentum": float(args.muon_momentum),
        "muon_weight_decay": float(args.muon_weight_decay),
        "muon_parameters": sum(parameter.numel() for parameter in muon_params),
        "muon_tensors": len(muon_params),
        "adam_lr": float(muon_adam_lr),
        "adam_weight_decay": float(muon_adam_weight_decay),
        "adam_parameters": sum(parameter.numel() for parameter in adam_params),
        "adam_tensors": len(adam_params),
        "muon_hidden_only": bool(args.muon_hidden_only),
        "muon_min_ndim": int(args.muon_min_ndim),
        "muon_platonic_kernel_view": str(args.muon_platonic_kernel_view),
        "muon_view_counts": muon_view_counts,
        "muon_first_names": muon_names[:12],
        "adam_first_names": adam_names[:12],
    }
    print("optimizer_resolved: " + json.dumps(args.optimizer_resolved, sort_keys=True))
    return SingleDeviceMuonWithAuxAdam(param_groups)


def resolve_scheduler_max_iters(args: argparse.Namespace, train_loader: DataLoader) -> int:
    if args.max_steps > 0:
        return int(args.max_steps)
    if args.max_epochs is None:
        raise ValueError("--max-steps <= 0 requires --max-epochs so the LR scheduler has a finite horizon")
    batches_per_epoch = int(len(train_loader))
    if batches_per_epoch <= 0:
        raise ValueError("train_loader has no batches; cannot resolve an epoch-limited LR scheduler horizon")
    return int(args.max_epochs) * batches_per_epoch


def resolve_scheduler_min_lrs(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> list[float]:
    lr_min = float(args.lr_min)
    if lr_min < 0.0:
        raise ValueError("--lr-min must be non-negative")
    if lr_min == 0.0:
        return [0.0 for _ in optimizer.param_groups]

    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "muon":
        reference_lr = float(args.muon_adam_lr if args.muon_adam_lr is not None else args.lr)
    else:
        reference_lr = float(args.lr)
    if reference_lr <= 0.0:
        raise ValueError("Reference learning rate must be positive when --lr-min is positive")

    min_lrs = []
    for group in optimizer.param_groups:
        group_lr = float(group["lr"])
        min_lrs.append(lr_min * group_lr / reference_lr)
    return min_lrs


def _config_values_match(actual, expected) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return list(actual) == expected if isinstance(actual, (list, tuple)) else False
    return actual == expected


def validate_allscaip_strict_config(model: torch.nn.Module, expected: dict | None, *, stage: str) -> None:
    if not expected:
        return
    actual = collect_allscaip_runtime_config(model)
    missing: list[str] = []
    mismatched: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            missing.append(key)
            continue
        actual_value = actual[key]
        if not _config_values_match(actual_value, expected_value):
            mismatched.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if mismatched:
            details.append("mismatched=[" + "; ".join(mismatched) + "]")
        raise RuntimeError(f"AllScAIP strict config check failed at {stage}: " + " ".join(details))


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class CudaPhaseTimer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.events: dict[str, list[torch.cuda.Event | None]] = {}

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.setdefault(name, [None, None])[0] = event

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.setdefault(name, [None, None])[1] = event

    @contextmanager
    def phase(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def results_ms(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        results: dict[str, float] = {}
        for name, (start, stop) in self.events.items():
            if start is not None and stop is not None:
                results[f"profile/{name}_ms"] = float(start.elapsed_time(stop))
        results["profile/max_mem_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
        return results


def maybe_init_wandb(args: argparse.Namespace):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required unless --wandb-mode disabled is used") from exc
    wandb_dir = Path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config=vars(args),
    )


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    max_graphs_per_batch: int | None,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    max_atoms: int | None,
    max_edges: int | None,
    seed: int,
    prefetch_factor: int | None,
    batching_mode: str = "random",
    bucket_window_size: int = 4096,
    bucket_shuffle_groups: int = 8,
) -> DataLoader:
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_omol,
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if max_atoms is not None:
        max_batch_size = int(max_graphs_per_batch or batch_size)
        sampler = OMolDynamicBatchSampler(
            dataset,
            max_batch_size=max_batch_size,
            max_atoms=max_atoms,
            max_edges=max_edges,
            shuffle=shuffle,
            seed=seed,
            batching_mode=batching_mode,
            bucket_window_size=bucket_window_size,
            bucket_shuffle_groups=bucket_shuffle_groups,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, **kwargs)
        loader.set_epoch = sampler.set_epoch  # type: ignore[attr-defined]
        return loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def summarize_loader_startup(name: str, loader: DataLoader, run) -> None:
    estimated_batches = len(loader)
    sampler = getattr(loader, "batch_sampler", None)
    if not isinstance(sampler, OMolDynamicBatchSampler):
        print(f"loader/{name}: len={estimated_batches} dynamic=false")
        if run is not None:
            run.summary[f"loader/{name}/len"] = int(estimated_batches)
        return

    start = time.perf_counter()
    batches = 0
    graphs = 0
    atoms = 0
    sum_n2 = 0
    max_graphs = 0
    max_atoms = 0
    max_sum_n2 = 0
    for batch_indices in sampler:
        counts = [int(sampler._get_num_atoms(int(idx))) for idx in batch_indices]
        batch_atoms = sum(counts)
        batch_sum_n2 = sum(count * count for count in counts)
        batches += 1
        graphs += len(counts)
        atoms += batch_atoms
        sum_n2 += batch_sum_n2
        max_graphs = max(max_graphs, len(counts))
        max_atoms = max(max_atoms, batch_atoms)
        max_sum_n2 = max(max_sum_n2, batch_sum_n2)

    scan_seconds = time.perf_counter() - start
    graphs_per_step = graphs / max(batches, 1)
    atoms_per_step = atoms / max(batches, 1)
    sum_n2_per_step = sum_n2 / max(batches, 1)
    print(
        f"loader/{name}: len_estimate={estimated_batches} len_scanned={batches} "
        f"graphs_per_step={graphs_per_step:.3f} atoms_per_step={atoms_per_step:.3f} "
        f"sum_n2_per_step={sum_n2_per_step:.3f} max_graphs_step={max_graphs} "
        f"max_atoms_step={max_atoms} max_sum_n2_step={max_sum_n2} "
        f"scan_seconds={scan_seconds:.3f}"
    )
    if run is not None:
        for key, value in {
            "len_estimate": float(estimated_batches),
            "len_scanned": float(batches),
            "graphs_per_step": float(graphs_per_step),
            "atoms_per_step": float(atoms_per_step),
            "sum_n2_per_step": float(sum_n2_per_step),
            "max_graphs_step": float(max_graphs),
            "max_atoms_step": float(max_atoms),
            "max_sum_n2_step": float(max_sum_n2),
            "scan_seconds": float(scan_seconds),
        }.items():
            run.summary[f"loader/{name}/{key}"] = value


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    if args.synthetic_samples > 0:
        base = SyntheticOMolDataset(
            args.synthetic_samples,
            seed=args.seed,
            min_atoms=args.synthetic_min_atoms,
            max_atoms=args.synthetic_max_atoms,
        )
        train_dataset, val_dataset = split_train_val(base, train_size=0.8, seed=args.seed)
        return train_dataset, val_dataset, val_dataset

    if args.train_data_path is None:
        raise ValueError("--train-data-path is required unless --synthetic-samples is positive")
    train_base = FairChemOMolDataset(args.train_data_path, keep_in_memory=args.keep_in_memory)
    train_base = apply_debug_subset(train_base, args.debug_subset)

    validation_mode = args.validation_mode.lower().replace("-", "_")
    if validation_mode == "train_split":
        train_dataset, val_dataset = split_train_val(train_base, train_size=args.train_size, seed=args.seed)
        test_dataset = val_dataset
        if args.val_data_path is not None:
            test_dataset = apply_debug_subset(
                FairChemOMolDataset(args.val_data_path, keep_in_memory=args.keep_in_memory),
                args.debug_subset,
            )
        return train_dataset, val_dataset, test_dataset
    if validation_mode == "heldout":
        if args.val_data_path is None:
            raise ValueError("--val-data-path is required when --validation-mode heldout")
        val_dataset = apply_debug_subset(
            FairChemOMolDataset(args.val_data_path, keep_in_memory=args.keep_in_memory),
            args.debug_subset,
        )
        return train_base, val_dataset, val_dataset
    raise ValueError("--validation-mode must be one of {'heldout', 'train_split'}")


def apply_rotation_augmentation(batch: OMolBatch, mode: str) -> OMolBatch:
    mode = mode.lower()
    if mode in {"off", "none", "false", "no"}:
        return batch
    rotations = random_rotation_matrices(batch.coords.shape[0], device=batch.coords.device, dtype=batch.coords.dtype)
    if mode in {"o3", "reflection", "reflect"}:
        reflect = torch.where(
            torch.rand(batch.coords.shape[0], device=batch.coords.device) < 0.5,
            -1.0,
            1.0,
        ).to(dtype=batch.coords.dtype)
        rotations[:, :, 0] = rotations[:, :, 0] * reflect[:, None]
    elif mode not in {"so3", "rotation", "rot"}:
        raise ValueError("--train-augmentation must be one of {'off', 'so3', 'o3'}")
    batch.coords = torch.einsum("bij,bnj->bni", rotations, batch.coords).masked_fill(batch.pad_mask[..., None], 0.0)
    batch.forces = torch.einsum("bij,bnj->bni", rotations, batch.forces).masked_fill(batch.pad_mask[..., None], 0.0)
    return batch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def close_dataset_handles(dataset: Dataset, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    marker = id(dataset)
    if marker in seen:
        return
    seen.add(marker)
    if hasattr(dataset, "close"):
        dataset.close()  # type: ignore[attr-defined]
    if isinstance(dataset, Subset):
        close_dataset_handles(dataset.dataset, seen)


def clone_model_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_model = unwrap_model(model)
    return {key: value.detach().cpu().clone() for key, value in state_model.state_dict().items()}


def build_omol_model(args: argparse.Namespace, *, device: torch.device) -> torch.nn.Module:
    backend = str(args.model_backend).lower().replace("-", "_")
    if backend == "matterformer":
        return MatterformerOMolForceField(
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
            tetra_pair_force_mode=args.tetra_pair_force_mode,
            tetra_pair_k_neighbors=args.tetra_pair_k_neighbors,
            tetra_pair_feature_dim=args.tetra_pair_feature_dim,
            tetra_pair_element_dim=args.tetra_pair_element_dim,
            tetra_pair_gate_init=args.tetra_pair_gate_init,
            tetra_pair_geometry_strict=args.tetra_pair_geometry_strict,
            force_head_mode=args.force_head_mode,
            readout_head_mode=args.readout_head_mode,
            tetra_readout_mode=args.tetra_readout_mode,
            tetra_irrep_scalar_input=args.tetra_irrep_scalar_input,
            readout_activation=args.readout_activation,
            runtime_mode=args.omol_runtime_mode,
        ).to(device)
    if backend == "allscaip_direct":
        allscaip_config = load_allscaip_config(args.allscaip_config_json)
        frequency_list = parse_int_list(args.allscaip_frequency_list)
        max_batch_size = args.allscaip_max_batch_size
        if max_batch_size is None:
            max_batch_size = max(int(args.batch_size), int(args.val_batch_size or args.batch_size))
        allscaip_compile = bool(args.compile if args.allscaip_compile is None else args.allscaip_compile)
        return MatterformerAllScAIPDirectForceField(
            allscaip_config,
            hidden_size=args.allscaip_hidden_size,
            num_layers=args.allscaip_num_layers,
            use_compile=allscaip_compile,
            use_padding=args.allscaip_use_padding,
            max_atoms=args.allscaip_max_atoms or args.max_atoms_per_batch or 12000,
            max_batch_size=max_batch_size,
            max_radius=args.allscaip_max_radius,
            knn_k=args.allscaip_knn_k,
            knn_pad_size=args.allscaip_knn_pad_size,
            atten_name=args.allscaip_atten_name,
            atten_num_heads=args.allscaip_atten_num_heads,
            freequency_list=frequency_list,
            mlp_dropout=args.dropout,
            atten_dropout=args.attn_dropout,
            use_chunked_graph=args.allscaip_use_chunked_graph,
            graph_chunk_size=args.allscaip_graph_chunk_size,
            preprocess_on_cpu=args.allscaip_preprocess_on_cpu,
        ).to(device)
    raise ValueError(f"Unknown --model-backend {args.model_backend!r}")


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    args: argparse.Namespace,
    best_val_total: float,
    global_step: int,
    epoch: int,
) -> dict[str, object]:
    return {
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "ema_state_dict": None if ema is None else ema.state_dict(),
        "args": vars(args),
        "best_val_total": float(best_val_total),
        "global_step": int(global_step),
        "epoch": int(epoch),
    }


def evaluate_with_ema(
    *,
    model: torch.nn.Module,
    ema: EMA | None,
    ema_ready: bool,
    criterion: OMolDirectForceLoss,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    bf16_enabled: bool,
) -> dict[str, float]:
    backup = None
    if ema is not None and ema_ready:
        backup = ema.apply(unwrap_model(model))
    try:
        return evaluate(
            model=model,
            criterion=criterion,
            loader=loader,
            device=device,
            max_batches=max_batches,
            bf16_enabled=bf16_enabled,
        )
    finally:
        if ema is not None and backup is not None:
            ema.restore(unwrap_model(model), backup)


def initialize_model_parameters(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    *,
    bf16_enabled: bool,
) -> None:
    if len(dataset) == 0:
        raise ValueError("Cannot initialize OMol model from an empty training dataset")
    init_batch = collate_omol([dataset[0]]).to(device)
    model.eval()
    with torch.no_grad():
        with make_autocast_context(device, bf16_enabled):
            _ = model(
                init_batch.atomic_numbers,
                init_batch.coords,
                init_batch.pad_mask,
                charge=init_batch.charge,
                spin=init_batch.spin,
            )


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    criterion: OMolDirectForceLoss,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    bf16_enabled: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    graphs = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        with make_autocast_context(device, bf16_enabled):
            predictions = model(
                batch.atomic_numbers,
                batch.coords,
                batch.pad_mask,
                charge=batch.charge,
                spin=batch.spin,
            )
            output = criterion(predictions, batch)
        batch_graphs = int(batch.energy.shape[0])
        graphs += batch_graphs
        for key, value in output.diagnostics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_graphs
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    return {key: value / max(graphs, 1) for key, value in totals.items()}


def log_metrics(run, metrics: dict[str, float], *, prefix: str, step: int, extra: dict[str, float] | None = None) -> None:
    if run is None:
        return
    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
    if extra:
        payload.update(extra)
    run.log(payload, step=step)


def batch_runtime_metrics(
    batch: OMolBatch,
    *,
    step_seconds: float,
    tokens_seen: int,
    padded_tokens_seen: int,
    train_seconds_seen: float,
    num_parameters: int,
    flops_coef: float,
) -> dict[str, float]:
    real_atoms = int(batch.num_atoms.sum().item())
    padded_slots = int(batch.atomic_numbers.numel())
    graphs = int(batch.energy.shape[0])
    max_atoms_graph = int(batch.num_atoms.max().item()) if batch.num_atoms.numel() > 0 else 0
    sum_n2 = int(batch.num_atoms.long().pow(2).sum().item())
    metrics = {
        "batch/graphs": float(graphs),
        "batch/real_atoms": float(real_atoms),
        "batch/padded_slots": float(padded_slots),
        "batch/padding_ratio": float(padded_slots / max(real_atoms, 1)),
        "batch/max_atoms_graph": float(max_atoms_graph),
        "batch/sum_n2": float(sum_n2),
        "perf/step_s": float(step_seconds),
        "perf/real_atoms_per_sec": float(real_atoms / max(step_seconds, 1e-12)),
        "perf/padded_slots_per_sec": float(padded_slots / max(step_seconds, 1e-12)),
        "perf/sum_n2_per_sec": float(sum_n2 / max(step_seconds, 1e-12)),
        "trainer/tokens_processed": float(tokens_seen),
        "trainer/padded_tokens_processed": float(padded_tokens_seen),
        "trainer/train_seconds": float(train_seconds_seen),
        "trainer/total_flops_used": float(flops_coef * tokens_seen * num_parameters),
    }
    if torch.cuda.is_available():
        metrics["profile/max_mem_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
        metrics["profile/max_mem_reserved_gb"] = float(torch.cuda.max_memory_reserved() / 1024**3)
    return metrics


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    device = default_device()
    args.model_backend = str(args.model_backend).lower().replace("-", "_")
    args.hybrid_config = load_hybrid_config(args.hybrid_config_json) if args.model_backend == "matterformer" else None
    args.allscaip_strict_config = (
        load_allscaip_config(args.allscaip_strict_config_json)
        if args.model_backend == "allscaip_direct"
        else None
    )

    train_dataset, val_dataset, test_dataset = build_datasets(args)
    train_loader = build_loader(
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
    val_loader = build_loader(
        val_dataset,
        batch_size=args.val_batch_size or args.batch_size,
        max_graphs_per_batch=args.max_graphs_per_batch_val or args.max_graphs_per_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_atoms=args.max_atoms_per_batch_val or args.max_atoms_per_batch,
        max_edges=args.max_edges_per_batch_val or args.max_edges_per_batch,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        batching_mode=args.batching_mode,
        bucket_window_size=args.bucket_window_size,
        bucket_shuffle_groups=args.bucket_shuffle_groups,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=args.val_batch_size or args.batch_size,
        max_graphs_per_batch=args.max_graphs_per_batch_val or args.max_graphs_per_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_atoms=args.max_atoms_per_batch_val or args.max_atoms_per_batch,
        max_edges=args.max_edges_per_batch_val or args.max_edges_per_batch,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        batching_mode=args.batching_mode,
        bucket_window_size=args.bucket_window_size,
        bucket_shuffle_groups=args.bucket_shuffle_groups,
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
    model = build_omol_model(args, device=device)
    if hasattr(model, "allscaip_config"):
        args.allscaip_config_resolved = dict(getattr(model, "allscaip_config"))
        validate_allscaip_strict_config(model, args.allscaip_strict_config, stage="model_init")
    initialize_model_parameters(model, train_dataset, device, bf16_enabled=args.bf16)
    if hasattr(model, "allscaip_config"):
        validate_allscaip_strict_config(model, args.allscaip_strict_config, stage="first_forward")
        args.allscaip_config_runtime = collect_allscaip_runtime_config(model)
    close_dataset_handles(train_dataset)
    close_dataset_handles(val_dataset)
    close_dataset_handles(test_dataset)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if args.compile and args.model_backend == "matterformer":
        if args.compile_scope == "model":
            model = torch.compile(model, mode=args.compile_mode)  # type: ignore[assignment]
        elif args.compile_scope == "trunk_flat":
            base_model = unwrap_model(model)
            if args.omol_runtime_mode == "internal_flat_tetra":
                base_model.trunk.forward_flat_tetra = torch.compile(  # type: ignore[method-assign]
                    base_model.trunk.forward_flat_tetra,
                    mode=args.compile_mode,
                )
            elif args.omol_runtime_mode == "internal_flat_hybrid":
                base_model.trunk.forward_flat_hybrid = torch.compile(  # type: ignore[method-assign]
                    base_model.trunk.forward_flat_hybrid,
                    mode=args.compile_mode,
                )
            else:
                base_model.trunk.forward = torch.compile(  # type: ignore[method-assign]
                    base_model.trunk.forward,
                    mode=args.compile_mode,
                )
            print(f"compile: compiled Matterformer trunk only scope={args.compile_scope} mode={args.compile_mode}")
        elif args.compile_scope == "none":
            print("compile: disabled by --compile-scope none")
        else:
            raise ValueError(f"Unknown --compile-scope {args.compile_scope!r}")
    elif args.compile:
        print("compile: AllScAIP backend skips the outer torch.compile wrapper; use --allscaip-compile to control its internal compile path")

    optimizer = build_optimizer(model, args)
    scheduler_max_iters = resolve_scheduler_max_iters(args, train_loader)
    scheduler_min_lrs = resolve_scheduler_min_lrs(optimizer, args)
    args.scheduler_resolved = {
        "name": "cosine_warmup",
        "warmup_steps": int(args.warmup_steps),
        "max_iters": int(scheduler_max_iters),
        "lr_min": float(args.lr_min),
        "base_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "min_lrs": [float(value) for value in scheduler_min_lrs],
    }
    print("scheduler_resolved: " + json.dumps(args.scheduler_resolved, sort_keys=True))
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=args.warmup_steps,
        max_iters=scheduler_max_iters,
        min_lrs=scheduler_min_lrs,
    )
    run = maybe_init_wandb(args)

    checkpoint_path = Path(args.output)
    latest_checkpoint_path = checkpoint_path.with_name("latest.pt")
    if args.save_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_total = float("inf")
    best_state_in_memory: dict[str, torch.Tensor] | None = None
    best_ema_state_in_memory: dict[str, torch.Tensor] | None = None
    global_step = 0
    epoch = 0
    tokens_seen = 0
    padded_tokens_seen = 0
    train_seconds_seen = 0.0
    skipped_updates = 0

    if args.resume_checkpoint is not None and args.warm_start_checkpoint is not None:
        raise ValueError("Only one of --resume-checkpoint or --warm-start-checkpoint may be set")
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_total = float(checkpoint.get("best_val_total", float("inf")))
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint.get("epoch", 0))
        print(f"resume_checkpoint: path={args.resume_checkpoint} step={global_step} best_val_total={best_val_total:.6f}")
    elif args.warm_start_checkpoint is not None:
        checkpoint = torch.load(args.warm_start_checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        print(f"warm_start_checkpoint: path={args.warm_start_checkpoint}")

    base_model = unwrap_model(model)
    num_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    non_trainable_parameters = num_parameters - trainable_parameters
    print(f"dataset: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    print(
        f"model: backend={args.model_backend} stream_type={getattr(base_model, 'stream_type', 'unknown')} "
        f"readout_head_mode={getattr(base_model, 'readout_head_mode', 'n/a')} "
        f"tetra_readout_mode={getattr(base_model, 'tetra_readout_mode', 'n/a')} "
        f"tetra_irrep_scalar_input={getattr(base_model, 'tetra_irrep_scalar_input', 'n/a')} "
        f"parameters={num_parameters} trainable={trainable_parameters} non_trainable={non_trainable_parameters}"
    )
    print(
        f"runtime: omol_runtime_mode={args.omol_runtime_mode} batching_mode={args.batching_mode} "
        f"bucket_window_size={args.bucket_window_size} bucket_shuffle_groups={args.bucket_shuffle_groups} "
        f"max_graphs_per_batch={args.max_graphs_per_batch} "
        f"max_graphs_per_batch_val={args.max_graphs_per_batch_val or args.max_graphs_per_batch}"
    )
    print(f"precision: bf16={args.bf16} matmul={args.float32_matmul_precision}")
    print(f"normalization: rmsd={args.normalizer_rmsd} energy_weight={args.energy_weight} force_weight={args.force_weight}")
    print("scheduler: " + json.dumps(args.scheduler_resolved, sort_keys=True))
    summarize_loader_startup("train", train_loader, run)
    summarize_loader_startup("val", val_loader, run)
    if hasattr(base_model, "allscaip_config"):
        print("allscaip_config_resolved: " + json.dumps(args.allscaip_config_runtime, sort_keys=True))
    if run is not None:
        run.summary["model/params/total"] = num_parameters
        run.summary["model/params/trainable"] = trainable_parameters
        run.summary["model/params/non_trainable"] = non_trainable_parameters
        run.summary["model/backend"] = args.model_backend
        log_metrics(
            run,
            {},
            prefix="model",
            step=0,
            extra={
                "model/params/total": float(num_parameters),
                "model/params/trainable": float(trainable_parameters),
                "model/params/non_trainable": float(non_trainable_parameters),
            },
        )

    while (args.max_steps <= 0 or global_step < args.max_steps) and (
        args.max_epochs is None or epoch < args.max_epochs
    ):
        epoch += 1
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)  # type: ignore[attr-defined]
        model.train()
        epoch_totals: dict[str, float] = {}
        epoch_graphs = 0
        data_wait_start = time.perf_counter()
        for batch in train_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            next_step = global_step + 1
            profile_this_step = (
                args.profile_steps > 0
                and next_step > args.profile_warmup_steps
                and next_step <= args.profile_warmup_steps + args.profile_steps
            )
            data_wait_ms = (time.perf_counter() - data_wait_start) * 1000.0
            step_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            timer = CudaPhaseTimer(profile_this_step)
            with timer.phase("h2d"):
                batch = batch.to(device)
            with timer.phase("augmentation"):
                batch = apply_rotation_augmentation(batch, args.train_augmentation)
            with timer.phase("forward"):
                with make_autocast_context(device, args.bf16):
                    predictions = model(
                        batch.atomic_numbers,
                        batch.coords,
                        batch.pad_mask,
                        charge=batch.charge,
                        spin=batch.spin,
                    )
                    output = criterion(predictions, batch)
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(output.loss.detach().float().item())
            skip_reason = ""
            if not math.isfinite(loss_value):
                skip_reason = "nonfinite_loss"
            elif args.skip_loss_above > 0.0 and loss_value > args.skip_loss_above:
                skip_reason = "loss_above_threshold"
            if skip_reason:
                step_seconds = time.perf_counter() - step_start
                global_step += 1
                skipped_updates += 1
                batch_graphs = int(batch.energy.shape[0])
                batch_real_atoms = int(batch.num_atoms.sum().item())
                batch_padded_slots = int(batch.atomic_numbers.numel())
                tokens_seen += batch_real_atoms
                padded_tokens_seen += batch_padded_slots
                train_seconds_seen += step_seconds
                skip_metrics = {
                    "trainer/global_step": float(global_step),
                    "trainer/epoch": float(epoch),
                    "trainer/skipped_updates": float(skipped_updates),
                    "trainer/skip_loss_value": loss_value,
                    "trainer/skip_threshold": float(args.skip_loss_above),
                    "trainer/skip_graphs": float(batch_graphs),
                    "trainer/skip_atoms": float(batch_real_atoms),
                    "trainer/skip_padded_tokens": float(batch_padded_slots),
                    "trainer/skip_step_seconds": float(step_seconds),
                }
                log_metrics(run, {}, prefix="skip", step=global_step, extra=skip_metrics)
                print(
                    f"skip_update step={global_step:07d} reason={skip_reason} "
                    f"loss={loss_value:.6g} threshold={args.skip_loss_above:.6g}"
                )
                data_wait_start = time.perf_counter()
                continue
            with timer.phase("backward"):
                output.loss.backward()
            with timer.phase("optimizer"):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                scheduler.step()
            step_seconds = time.perf_counter() - step_start
            global_step += 1
            if ema is not None and global_step >= args.ema_warmup_steps:
                ema.update(unwrap_model(model))

            if profile_this_step:
                profile_metrics = {"profile/data_wait_ms": float(data_wait_ms)}
                profile_metrics.update(timer.results_ms())
                log_metrics(run, {}, prefix="profile", step=global_step, extra=profile_metrics)
                rendered = " ".join(f"{key}={value:.3f}" for key, value in sorted(profile_metrics.items()))
                print(f"profile step={global_step:07d} {rendered}")

            batch_graphs = int(batch.energy.shape[0])
            batch_real_atoms = int(batch.num_atoms.sum().item())
            batch_padded_slots = int(batch.atomic_numbers.numel())
            tokens_seen += batch_real_atoms
            padded_tokens_seen += batch_padded_slots
            train_seconds_seen += step_seconds
            epoch_graphs += batch_graphs
            for key, value in output.diagnostics.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value.detach().item()) * batch_graphs

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                metrics = {key: value / max(epoch_graphs, 1) for key, value in epoch_totals.items()}
                extra = {
                    "trainer/global_step": float(global_step),
                    "trainer/epoch": float(epoch),
                    "optim/lr": float(optimizer.param_groups[0]["lr"]),
                    "optim/grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                }
                extra.update(
                    batch_runtime_metrics(
                        batch,
                        step_seconds=step_seconds,
                        tokens_seen=tokens_seen,
                        padded_tokens_seen=padded_tokens_seen,
                        train_seconds_seen=train_seconds_seen,
                        num_parameters=num_parameters,
                        flops_coef=args.flops_coef,
                    )
                )
                extra.update(
                    {f"train_step/{key}": float(value.detach().item()) for key, value in output.diagnostics.items()}
                )
                base_model = unwrap_model(model)
                if base_model.stream_type == "tetra":
                    extra.update({key: float(value) for key, value in base_model.collect_sg_diagnostics().items()})
                log_metrics(run, metrics, prefix="train", step=global_step, extra=extra)

            if args.val_estimate_every_steps > 0 and global_step % args.val_estimate_every_steps == 0:
                val_estimate = evaluate_with_ema(
                    model=model,
                    ema=ema,
                    ema_ready=global_step >= args.ema_warmup_steps,
                    criterion=criterion,
                    loader=val_loader,
                    device=device,
                    max_batches=args.val_estimate_batches,
                    bf16_enabled=args.bf16,
                )
                log_metrics(run, val_estimate, prefix="val_estimate", step=global_step)

            if args.full_val_every_steps > 0 and global_step % args.full_val_every_steps == 0:
                val_metrics = evaluate_with_ema(
                    model=model,
                    ema=ema,
                    ema_ready=global_step >= args.ema_warmup_steps,
                    criterion=criterion,
                    loader=val_loader,
                    device=device,
                    max_batches=args.limit_val_batches,
                    bf16_enabled=args.bf16,
                )
                val_total = 0.5 * (val_metrics.get("e_mae_per_atom", 0.0) + val_metrics.get("f_mae", 0.0))
                log_metrics(run, val_metrics, prefix="val", step=global_step, extra={"val/total": val_total})
                is_best = val_total < best_val_total
                if is_best:
                    best_val_total = val_total
                    best_state_in_memory = clone_model_state_to_cpu(model)
                    best_ema_state_in_memory = None if ema is None else ema.state_dict()
                if args.save_checkpoint:
                    payload = build_checkpoint_payload(
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        best_val_total=best_val_total,
                        global_step=global_step,
                        epoch=epoch,
                    )
                    torch.save(payload, latest_checkpoint_path)
                    if is_best:
                        torch.save(payload, checkpoint_path)
            data_wait_start = time.perf_counter()

        train_epoch = {key: value / max(epoch_graphs, 1) for key, value in epoch_totals.items()}
        print(
            f"epoch={epoch:03d} step={global_step:07d} "
            f"loss={train_epoch.get('loss', 0.0):.6f} "
            f"e_mae_per_atom={train_epoch.get('e_mae_per_atom', 0.0):.3f} "
            f"f_mae={train_epoch.get('f_mae', 0.0):.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

    val_metrics = evaluate_with_ema(
        model=model,
        ema=ema,
        ema_ready=global_step >= args.ema_warmup_steps,
        criterion=criterion,
        loader=val_loader,
        device=device,
        max_batches=args.limit_val_batches,
        bf16_enabled=args.bf16,
    )
    val_total = 0.5 * (val_metrics.get("e_mae_per_atom", 0.0) + val_metrics.get("f_mae", 0.0))
    is_best = val_total < best_val_total
    if is_best:
        best_val_total = val_total
        best_state_in_memory = clone_model_state_to_cpu(model)
        best_ema_state_in_memory = None if ema is None else ema.state_dict()
    if args.save_checkpoint:
        payload = build_checkpoint_payload(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            best_val_total=best_val_total,
            global_step=global_step,
            epoch=epoch,
        )
        torch.save(payload, latest_checkpoint_path)
        if is_best:
            torch.save(payload, checkpoint_path)
    print(
        f"final_val step={global_step:07d} total={val_total:.3f} "
        f"e_mae_per_atom={val_metrics.get('e_mae_per_atom', 0.0):.3f} "
        f"f_mae={val_metrics.get('f_mae', 0.0):.3f} best_total={best_val_total:.3f}"
    )
    log_metrics(run, val_metrics, prefix="val", step=global_step, extra={"val/total": val_total})

    if best_state_in_memory is not None:
        unwrap_model(model).load_state_dict(best_state_in_memory)
        if ema is not None and best_ema_state_in_memory is not None:
            ema.load_state_dict(best_ema_state_in_memory)
    elif args.save_checkpoint and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
    test_metrics = evaluate_with_ema(
        model=model,
        ema=ema,
        ema_ready=global_step >= args.ema_warmup_steps,
        criterion=criterion,
        loader=test_loader,
        device=device,
        max_batches=args.limit_test_batches,
        bf16_enabled=args.bf16,
    )
    test_total = 0.5 * (test_metrics.get("e_mae_per_atom", 0.0) + test_metrics.get("f_mae", 0.0))
    print(
        f"test total={test_total:.3f} "
        f"e_mae_per_atom={test_metrics.get('e_mae_per_atom', 0.0):.3f} "
        f"f_mae={test_metrics.get('f_mae', 0.0):.3f}"
    )
    log_metrics(run, test_metrics, prefix="test", step=global_step, extra={"test/total": test_total, "val/best_total": best_val_total})
    if run is not None:
        run.summary["val/best_total"] = best_val_total
        run.summary["test/total"] = test_total
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Matterformer direct-force model on OMol")
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--validation-mode", type=str, default="heldout", choices=["heldout", "train_split"])
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument("--keep-in-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-subset", type=str, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--synthetic-min-atoms", type=int, default=2)
    parser.add_argument("--synthetic-max-atoms", type=int, default=8)
    parser.add_argument("--output", type=str, default="./outputs/Matterformer_OMol/checkpoints/omol_forcefield.pt")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--element-refs-json", type=str, default="configs/omol/element_refs.json")
    parser.add_argument(
        "--model-backend",
        type=str,
        default="matterformer",
        choices=["matterformer", "allscaip_direct", "allscaip-direct"],
    )
    parser.add_argument("--hybrid-config-json", type=str, default="configs/omol/scalar_sit_d768_l8.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--max-graphs-per-batch", type=int, default=None)
    parser.add_argument("--max-graphs-per-batch-val", type=int, default=None)
    parser.add_argument("--max-atoms-per-batch", type=int, default=None)
    parser.add_argument("--max-atoms-per-batch-val", type=int, default=None)
    parser.add_argument("--max-edges-per-batch", type=int, default=None)
    parser.add_argument("--max-edges-per-batch-val", type=int, default=None)
    parser.add_argument("--batching-mode", type=str, default="random", choices=["random", "bucketed"])
    parser.add_argument("--bucket-window-size", type=int, default=4096)
    parser.add_argument("--bucket-shuffle-groups", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--muon-adam-lr",
        type=float,
        default=None,
        help="Aux Adam learning rate for parameters not assigned to Muon; defaults to --lr.",
    )
    parser.add_argument(
        "--muon-adam-weight-decay",
        type=float,
        default=None,
        help="Aux Adam weight decay for parameters not assigned to Muon; defaults to --weight-decay.",
    )
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-eps", type=float, default=1e-10)
    parser.add_argument(
        "--muon-hidden-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using Muon, restrict Muon updates to hidden trunk block matrix weights.",
    )
    parser.add_argument("--muon-min-ndim", type=int, default=2)
    parser.add_argument(
        "--muon-exclude-name-fragments",
        type=str,
        default="embed,embedding,head,readout,rope,freq",
        help="Comma-separated lowercase name fragments excluded from the Muon parameter group.",
    )
    parser.add_argument(
        "--muon-platonic-kernel-view",
        type=str,
        default="slice",
        choices=["slice", "conv"],
        help=(
            "Muon view for 3D PlatonicLinear kernels. 'slice' keeps [G,out,in] batched "
            "matrix updates; 'conv' uses [out,G*in] filter-bank updates."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help=(
            "Final cosine LR floor for the reference Adam LR. For Muon, this is applied to the aux Adam "
            "LR and scaled proportionally for the Muon LR so both groups keep the same decay ratio."
        ),
    )
    parser.add_argument("--normalizer-rmsd", type=float, default=1.433569)
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--energy-loss", type=str, default="per_atom_mae", choices=["mae", "per_atom_mae"])
    parser.add_argument("--force-loss", type=str, default="l2norm", choices=["mae", "l2norm"])
    parser.add_argument(
        "--skip-loss-above",
        type=float,
        default=0.0,
        help="Skip backward/optimizer/scheduler update when scalar train loss exceeds this value; <=0 disables.",
    )
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--chgspin-mode", type=str, default="add", choices=["off", "add", "concat"])
    parser.add_argument("--chgspin-emb-dim", type=int, default=None)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--pair-n-rbf", type=int, default=16)
    parser.add_argument("--pair-rbf-max", type=float, default=6.0)
    parser.add_argument("--tetra-pair-force-mode", type=str, default="off", choices=["off", "residual"])
    parser.add_argument("--tetra-pair-k-neighbors", type=int, default=30)
    parser.add_argument("--tetra-pair-feature-dim", type=int, default=128)
    parser.add_argument("--tetra-pair-element-dim", type=int, default=32)
    parser.add_argument("--tetra-pair-gate-init", type=float, default=0.0)
    parser.add_argument("--tetra-pair-geometry-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--force-head-mode",
        default="auto",
        choices=["auto", "pairwise", "direct", "direct_3d", "non_equivariant", "tetra_vector"],
    )
    parser.add_argument("--readout-head-mode", type=str, default="dense", choices=["dense", "platonic"])
    parser.add_argument("--tetra-readout-mode", type=str, default="platonic", choices=["platonic", "irrep"])
    parser.add_argument("--tetra-irrep-scalar-input", type=str, default="rho1", choices=["rho1", "invariants"])
    parser.add_argument("--readout-activation", type=str, default=None, choices=["gelu", "silu", "relu", "mish", "sin"])
    parser.add_argument(
        "--omol-runtime-mode",
        type=str,
        default="padded",
        choices=["padded", "internal_flat_tetra", "internal_flat_hybrid"],
    )
    parser.add_argument("--allscaip-config-json", type=str, default=None)
    parser.add_argument("--allscaip-strict-config-json", type=str, default=None)
    parser.add_argument("--allscaip-hidden-size", type=int, default=128)
    parser.add_argument("--allscaip-num-layers", type=int, default=5)
    parser.add_argument("--allscaip-atten-num-heads", type=int, default=2)
    parser.add_argument("--allscaip-max-atoms", type=int, default=None)
    parser.add_argument("--allscaip-max-batch-size", type=int, default=None)
    parser.add_argument("--allscaip-max-radius", type=float, default=6.0)
    parser.add_argument("--allscaip-knn-k", type=int, default=30)
    parser.add_argument("--allscaip-knn-pad-size", type=int, default=30)
    parser.add_argument("--allscaip-atten-name", type=str, default="memory_efficient", choices=["math", "memory_efficient", "flash"])
    parser.add_argument("--allscaip-frequency-list", type=str, default=None)
    parser.add_argument("--allscaip-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allscaip-use-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allscaip-use-chunked-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allscaip-graph-chunk-size", type=int, default=512)
    parser.add_argument("--allscaip-preprocess-on-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-augmentation", type=str, default="o3", choices=["off", "so3", "o3"])
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--float32-matmul-precision", type=str, default="highest")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument(
        "--compile-scope",
        type=str,
        default="model",
        choices=["model", "trunk_flat", "none"],
        help=(
            "Matterformer torch.compile target. 'model' compiles the whole OMol wrapper; "
            "'trunk_flat' compiles only the flat trunk path and leaves dynamic padding/scatter eager."
        ),
    )
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--flops-coef", type=float, default=72.0)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--profile-warmup-steps", type=int, default=5)
    parser.add_argument("--val-estimate-every-steps", type=int, default=500)
    parser.add_argument("--val-estimate-batches", type=int, default=16)
    parser.add_argument("--full-val-every-steps", type=int, default=2000)
    parser.add_argument("--limit-val-batches", type=int, default=500)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default="Matterformer_OMol")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=str, default="./outputs/Matterformer_OMol/wandb")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
