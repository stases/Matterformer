#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager, nullcontext
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
from matterformer.models import MatterformerOMolForceField
from matterformer.tasks import OMolDirectForceLoss, load_omol_element_references
from matterformer.utils import EMA, CosineWarmupScheduler, default_device, random_rotation_matrices, seed_everything


def load_hybrid_config(value: str | None) -> dict | None:
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
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    max_atoms: int | None,
    max_edges: int | None,
    seed: int,
    prefetch_factor: int | None,
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
        sampler = OMolDynamicBatchSampler(
            dataset,
            max_batch_size=batch_size,
            max_atoms=max_atoms,
            max_edges=max_edges,
            shuffle=shuffle,
            seed=seed,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, **kwargs)
        loader.set_epoch = sampler.set_epoch  # type: ignore[attr-defined]
        return loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


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
    model: MatterformerOMolForceField,
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
    model: MatterformerOMolForceField,
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
    model: MatterformerOMolForceField,
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


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    device = default_device()
    args.hybrid_config = load_hybrid_config(args.hybrid_config_json)

    train_dataset, val_dataset, test_dataset = build_datasets(args)
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
    val_loader = build_loader(
        val_dataset,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_atoms=args.max_atoms_per_batch_val or args.max_atoms_per_batch,
        max_edges=args.max_edges_per_batch_val or args.max_edges_per_batch,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_atoms=args.max_atoms_per_batch_val or args.max_atoms_per_batch,
        max_edges=args.max_edges_per_batch_val or args.max_edges_per_batch,
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
    initialize_model_parameters(model, train_dataset, device, bf16_enabled=args.bf16)
    close_dataset_handles(train_dataset)
    close_dataset_handles(val_dataset)
    close_dataset_handles(test_dataset)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, warmup=args.warmup_steps, max_iters=args.max_steps)
    run = maybe_init_wandb(args)

    checkpoint_path = Path(args.output)
    if args.save_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_total = float("inf")
    best_state_in_memory: dict[str, torch.Tensor] | None = None
    best_ema_state_in_memory: dict[str, torch.Tensor] | None = None
    global_step = 0
    epoch = 0

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
    print(f"dataset: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    print(f"model: stream_type={base_model.stream_type} parameters={num_parameters}")
    print(f"precision: bf16={args.bf16} matmul={args.float32_matmul_precision}")
    print(f"normalization: rmsd={args.normalizer_rmsd} energy_weight={args.energy_weight} force_weight={args.force_weight}")

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
            if profile_this_step and torch.cuda.is_available():
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
            with timer.phase("backward"):
                output.loss.backward()
            with timer.phase("optimizer"):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                scheduler.step()
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
                if val_total < best_val_total:
                    best_val_total = val_total
                    best_state_in_memory = clone_model_state_to_cpu(model)
                    best_ema_state_in_memory = None if ema is None else ema.state_dict()
                    if args.save_checkpoint:
                        torch.save(
                            build_checkpoint_payload(
                                model=model,
                                ema=ema,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                args=args,
                                best_val_total=best_val_total,
                                global_step=global_step,
                                epoch=epoch,
                            ),
                            checkpoint_path,
                        )
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
    if val_total < best_val_total:
        best_val_total = val_total
        best_state_in_memory = clone_model_state_to_cpu(model)
        best_ema_state_in_memory = None if ema is None else ema.state_dict()
        if args.save_checkpoint:
            torch.save(
                build_checkpoint_payload(
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    best_val_total=best_val_total,
                    global_step=global_step,
                    epoch=epoch,
                ),
                checkpoint_path,
            )
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
    parser.add_argument("--hybrid-config-json", type=str, default="configs/omol/scalar_sit_d768_l8.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--max-atoms-per-batch", type=int, default=None)
    parser.add_argument("--max-atoms-per-batch-val", type=int, default=None)
    parser.add_argument("--max-edges-per-batch", type=int, default=None)
    parser.add_argument("--max-edges-per-batch-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--normalizer-rmsd", type=float, default=1.433569)
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--energy-loss", type=str, default="per_atom_mae", choices=["mae", "per_atom_mae"])
    parser.add_argument("--force-loss", type=str, default="l2norm", choices=["mae", "l2norm"])
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
    parser.add_argument(
        "--force-head-mode",
        default="auto",
        choices=["auto", "pairwise", "direct", "direct_3d", "non_equivariant", "tetra_vector"],
    )
    parser.add_argument("--train-augmentation", type=str, default="o3", choices=["off", "so3", "o3"])
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--float32-matmul-precision", type=str, default="highest")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=10)
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
