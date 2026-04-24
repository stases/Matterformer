#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader, Subset

from matterformer.data import BWDBDataset, collate_mof_stage1
from matterformer.geometry.lattice import lattice_latent_to_gram
from matterformer.models import MOFStage1EDMModel
from matterformer.tasks import (
    MOFStage1EDMLoss,
    MOFStage1EDMPreconditioner,
    lattice_latent_to_lattice_params,
    lattice_params_to_lattice_latent,
    mod1,
    mof_stage1_edm_sampler,
    wrap_frac,
)
from matterformer.utils import CosineWarmupScheduler, EMA, default_device, seed_everything


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_configure_cuda_sdpa_from_env() -> None:
    if os.environ.get("DISABLE_CUDNN_SDPA", "0") == "1":
        torch.backends.cuda.enable_cudnn_sdp(False)


def sampler_kwargs_from_args(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "num_steps": args.sample_num_steps,
        "sigma_min": args.sample_sigma_min,
        "sigma_max": args.sample_sigma_max,
        "rho": args.sample_rho,
        "s_churn": args.sample_s_churn,
        "s_min": args.sample_s_min,
        "s_max": args.sample_s_max,
        "s_noise": args.sample_s_noise,
    }


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


def maybe_configure_wandb(
    run,
    args: argparse.Namespace,
    *,
    train_dataset: BWDBDataset,
    val_dataset: BWDBDataset,
    test_dataset: BWDBDataset,
    pseudo_match_count: int,
    model: MOFStage1EDMModel,
) -> None:
    if run is None:
        return
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run.config.update(
        {
            "dataset": {
                "name": "BWDB-stage1",
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "data_dir": str(args.data_dir),
                "max_num_atoms": args.max_num_atoms,
            },
            "model": {
                "num_parameters": num_parameters,
                "num_trainable_parameters": num_trainable_parameters,
                "attn_type": args.attn_type,
                "simplicial_geom_mode": args.simplicial_geom_mode,
                "simplicial_impl": args.simplicial_impl,
                "simplicial_precision": args.simplicial_precision,
                "lattice_repr": args.lattice_repr,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "mlp_ratio": args.mlp_ratio,
                "pbc_radius": args.pbc_radius,
            },
            "loss": {
                "p_mean": args.p_mean,
                "p_std": args.p_std,
                "sigma_data_coord": args.sigma_data_coord,
                "sigma_data_lattice": args.sigma_data_lattice,
                "lattice_repr": args.lattice_repr,
                "coord_weight": args.coord_weight,
                "lattice_weight": args.lattice_weight,
                "align_global_shift": args.align_global_shift,
            },
            "logging": {
                "log_every_steps": args.log_every_steps,
                "val_estimate_every_steps": args.val_estimate_every_steps,
                "val_estimate_batches": args.val_estimate_batches,
                "full_val_every_steps": args.full_val_every_steps,
                "pseudo_match_every_steps": args.pseudo_match_every_steps,
                "pseudo_match_num_samples": pseudo_match_count,
                "pseudo_match_batch_size": args.pseudo_match_batch_size,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
            },
            "ema": {
                "decay": args.ema_decay,
                "use_for_sampling": args.ema_use_for_sampling,
            },
            "sampler": sampler_kwargs_from_args(args),
            "pseudo_match": {
                "stol": args.pseudo_match_stol,
                "ltol": args.pseudo_match_ltol,
                "angle_tol": args.pseudo_match_angle_tol,
            },
            "checkpointing": {
                "save_checkpoint": args.save_checkpoint,
                "output": args.output,
            },
        },
        allow_val_change=True,
    )
    run.summary["dataset/train_samples"] = len(train_dataset)
    run.summary["dataset/val_samples"] = len(val_dataset)
    run.summary["dataset/test_samples"] = len(test_dataset)
    run.summary["dataset/pseudo_match_samples"] = pseudo_match_count
    run.summary["model/num_parameters"] = num_parameters
    run.summary["model/num_trainable_parameters"] = num_trainable_parameters


def initialize_model_parameters(
    net: MOFStage1EDMPreconditioner,
    criterion: MOFStage1EDMLoss,
    dataset: BWDBDataset,
    device: torch.device,
    *,
    bf16_enabled: bool = False,
) -> None:
    if len(dataset) == 0:
        raise ValueError("Cannot initialize model parameters from an empty dataset")
    init_batch = collate_mof_stage1([dataset[0]]).to(device)
    with torch.no_grad():
        with make_autocast_context(device, bf16_enabled):
            _ = criterion(net, init_batch)


def evaluate_loss(
    net: MOFStage1EDMPreconditioner,
    criterion: MOFStage1EDMLoss,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    bf16_enabled: bool = False,
) -> dict[str, float]:
    net.eval()
    total = {
        "loss_total": 0.0,
        "coord_loss": 0.0,
        "lattice_loss": 0.0,
        "coord_frac_rmse": 0.0,
        "length_mae": 0.0,
        "angle_mae": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            with make_autocast_context(device, bf16_enabled):
                loss, diagnostics = criterion(net, batch)
            batch_size = int(batch.num_blocks.shape[0])
            total["loss_total"] += float(loss.item()) * batch_size
            total["coord_loss"] += float(diagnostics["coord_loss"].item()) * batch_size
            total["lattice_loss"] += float(diagnostics["lattice_loss"].item()) * batch_size
            total["coord_frac_rmse"] += float(diagnostics["coord_frac_rmse"].item()) * batch_size
            total["length_mae"] += float(diagnostics["length_mae"].item()) * batch_size
            total["angle_mae"] += float(diagnostics["angle_mae"].item()) * batch_size
            count += batch_size
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    denom = max(count, 1)
    return {key: value / denom for key, value in total.items()}


def clone_model_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def maybe_clone_ema_state(ema: EMA | None) -> dict[str, torch.Tensor] | None:
    if ema is None:
        return None
    return ema.state_dict()


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    args: argparse.Namespace,
    best_val: float,
    best_pseudo_match_rate: float,
    global_step: int,
    epoch: int,
    last_full_val_step: int,
    last_full_val: dict[str, float] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
        "best_val_loss": float(best_val),
        "best_pseudo_match_rate": float(best_pseudo_match_rate),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "last_full_val_step": int(last_full_val_step),
    }
    if last_full_val is not None:
        payload["last_full_val"] = {key: float(value) for key, value in last_full_val.items()}
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
        payload["ema_decay"] = args.ema_decay
    return payload


def build_pseudo_match_loader(
    dataset: BWDBDataset,
    *,
    seed: int,
    num_samples: int,
    batch_size: int,
) -> tuple[DataLoader | None, int]:
    if len(dataset) == 0 or num_samples <= 0:
        return None, 0
    actual_count = min(int(num_samples), len(dataset))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1729)
    indices = torch.randperm(len(dataset), generator=generator)[:actual_count].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=max(1, min(int(batch_size), actual_count)),
        shuffle=False,
        collate_fn=collate_mof_stage1,
        num_workers=0,
    )
    return loader, actual_count


def _block_signature(
    block_feature_row: torch.Tensor,
    block_type_id: int,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    counts = torch.round(block_feature_row).to(dtype=torch.long)
    nz = torch.nonzero(counts, as_tuple=False).flatten().tolist()
    return (
        int(block_type_id),
        tuple((int(index) + 1, int(counts[index].item())) for index in nz),
    )


def _translation_align(pred_frac: torch.Tensor, gt_frac: torch.Tensor) -> torch.Tensor:
    shift = wrap_frac((pred_frac - gt_frac).mean(dim=0, keepdim=True))
    return mod1(pred_frac - shift)


def _pseudo_match_metrics_one(
    *,
    pred_frac: torch.Tensor,
    pred_lattice_params: torch.Tensor,
    gt_frac: torch.Tensor,
    gt_lattice_params: torch.Tensor,
    block_features: torch.Tensor,
    block_type_ids: torch.Tensor,
    stol: float,
    ltol: float,
    angle_tol: float,
    lattice_repr: str,
) -> dict[str, float]:
    pred_frac = _translation_align(mod1(pred_frac), mod1(gt_frac))
    gt_frac = mod1(gt_frac)

    signatures = [
        _block_signature(block_features[index], int(block_type_ids[index].item()))
        for index in range(block_features.shape[0])
    ]
    signature_to_indices: dict[tuple[int, tuple[tuple[int, int], ...]], list[int]] = {}
    for index, signature in enumerate(signatures):
        signature_to_indices.setdefault(signature, []).append(index)

    gram = lattice_latent_to_gram(
        lattice_params_to_lattice_latent(
            gt_lattice_params[None, :],
            lattice_repr=lattice_repr,
        ),
        lattice_repr=lattice_repr,
    )[0].to(dtype=pred_frac.dtype, device=pred_frac.device)
    volume = torch.sqrt(torch.linalg.det(gram).clamp_min(1e-12))
    scale = (volume / float(max(len(signatures), 1))).clamp_min(1e-12).pow(1.0 / 3.0)

    matched_norm_dists: list[torch.Tensor] = []
    matched_frac_dists: list[torch.Tensor] = []
    for indices in signature_to_indices.values():
        pred_group = pred_frac[indices]
        gt_group = gt_frac[indices]
        diff = wrap_frac(pred_group[:, None, :] - gt_group[None, :, :])
        dist2_cart = torch.einsum("mnj,jk,mnk->mn", diff, gram, diff).clamp_min(0.0)
        dist_norm = torch.sqrt(dist2_cart) / scale
        dist_frac = torch.sqrt(diff.square().sum(dim=-1))
        row_ind, col_ind = linear_sum_assignment(dist_norm.detach().cpu().numpy())
        matched_norm_dists.append(dist_norm[row_ind, col_ind])
        matched_frac_dists.append(dist_frac[row_ind, col_ind])

    if matched_norm_dists:
        matched_norm = torch.cat(matched_norm_dists)
        matched_frac = torch.cat(matched_frac_dists)
        coord_norm_rmse = torch.sqrt(matched_norm.square().mean())
        coord_frac_rmse = torch.sqrt(matched_frac.square().mean())
        coord_max_norm_error = matched_norm.max()
        coord_match = bool((matched_norm <= stol).all().item())
    else:
        coord_norm_rmse = torch.tensor(0.0, device=pred_frac.device)
        coord_frac_rmse = torch.tensor(0.0, device=pred_frac.device)
        coord_max_norm_error = torch.tensor(0.0, device=pred_frac.device)
        coord_match = True

    length_errors = (pred_lattice_params[:3] - gt_lattice_params[:3]).abs()
    angle_errors = (pred_lattice_params[3:] - gt_lattice_params[3:]).abs()
    length_rel_errors = length_errors / gt_lattice_params[:3].abs().clamp_min(1e-8)
    lattice_match = bool(
        (length_rel_errors.max() <= ltol).item()
        and (angle_errors.max() <= angle_tol).item()
    )
    return {
        "match_rate": float(coord_match and lattice_match),
        "coord_frac_rmse": float(coord_frac_rmse.item()),
        "coord_norm_rmse": float(coord_norm_rmse.item()),
        "coord_max_norm_error": float(coord_max_norm_error.item()),
        "length_mae": float(length_errors.mean().item()),
        "angle_mae": float(angle_errors.mean().item()),
        "length_rel_mae": float(length_rel_errors.mean().item()),
        "length_rel_max": float(length_rel_errors.max().item()),
        "angle_max": float(angle_errors.max().item()),
    }


def evaluate_pseudo_match(
    run,
    model: torch.nn.Module,
    ema: EMA | None,
    net: MOFStage1EDMPreconditioner,
    loader: DataLoader,
    device: torch.device,
    *,
    global_step: int,
    epoch_float: float,
    split_name: str,
    sampler_kwargs: dict[str, float | int],
    use_ema: bool,
    stol: float,
    ltol: float,
    angle_tol: float,
    lattice_repr: str,
    bf16_enabled: bool = False,
) -> dict[str, float]:
    ema_backup = None
    if use_ema and ema is not None:
        ema_backup = ema.apply(model)

    totals = {
        "match_rate": 0.0,
        "coord_frac_rmse": 0.0,
        "coord_norm_rmse": 0.0,
        "coord_max_norm_error": 0.0,
        "length_mae": 0.0,
        "angle_mae": 0.0,
        "length_rel_mae": 0.0,
        "length_rel_max": 0.0,
        "angle_max": 0.0,
    }
    count = 0

    try:
        net.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with make_autocast_context(device, bf16_enabled):
                    pred_frac, pred_lattice_latent, _ = mof_stage1_edm_sampler(
                        net,
                        batch.block_features,
                        batch.block_type_ids,
                        batch.num_blocks,
                        **sampler_kwargs,
                    )
                pred_frac = pred_frac.float()
                pred_lattice_params = lattice_latent_to_lattice_params(
                    pred_lattice_latent.float(),
                    lattice_repr=lattice_repr,
                )
                for batch_index in range(batch.num_blocks.shape[0]):
                    num_blocks = int(batch.num_blocks[batch_index].item())
                    if num_blocks <= 0:
                        continue
                    metrics = _pseudo_match_metrics_one(
                        pred_frac=pred_frac[batch_index, :num_blocks],
                        pred_lattice_params=pred_lattice_params[batch_index],
                        gt_frac=batch.block_com_frac[batch_index, :num_blocks].float(),
                        gt_lattice_params=batch.lattice[batch_index].float(),
                        block_features=batch.block_features[batch_index, :num_blocks].float(),
                        block_type_ids=batch.block_type_ids[batch_index, :num_blocks].long(),
                        stol=stol,
                        ltol=ltol,
                        angle_tol=angle_tol,
                        lattice_repr=lattice_repr,
                    )
                    for key, value in metrics.items():
                        totals[key] += float(value)
                    count += 1
    finally:
        if ema_backup is not None:
            ema.restore(model, ema_backup)

    denom = max(count, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["num_samples"] = float(count)

    log_payload = {
        "trainer/global_step": global_step,
        "trainer/epoch": epoch_float,
    }
    for key, value in metrics.items():
        log_payload[f"{split_name}/{key}"] = value
    if run is not None:
        run.log(log_payload, step=global_step)
    else:
        metrics_str = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"step={global_step:07d} {split_name} {metrics_str}")
    return metrics


def main(args: argparse.Namespace) -> None:
    maybe_configure_cuda_sdpa_from_env()
    seed_everything(args.seed)
    device = default_device()

    train_dataset = BWDBDataset(
        args.data_dir,
        split="train",
        sample_limit=args.train_sample_limit,
        max_num_atoms=args.max_num_atoms,
    )
    val_dataset = BWDBDataset(
        args.data_dir,
        split="val",
        sample_limit=args.val_sample_limit,
        max_num_atoms=args.max_num_atoms,
    )
    test_dataset = BWDBDataset(
        args.data_dir,
        split="test",
        sample_limit=args.test_sample_limit,
        max_num_atoms=args.max_num_atoms,
    )
    if len(train_dataset) == 0:
        raise ValueError("train_dataset is empty after filtering")
    if len(val_dataset) == 0:
        raise ValueError("val_dataset is empty after filtering")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mof_stage1,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mof_stage1,
        num_workers=args.num_workers,
    )
    pseudo_match_loader, pseudo_match_count = build_pseudo_match_loader(
        val_dataset,
        seed=args.seed,
        num_samples=args.pseudo_match_num_samples,
        batch_size=args.pseudo_match_batch_size,
    )
    run = maybe_init_wandb(args)

    block_feature_dim = int(train_dataset[0].block_element_count_vecs.shape[-1])
    model = MOFStage1EDMModel(
        block_feature_dim=block_feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        attn_type=args.attn_type,
        simplicial_geom_mode=args.simplicial_geom_mode,
        simplicial_impl=args.simplicial_impl,
        simplicial_precision=args.simplicial_precision,
        use_geometry_bias=not args.disable_geometry_bias,
        lattice_repr=args.lattice_repr,
        pbc_radius=args.pbc_radius,
    ).to(device)
    net = MOFStage1EDMPreconditioner(
        model,
        sigma_min=args.sample_sigma_min,
        sigma_max=args.sample_sigma_max,
        sigma_data_coord=args.sigma_data_coord,
        sigma_data_lattice=args.sigma_data_lattice,
        lattice_repr=args.lattice_repr,
    ).to(device)
    criterion = MOFStage1EDMLoss(
        p_mean=args.p_mean,
        p_std=args.p_std,
        sigma_data_coord=args.sigma_data_coord,
        sigma_data_lattice=args.sigma_data_lattice,
        lattice_repr=args.lattice_repr,
        coord_weight=args.coord_weight,
        lattice_weight=args.lattice_weight,
        align_global_shift=args.align_global_shift,
    )
    initialize_model_parameters(net, criterion, train_dataset, device, bf16_enabled=args.bf16)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    maybe_configure_wandb(
        run,
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pseudo_match_count=pseudo_match_count,
        model=model,
    )

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        "dataset: "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"pseudo_match_subset={pseudo_match_count}"
    )
    print(
        "model: "
        f"num_parameters={num_parameters} num_trainable_parameters={num_trainable_parameters}"
    )
    print(
        "schedule: "
        f"warmup_steps={args.warmup_steps} max_steps={args.max_steps}"
    )
    print(
        "loss: "
        f"p_mean={args.p_mean} p_std={args.p_std} "
        f"sigma_data_coord={args.sigma_data_coord} sigma_data_lattice={args.sigma_data_lattice} "
        f"lattice_repr={args.lattice_repr} "
        f"coord_weight={args.coord_weight} lattice_weight={args.lattice_weight} "
        f"align_global_shift={args.align_global_shift}"
    )
    print(
        "sampler: "
        f"num_steps={args.sample_num_steps} sigma_min={args.sample_sigma_min} "
        f"sigma_max={args.sample_sigma_max} rho={args.sample_rho} "
        f"s_churn={args.sample_s_churn} s_min={args.sample_s_min} "
        f"s_max={args.sample_s_max} s_noise={args.sample_s_noise}"
    )
    print(
        "pseudo_match: "
        f"every={args.pseudo_match_every_steps} num_samples={pseudo_match_count} "
        f"batch_size={args.pseudo_match_batch_size} stol={args.pseudo_match_stol} "
        f"ltol={args.pseudo_match_ltol} angle_tol={args.pseudo_match_angle_tol}"
    )
    print(
        "ema: "
        f"decay={args.ema_decay} use_for_sampling={args.ema_use_for_sampling}"
    )
    print(f"precision: bf16={args.bf16}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=args.warmup_steps,
        max_iters=args.max_steps,
    )

    checkpoint_path = Path(args.output)
    if args.save_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_pseudo_match_rate = float("-inf")
    best_state_in_memory: dict[str, torch.Tensor] | None = None
    best_ema_state_in_memory: dict[str, torch.Tensor] | None = None
    global_step = 0
    last_full_val_step = -1
    last_pseudo_match_step = -1
    last_full_val: dict[str, float] | None = None
    sampler_kwargs = sampler_kwargs_from_args(args)
    batches_per_epoch = len(train_loader)
    epoch = 0

    if args.resume_checkpoint is not None and args.warm_start_checkpoint is not None:
        raise ValueError("Only one of --resume-checkpoint or --warm-start-checkpoint may be set")

    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location=device)
        required_keys = ("model_state", "optimizer_state", "scheduler_state", "global_step")
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(
                f"Checkpoint {resume_path} is missing resume fields: {', '.join(missing_keys)}"
            )
        model.load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        best_pseudo_match_rate = float(checkpoint.get("best_pseudo_match_rate", float("-inf")))
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint.get("epoch", 0))
        last_full_val_step = int(checkpoint.get("last_full_val_step", -1))
        checkpoint_last_full_val = checkpoint.get("last_full_val")
        if checkpoint_last_full_val is not None:
            last_full_val = {key: float(value) for key, value in checkpoint_last_full_val.items()}
        best_state_in_memory = clone_model_state_to_cpu(model)
        best_ema_state_in_memory = maybe_clone_ema_state(ema)
        print(
            "resume_checkpoint: "
            f"path={resume_path} global_step={global_step} epoch={epoch} "
            f"best_val_loss={best_val:.6f} best_pseudo_match_rate={best_pseudo_match_rate:.6f}"
        )
        if (
            args.save_checkpoint
            and checkpoint_path.expanduser().resolve() != resume_path.expanduser().resolve()
        ):
            torch.save(checkpoint, checkpoint_path)
    elif args.warm_start_checkpoint is not None:
        warm_start_path = Path(args.warm_start_checkpoint)
        checkpoint = torch.load(warm_start_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        print(
            "warm_start_checkpoint: "
            f"path={warm_start_path} source_best_val_loss="
            f"{float(checkpoint.get('best_val_loss', float('nan'))):.6f}"
        )

    while global_step < args.max_steps:
        epoch += 1
        net.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            if global_step >= args.max_steps:
                break

            batch = batch.to(device)
            with make_autocast_context(device, args.bf16):
                loss, diagnostics = criterion(net, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            global_step += 1

            epoch_float = (epoch - 1) + batch_idx / max(batches_per_epoch, 1)
            lr = optimizer.param_groups[0]["lr"]
            grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                log_payload = {
                    "trainer/global_step": global_step,
                    "trainer/epoch": epoch_float,
                    "train/loss_total": float(loss.item()),
                    "train/coord_loss": float(diagnostics["coord_loss"].item()),
                    "train/lattice_loss": float(diagnostics["lattice_loss"].item()),
                    "train/coord_frac_rmse": float(diagnostics["coord_frac_rmse"].item()),
                    "train/length_mae": float(diagnostics["length_mae"].item()),
                    "train/angle_mae": float(diagnostics["angle_mae"].item()),
                    "train/num_blocks_mean": float(batch.num_blocks.float().mean().item()),
                    "train/num_atoms_mean": float(batch.num_atoms.float().mean().item()),
                    "noise/sigma_mean": float(diagnostics["sigma"].mean().item()),
                    "noise/sigma_std": float(diagnostics["sigma"].std(unbiased=False).item()),
                    "noise/sigma_min": float(diagnostics["sigma"].min().item()),
                    "noise/sigma_max": float(diagnostics["sigma"].max().item()),
                    "optim/grad_norm": grad_norm_value,
                    "optim/lr": lr,
                }
                if run is not None:
                    run.log(log_payload, step=global_step)
                else:
                    print(
                        f"step={global_step:07d} epoch={epoch_float:.3f} "
                        f"train_loss={log_payload['train/loss_total']:.6f} "
                        f"coord_rmse={log_payload['train/coord_frac_rmse']:.6f} "
                        f"length_mae={log_payload['train/length_mae']:.6f} "
                        f"angle_mae={log_payload['train/angle_mae']:.6f}"
                    )

            if (
                args.val_estimate_every_steps > 0
                and global_step % args.val_estimate_every_steps == 0
            ):
                val_est = evaluate_loss(
                    net,
                    criterion,
                    val_loader,
                    device,
                    max_batches=args.val_estimate_batches,
                    bf16_enabled=args.bf16,
                )
                if run is not None:
                    run.log(
                        {
                            "trainer/global_step": global_step,
                            "trainer/epoch": epoch_float,
                            "val_estimate/loss_total": val_est["loss_total"],
                            "val_estimate/coord_loss": val_est["coord_loss"],
                            "val_estimate/lattice_loss": val_est["lattice_loss"],
                            "val_estimate/coord_frac_rmse": val_est["coord_frac_rmse"],
                            "val_estimate/length_mae": val_est["length_mae"],
                            "val_estimate/angle_mae": val_est["angle_mae"],
                        },
                        step=global_step,
                    )
                else:
                    print(
                        f"step={global_step:07d} val_estimate "
                        f"loss={val_est['loss_total']:.6f} "
                        f"coord_rmse={val_est['coord_frac_rmse']:.6f} "
                        f"length_mae={val_est['length_mae']:.6f} "
                        f"angle_mae={val_est['angle_mae']:.6f}"
                    )

            if args.full_val_every_steps > 0 and global_step % args.full_val_every_steps == 0:
                val_metrics = evaluate_loss(
                    net,
                    criterion,
                    val_loader,
                    device,
                    bf16_enabled=args.bf16,
                )
                last_full_val = val_metrics
                last_full_val_step = global_step
                best_val = min(best_val, val_metrics["loss_total"])
                if run is not None:
                    run.log(
                        {
                            "trainer/global_step": global_step,
                            "trainer/epoch": epoch_float,
                            "val/loss_total": val_metrics["loss_total"],
                            "val/coord_loss": val_metrics["coord_loss"],
                            "val/lattice_loss": val_metrics["lattice_loss"],
                            "val/coord_frac_rmse": val_metrics["coord_frac_rmse"],
                            "val/length_mae": val_metrics["length_mae"],
                            "val/angle_mae": val_metrics["angle_mae"],
                            "val/best_loss": best_val,
                        },
                        step=global_step,
                    )
                else:
                    print(
                        f"step={global_step:07d} val "
                        f"loss={val_metrics['loss_total']:.6f} "
                        f"coord_rmse={val_metrics['coord_frac_rmse']:.6f} "
                        f"length_mae={val_metrics['length_mae']:.6f} "
                        f"angle_mae={val_metrics['angle_mae']:.6f} "
                        f"best_val={best_val:.6f}"
                    )
                if val_metrics["loss_total"] <= best_val:
                    best_state_in_memory = clone_model_state_to_cpu(model)
                    best_ema_state_in_memory = maybe_clone_ema_state(ema)
                    if args.save_checkpoint:
                        torch.save(
                            build_checkpoint_payload(
                                model=model,
                                ema=ema,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                args=args,
                                best_val=best_val,
                                best_pseudo_match_rate=best_pseudo_match_rate,
                                global_step=global_step,
                                epoch=epoch,
                                last_full_val_step=last_full_val_step,
                                last_full_val=last_full_val,
                            ),
                            checkpoint_path,
                        )

            if (
                pseudo_match_loader is not None
                and args.pseudo_match_every_steps > 0
                and global_step % args.pseudo_match_every_steps == 0
            ):
                pseudo_metrics = evaluate_pseudo_match(
                    run,
                    model,
                    ema,
                    net,
                    pseudo_match_loader,
                    device,
                    global_step=global_step,
                    epoch_float=epoch_float,
                    split_name="pseudo_match",
                    sampler_kwargs=sampler_kwargs,
                    use_ema=args.ema_use_for_sampling,
                    stol=args.pseudo_match_stol,
                    ltol=args.pseudo_match_ltol,
                    angle_tol=args.pseudo_match_angle_tol,
                    lattice_repr=args.lattice_repr,
                    bf16_enabled=args.bf16,
                )
                last_pseudo_match_step = global_step
                if math.isfinite(pseudo_metrics["match_rate"]):
                    best_pseudo_match_rate = max(best_pseudo_match_rate, pseudo_metrics["match_rate"])
                if run is not None:
                    run.log(
                        {
                            "trainer/global_step": global_step,
                            "trainer/epoch": epoch_float,
                            "pseudo_match/best_rate": best_pseudo_match_rate,
                        },
                        step=global_step,
                    )

    if last_full_val is None or last_full_val_step != global_step:
        last_full_val = evaluate_loss(
            net,
            criterion,
            val_loader,
            device,
            bf16_enabled=args.bf16,
        )
        last_full_val_step = global_step
        best_val = min(best_val, last_full_val["loss_total"])

    if pseudo_match_loader is not None and last_pseudo_match_step != global_step:
        pseudo_metrics = evaluate_pseudo_match(
            run,
            model,
            ema,
            net,
            pseudo_match_loader,
            device,
            global_step=global_step,
            epoch_float=float(epoch),
            split_name="pseudo_match",
            sampler_kwargs=sampler_kwargs,
            use_ema=args.ema_use_for_sampling,
            stol=args.pseudo_match_stol,
            ltol=args.pseudo_match_ltol,
            angle_tol=args.pseudo_match_angle_tol,
            lattice_repr=args.lattice_repr,
            bf16_enabled=args.bf16,
        )
        if math.isfinite(pseudo_metrics["match_rate"]):
            best_pseudo_match_rate = max(best_pseudo_match_rate, pseudo_metrics["match_rate"])

    print(
        f"final_val step={global_step:07d} "
        f"val_loss={last_full_val['loss_total']:.6f} "
        f"val_coord_loss={last_full_val['coord_loss']:.6f} "
        f"val_lattice_loss={last_full_val['lattice_loss']:.6f} "
        f"val_coord_frac_rmse={last_full_val['coord_frac_rmse']:.6f} "
        f"val_length_mae={last_full_val['length_mae']:.6f} "
        f"val_angle_mae={last_full_val['angle_mae']:.6f} "
        f"best_val_loss={best_val:.6f} "
        f"best_pseudo_match_rate={best_pseudo_match_rate:.6f}"
    )

    if best_state_in_memory is not None:
        model.load_state_dict(best_state_in_memory)
        if ema is not None and best_ema_state_in_memory is not None:
            ema.load_state_dict(best_ema_state_in_memory)
    elif args.save_checkpoint and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])

    if run is not None:
        run.summary["val/best_loss"] = best_val
        run.summary["pseudo_match/best_rate"] = best_pseudo_match_rate
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Matterformer BWDB stage-1 EDM")
    parser.add_argument("--data-dir", type=str, default="./data/mofs/bwdb")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/Matterformer_BWDB_stage1/checkpoints/mof_stage1_edm.pt",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--val-sample-limit", type=int, default=None)
    parser.add_argument("--test-sample-limit", type=int, default=None)
    parser.add_argument("--max-num-atoms", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--attn-type", type=str, default="simplicial", choices=["mha", "simplicial"])
    parser.add_argument(
        "--simplicial-geom-mode",
        type=str,
        default="factorized",
        choices=["none", "factorized", "angle_residual"],
    )
    parser.add_argument(
        "--simplicial-impl",
        type=str,
        default="auto",
        choices=["auto", "torch", "triton"],
    )
    parser.add_argument(
        "--simplicial-precision",
        type=str,
        default="bf16_tc",
        choices=["bf16_tc", "tf32", "ieee_fp32"],
    )
    parser.add_argument("--disable-geometry-bias", action="store_true")
    parser.add_argument("--lattice-repr", type=str, default="y1", choices=["y1", "ltri"])
    parser.add_argument("--pbc-radius", type=int, default=1)
    parser.add_argument("--p-mean", type=float, default=-1.2)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--sigma-data-coord", type=float, default=0.5)
    parser.add_argument("--sigma-data-lattice", type=float, default=0.5)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--lattice-weight", type=float, default=1.0)
    parser.add_argument(
        "--align-global-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample-num-steps", type=int, default=100)
    parser.add_argument("--sample-sigma-min", type=float, default=0.002)
    parser.add_argument("--sample-sigma-max", type=float, default=10.0)
    parser.add_argument("--sample-rho", type=float, default=7.0)
    parser.add_argument("--sample-s-churn", type=float, default=30.0)
    parser.add_argument("--sample-s-min", type=float, default=0.0)
    parser.add_argument("--sample-s-max", type=float, default=float("inf"))
    parser.add_argument("--sample-s-noise", type=float, default=1.003)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-use-for-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--wandb-project", type=str, default="Matterformer_BWDB_stage1")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="./outputs/Matterformer_BWDB_stage1/wandb",
    )
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--val-estimate-every-steps", type=int, default=500)
    parser.add_argument("--val-estimate-batches", type=int, default=16)
    parser.add_argument("--full-val-every-steps", type=int, default=5_000)
    parser.add_argument("--pseudo-match-every-steps", type=int, default=10_000)
    parser.add_argument("--pseudo-match-num-samples", type=int, default=1_024)
    parser.add_argument("--pseudo-match-batch-size", type=int, default=64)
    parser.add_argument("--pseudo-match-stol", type=float, default=0.30)
    parser.add_argument("--pseudo-match-ltol", type=float, default=0.20)
    parser.add_argument("--pseudo-match-angle-tol", type=float, default=10.0)
    main(parser.parse_args())
