#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from tqdm import tqdm, trange

from matterformer.data import (
    SYNOMOL_TRANSFER_SPLITS,
    SynOMolTransferConfig,
    calibrate_synomol_transfer_component_scales,
    compute_synomol_transfer_labels_batch_local,
    pack_synomol_transfer_samples,
    propose_synomol_transfer_inputs,
    relax_coords_batch,
    langevin_coords_batch,
    summarize_synomol_transfer_samples,
    synomol_transfer_atom_type_table,
)
from matterformer.data.synomol_transfer import (
    _HELDOUT_COMBO_SET,
    _active_type_triples,
    _distort_coords,
    _generate_synomol_transfer_inputs,
    _repair_heldout_type_combos,
    _sample_is_acceptable,
    _split_allows_active_heldout,
    generate_synomol_transfer_sample_with_stats,
)


DEFAULT_SPLIT_SIZES = {
    "train": 100_000,
    "val": 10_000,
    "test_iid": 10_000,
    "test_type_combo": 10_000,
    "test_motif": 10_000,
    "test_size": 10_000,
    "test_perturb": 10_000,
}
SMOKE_SPLIT_SIZES = {
    "train": 1_000,
    "val": 200,
    "test_iid": 200,
    "test_type_combo": 200,
    "test_motif": 200,
    "test_size": 200,
    "test_perturb": 200,
}
FULL_SPLIT_SIZES = {
    "train": 200_000,
    "val": 50_000,
    "test_iid": 50_000,
    "test_type_combo": 50_000,
    "test_motif": 50_000,
    "test_size": 50_000,
    "test_perturb": 50_000,
}


def parse_num_atoms(value: str) -> int | tuple[int, int]:
    value = value.strip()
    separator = ":" if ":" in value else "-"
    if separator in value:
        left, right = value.split(separator, maxsplit=1)
        return int(left), int(right)
    return int(value)


def build_config(args: argparse.Namespace, *, length: int) -> SynOMolTransferConfig:
    preset_steps = {
        "smoke": (2, 1, 1),
        "default": (16, 4, 8),
        "full": (64, 16, 32),
    }
    default_relax_steps, default_relaxation_snapshot_steps, default_langevin_steps = preset_steps[args.preset]
    default_num_atoms = "8:32" if args.preset == "smoke" else "16:96"
    default_size_ood_num_atoms = "48:64" if args.preset == "smoke" else "128:256"
    if args.preset == "full":
        default_num_atoms = "16:128"
        default_size_ood_num_atoms = "160:350"
    return SynOMolTransferConfig(
        num_atoms=parse_num_atoms(args.num_atoms or default_num_atoms),
        size_ood_num_atoms=parse_num_atoms(args.size_ood_num_atoms or default_size_ood_num_atoms),
        length=length,
        seed=args.seed,
        density=args.density,
        cutoff=args.cutoff,
        cutoff_delta=args.cutoff_delta,
        radial_basis_size=args.radial_basis_size,
        angle_lmax=args.angle_lmax,
        relax_steps=args.relax_steps if args.relax_steps is not None else default_relax_steps,
        relaxation_snapshot_steps=(
            args.relaxation_snapshot_steps
            if args.relaxation_snapshot_steps is not None
            else default_relaxation_snapshot_steps
        ),
        langevin_steps=args.langevin_steps if args.langevin_steps is not None else default_langevin_steps,
        pair_scale=args.pair_scale,
        angle_scale=args.angle_scale,
        motif_scale=args.motif_scale,
        many_body_scale=args.many_body_scale,
        generation_pair_scale=args.generation_pair_scale,
        generation_angle_scale=args.generation_angle_scale,
        generation_motif_scale=args.generation_motif_scale,
        generation_many_body_scale=args.generation_many_body_scale,
        force_rms_min=args.force_rms_min,
        force_rms_max=args.force_rms_max,
        max_abs_energy=args.max_abs_energy,
        max_abs_energy_per_atom=args.max_abs_energy_per_atom,
    )


def split_sizes_from_args(args: argparse.Namespace) -> dict[str, int]:
    if args.preset == "smoke":
        sizes = dict(SMOKE_SPLIT_SIZES)
    elif args.preset == "full":
        sizes = dict(FULL_SPLIT_SIZES)
    else:
        sizes = dict(DEFAULT_SPLIT_SIZES)
    overrides = {
        "train": args.train_size,
        "val": args.val_size,
        "test_iid": args.test_iid_size,
        "test_type_combo": args.test_type_combo_size,
        "test_motif": args.test_motif_size,
        "test_size": args.test_size_size,
        "test_perturb": args.test_perturb_size,
    }
    for split, value in overrides.items():
        if value is not None:
            sizes[split] = int(value)
    return sizes


def write_split(
    *,
    root: Path,
    base_config: SynOMolTransferConfig,
    config_name: str,
    split: str,
    size: int,
    metadata: dict[str, object],
    force: bool,
    num_workers: int,
    backend: str,
    generation_device: str,
    generation_batch_size: int,
    k_label: int,
    max_gpu_work: int,
    max_label_cap_hit_fraction: float,
    allow_label_cap_hits: bool,
    chunksize: int | None,
    shard_size: int | None,
) -> Path:
    output_path = root / config_name / f"{split}.pt"
    stats_path = root / config_name / f"{split}_stats.json"
    manifest_path = root / config_name / f"{split}_manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not force:
        existing = manifest_path if manifest_path.exists() else output_path
        print(f"exists, skipping: {existing}")
        return existing

    split_config = SynOMolTransferConfig.from_dict({**asdict(base_config), "length": int(size)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writing_sharded = shard_size is not None and shard_size > 0 and shard_size < size
    _cleanup_stale_split_outputs(output_path.parent, split, writing_sharded=writing_sharded)
    start_time = time.perf_counter()
    if writing_sharded:
        shard_entries = []
        summary_accumulator = _new_summary_accumulator()
        for shard_index, start in enumerate(range(0, size, shard_size)):
            end = min(start + shard_size, size)
            generated = _generate_range(
                start=start,
                end=end,
                split=split,
                split_config=split_config,
                num_workers=num_workers,
                backend=backend,
                generation_device=generation_device,
                generation_batch_size=generation_batch_size,
                k_label=k_label,
                max_gpu_work=max_gpu_work,
                chunksize=chunksize,
            )
            samples = [sample for sample, _ in generated]
            generation_stats = [stats for _, stats in generated]
            _accumulate_summary(summary_accumulator, samples, generation_stats)
            shard_stats = summarize_synomol_transfer_samples(samples, generation_stats=generation_stats)
            _enforce_label_cap_policy(
                shard_stats,
                split=split,
                max_label_cap_hit_fraction=max_label_cap_hit_fraction,
                allow_label_cap_hits=allow_label_cap_hits,
            )
            shard_name = f"{split}_{shard_index:05d}.pt"
            shard_path = root / config_name / shard_name
            packed = pack_synomol_transfer_samples(samples, config=split_config, split=split, metadata=metadata)
            torch.save(packed, shard_path)
            shard_entries.append({"path": shard_name, "num_samples": len(samples)})
        split_stats = _finalize_summary(summary_accumulator)
        split_stats = _add_timing_stats(split_stats, elapsed=time.perf_counter() - start_time)
        _enforce_label_cap_policy(
            split_stats,
            split=split,
            max_label_cap_hit_fraction=max_label_cap_hit_fraction,
            allow_label_cap_hits=allow_label_cap_hits,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "format": "synomol_transfer_sharded_v1",
                    "split": split,
                    "num_samples": size,
                    "shards": shard_entries,
                    "stats": split_stats,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        generated = _generate_range(
            start=0,
            end=size,
            split=split,
            split_config=split_config,
            num_workers=num_workers,
            backend=backend,
            generation_device=generation_device,
            generation_batch_size=generation_batch_size,
            k_label=k_label,
            max_gpu_work=max_gpu_work,
            chunksize=chunksize,
        )
        samples = [sample for sample, _ in generated]
        generation_stats = [stats for _, stats in generated]
        split_stats = summarize_synomol_transfer_samples(samples, generation_stats=generation_stats)
        split_stats = _add_timing_stats(split_stats, elapsed=time.perf_counter() - start_time)
        _enforce_label_cap_policy(
            split_stats,
            split=split,
            max_label_cap_hit_fraction=max_label_cap_hit_fraction,
            allow_label_cap_hits=allow_label_cap_hits,
        )
        packed_metadata = {**metadata, "split_stats": split_stats}
        packed = pack_synomol_transfer_samples(samples, config=split_config, split=split, metadata=packed_metadata)
        torch.save(packed, output_path)
    stats_path.write_text(json.dumps(split_stats, indent=2, sort_keys=True) + "\n")
    print(f"wrote {size} samples: {manifest_path if manifest_path.exists() else output_path}")
    print(f"wrote stats: {stats_path}")
    return manifest_path if manifest_path.exists() else output_path


def _generate_range(
    *,
    start: int,
    end: int,
    split: str,
    split_config: SynOMolTransferConfig,
    num_workers: int,
    backend: str,
    generation_device: str,
    generation_batch_size: int,
    k_label: int,
    max_gpu_work: int,
    chunksize: int | None,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    size = int(end - start)
    if backend == "gpu-batched":
        return _generate_range_gpu_batched(
            start=start,
            end=end,
            split=split,
            split_config=split_config,
            generation_device=generation_device,
            generation_batch_size=generation_batch_size,
            k_label=k_label,
            max_gpu_work=max_gpu_work,
        )
    if backend == "gpu-full":
        return _generate_range_gpu_full(
            start=start,
            end=end,
            split=split,
            split_config=split_config,
            generation_device=generation_device,
            generation_batch_size=generation_batch_size,
            k_label=k_label,
            max_gpu_work=max_gpu_work,
        )
    if num_workers > 0:
        effective_chunksize = chunksize or max(1, size // max(num_workers * 64, 1))
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as pool:
            return list(
                tqdm_map(
                    pool.map(_generate_one_sample, _iter_tasks(start, end, split, split_config), chunksize=effective_chunksize),
                    total=size,
                    desc=f"SynOMol-Transfer {split}",
                )
            )
    return [
        _generate_one_sample((index, split, split_config))
        for index in trange(start, end, desc=f"SynOMol-Transfer {split}", unit="sample")
    ]


def _generate_range_gpu_batched(
    *,
    start: int,
    end: int,
    split: str,
    split_config: SynOMolTransferConfig,
    generation_device: str,
    generation_batch_size: int,
    k_label: int,
    max_gpu_work: int,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if generation_device != "cuda":
        raise ValueError("gpu-batched generation requires --generation-device cuda")
    if generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    device = torch.device(generation_device)
    pending = list(range(start, end))
    attempts = {index: 0 for index in pending}
    rejections = {
        index: {"heldout_type_combo": 0, "sample_filter": 0}
        for index in pending
    }
    accepted: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    progress = tqdm(total=end - start, desc=f"SynOMol-Transfer {split} gpu-batched", unit="sample")
    try:
        while pending:
            candidates = []
            deferred: list[int] = []
            for index in pending:
                if len(candidates) >= generation_batch_size:
                    deferred.append(index)
                    continue
                if attempts[index] >= split_config.max_resample_attempts:
                    raise RuntimeError(f"failed to generate sample {index} for split={split!r}: max attempts exceeded")
                atom_types, coords, metadata = _generate_synomol_transfer_inputs(
                    index=index,
                    split=split,
                    epoch=0,
                    config=split_config,
                    attempt=attempts[index],
                )
                attempts[index] += 1
                active_triples = _active_type_triples(atom_types, coords, split_config)
                has_heldout = bool(active_triples & _HELDOUT_COMBO_SET)
                if not _split_allows_active_heldout(split, has_heldout):
                    rejections[index]["heldout_type_combo"] += 1
                    deferred.append(index)
                    continue
                metadata = dict(metadata)
                metadata["has_heldout_type_combo"] = has_heldout
                candidates.append((index, atom_types, coords, metadata))
            if candidates:
                for candidate_chunk in _chunk_candidates_for_gpu(
                    candidates,
                    max_batch_size=generation_batch_size,
                    k_label=k_label,
                    angle_lmax=split_config.angle_lmax,
                    max_gpu_work=max_gpu_work,
                ):
                    labeled = _label_candidates_on_device(
                        candidate_chunk,
                        split_config,
                        split=split,
                        device=device,
                        k_label=k_label,
                    )
                    for index, sample, stats in labeled:
                        if _sample_is_acceptable(
                            sample["atom_types"],
                            sample["coords"],
                            sample["energy"],
                            sample["forces"],
                            sample["component_energies"],
                            split_config,
                        ):
                            stats["attempts"] = int(attempts[index])
                            stats["rejection_reasons"] = dict(rejections[index])
                            sample["accepted_attempts"] = int(attempts[index])
                            sample["global_sample_id"] = int(index)
                            accepted[index] = (sample, stats)
                            progress.update(1)
                        else:
                            rejections[index]["sample_filter"] += 1
                            deferred.append(index)
            pending = [index for index in deferred if index not in accepted]
    finally:
        progress.close()
    return [accepted[index] for index in range(start, end)]


def _generate_range_gpu_full(
    *,
    start: int,
    end: int,
    split: str,
    split_config: SynOMolTransferConfig,
    generation_device: str,
    generation_batch_size: int,
    k_label: int,
    max_gpu_work: int,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if generation_device != "cuda":
        raise ValueError("gpu-full generation requires --generation-device cuda")
    if generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    device = torch.device(generation_device)
    pending = list(range(start, end))
    attempts = {index: 0 for index in pending}
    rejections = {
        index: {"heldout_type_combo": 0, "sample_filter": 0}
        for index in pending
    }
    accepted: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    progress = tqdm(total=end - start, desc=f"SynOMol-Transfer {split} gpu-full", unit="sample")
    try:
        while pending:
            proposals = []
            deferred: list[int] = []
            for index in pending:
                if len(proposals) >= generation_batch_size:
                    deferred.append(index)
                    continue
                if attempts[index] >= split_config.max_resample_attempts:
                    raise RuntimeError(f"failed to generate sample {index} for split={split!r}: max attempts exceeded")
                atom_types, coords, metadata = propose_synomol_transfer_inputs(
                    index=index,
                    split=split,
                    epoch=0,
                    config=split_config,
                    attempt=attempts[index],
                )
                attempts[index] += 1
                proposals.append((index, atom_types, coords, metadata))
            if proposals:
                for proposal_chunk in _chunk_candidates_for_gpu(
                    proposals,
                    max_batch_size=generation_batch_size,
                    k_label=k_label,
                    angle_lmax=split_config.angle_lmax,
                    max_gpu_work=max_gpu_work,
                ):
                    processed = _postprocess_candidates_gpu_full(
                        proposal_chunk,
                        split_config,
                        split=split,
                        device=device,
                        k_label=k_label,
                    )
                    label_ready = []
                    for index, atom_types, coords, metadata in processed:
                        active_triples = _active_type_triples(atom_types, coords, split_config)
                        has_heldout = bool(active_triples & _HELDOUT_COMBO_SET)
                        if not _split_allows_active_heldout(split, has_heldout):
                            rejections[index]["heldout_type_combo"] += 1
                            deferred.append(index)
                            continue
                        metadata = dict(metadata)
                        metadata["has_heldout_type_combo"] = has_heldout
                        label_ready.append((index, atom_types, coords, metadata))
                    if not label_ready:
                        continue
                    for label_chunk in _chunk_candidates_for_gpu(
                        label_ready,
                        max_batch_size=generation_batch_size,
                        k_label=k_label,
                        angle_lmax=split_config.angle_lmax,
                        max_gpu_work=max_gpu_work,
                    ):
                        labeled = _label_candidates_on_device(
                            label_chunk,
                            split_config,
                            split=split,
                            device=device,
                            k_label=k_label,
                        )
                        for index, sample, stats in labeled:
                            if _sample_is_acceptable(
                                sample["atom_types"],
                                sample["coords"],
                                sample["energy"],
                                sample["forces"],
                                sample["component_energies"],
                                split_config,
                            ):
                                stats["attempts"] = int(attempts[index])
                                stats["rejection_reasons"] = dict(rejections[index])
                                stats["backend"] = "gpu-full"
                                sample["accepted_attempts"] = int(attempts[index])
                                sample["global_sample_id"] = int(index)
                                accepted[index] = (sample, stats)
                                progress.update(1)
                            else:
                                rejections[index]["sample_filter"] += 1
                                deferred.append(index)
            pending = [index for index in deferred if index not in accepted]
    finally:
        progress.close()
    return [accepted[index] for index in range(start, end)]


def _postprocess_candidates_gpu_full(
    candidates: list[tuple[int, torch.Tensor, torch.Tensor, dict[str, object]]],
    config: SynOMolTransferConfig,
    *,
    split: str,
    device: torch.device,
    k_label: int,
) -> list[tuple[int, torch.Tensor, torch.Tensor, dict[str, object]]]:
    processed_by_index: dict[int, tuple[int, torch.Tensor, torch.Tensor, dict[str, object]]] = {}
    table = synomol_transfer_atom_type_table()
    for sample_kind in sorted({str(metadata["sample_kind"]) for _, _, _, metadata in candidates}):
        group = [candidate for candidate in candidates if str(candidate[3]["sample_kind"]) == sample_kind]
        max_atoms = max(int(coords.shape[0]) for _, _, coords, _ in group)
        batch_size = len(group)
        atom_types = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
        coords_batch = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32, device=device)
        mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)
        for batch_idx, (_, sample_atom_types, sample_coords, _) in enumerate(group):
            count = int(sample_atom_types.shape[0])
            atom_types[batch_idx, :count] = sample_atom_types.to(device=device)
            coords_batch[batch_idx, :count] = sample_coords.to(device=device)
            mask[batch_idx, :count] = True
        if sample_kind != "collision":
            relax_steps = config.relaxation_snapshot_steps if sample_kind == "relaxation" else config.relax_steps
            coords_batch = relax_coords_batch(atom_types, coords_batch, mask, config, steps=relax_steps, k_label=k_label)
            if sample_kind == "langevin":
                sample_seeds = torch.tensor(
                    [int(metadata.get("proposal_seed", 0)) + 1_000_003 for _, _, _, metadata in group],
                    device=device,
                    dtype=torch.long,
                )
                coords_batch = langevin_coords_batch(
                    atom_types,
                    coords_batch,
                    mask,
                    config,
                    steps=config.langevin_steps,
                    k_label=k_label,
                    sample_seeds=sample_seeds,
                )
        coords_cpu = coords_batch.detach().cpu()
        for batch_idx, (index, sample_atom_types, _, metadata) in enumerate(group):
            count = int(sample_atom_types.shape[0])
            sample_coords = coords_cpu[batch_idx, :count].contiguous()
            sample_metadata = dict(metadata)
            generator = torch.Generator().manual_seed(int(sample_metadata.get("proposal_seed", 0)) + 2_000_033)
            if sample_kind in {"bond_distortion", "angle_distortion", "shell_distortion"}:
                sample_coords = _distort_coords(
                    sample_coords,
                    sample_atom_types,
                    str(sample_metadata["primary_motif"]),
                    sample_kind,
                    split,
                    table,
                    config,
                    generator,
                )
            repair_count = 0
            sample_atom_types = sample_atom_types.clone()
            if split != "test_type_combo":
                sample_atom_types, repair_count = _repair_heldout_type_combos(
                    sample_atom_types,
                    sample_coords,
                    config,
                    generator,
                )
            sample_metadata["heldout_repair_count"] = int(repair_count)
            if sample_atom_types.shape[0] >= 3:
                sample_metadata["primary_triple"] = (
                    int(sample_atom_types[0].item()),
                    int(sample_atom_types[1].item()),
                    int(sample_atom_types[2].item()),
                )
            processed_by_index[index] = (index, sample_atom_types, sample_coords.to(dtype=torch.float32), sample_metadata)
    return [processed_by_index[index] for index, _, _, _ in candidates]


def _label_candidates_on_device(
    candidates,
    config: SynOMolTransferConfig,
    *,
    split: str,
    device: torch.device,
    k_label: int,
) -> list[tuple[int, dict[str, object], dict[str, object]]]:
    max_atoms = max(int(coords.shape[0]) for _, _, coords, _ in candidates)
    batch_size = len(candidates)
    atom_types = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
    coords_batch = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32, device=device)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)
    for batch_idx, (_, sample_atom_types, sample_coords, _) in enumerate(candidates):
        count = int(sample_atom_types.shape[0])
        atom_types[batch_idx, :count] = sample_atom_types.to(device=device)
        coords_batch[batch_idx, :count] = sample_coords.to(device=device)
        mask[batch_idx, :count] = True
    energy, forces, components, label_stats = compute_synomol_transfer_labels_batch_local(
        atom_types,
        coords_batch,
        mask,
        config,
        k_label=k_label,
        return_stats=True,
    )
    results = []
    for batch_idx, (index, sample_atom_types, sample_coords, metadata) in enumerate(candidates):
        count = int(sample_atom_types.shape[0])
        sample_components = {
            name: components[name][batch_idx].detach().cpu()
            for name in components
        }
        sample = {
            "atom_types": sample_atom_types.detach().cpu(),
            "coords": sample_coords.detach().cpu(),
            "forces": forces[batch_idx, :count].detach().cpu(),
            "energy": energy[batch_idx].detach().cpu(),
            "component_energies": sample_components,
            "num_atoms": count,
            "idx": int(index),
            "system_id": int(metadata["system_id"]),
            "sample_kind": str(metadata["sample_kind"]),
            "primary_motif": str(metadata["primary_motif"]),
            "primary_triple": torch.as_tensor(metadata["primary_triple"], dtype=torch.long),
            "original_primary_triple": torch.as_tensor(metadata["original_primary_triple"], dtype=torch.long),
            "has_heldout_type_combo": bool(metadata["has_heldout_type_combo"]),
            "heldout_repair_count": int(metadata["heldout_repair_count"]),
            "proposal_seed": int(metadata.get("proposal_seed", -1)),
            "accepted_attempts": 1,
            "global_sample_id": int(index),
            "shard_id": -1,
            "motif_labels": torch.as_tensor(metadata["motif_labels"], dtype=torch.long),
            "split": split,
        }
        stats = {
            "accepted_sample_kind": str(metadata["sample_kind"]),
            "accepted_primary_motif": str(metadata["primary_motif"]),
            "heldout_repair_count": int(metadata["heldout_repair_count"]),
            "label_neighbor_stats": dict(label_stats),
        }
        results.append((index, sample, stats))
    return results


def _chunk_candidates_for_gpu(
    candidates: list[tuple[int, torch.Tensor, torch.Tensor, dict[str, object]]],
    *,
    max_batch_size: int,
    k_label: int,
    angle_lmax: int,
    max_gpu_work: int,
):
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if not candidates:
        return
    index = 0
    while index < len(candidates):
        chunk: list[tuple[int, torch.Tensor, torch.Tensor, dict[str, object]]] = []
        max_atoms = 0
        while index < len(candidates) and len(chunk) < max_batch_size:
            candidate = candidates[index]
            candidate_atoms = int(candidate[2].shape[0])
            next_max_atoms = max(max_atoms, candidate_atoms)
            next_size = len(chunk) + 1
            work = next_size * next_max_atoms * int(k_label) * int(k_label) * (int(angle_lmax) + 1)
            if chunk and max_gpu_work > 0 and work > max_gpu_work:
                break
            chunk.append(candidate)
            max_atoms = next_max_atoms
            index += 1
        if not chunk:
            chunk.append(candidates[index])
            index += 1
        yield chunk


def _cleanup_stale_split_outputs(cache_dir: Path, split: str, *, writing_sharded: bool) -> None:
    fixed_path = cache_dir / f"{split}.pt"
    manifest_path = cache_dir / f"{split}_manifest.json"
    if writing_sharded:
        if fixed_path.exists():
            fixed_path.unlink()
        return
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest.get("shards", []):
                shard_name = entry["path"] if isinstance(entry, dict) else str(entry)
                shard_path = cache_dir / shard_name
                if shard_path.exists():
                    shard_path.unlink()
        except json.JSONDecodeError:
            for shard_path in cache_dir.glob(f"{split}_*.pt"):
                shard_path.unlink()
        manifest_path.unlink()


def _neighbor_cap_hit_fraction(stats: dict[str, object]) -> float:
    label_stats = stats.get("label_neighbor_stats")
    if not isinstance(label_stats, dict):
        return 0.0
    cap_stats = label_stats.get("neighbor_cap_hit_fraction")
    if isinstance(cap_stats, dict):
        return float(cap_stats.get("max", cap_stats.get("mean", 0.0)))
    if cap_stats is None:
        return 0.0
    return float(cap_stats)


def _enforce_label_cap_policy(
    stats: dict[str, object],
    *,
    split: str,
    max_label_cap_hit_fraction: float,
    allow_label_cap_hits: bool,
) -> None:
    if allow_label_cap_hits:
        stats["label_cap_policy"] = {
            "allow_label_cap_hits": True,
            "max_label_cap_hit_fraction": float(max_label_cap_hit_fraction),
        }
        return
    cap_fraction = _neighbor_cap_hit_fraction(stats)
    stats["label_cap_policy"] = {
        "allow_label_cap_hits": False,
        "max_label_cap_hit_fraction": float(max_label_cap_hit_fraction),
        "observed_neighbor_cap_hit_fraction": float(cap_fraction),
    }
    if cap_fraction > max_label_cap_hit_fraction:
        raise RuntimeError(
            f"SynOMol-Transfer {split} GPU label cap hit fraction {cap_fraction:.6g} "
            f"exceeds threshold {max_label_cap_hit_fraction:.6g}; increase --k-label "
            "or pass --allow-label-cap-hits for an explicitly capped-oracle run"
        )


def _iter_tasks(start: int, end: int, split: str, split_config: SynOMolTransferConfig):
    for index in range(start, end):
        yield index, split, split_config


def _worker_init() -> None:
    torch.set_num_threads(1)


def _add_timing_stats(stats: dict[str, object], *, elapsed: float) -> dict[str, object]:
    accepted = int(stats.get("accepted", 0))
    stats = dict(stats)
    stats["timing"] = {
        "total_seconds": float(elapsed),
        "seconds_per_sample": float(elapsed) / float(max(accepted, 1)),
        "samples_per_second": float(accepted) / float(max(elapsed, 1.0e-12)),
    }
    return stats


def _new_summary_accumulator() -> dict[str, object]:
    return {
        "num_atoms": [],
        "force_rms": [],
        "energy_per_atom": [],
        "component_energies": {name: [] for name in ("pair", "angle", "motif", "many_body")},
        "component_energies_per_atom": {name: [] for name in ("pair", "angle", "motif", "many_body")},
        "sample_kind_counts": {},
        "primary_motif_counts": {},
        "heldout_type_combo_count": 0,
        "heldout_repair_count": 0,
        "attempts": [],
        "rejections": {},
        "heldout_repair_count_from_generation": 0,
        "label_neighbor_stats": {
            "mean_label_neighbors": [],
            "q95_label_neighbors": [],
            "max_label_neighbors": [],
            "neighbor_cap_hit_fraction": [],
        },
    }


def _accumulate_summary(
    accumulator: dict[str, object],
    samples: list[dict[str, object]],
    generation_stats: list[dict[str, object]],
) -> None:
    for sample in samples:
        num_atoms = int(sample["num_atoms"])
        accumulator["num_atoms"].append(num_atoms)
        forces = torch.as_tensor(sample["forces"], dtype=torch.float32)
        accumulator["force_rms"].append(float(torch.sqrt((forces * forces).sum(dim=-1).mean()).item()))
        energy = float(torch.as_tensor(sample["energy"], dtype=torch.float32).item())
        accumulator["energy_per_atom"].append(energy / float(max(num_atoms, 1)))
        for name in ("pair", "angle", "motif", "many_body"):
            value = float(torch.as_tensor(sample["component_energies"][name], dtype=torch.float32).item())
            accumulator["component_energies"][name].append(value)
            accumulator["component_energies_per_atom"][name].append(value / float(max(num_atoms, 1)))
        _increment_count(accumulator["sample_kind_counts"], str(sample["sample_kind"]))
        _increment_count(accumulator["primary_motif_counts"], str(sample["primary_motif"]))
        accumulator["heldout_type_combo_count"] += int(bool(sample["has_heldout_type_combo"]))
        accumulator["heldout_repair_count"] += int(sample.get("heldout_repair_count", 0))
    for stat in generation_stats:
        accumulator["attempts"].append(int(stat.get("attempts", 1)))
        accumulator["heldout_repair_count_from_generation"] += int(stat.get("heldout_repair_count", 0))
        for reason, count in dict(stat.get("rejection_reasons", {})).items():
            rejections = accumulator["rejections"]
            rejections[str(reason)] = int(rejections.get(str(reason), 0)) + int(count)
        for name, value in dict(stat.get("label_neighbor_stats", {})).items():
            if name in accumulator["label_neighbor_stats"]:
                accumulator["label_neighbor_stats"][name].append(float(value))


def _finalize_summary(accumulator: dict[str, object]) -> dict[str, object]:
    accepted = len(accumulator["num_atoms"])
    attempts = [int(value) for value in accumulator["attempts"]]
    total_attempts = int(sum(attempts))
    rejections = dict(accumulator["rejections"])
    summary = {
        "requested": accepted,
        "accepted": accepted,
        "num_atoms": _numeric_summary(accumulator["num_atoms"]),
        "force_rms": _numeric_summary(accumulator["force_rms"]),
        "energy_per_atom": _numeric_summary(accumulator["energy_per_atom"]),
        "component_energies": {
            name: _numeric_summary(values)
            for name, values in accumulator["component_energies"].items()
        },
        "component_energies_per_atom": {
            name: _numeric_summary(values)
            for name, values in accumulator["component_energies_per_atom"].items()
        },
        "sample_kind_counts": dict(accumulator["sample_kind_counts"]),
        "primary_motif_counts": dict(accumulator["primary_motif_counts"]),
        "heldout_type_combo_count": int(accumulator["heldout_type_combo_count"]),
        "heldout_repair_count": int(accumulator["heldout_repair_count"]),
        "attempts": total_attempts,
        "mean_attempts_per_accepted": float(total_attempts) / float(max(accepted, 1)),
        "max_attempts": int(max(attempts) if attempts else 0),
        "rejections": rejections,
        "rejection_rate": float(sum(rejections.values())) / float(max(total_attempts, 1)),
        "heldout_repair_count_from_generation": int(accumulator["heldout_repair_count_from_generation"]),
    }
    label_neighbor_stats = {
        name: _numeric_summary(values)
        for name, values in accumulator["label_neighbor_stats"].items()
        if values
    }
    if label_neighbor_stats:
        summary["label_neighbor_stats"] = label_neighbor_stats
    return summary


def _numeric_summary(values) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {"min": 0.0, "q05": 0.0, "median": 0.0, "mean": 0.0, "q95": 0.0, "max": 0.0}
    return {
        "min": float(tensor.min().item()),
        "q05": float(torch.quantile(tensor, 0.05).item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "mean": float(tensor.mean().item()),
        "q95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def _increment_count(counts: dict[str, int], value: str) -> None:
    counts[value] = int(counts.get(value, 0)) + 1


def _generate_one_sample(task: tuple[int, str, SynOMolTransferConfig]) -> tuple[dict[str, object], dict[str, object]]:
    torch.set_num_threads(1)
    index, split, config = task
    return generate_synomol_transfer_sample_with_stats(
        index=index,
        split=split,
        epoch=0,
        config=config,
    )


def tqdm_map(iterator, *, total: int, desc: str):
    yield from tqdm(iterator, total=total, desc=desc, unit="sample")


def write_manifest(
    root: Path,
    *,
    preset: str,
    config_name: str,
    config: SynOMolTransferConfig,
    metadata: dict[str, object],
    status: str,
) -> None:
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    aliases = dict(manifest.get("aliases", {}))
    aliases[preset] = config_name
    aliases[f"synomol_transfer_v1_{preset}"] = config_name
    configs = dict(manifest.get("configs", {}))
    configs[config_name] = {
        "preset": preset,
        "status": status,
        "config": asdict(config),
        "metadata": metadata,
    }
    manifest = {"aliases": aliases, "configs": configs}
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(args: argparse.Namespace) -> None:
    if args.backend == "serial-cpu":
        args.num_workers = 0
    if args.backend == "multiprocess-cpu" and args.num_workers <= 0:
        raise ValueError("--backend multiprocess-cpu requires --num-workers > 0")
    if args.backend in {"gpu-batched", "gpu-full"}:
        if args.generation_device != "cuda":
            raise ValueError(f"--backend {args.backend} requires --generation-device cuda")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for {args.backend} generation")
    split_sizes = split_sizes_from_args(args)
    base_config = build_config(args, length=split_sizes["train"])
    metadata: dict[str, object] = {
        "preset": args.preset,
        "backend": args.backend,
        "generation_device": args.generation_device,
        "generation_batch_size": int(args.generation_batch_size),
        "k_label": int(args.k_label),
        "max_gpu_work": int(args.max_gpu_work),
        "allow_label_cap_hits": bool(args.allow_label_cap_hits),
        "max_label_cap_hit_fraction": float(args.max_label_cap_hit_fraction),
    }
    if args.calibrate_components:
        calibration_size = args.calibration_size
        if calibration_size is None:
            calibration_size = {"smoke": 64, "default": 512, "full": 2048}[args.preset]
        base_config, calibration_metadata = calibrate_synomol_transfer_component_scales(
            base_config,
            split="train",
            calibration_size=calibration_size,
        )
        metadata["calibration"] = calibration_metadata
    if args.config_name is None:
        config_name = base_config.cache_name()
        if args.backend == "gpu-full" or args.allow_label_cap_hits:
            config_name = f"{config_name}_{args.backend}_k{args.k_label}"
    else:
        config_name = args.config_name
    root = Path(args.root)
    write_manifest(
        root,
        preset=args.preset,
        config_name=config_name,
        config=base_config,
        metadata=metadata,
        status="incomplete",
    )
    for split in SYNOMOL_TRANSFER_SPLITS:
        write_split(
            root=root,
            base_config=base_config,
            config_name=config_name,
            split=split,
            size=split_sizes[split],
            metadata=metadata,
            force=args.force,
            num_workers=args.num_workers,
            backend=args.backend,
            generation_device=args.generation_device,
            generation_batch_size=args.generation_batch_size,
            k_label=args.k_label,
            max_gpu_work=args.max_gpu_work,
            max_label_cap_hit_fraction=args.max_label_cap_hit_fraction,
            allow_label_cap_hits=args.allow_label_cap_hits,
            chunksize=args.chunksize,
            shard_size=args.shard_size,
        )
    write_manifest(
        root,
        preset=args.preset,
        config_name=config_name,
        config=base_config,
        metadata=metadata,
        status="complete",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materialize fixed SynOMol-Transfer splits.")
    parser.add_argument("--root", type=Path, default=Path("data/synomol_transfer"))
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--preset", choices=("default", "smoke", "full"), default="default")
    parser.add_argument("--num-atoms", type=str, default=None)
    parser.add_argument("--size-ood-num-atoms", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-iid-size", type=int, default=None)
    parser.add_argument("--test-type-combo-size", type=int, default=None)
    parser.add_argument("--test-motif-size", type=int, default=None)
    parser.add_argument("--test-size-size", type=int, default=None)
    parser.add_argument("--test-perturb-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density", type=float, default=0.18)
    parser.add_argument("--cutoff", type=float, default=4.2)
    parser.add_argument("--cutoff-delta", type=float, default=0.7)
    parser.add_argument("--radial-basis-size", type=int, default=8)
    parser.add_argument("--angle-lmax", type=int, default=4)
    parser.add_argument("--relax-steps", type=int, default=None)
    parser.add_argument("--relaxation-snapshot-steps", type=int, default=None)
    parser.add_argument("--langevin-steps", type=int, default=None)
    parser.add_argument("--pair-scale", type=float, default=0.04)
    parser.add_argument("--angle-scale", type=float, default=0.006)
    parser.add_argument("--motif-scale", type=float, default=0.08)
    parser.add_argument("--many-body-scale", type=float, default=0.10)
    parser.add_argument("--generation-pair-scale", type=float, default=0.08)
    parser.add_argument("--generation-angle-scale", type=float, default=0.012)
    parser.add_argument("--generation-motif-scale", type=float, default=0.08)
    parser.add_argument("--generation-many-body-scale", type=float, default=0.0)
    parser.add_argument("--force-rms-min", type=float, default=1.0e-5)
    parser.add_argument("--force-rms-max", type=float, default=80.0)
    parser.add_argument("--max-abs-energy", type=float, default=1.0e5)
    parser.add_argument("--max-abs-energy-per-atom", type=float, default=1.0e3)
    parser.add_argument("--calibration-size", type=int, default=None)
    parser.add_argument("--backend", choices=("serial-cpu", "multiprocess-cpu", "gpu-batched", "gpu-full"), default="serial-cpu")
    parser.add_argument("--generation-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--k-label", type=int, default=64)
    parser.add_argument("--max-gpu-work", type=int, default=80_000_000)
    parser.add_argument("--max-label-cap-hit-fraction", type=float, default=0.0)
    parser.add_argument("--allow-label-cap-hits", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--calibrate-components", dest="calibrate_components", action="store_true", default=True)
    parser.add_argument("--no-calibrate-components", dest="calibrate_components", action="store_false")
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
