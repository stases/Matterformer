#!/usr/bin/env python3
"""Download preprocessed AtomMOF datasets into the Matterformer data layout."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


HF_TREE_URL = "https://huggingface.co/api/datasets/{repo_id}/tree/main/{dataset}?recursive=1&expand=1"
HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{path}?download=true"


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def fetch_remote_files(repo_id: str, dataset: str) -> list[RemoteFile]:
    url = HF_TREE_URL.format(repo_id=repo_id, dataset=dataset)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)

    files: list[RemoteFile] = []
    for entry in payload:
        if entry.get("type") != "file":
            continue
        path = entry.get("path")
        size = int(entry.get("size", 0))
        if not isinstance(path, str) or not path.startswith(f"{dataset}/"):
            continue
        files.append(RemoteFile(path=path, size=size))

    if not files:
        raise RuntimeError(f"No files found for dataset '{dataset}' in repo '{repo_id}'.")
    return sorted(files, key=lambda item: item.path)


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _download_response(url: str, start_byte: int):
    headers = {}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=120)


def download_file(repo_id: str, remote_file: RemoteFile, local_root: Path, force: bool) -> Path:
    destination = local_root / remote_file.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")

    if force:
        if destination.exists():
            destination.unlink()
        if partial.exists():
            partial.unlink()

    if destination.exists() and destination.stat().st_size == remote_file.size:
        print(f"skip  {remote_file.path} ({format_bytes(remote_file.size)})")
        return destination

    resume_from = partial.stat().st_size if partial.exists() else 0
    url = HF_RESOLVE_URL.format(
        repo_id=repo_id,
        path=urllib.parse.quote(remote_file.path),
    )

    with _download_response(url, resume_from) as response:
        status = getattr(response, "status", None)
        mode = "ab" if resume_from > 0 and status == 206 else "wb"
        if mode == "wb":
            resume_from = 0

        print(
            f"fetch {remote_file.path} "
            f"({format_bytes(remote_file.size)}) -> {destination}"
        )

        written = resume_from
        last_report = time.time()
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_report >= 5:
                    pct = 100.0 * written / max(remote_file.size, 1)
                    print(f"  ... {pct:5.1f}% ({format_bytes(written)}/{format_bytes(remote_file.size)})")
                    last_report = now

    final_size = partial.stat().st_size
    if final_size != remote_file.size:
        raise RuntimeError(
            f"Downloaded size mismatch for {remote_file.path}: "
            f"expected {remote_file.size}, got {final_size}."
        )

    partial.replace(destination)
    return destination


def write_manifest(repo_id: str, dataset: str, local_root: Path, files: list[RemoteFile]) -> Path:
    dataset_root = local_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "_source_manifest.json"
    manifest = {
        "repo_id": repo_id,
        "dataset": dataset,
        "source_repo_url": f"https://huggingface.co/datasets/{repo_id}",
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [{"path": item.path, "size": item.size} for item in files],
        "total_bytes": sum(item.size for item in files),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a preprocessed AtomMOF dataset into Matterformer data/mofs/."
    )
    parser.add_argument(
        "--dataset",
        choices=("bwdb", "odac25"),
        default="bwdb",
        help="Which AtomMOF dataset subtree to download.",
    )
    parser.add_argument(
        "--repo-id",
        default="nayoung10/AtomMOF-data",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("data") / "mofs",
        help="Root directory under which the dataset subtree is stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if local copies already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_root = args.local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    files = fetch_remote_files(args.repo_id, args.dataset)
    total_bytes = sum(item.size for item in files)
    print(
        f"Found {len(files)} files for {args.dataset} "
        f"({format_bytes(total_bytes)}) in {args.repo_id}."
    )
    print(f"Downloading into {local_root / args.dataset}")

    for remote_file in files:
        download_file(args.repo_id, remote_file, local_root=local_root, force=args.force)

    manifest_path = write_manifest(args.repo_id, args.dataset, local_root=local_root, files=files)
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
