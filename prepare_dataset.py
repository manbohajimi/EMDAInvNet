"""Create deterministic CSV manifests from paired 3DInvNet-style folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


SUPPORTED = {".mat", ".npy", ".npz"}


def indexed_files(folder: Path) -> dict[str, Path]:
    return {
        path.stem: path.resolve()
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    }


def write_manifest(path: Path, records: list[tuple[str, Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "input", "target"])
        writer.writeheader()
        for sample_id, input_path, target_path in records:
            writer.writerow({"id": sample_id, "input": input_path, "target": target_path})


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair volume files and write train/val/test manifests")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/manifests"), type=Path)
    parser.add_argument("--name", default="dataset_ii")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write all pairs to one manifest and skip random splitting",
    )
    parser.add_argument("--train", type=float, default=0.95)
    parser.add_argument("--val", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.train <= 0 or args.val < 0 or args.train + args.val >= 1:
        raise ValueError("ratios require train > 0, val >= 0 and train + val < 1")
    inputs = indexed_files(args.inputs)
    targets = indexed_files(args.targets)
    shared = sorted(inputs.keys() & targets.keys())
    if not shared:
        raise ValueError("no matching file stems found")
    missing_targets = sorted(inputs.keys() - targets.keys())
    missing_inputs = sorted(targets.keys() - inputs.keys())
    if missing_targets or missing_inputs:
        raise ValueError(
            f"unpaired files: missing targets={missing_targets[:5]}, missing inputs={missing_inputs[:5]}"
        )
    random.Random(args.seed).shuffle(shared)
    records = [(key, inputs[key], targets[key]) for key in shared]
    if args.manifest:
        write_manifest(args.manifest, records)
        print(f"all: {len(records)} -> {args.manifest.resolve()}")
        return
    n_train = int(len(records) * args.train)
    n_val = int(len(records) * args.val)
    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }
    for split, split_records in splits.items():
        path = args.output_dir / f"{args.name}_{split}.csv"
        write_manifest(path, split_records)
        print(f"{split}: {len(split_records)} -> {path.resolve()}")


if __name__ == "__main__":
    main()
