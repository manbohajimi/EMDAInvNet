from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_prediction(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        volume = np.load(path)
    else:
        with np.load(path) as archive:
            volume = archive[key]
    volume = np.asarray(volume).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"expected 3-D data, got {volume.shape}")
    return volume


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize orthogonal EMDAInvNet slices")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key", default="permittivity")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    args = parser.parse_args()
    volume = load_prediction(args.input, args.key)
    centers = tuple(size // 2 for size in volume.shape)
    slices = (
        volume[centers[0], :, :],
        volume[:, centers[1], :],
        volume[:, :, centers[2]],
    )
    labels = ("D center", "H center", "W center")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    image = None
    for axis, image_slice, label in zip(axes, slices, labels):
        image = axis.imshow(image_slice, cmap="viridis", vmin=args.vmin, vmax=args.vmax)
        axis.set_title(label)
        axis.set_xlabel("voxel")
        axis.set_ylabel("voxel")
    assert image is not None
    figure.colorbar(image, ax=axes, label=args.key, shrink=0.85)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()

