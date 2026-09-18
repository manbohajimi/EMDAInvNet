"""Paired 3-D GPR volume loading and paper preprocessing utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset


def _pick_array(container: dict[str, Any], key: str | None, path: Path) -> np.ndarray:
    if key and key in container:
        return np.asarray(container[key])
    candidates = [
        np.asarray(value)
        for name, value in container.items()
        if not name.startswith("__") and isinstance(value, np.ndarray) and value.ndim >= 3
    ]
    if len(candidates) != 1:
        names = [name for name in container if not name.startswith("__")]
        raise KeyError(f"cannot infer array key in {path}; available keys: {names}")
    return candidates[0]


def load_volume(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if key and key in archive:
                array = archive[key]
            elif len(archive.files) == 1:
                array = archive[archive.files[0]]
            else:
                raise KeyError(f"specify key for {path}; available keys: {archive.files}")
    elif suffix == ".mat":
        array = _pick_array(scipy.io.loadmat(path), key, path)
    else:
        raise ValueError(f"unsupported volume format: {path}")
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D volume in {path}, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values found in {path}")
    return np.ascontiguousarray(array)


def resize_volume(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if volume.shape == shape:
        return volume
    tensor = torch.from_numpy(volume)[None, None]
    resized = F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
    return resized[0, 0].numpy()


def suppress_direct_wave(volume: np.ndarray, spatial_axes: tuple[int, int] = (1, 2)) -> np.ndarray:
    """Subtract the mean trace, matching the paper's mean-filter clutter removal.

    The time/depth axis is assumed to be axis 0. Change ``spatial_axes`` when the
    source data uses a different convention.
    """

    return volume - volume.mean(axis=spatial_axes, keepdims=True)


def normalize(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum <= minimum:
        raise ValueError("normalization maximum must exceed minimum")
    return np.clip((array - minimum) / (maximum - minimum), 0.0, 1.0)


def target_affine(data_config: dict[str, Any]) -> tuple[float, float]:
    """Return physical offset and scale for the configured target transform."""

    mode = str(data_config.get("target_normalization", "minmax")).lower()
    if mode == "scale":
        return 0.0, float(data_config["target_scale"])
    if mode == "minmax":
        minimum = float(data_config["target_min"])
        maximum = float(data_config["target_max"])
        return minimum, maximum - minimum
    if mode == "none":
        return 0.0, 1.0
    raise ValueError(f"unsupported target_normalization: {mode}")


def normalize_target_array(array: np.ndarray, mode: str, offset: float, scale: float) -> np.ndarray:
    if mode == "none":
        return array
    if scale <= 0:
        raise ValueError("target normalization scale must be positive")
    return np.clip((array - offset) / scale, 0.0, 1.0)


def denormalize_target_array(array: np.ndarray, data_config: dict[str, Any]) -> np.ndarray:
    offset, scale = target_affine(data_config)
    return array * scale + offset


def normalized_target_threshold(data_config: dict[str, Any]) -> float:
    if "target_threshold_phys" in data_config:
        offset, scale = target_affine(data_config)
        return (float(data_config["target_threshold_phys"]) - offset) / scale
    return float(data_config.get("target_threshold", 0.1))


class PairedVolumeDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """CSV manifest dataset with columns ``input,target`` and optional ``id``."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        input_key: str | None = "clean_data",
        target_key: str | None = "mask",
        input_shape: tuple[int, int, int] = (64, 64, 64),
        input_min: float = -9.0,
        input_max: float = 9.0,
        target_min: float = 4.0,
        target_max: float = 49.92,
        target_normalization: str = "minmax",
        target_scale: float | None = None,
        normalize_target: bool | None = None,
        direct_wave_suppression: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.base_dir = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.records = list(csv.DictReader(handle))
        if not self.records:
            raise ValueError(f"empty manifest: {self.manifest}")
        required = {"input", "target"}
        if not required.issubset(self.records[0]):
            raise ValueError(f"manifest needs columns {sorted(required)}")
        self.input_key = input_key
        self.target_key = target_key
        self.input_shape = tuple(int(v) for v in input_shape)
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        # ``normalize_target`` remains only for old configs. New paper configs
        # must state the exact transform explicitly.
        if normalize_target is not None:
            target_normalization = "minmax" if normalize_target else "none"
        self.target_normalization = str(target_normalization).lower()
        if self.target_normalization == "scale":
            self.target_offset = 0.0
            self.target_scale = float(target_scale if target_scale is not None else target_max)
        elif self.target_normalization == "minmax":
            self.target_offset = self.target_min
            self.target_scale = self.target_max - self.target_min
        elif self.target_normalization == "none":
            self.target_offset = 0.0
            self.target_scale = 1.0
        else:
            raise ValueError(f"unsupported target_normalization: {self.target_normalization}")
        self.direct_wave_suppression = direct_wave_suppression

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        record = self.records[index]
        input_path = self._resolve(record["input"])
        target_path = self._resolve(record["target"])
        radar = load_volume(input_path, self.input_key)
        target = load_volume(target_path, self.target_key)
        if self.direct_wave_suppression:
            radar = suppress_direct_wave(radar)
        radar = resize_volume(radar, self.input_shape)
        target = resize_volume(target, self.input_shape)
        radar = normalize(radar, self.input_min, self.input_max)
        target = normalize_target_array(
            target, self.target_normalization, self.target_offset, self.target_scale
        )
        sample_id = record.get("id") or input_path.stem
        return torch.from_numpy(radar)[None], torch.from_numpy(target)[None], sample_id


def dataset_from_config(manifest: Path, data_config: dict[str, Any]) -> PairedVolumeDataset:
    return PairedVolumeDataset(
        manifest,
        input_key=data_config.get("input_key"),
        target_key=data_config.get("target_key"),
        input_shape=tuple(data_config.get("input_shape", (64, 64, 64))),
        input_min=data_config.get("input_min", -9.0),
        input_max=data_config.get("input_max", 9.0),
        target_min=data_config.get("target_min", 4.0),
        target_max=data_config.get("target_max", 49.92),
        target_normalization=data_config.get("target_normalization", "minmax"),
        target_scale=data_config.get("target_scale"),
        normalize_target=data_config.get("normalize_target"),
        direct_wave_suppression=data_config.get("direct_wave_suppression", False),
    )
