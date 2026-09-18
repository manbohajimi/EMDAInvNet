"""Metrics defined by equations (12)-(17) of the EMDAInvNet paper."""

from __future__ import annotations

import torch
from torch import Tensor


def _per_sample_mean(x: Tensor) -> Tensor:
    return x.flatten(1).mean(dim=1)


def volume_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.1,
    epsilon: float = 1e-8,
    data_range: float = 1.0,
    mape_min: float | None = None,
    mape_max: float | None = None,
) -> dict[str, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    prediction = prediction.float()
    target = target.float()
    difference = prediction - target
    mae = _per_sample_mean(difference.abs())
    mse = _per_sample_mean(difference.square())
    if mape_min is not None and mape_max is not None:
        scale = mape_max - mape_min
        target_for_mape = target * scale + mape_min
        difference_for_mape = difference * scale
    else:
        target_for_mape = target
        difference_for_mape = difference
    mape = _per_sample_mean(
        difference_for_mape.abs() / (target_for_mape.abs() + epsilon)
    ) * 100.0

    dims = tuple(range(1, target.ndim))
    mu_target = target.mean(dim=dims)
    mu_prediction = prediction.mean(dim=dims)
    centered_target = target - mu_target.view(-1, *([1] * (target.ndim - 1)))
    centered_prediction = prediction - mu_prediction.view(-1, *([1] * (target.ndim - 1)))
    variance_target = centered_target.square().mean(dim=dims)
    variance_prediction = centered_prediction.square().mean(dim=dims)
    covariance = (centered_target * centered_prediction).mean(dim=dims)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = (
        (2 * mu_target * mu_prediction + c1) * (2 * covariance + c2)
        / (
            (mu_target.square() + mu_prediction.square() + c1)
            * (variance_target + variance_prediction + c2)
        )
    )

    predicted_mask = prediction > threshold
    target_mask = target > threshold
    intersection = (predicted_mask & target_mask).flatten(1).sum(dim=1).float()
    union = (predicted_mask | target_mask).flatten(1).sum(dim=1).float()
    predicted_size = predicted_mask.flatten(1).sum(dim=1).float()
    target_size = target_mask.flatten(1).sum(dim=1).float()
    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * intersection + epsilon) / (predicted_size + target_size + epsilon)
    return {"mae": mae, "mse": mse, "mape": mape, "ssim": ssim, "iou": iou, "dice": dice}


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, Tensor]) -> None:
        batch = next(iter(metrics.values())).numel()
        for name, values in metrics.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(values.sum().item())
        self.count += batch

    def compute(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("no samples accumulated")
        return {name: total / self.count for name, total in self.sums.items()}
