"""Fail-fast audit for the paper-faithful Dataset-II training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from emdainvnet.config import load_config, resolve_project_path
from emdainvnet.data import dataset_from_config, load_volume, normalized_target_threshold
from emdainvnet.model import EMDAInvNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", help="defaults to data.train_manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", help="optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    manifest = resolve_project_path(config, args.manifest or data_config["train_manifest"])
    dataset = dataset_from_config(manifest, data_config)
    first_target_path = dataset._resolve(dataset.records[0]["target"])
    raw_target = load_volume(first_target_path, data_config.get("target_key"))
    raw_target_min = float(raw_target.min())
    raw_target_max = float(raw_target.max())
    expected_physical_min = float(data_config["target_min"])
    expected_physical_max = float(data_config["target_max"])
    if abs(raw_target_min - expected_physical_min) > 1e-3:
        raise AssertionError(
            f"raw target background mismatch: {raw_target_min} != {expected_physical_min}"
        )
    if raw_target_max > expected_physical_max + 1e-3:
        raise AssertionError(
            f"raw target exceeds paper Dataset-II range: {raw_target_max} > "
            f"{expected_physical_max}"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    radar, target, sample_ids = next(iter(loader))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = EMDAInvNet(**config["model"]).to(device)
    radar = radar.to(device)
    target = target.to(device)
    model.train()
    prediction = model(radar)
    loss = nn.L1Loss()(prediction, target)
    loss.backward()

    parameter_count = model.num_parameters()
    expected_parameters = 16_637_852
    is_paper_default = (
        int(config["model"].get("base_channels", 8)) == 8
        and bool(config["model"].get("use_emsfa", True))
        and bool(config["model"].get("use_dilated", True))
    )
    if is_paper_default and parameter_count != expected_parameters:
        raise AssertionError(
            f"paper model parameter contract changed: {parameter_count:,} != "
            f"{expected_parameters:,}"
        )
    expected_shape = tuple(int(value) for value in data_config["input_shape"])
    if tuple(radar.shape[2:]) != expected_shape or tuple(target.shape[2:]) != expected_shape:
        raise AssertionError(
            f"preprocessing shape mismatch: radar={tuple(radar.shape)}, "
            f"target={tuple(target.shape)}, expected={expected_shape}"
        )
    if not (0.0 <= float(radar.min()) and float(radar.max()) <= 1.0):
        raise AssertionError("input normalization must produce [0, 1]")
    if not (0.0 <= float(target.min()) and float(target.max()) <= 1.0):
        raise AssertionError("sigmoid target must be in [0, 1]")

    threshold = normalized_target_threshold(data_config)
    gradient_norm = sum(
        float(parameter.grad.detach().norm().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    input_delta = (
        float((prediction[0] - prediction[1]).abs().mean().item())
        if prediction.shape[0] >= 2
        else None
    )
    report = {
        "config": str(Path(args.config).resolve()),
        "manifest": str(manifest),
        "samples": list(sample_ids),
        "parameter_count": parameter_count,
        "radar_shape": list(radar.shape),
        "radar_min": float(radar.min().item()),
        "radar_max": float(radar.max().item()),
        "target_min": float(target.min().item()),
        "target_max": float(target.max().item()),
        "raw_target_min": raw_target_min,
        "raw_target_max": raw_target_max,
        "target_mean": float(target.mean().item()),
        "foreground_fraction": float((target >= threshold).float().mean().item()),
        "prediction_min": float(prediction.min().item()),
        "prediction_max": float(prediction.max().item()),
        "prediction_std": float(prediction.std().item()),
        "prediction_input_delta": input_delta,
        "plain_mae": float(loss.item()),
        "gradient_norm_sum": gradient_norm,
        "target_threshold_normalized": threshold,
    }
    if report["prediction_std"] <= 0.0 or gradient_norm <= 0.0:
        raise AssertionError("untrained model must have nonconstant output and nonzero gradients")
    if input_delta is not None and input_delta <= 0.0:
        raise AssertionError("untrained model output must depend on the input")

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
