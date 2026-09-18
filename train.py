from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from emdainvnet.checkpoint import load_model_weights, save_checkpoint
from emdainvnet.config import load_config, resolve_project_path
from emdainvnet.data import dataset_from_config, normalized_target_threshold, target_affine
from emdainvnet.metrics import MetricAccumulator, volume_metrics
from emdainvnet.model import EMDAInvNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EMDAInvNet reproduction")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--pretrained", help="model checkpoint used for measured-data fine-tuning")
    parser.add_argument("--resume", help="resume model, optimizer and scheduler state")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-emsfa", action="store_true")
    parser.add_argument("--disable-dilated", action="store_true")
    parser.add_argument(
        "--dilated-branch-block", choices=("conv", "conv_in_lrelu")
    )
    parser.add_argument(
        "--residual-order", choices=("add_then_activate", "activate_then_add")
    )
    parser.add_argument(
        "--transition-block", choices=("conv", "conv_in_lrelu")
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    data_config: dict,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    prediction_sum = 0.0
    prediction_sq_sum = 0.0
    prediction_count = 0
    prediction_min = float("inf")
    prediction_max = float("-inf")
    input_delta_sum = 0.0
    input_delta_count = 0
    foreground_sum = 0.0
    foreground_count = 0
    background_sum = 0.0
    background_count = 0
    threshold = normalized_target_threshold(data_config)
    for radar, target, _ in loader:
        radar = radar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(radar)
        prediction_sum += float(prediction.sum().item())
        prediction_sq_sum += float(prediction.square().sum().item())
        prediction_count += prediction.numel()
        prediction_min = min(prediction_min, float(prediction.min().item()))
        prediction_max = max(prediction_max, float(prediction.max().item()))
        if prediction.shape[0] >= 2:
            input_delta_sum += float((prediction[0] - prediction[1]).abs().mean().item())
            input_delta_count += 1
        foreground = target >= threshold
        background = ~foreground
        foreground_sum += float(prediction[foreground].sum().item())
        foreground_count += int(foreground.sum().item())
        background_sum += float(prediction[background].sum().item())
        background_count += int(background.sum().item())
        target_offset, target_scale = target_affine(data_config)
        accumulator.update(
            volume_metrics(
                prediction,
                target,
                threshold=threshold,
                mape_min=target_offset,
                mape_max=target_offset + target_scale,
            )
        )
    result = accumulator.compute()
    prediction_mean = prediction_sum / max(prediction_count, 1)
    prediction_variance = max(
        prediction_sq_sum / max(prediction_count, 1) - prediction_mean**2,
        0.0,
    )
    result.update(
        {
            "pred_mean": prediction_mean,
            "pred_std": prediction_variance**0.5,
            "pred_min": prediction_min,
            "pred_max": prediction_max,
            "pred_input_delta": input_delta_sum / max(input_delta_count, 1),
            "pred_on_target": foreground_sum / max(foreground_count, 1),
            "pred_on_background": background_sum / max(background_count, 1),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    train_config = config["training"]
    if args.train_manifest:
        data_config["train_manifest"] = args.train_manifest
    if args.val_manifest:
        data_config["val_manifest"] = args.val_manifest
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.disable_emsfa:
        config["model"]["use_emsfa"] = False
    if args.disable_dilated:
        config["model"]["use_dilated"] = False
    if args.dilated_branch_block:
        config["model"]["dilated_branch_block"] = args.dilated_branch_block
    if args.residual_order:
        config["model"]["residual_order"] = args.residual_order
    if args.transition_block:
        config["model"]["transition_block"] = args.transition_block
    set_seed(int(config.get("seed", 2026)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train_manifest = resolve_project_path(config, data_config["train_manifest"])
    val_manifest = resolve_project_path(config, data_config["val_manifest"])
    train_dataset = dataset_from_config(train_manifest, data_config)
    val_dataset = dataset_from_config(val_manifest, data_config)
    batch_size = int(train_config["batch_size"])
    workers = int(data_config.get("num_workers", 0))
    train_loader = make_loader(train_dataset, batch_size, workers, True)
    val_loader = make_loader(val_dataset, batch_size, workers, False)

    model = EMDAInvNet(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_config["lr_decay"])
    )
    criterion = nn.L1Loss()
    use_amp = bool(train_config.get("amp", True)) and not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    if args.pretrained:
        load_model_weights(model, args.pretrained, device)
    if args.resume:
        checkpoint = load_model_weights(model, args.resume, device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    output_dir = resolve_project_path(config, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in config.items() if not k.startswith("_")}, handle, indent=2)
    history_path = output_dir / "metrics.csv"
    best = float("inf")
    selection_metric = train_config.get("selection_metric", "mae")
    epochs = int(train_config["epochs"])
    milestone_epochs = {int(value) for value in train_config.get("milestone_epochs", [])}
    print(
        f"device={device} parameters={model.num_parameters():,} "
        f"train={len(train_dataset)} val={len(val_dataset)} amp={use_amp}"
    )

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        for radar, target, _ in train_loader:
            radar = radar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(radar)
                loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * radar.shape[0]
            seen += radar.shape[0]
        metrics = validate(model, val_loader, device, data_config)
        train_loss = running_loss / seen
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_mae": train_loss,
            **{f"val_{key}": value for key, value in metrics.items()},
            "seconds": time.time() - started,
        }
        write_header = not history_path.exists()
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        score = metrics[selection_metric]
        if score < best:
            best = score
            save_checkpoint(
                output_dir / "best.pt",
                model,
                epoch=epoch,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
            )
        every = int(train_config.get("checkpoint_every", 10))
        if epoch % every == 0 or epoch == epochs:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                epoch=epoch,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
            )
        if epoch in milestone_epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                epoch=epoch,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
            )
        scheduler.step()
        formatted = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
        print(f"epoch={epoch:03d} train_mae={train_loss:.6f} {formatted} lr={lr:.3e}")
        if (
            metrics["pred_std"] < float(train_config.get("collapse_std_threshold", 1e-4))
            and metrics["pred_input_delta"]
            < float(train_config.get("collapse_input_delta_threshold", 1e-5))
        ):
            print(
                "WARNING possible constant-output collapse: "
                f"pred_std={metrics['pred_std']:.3e}, "
                f"pred_input_delta={metrics['pred_input_delta']:.3e}"
            )


if __name__ == "__main__":
    main()
