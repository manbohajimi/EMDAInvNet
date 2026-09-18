from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    metrics: dict[str, float] | None = None,
) -> None:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_model_weights(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}

