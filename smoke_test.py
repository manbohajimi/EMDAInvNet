from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from emdainvnet.metrics import volume_metrics
from emdainvnet.model import EMDAInvNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMDAInvNet(base_channels=2).to(device)
    radar = torch.randn(1, 1, 32, 32, 32, device=device)
    target = torch.rand(1, 1, 32, 32, 32, device=device)
    prediction = model(radar)
    assert prediction.shape == target.shape
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    metrics = volume_metrics(prediction.detach(), target, threshold=0.1)
    assert all(torch.isfinite(values).all() for values in metrics.values())
    print(
        f"OK device={device} shape={tuple(prediction.shape)} "
        f"parameters={model.num_parameters():,} loss={loss.item():.6f}"
    )


if __name__ == "__main__":
    main()

