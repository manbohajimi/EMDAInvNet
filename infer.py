from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from emdainvnet.checkpoint import load_model_weights
from emdainvnet.config import load_config
from emdainvnet.data import (
    denormalize_target_array,
    load_volume,
    normalize,
    resize_volume,
    suppress_direct_wave,
)
from emdainvnet.model import EMDAInvNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EMDAInvNet on one 3-D volume")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-key")
    parser.add_argument("--output", required=True, help="output .npz file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--suppress-direct-wave", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    volume = load_volume(args.input, args.input_key or data_config.get("input_key"))
    if args.suppress_direct_wave:
        volume = suppress_direct_wave(volume)
    volume = resize_volume(volume, tuple(data_config["input_shape"]))
    volume = normalize(volume, data_config["input_min"], data_config["input_max"])
    device = torch.device(args.device)
    model = EMDAInvNet(**config["model"]).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()
    tensor = torch.from_numpy(volume)[None, None].to(device)
    with torch.no_grad():
        normalized = model(tensor)[0, 0].cpu().numpy()
    permittivity = denormalize_target_array(normalized, data_config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        normalized_permittivity=normalized.astype(np.float32),
        permittivity=permittivity.astype(np.float32),
    )
    print(f"saved {output.resolve()} shape={normalized.shape}")


if __name__ == "__main__":
    main()
