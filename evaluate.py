from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from emdainvnet.checkpoint import load_model_weights
from emdainvnet.config import load_config, resolve_project_path
from emdainvnet.data import dataset_from_config, normalized_target_threshold, target_affine
from emdainvnet.metrics import MetricAccumulator, volume_metrics
from emdainvnet.model import EMDAInvNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EMDAInvNet")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    manifest = resolve_project_path(
        config, args.manifest or data_config["test_manifest"]
    )
    dataset = dataset_from_config(manifest, data_config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = EMDAInvNet(**config["model"]).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()
    accumulator = MetricAccumulator()
    per_case = []
    with torch.no_grad():
        for radar, target, sample_ids in loader:
            prediction = model(radar.to(device))
            target_offset, target_scale = target_affine(data_config)
            case = volume_metrics(
                prediction,
                target.to(device),
                threshold=normalized_target_threshold(data_config),
                mape_min=target_offset,
                mape_max=target_offset + target_scale,
            )
            accumulator.update(case)
            per_case.append({"id": sample_ids[0], **{k: float(v.item()) for k, v in case.items()}})
    result = {"mean": accumulator.compute(), "cases": per_case}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
