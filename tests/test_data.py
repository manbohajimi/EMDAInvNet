from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emdainvnet.data import (
    denormalize_target_array,
    normalize_target_array,
    normalized_target_threshold,
    target_affine,
)


def test_paper_minmax_uses_dataset_ii_physical_range() -> None:
    config = {
        "target_normalization": "minmax",
        "target_min": 4.0,
        "target_max": 27.0,
        "target_threshold_phys": 6.0,
    }
    offset, scale = target_affine(config)
    physical = np.array([4.0, 8.0, 27.0], dtype=np.float32)
    normalized = normalize_target_array(physical, "minmax", offset, scale)
    assert np.allclose(normalized, (physical - 4.0) / 23.0)
    assert normalized[0] == 0.0
    assert np.allclose(denormalize_target_array(normalized, config), physical)
    assert np.isclose(normalized_target_threshold(config), 2.0 / 23.0)
