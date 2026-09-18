from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emdainvnet.metrics import volume_metrics
from emdainvnet.model import DilatedBlock, EMDAInvNet


def test_shape_and_range() -> None:
    model = EMDAInvNet(base_channels=2).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32, 32))
    assert output.shape == (1, 1, 32, 32, 32)
    assert output.min() >= 0
    assert output.max() <= 1


def test_perfect_metrics() -> None:
    target = torch.zeros(2, 1, 8, 8, 8)
    target[:, :, 2:6, 2:6, 2:6] = 1
    metrics = volume_metrics(target, target)
    assert torch.allclose(metrics["mae"], torch.zeros(2))
    assert torch.allclose(metrics["mse"], torch.zeros(2))
    assert torch.allclose(metrics["ssim"], torch.ones(2))
    assert torch.allclose(metrics["iou"], torch.ones(2))
    assert torch.allclose(metrics["dice"], torch.ones(2))


def test_paper_channel_contracts() -> None:
    model = EMDAInvNet(base_channels=2)
    # Fig. 2: each EMSFA convolution uses C channels; the transition expands
    # the compact feature to 3C; every Eq. (10) atrous branch preserves width.
    assert model.down1.enhance.local.conv1[0].out_channels == 2
    assert isinstance(model.down1.enhance.transition, torch.nn.Sequential)
    assert model.down1.enhance.transition[0].out_channels == 6
    assert all(
        isinstance(branch, torch.nn.Sequential) and branch[0].out_channels == 6
        for branch in model.down1.enhance.context.branches
    )
    assert not any(module.affine for module in model.modules() if isinstance(module, torch.nn.InstanceNorm3d))
    assert isinstance(model.bridge[1], DilatedBlock)


def _identity_dilated_block(residual_order: str) -> DilatedBlock:
    block = DilatedBlock(1, rates=(1,), residual_order=residual_order)
    block.branches = torch.nn.ModuleList([torch.nn.Identity()])
    block.fusion = torch.nn.Identity()
    block.norm = torch.nn.Identity()
    return block


def test_default_dilated_residual_activates_after_addition() -> None:
    block = _identity_dilated_block("add_then_activate")
    value = -torch.ones(1, 1, 2, 2, 2)
    expected = torch.nn.functional.leaky_relu(
        value + value, negative_slope=0.01, inplace=False
    )
    assert torch.allclose(block(value), expected)


def test_literal_equation_variant_adds_residual_after_activation() -> None:
    block = _identity_dilated_block("activate_then_add")
    value = -torch.ones(1, 1, 2, 2, 2)
    expected = torch.nn.functional.leaky_relu(
        value, negative_slope=0.01, inplace=False
    ) + value
    assert torch.allclose(block(value), expected)
