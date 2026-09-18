"""Paper-faithful 3-D EMDAInvNet building blocks.

The public paper specifies the topology, equations (3)-(11), channel ratios,
InstanceNorm, LeakyReLU and dilation rates, but not the base width, attention
reduction ratio or spatial-attention kernel. Those are constructor arguments.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvINLReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            # The paper states InstanceNorm3d but does not state learnable
            # affine parameters. Keep PyTorch's standard affine=False.
            nn.InstanceNorm3d(out_channels, affine=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


class ChannelAttention3d(nn.Module):
    """Equation (6): shared MLP over global average and max descriptors."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg = F.adaptive_avg_pool3d(x, 1)
        maximum = F.adaptive_max_pool3d(x, 1)
        return torch.sigmoid(self.mlp(avg) + self.mlp(maximum))


class SpatialAttention3d(nn.Module):
    """Equation (7): channel mean/max followed by a 3-D convolution."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("spatial attention kernel size must be odd")
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        descriptor = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1
        )
        return torch.sigmoid(self.conv(descriptor))


class ChannelSpatialAttention3d(nn.Module):
    """Sequential channel and spatial attention used by the Fig. 2 bridge."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention3d(channels, reduction)
        self.spatial = SpatialAttention3d(spatial_kernel)

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)


class EMSFA(nn.Module):
    """Enhanced multiscale fusion attention, equations (4)-(8).

    Three serial 3x3x3 convolutions expose progressively larger local fields.
    Their outputs are concatenated, sequentially reweighted by channel and
    spatial attention, then fused by the 1x1x1 convolution shown in Fig. 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.conv1 = ConvINLReLU(in_channels, out_channels)
        self.conv2 = ConvINLReLU(out_channels, out_channels)
        self.conv3 = ConvINLReLU(out_channels, out_channels)
        aggregated_channels = 3 * out_channels
        self.channel_attention = ChannelAttention3d(aggregated_channels, reduction)
        self.spatial_attention = SpatialAttention3d(spatial_kernel)
        self.fusion = nn.Conv3d(aggregated_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        aggregated = torch.cat((c1, c2, c3), dim=1)
        weighted = aggregated * self.channel_attention(aggregated)
        weighted = weighted * self.spatial_attention(weighted)
        return self.fusion(weighted)


class DilatedBlock(nn.Module):
    """Parallel atrous context aggregation with residual fusion, (10)-(11)."""

    def __init__(self, channels: int, rates: Sequence[int] = (1, 2, 4, 8)) -> None:
        super().__init__()
        if not rates:
            raise ValueError("at least one dilation rate is required")
        self.branches = nn.ModuleList(
            ConvINLReLU(channels, channels, dilation=int(rate))
            for rate in rates
        )
        merged_channels = channels * len(rates)
        self.fusion = nn.Conv3d(merged_channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=False)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(self.norm(self.fusion(merged)) + x)


class FeatureEnhancementUnit(nn.Module):
    """EMSFA -> transition convolution -> DilatedBlock, as in Fig. 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_emsfa: bool = True,
        use_dilated: bool = True,
        reduction: int = 16,
        spatial_kernel: int = 7,
        rates: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        # Fig. 2 labels the transition as a 1x1 convolution expanding the
        # compact EMSFA representation to three times as many channels.
        if out_channels % 3:
            raise ValueError("feature-enhancement output channels must be divisible by 3")
        compact_channels = out_channels // 3
        self.local = (
            EMSFA(
                in_channels,
                compact_channels,
                reduction=reduction,
                spatial_kernel=spatial_kernel,
            )
            if use_emsfa
            else ConvINLReLU(in_channels, compact_channels)
        )
        self.transition = ConvINLReLU(compact_channels, out_channels, kernel_size=1)
        self.context = DilatedBlock(out_channels, rates) if use_dilated else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.context(self.transition(self.local(x)))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: object) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(2, stride=2)
        self.enhance = FeatureEnhancementUnit(in_channels, out_channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.enhance(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, **kwargs: object
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.enhance = FeatureEnhancementUnit(
            out_channels + skip_channels, out_channels, **kwargs
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.enhance(torch.cat((x, skip), dim=1))


class EMDAInvNet(nn.Module):
    """3-D encoder-decoder shown in Fig. 2 of Kong et al. (JSEN 2026).

    Channel ratios follow the figure: C -> 3C -> 6C -> 12C -> 24C -> 32C,
    then 12C -> 6C -> 3C -> 3C/2. The base width C is not reported.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        *,
        use_emsfa: bool = True,
        use_dilated: bool = True,
        attention_reduction: int = 16,
        spatial_kernel: int = 7,
        dilation_rates: Sequence[int] = (1, 2, 4, 8),
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if base_channels < 2 or base_channels % 2:
            raise ValueError("base_channels must be an even integer >= 2")
        c = base_channels
        common = dict(
            use_emsfa=use_emsfa,
            use_dilated=use_dilated,
            reduction=attention_reduction,
            spatial_kernel=spatial_kernel,
            rates=dilation_rates,
        )
        self.stem = ConvINLReLU(in_channels, c)
        self.down1 = DownBlock(c, 3 * c, **common)
        self.down2 = DownBlock(3 * c, 6 * c, **common)
        self.down3 = DownBlock(6 * c, 12 * c, **common)
        self.down4 = DownBlock(12 * c, 24 * c, **common)
        self.bridge = nn.Sequential(
            ConvINLReLU(24 * c, 32 * c),
            DilatedBlock(32 * c, dilation_rates),
            ChannelSpatialAttention3d(32 * c, attention_reduction, spatial_kernel),
        )
        self.up4 = UpBlock(32 * c, 12 * c, 12 * c, **common)
        self.up3 = UpBlock(12 * c, 6 * c, 6 * c, **common)
        self.up2 = UpBlock(6 * c, 3 * c, 3 * c, **common)
        self.up1 = UpBlock(3 * c, c, 3 * c // 2, **common)
        self.head = nn.Conv3d(3 * c // 2, out_channels, kernel_size=1)
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")
        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        x = self.bridge(s4)
        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = self.head(x)
        return torch.sigmoid(x) if self.output_activation == "sigmoid" else x

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
