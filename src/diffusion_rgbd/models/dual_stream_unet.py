import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, F.max_pool2d(skip, 2)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DualStreamUNet(nn.Module):
    expects_dict_inputs = True

    def __init__(self, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        self.rgb1 = DownBlock(3, base_channels)
        self.rgb2 = DownBlock(base_channels, base_channels * 2)
        self.rgb3 = DownBlock(base_channels * 2, base_channels * 4)

        self.depth1 = DownBlock(1, base_channels)
        self.depth2 = DownBlock(base_channels, base_channels * 2)
        self.depth3 = DownBlock(base_channels * 2, base_channels * 4)

        self.fuse1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.fuse2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.fuse3 = nn.Conv2d(base_channels * 8, base_channels * 4, 1)

        self.bridge = ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        rgb = inputs["rgb"]
        depth = inputs["depth"]
        mask = inputs.get("modality_mask")

        if mask is not None:
            rgb = rgb * mask[:, 0:1, None, None]
            depth = depth * mask[:, 1:2, None, None]

        r1, rp1 = self.rgb1(rgb)
        r2, rp2 = self.rgb2(rp1)
        r3, rp3 = self.rgb3(rp2)

        d1, dp1 = self.depth1(depth)
        d2, dp2 = self.depth2(dp1)
        d3, dp3 = self.depth3(dp2)

        s1 = self.fuse1(torch.cat([r1, d1], dim=1))
        s2 = self.fuse2(torch.cat([r2, d2], dim=1))
        s3 = self.fuse3(torch.cat([r3, d3], dim=1))

        x = self.bridge((rp3 + dp3) * 0.5)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)
