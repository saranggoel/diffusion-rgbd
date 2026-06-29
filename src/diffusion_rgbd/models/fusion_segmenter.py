from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.stage1 = EncoderStage(in_channels, base_channels)
        self.stage2 = EncoderStage(base_channels, base_channels * 2)
        self.stage3 = EncoderStage(base_channels * 2, base_channels * 4)
        self.stage4 = EncoderStage(base_channels * 4, latent_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s1 = self.stage1(x)
        s2 = self.stage2(F.max_pool2d(s1, kernel_size=2))
        s3 = self.stage3(F.max_pool2d(s2, kernel_size=2))
        s4 = self.stage4(F.max_pool2d(s3, kernel_size=2))
        return [s1, s2, s3, s4]


class FusionBlock(nn.Module):
    def __init__(self, channels: int, use_transformer: bool, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.rgb_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid())
        self.depth_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            ConvBNAct(channels, channels),
        )
        self.mask_embed = nn.Linear(2, channels)
        self.use_transformer = use_transformer
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=channels * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.transformer = None

    def forward(
        self,
        rgb_feature: torch.Tensor,
        depth_feature: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        rgb_weight, depth_weight = modality_mask[:, 0:1], modality_mask[:, 1:2]
        rgb_feature = rgb_feature * rgb_weight[:, :, None, None]
        depth_feature = depth_feature * depth_weight[:, :, None, None]

        paired = torch.cat([rgb_feature, depth_feature], dim=1)
        rgb_rectified = rgb_feature + self.rgb_gate(paired) * depth_feature
        depth_rectified = depth_feature + self.depth_gate(paired) * rgb_feature
        fused = self.fuse(torch.cat([rgb_rectified, depth_rectified], dim=1))

        mask_bias = self.mask_embed(modality_mask).to(dtype=fused.dtype)
        fused = fused + mask_bias[:, :, None, None]

        if self.transformer is None:
            return fused

        batch, channels, height, width = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SegmentationDecoder(nn.Module):
    def __init__(self, base_channels: int, latent_channels: int, num_classes: int) -> None:
        super().__init__()
        self.up3 = EncoderStage(latent_channels + base_channels * 4, base_channels * 4)
        self.up2 = EncoderStage(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = EncoderStage(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, latent: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = skips[0], skips[1], skips[2]
        x = F.interpolate(latent, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, s3], dim=1))
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, s2], dim=1))
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, s1], dim=1))
        return self.head(x)


class TransformerFusionSegmenter(nn.Module):
    expects_dict_inputs = True

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        latent_channels: int = 128,
        num_heads: int = 4,
        transformer_scales: tuple[int, ...] = (3,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rgb_encoder = ModalityEncoder(3, base_channels, latent_channels)
        self.depth_encoder = ModalityEncoder(1, base_channels, latent_channels)
        channels = [base_channels, base_channels * 2, base_channels * 4, latent_channels]
        self.fusion_blocks = nn.ModuleList(
            [
                FusionBlock(
                    channels=channel,
                    use_transformer=scale_idx in transformer_scales,
                    num_heads=max(1, min(num_heads, channel)),
                    dropout=dropout,
                )
                for scale_idx, channel in enumerate(channels)
            ]
        )
        self.decoder = SegmentationDecoder(base_channels, latent_channels, num_classes)

    @property
    def latent_channels(self) -> int:
        return self.fusion_blocks[-1].mask_embed.out_features

    def encode(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        modality_mask = inputs["modality_mask"].float()
        rgb_features = self.rgb_encoder(inputs["rgb"])
        depth_features = self.depth_encoder(inputs["depth"])
        return [
            block(rgb_feature, depth_feature, modality_mask)
            for block, rgb_feature, depth_feature in zip(self.fusion_blocks, rgb_features, depth_features)
        ]

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_dict: bool = False,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        fused_features = self.encode(inputs)
        latent = fused_features[-1]
        logits = self.decoder(latent, fused_features[:-1])
        if return_dict:
            return {"logits": logits, "latent": latent, "fused_features": fused_features}
        return logits
