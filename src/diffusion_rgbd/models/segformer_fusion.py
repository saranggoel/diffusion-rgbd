from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SegFormerFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.rgb_proj = ConvGNAct(in_channels, out_channels, kernel_size=1)
        self.depth_proj = ConvGNAct(in_channels, out_channels, kernel_size=1)
        self.rgb_gate = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=1), nn.Sigmoid())
        self.depth_gate = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            ConvGNAct(out_channels * 2, out_channels, kernel_size=1, dropout=dropout),
            ConvGNAct(out_channels, out_channels, kernel_size=3, dropout=dropout),
        )
        self.mask_embed = nn.Linear(2, out_channels)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        depth_feature: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        rgb_feature = self.rgb_proj(rgb_feature)
        depth_feature = self.depth_proj(depth_feature)

        rgb_weight, depth_weight = modality_mask[:, 0:1], modality_mask[:, 1:2]
        rgb_feature = rgb_feature * rgb_weight[:, :, None, None]
        depth_feature = depth_feature * depth_weight[:, :, None, None]

        paired = torch.cat([rgb_feature, depth_feature], dim=1)
        rgb_rectified = rgb_feature + self.rgb_gate(paired) * depth_feature
        depth_rectified = depth_feature + self.depth_gate(paired) * rgb_feature
        fused = self.fuse(torch.cat([rgb_rectified, depth_rectified], dim=1))

        mask_bias = self.mask_embed(modality_mask.float()).to(dtype=fused.dtype)
        return fused + mask_bias[:, :, None, None]


class SegFormerFusionDecoder(nn.Module):
    def __init__(self, channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvGNAct(channels * 4, channels, kernel_size=1, dropout=dropout),
            ConvGNAct(channels, channels, kernel_size=3, dropout=dropout),
        )
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(
        self,
        latent: torch.Tensor,
        skips: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        features = [*skips, latent]
        target_size = features[0].shape[-2:]
        resized = [
            feature
            if feature.shape[-2:] == target_size
            else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            for feature in features
        ]
        logits = self.head(self.fuse(torch.cat(resized, dim=1)))
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


class SegFormerFusionSegmenter(nn.Module):
    expects_dict_inputs = True

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "nvidia/mit-b2",
        pretrained: bool = True,
        latent_channels: int = 256,
        dropout: float = 0.0,
        depth_mean: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
        depth_std: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        segformer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.rgb_encoder, hidden_sizes = build_segformer_encoder(backbone_name, pretrained, segformer_config)
        self.depth_encoder, depth_hidden_sizes = build_segformer_encoder(backbone_name, pretrained, segformer_config)

        self.fusion_blocks = nn.ModuleList(
            [
                SegFormerFusionBlock(
                    in_channels=int(hidden_size),
                    out_channels=latent_channels,
                    dropout=dropout,
                )
                for hidden_size in hidden_sizes
            ]
        )
        self.decoder = SegFormerFusionDecoder(latent_channels, num_classes, dropout=dropout)
        self._last_output_size: tuple[int, int] | None = None

        self.register_buffer("depth_mean", torch.tensor(depth_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("depth_std", torch.tensor(depth_std, dtype=torch.float32).view(1, 3, 1, 1))

        if freeze_backbone:
            for encoder in (self.rgb_encoder, self.depth_encoder):
                for param in encoder.parameters():
                    param.requires_grad_(False)

        if gradient_checkpointing:
            for encoder in (self.rgb_encoder, self.depth_encoder):
                if hasattr(encoder, "gradient_checkpointing_enable"):
                    encoder.gradient_checkpointing_enable()

    @property
    def latent_channels(self) -> int:
        return self.fusion_blocks[-1].mask_embed.out_features

    def encode(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        self._last_output_size = tuple(inputs["rgb"].shape[-2:])
        modality_mask = inputs["modality_mask"].float()
        rgb_features = self.encoder_features(self.rgb_encoder, inputs["rgb"])
        depth = inputs["depth"].repeat(1, 3, 1, 1)
        depth = (depth - self.depth_mean.to(depth.dtype)) / self.depth_std.to(depth.dtype).clamp_min(1e-6)
        depth_features = self.encoder_features(self.depth_encoder, depth)
        return [
            block(rgb_feature, depth_feature, modality_mask)
            for block, rgb_feature, depth_feature in zip(self.fusion_blocks, rgb_features, depth_features)
        ]

    def decode(self, latent: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        return self.decoder(latent, skips, output_size=self._last_output_size)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_dict: bool = False,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        fused_features = self.encode(inputs)
        latent = fused_features[-1]
        logits = self.decode(latent, fused_features[:-1])
        if return_dict:
            return {"logits": logits, "latent": latent, "fused_features": fused_features}
        return logits

    @staticmethod
    def encoder_features(encoder: nn.Module, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        output = encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = list(output.hidden_states or [])
        return hidden_states[-4:]


def build_segformer_encoder(
    backbone_name: str,
    pretrained: bool,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[nn.Module, list[int]]:
    from transformers import SegformerConfig, SegformerModel

    if pretrained:
        config = SegformerConfig.from_pretrained(backbone_name)
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        config.output_hidden_states = True
        encoder = SegformerModel.from_pretrained(backbone_name, config=config)
    else:
        config_kwargs = dict(config_overrides or {})
        config = SegformerConfig(**config_kwargs)
        config.output_hidden_states = True
        encoder = SegformerModel(config)
    return encoder, [int(size) for size in config.hidden_sizes]
