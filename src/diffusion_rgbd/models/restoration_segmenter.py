from typing import Any

import torch
import torch.nn as nn

from diffusion_rgbd.models.fusion_segmenter import TransformerFusionSegmenter


class LatentRestorationRefiner(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or channels * 2
        self.modality_embedding = nn.Linear(2, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        mask_bias = self.modality_embedding(modality_mask.float()).to(dtype=latent.dtype)
        return self.layers(latent + mask_bias[:, :, None, None])


class LatentRestorationSegmenter(nn.Module):
    expects_dict_inputs = True

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        latent_channels: int = 128,
        num_heads: int = 4,
        transformer_scales: tuple[int, ...] = (3,),
        dropout: float = 0.0,
        fusion_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.fusion = fusion_model or TransformerFusionSegmenter(
            num_classes=num_classes,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_heads=num_heads,
            transformer_scales=transformer_scales,
            dropout=dropout,
        )
        actual_latent_channels = int(getattr(self.fusion, "latent_channels", latent_channels))
        self.restoration_refiner = LatentRestorationRefiner(actual_latent_channels)

    def encode(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return self.fusion.encode(inputs)

    def decode(self, latent: torch.Tensor, fused_features: list[torch.Tensor]) -> torch.Tensor:
        if hasattr(self.fusion, "decode"):
            return self.fusion.decode(latent, fused_features[:-1])
        return self.fusion.decoder(latent, fused_features[:-1])

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_dict: bool = False,
        teacher_latent: torch.Tensor | None = None,
        use_restoration: bool = True,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        fused_features = self.encode(inputs)
        latent = fused_features[-1]

        if not use_restoration:
            logits = self.decode(latent, fused_features)
            if return_dict:
                return {
                    "logits": logits,
                    "latent": latent,
                    "restored_latent": latent,
                    "fused_features": fused_features,
                }
            return logits

        residual = self.restoration_refiner(latent, inputs["modality_mask"])
        restored_latent = latent + residual
        logits = self.decode(restored_latent, fused_features)

        if return_dict:
            target_residual = teacher_latent - latent if teacher_latent is not None else torch.zeros_like(latent)
            return {
                "logits": logits,
                "latent": latent,
                "restored_latent": restored_latent,
                "residual": residual,
                "target_residual": target_residual,
                "fused_features": fused_features,
            }
        return logits
