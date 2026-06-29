from typing import Any

import torch.nn as nn

from diffusion_rgbd.models.fusion_segmenter import TransformerFusionSegmenter
from diffusion_rgbd.models.restoration_segmenter import LatentRestorationSegmenter
from diffusion_rgbd.models.segformer_fusion import SegFormerFusionSegmenter


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    name = model_cfg.get("name", "segformer_fusion")

    if name == "transformer_fusion":
        return TransformerFusionSegmenter(
            num_classes=int(data_cfg["num_classes"]),
            base_channels=int(model_cfg.get("base_channels", 32)),
            latent_channels=int(model_cfg.get("latent_channels", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            transformer_scales=tuple(model_cfg.get("transformer_scales", [3])),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )

    if name == "segformer_fusion":
        return build_segformer_fusion_model(cfg)

    if name == "segformer_latent_restoration":
        fusion_model = build_segformer_fusion_model(cfg)
        return LatentRestorationSegmenter(
            num_classes=int(data_cfg["num_classes"]),
            latent_channels=int(model_cfg.get("latent_channels", 256)),
            fusion_model=fusion_model,
        )

    return build_segformer_fusion_model(cfg)


def build_segformer_fusion_model(cfg: dict[str, Any]) -> SegFormerFusionSegmenter:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    return SegFormerFusionSegmenter(
        num_classes=int(data_cfg["num_classes"]),
        backbone_name=str(model_cfg.get("backbone_name", "nvidia/mit-b2")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        latent_channels=int(model_cfg.get("latent_channels", 256)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        depth_mean=model_cfg.get("depth_mean", [0.5, 0.5, 0.5]),
        depth_std=model_cfg.get("depth_std", [0.5, 0.5, 0.5]),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        segformer_config=model_cfg.get("segformer_config"),
    )
