from __future__ import annotations

from typing import Any

import torch.nn as nn

from diffusion_rgbd.models.simple_unet import SimpleUNet


MODALITY_CHANNELS = {
    "rgb": 3,
    "depth": 1,
    "rgbd": 4,
}


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    name = model_cfg.get("name", "simple_unet")
    modality = model_cfg.get("modality", "rgbd")
    if modality not in MODALITY_CHANNELS:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of {sorted(MODALITY_CHANNELS)}")
    if name != "simple_unet":
        raise ValueError(f"Unknown model '{name}'. Only simple_unet is implemented in the scaffold.")
    return SimpleUNet(
        in_channels=MODALITY_CHANNELS[modality],
        num_classes=int(data_cfg["num_classes"]),
        base_channels=int(model_cfg.get("base_channels", 32)),
    )

