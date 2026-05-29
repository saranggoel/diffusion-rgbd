from __future__ import annotations

from typing import Any

import torch

from diffusion_rgbd.corruptions import apply_condition


def prepare_inputs(
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    condition: str = "clean_rgbd",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    modality = cfg.get("model", {}).get("modality", "rgbd")

    rgb = batch["rgb"].float()
    depth = batch["depth"].float()
    labels = batch["label"].long()

    rgb, depth = apply_condition(
        rgb,
        depth,
        condition=condition,
        severity=int(eval_cfg.get("severity", 1)),
        rgb_corruption=eval_cfg.get("rgb_corruption", "mixed"),
        depth_corruption=eval_cfg.get("depth_corruption", "mixed"),
        generator=generator,
    )

    rgb = normalize_rgb(rgb, data_cfg.get("rgb_mean"), data_cfg.get("rgb_std"))

    if modality == "rgb":
        inputs = rgb
    elif modality == "depth":
        inputs = depth
    elif modality == "rgbd":
        inputs = torch.cat([rgb, depth], dim=1)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return inputs, labels


def normalize_rgb(
    rgb: torch.Tensor,
    mean: list[float] | tuple[float, ...] | None,
    std: list[float] | tuple[float, ...] | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return rgb
    mean_tensor = torch.tensor(mean, device=rgb.device, dtype=rgb.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=rgb.device, dtype=rgb.dtype).view(1, -1, 1, 1)
    return (rgb - mean_tensor) / std_tensor.clamp_min(1e-6)

