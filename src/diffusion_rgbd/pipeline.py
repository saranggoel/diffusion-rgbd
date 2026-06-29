from typing import Any

import torch

from diffusion_rgbd.corruptions import apply_condition, condition_to_modality_mask


DICT_INPUT_MODELS = {
    "transformer_fusion",
    "segformer_fusion",
    "segformer_latent_restoration",
}
RESTORATION_MODELS = {"segformer_latent_restoration"}


def prepare_inputs(
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    condition: str = "clean_rgbd",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    model_name = cfg.get("model", {}).get("name", "segformer_fusion")
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

    modality_mask = condition_to_modality_mask(
        condition=condition,
        batch_size=rgb.shape[0],
        device=rgb.device,
        dtype=rgb.dtype,
    )
    rgb = normalize_rgb(rgb, data_cfg.get("rgb_mean"), data_cfg.get("rgb_std"))

    if model_name in DICT_INPUT_MODELS:
        inputs = {
            "rgb": rgb,
            "depth": depth,
            "modality_mask": modality_mask,
        }
        return inputs, labels

    if modality == "rgb":
        inputs = rgb
    elif modality == "depth":
        inputs = depth
    else:
        inputs = torch.cat([rgb, depth], dim=1)
    return inputs, labels


def extract_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def forward_for_condition(
    model: torch.nn.Module,
    inputs: torch.Tensor | dict[str, torch.Tensor],
    cfg: dict[str, Any],
    condition: str,
) -> torch.Tensor:
    model_name = cfg.get("model", {}).get("name", "segformer_fusion")
    if model_name in RESTORATION_MODELS and condition == "clean_rgbd":
        return extract_logits(model(inputs, use_restoration=False))
    return extract_logits(model(inputs))


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
