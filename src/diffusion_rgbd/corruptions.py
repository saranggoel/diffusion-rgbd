from __future__ import annotations

import torch
import torch.nn.functional as F


VALID_CONDITIONS = {
    "clean_rgbd",
    "rgb_only",
    "depth_only",
    "rgb_corrupt",
    "depth_corrupt",
    "both_corrupt",
}


def apply_condition(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    condition: str,
    severity: int = 1,
    rgb_corruption: str = "mixed",
    depth_corruption: str = "mixed",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply missing/corrupted-modality condition to a batch."""
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Expected one of {sorted(VALID_CONDITIONS)}")

    rgb_out = rgb.clone()
    depth_out = depth.clone()

    if condition == "rgb_only":
        depth_out.zero_()
    elif condition == "depth_only":
        rgb_out.zero_()
    elif condition == "rgb_corrupt":
        rgb_out = corrupt_rgb(rgb_out, rgb_corruption, severity, generator)
    elif condition == "depth_corrupt":
        depth_out = corrupt_depth(depth_out, depth_corruption, severity, generator)
    elif condition == "both_corrupt":
        rgb_out = corrupt_rgb(rgb_out, rgb_corruption, severity, generator)
        depth_out = corrupt_depth(depth_out, depth_corruption, severity, generator)

    return rgb_out, depth_out


def corrupt_rgb(
    rgb: torch.Tensor,
    kind: str = "mixed",
    severity: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    severity = max(1, int(severity))
    out = rgb
    if kind in {"mixed", "low_light"}:
        factor = max(0.15, 1.0 - 0.25 * severity)
        out = out * factor
    if kind in {"mixed", "noise"}:
        std = 0.04 * severity
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        out = out + noise * std
    if kind in {"mixed", "blur"}:
        kernel_size = 2 * severity + 1
        out = _avg_blur(out, kernel_size)
    if kind in {"mixed", "occlusion"}:
        out = _random_patches(out, value=0.0, severity=severity, generator=generator)
    if kind not in {"mixed", "low_light", "noise", "blur", "occlusion"}:
        raise ValueError(f"Unknown RGB corruption: {kind}")
    return out.clamp(0.0, 1.0)


def corrupt_depth(
    depth: torch.Tensor,
    kind: str = "mixed",
    severity: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    severity = max(1, int(severity))
    out = depth
    if kind in {"mixed", "noise"}:
        std = 0.03 * severity
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        out = out + noise * std
    if kind in {"mixed", "holes"}:
        out = _random_patches(out, value=0.0, severity=severity + 1, generator=generator)
    if kind in {"mixed", "invalid_mask"}:
        drop_prob = min(0.08 * severity, 0.5)
        random_values = torch.rand(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        mask = random_values > drop_prob
        out = out * mask
    if kind not in {"mixed", "noise", "holes", "invalid_mask"}:
        raise ValueError(f"Unknown depth corruption: {kind}")
    return out.clamp(0.0, 1.0)


def _avg_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    channels = x.shape[1]
    weight = torch.ones(channels, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    weight = weight / float(kernel_size * kernel_size)
    padding = kernel_size // 2
    return F.conv2d(x, weight, padding=padding, groups=channels)


def _random_patches(
    x: torch.Tensor,
    value: float,
    severity: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    out = x.clone()
    batch, _channels, height, width = out.shape
    patch_count = max(1, severity)
    max_patch_h = max(4, height // max(6 - min(severity, 4), 2))
    max_patch_w = max(4, width // max(6 - min(severity, 4), 2))

    for b_idx in range(batch):
        for _ in range(patch_count):
            patch_h = _randint(max(2, max_patch_h // 3), max_patch_h + 1, x.device, generator)
            patch_w = _randint(max(2, max_patch_w // 3), max_patch_w + 1, x.device, generator)
            top = _randint(0, max(1, height - patch_h + 1), x.device, generator)
            left = _randint(0, max(1, width - patch_w + 1), x.device, generator)
            out[b_idx, :, top : top + patch_h, left : left + patch_w] = value
    return out


def _randint(
    low: int,
    high: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> int:
    return int(torch.randint(low, high, (1,), device=device, generator=generator).item())
