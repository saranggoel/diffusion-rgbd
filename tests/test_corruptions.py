import torch

from diffusion_rgbd.corruptions import apply_condition


def test_rgb_only_zeros_depth() -> None:
    rgb = torch.ones(2, 3, 8, 8)
    depth = torch.ones(2, 1, 8, 8)
    out_rgb, out_depth = apply_condition(rgb, depth, condition="rgb_only")
    assert torch.allclose(out_rgb, rgb)
    assert out_depth.sum().item() == 0.0


def test_depth_only_zeros_rgb() -> None:
    rgb = torch.ones(2, 3, 8, 8)
    depth = torch.ones(2, 1, 8, 8)
    out_rgb, out_depth = apply_condition(rgb, depth, condition="depth_only")
    assert out_rgb.sum().item() == 0.0
    assert torch.allclose(out_depth, depth)


def test_corruptions_preserve_shape_and_range() -> None:
    rgb = torch.ones(2, 3, 16, 16) * 0.5
    depth = torch.ones(2, 1, 16, 16) * 0.5
    out_rgb, out_depth = apply_condition(rgb, depth, condition="both_corrupt", severity=2)
    assert out_rgb.shape == rgb.shape
    assert out_depth.shape == depth.shape
    assert 0.0 <= out_rgb.min().item() <= out_rgb.max().item() <= 1.0
    assert 0.0 <= out_depth.min().item() <= out_depth.max().item() <= 1.0

