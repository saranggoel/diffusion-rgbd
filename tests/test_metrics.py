import torch

from diffusion_rgbd.metrics import SegmentationMetrics


def test_segmentation_metrics_perfect_prediction() -> None:
    metrics = SegmentationMetrics(num_classes=3, ignore_index=255)
    target = torch.tensor([[[0, 1], [2, 255]]])
    logits = torch.full((1, 3, 2, 2), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 1, 0, 1] = 10.0
    logits[0, 2, 1, 0] = 10.0
    metrics.update(logits, target)
    computed = metrics.compute()
    assert computed["miou"] == 1.0
    assert computed["pixel_accuracy"] == 1.0
    assert computed["mean_class_accuracy"] == 1.0


def test_segmentation_metrics_counts_false_positive() -> None:
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    target = torch.tensor([[[0, 1], [1, 1]]])
    logits = torch.tensor(
        [
            [
                [[3.0, 3.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0]],
            ]
        ]
    )
    metrics.update(logits, target)
    computed = metrics.compute()
    assert round(computed["pixel_accuracy"], 4) == 0.75
    assert round(computed["miou"], 4) == 0.5833

