import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255, device: str | torch.device = "cpu") -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)

    def reset(self) -> None:
        self.confusion.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        target = target.to(pred.device)
        valid = target != self.ignore_index
        valid &= target >= 0
        valid &= target < self.num_classes
        if not valid.any():
            return
        target_flat = target[valid].long()
        pred_flat = pred[valid].long().clamp(0, self.num_classes - 1)
        indices = target_flat * self.num_classes + pred_flat
        counts = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).to(self.confusion.device)

    def compute(self) -> dict[str, float]:
        confusion = self.confusion.float()
        true_positive = confusion.diag()
        gt_count = confusion.sum(dim=1)
        pred_count = confusion.sum(dim=0)
        union = gt_count + pred_count - true_positive

        valid_iou = union > 0
        iou = torch.zeros_like(true_positive)
        iou[valid_iou] = true_positive[valid_iou] / union[valid_iou].clamp_min(1.0)

        valid_acc = gt_count > 0
        class_acc = torch.zeros_like(true_positive)
        class_acc[valid_acc] = true_positive[valid_acc] / gt_count[valid_acc].clamp_min(1.0)

        total = confusion.sum().clamp_min(1.0)
        pixel_accuracy = true_positive.sum() / total

        return {
            "miou": float(iou[valid_iou].mean().item()) if valid_iou.any() else 0.0,
            "pixel_accuracy": float(pixel_accuracy.item()),
            "mean_class_accuracy": float(class_acc[valid_acc].mean().item()) if valid_acc.any() else 0.0,
        }

    def per_class_iou(self) -> list[float | None]:
        confusion = self.confusion.float()
        tp = confusion.diag()
        union = confusion.sum(dim=1) + confusion.sum(dim=0) - tp
        iou = torch.full((self.num_classes,), float("nan"), device=confusion.device)
        valid = union > 0
        iou[valid] = tp[valid] / union[valid]
        values: list[float | None] = []
        for value in iou.cpu():
            values.append(None if torch.isnan(value) else float(value))
        return values
