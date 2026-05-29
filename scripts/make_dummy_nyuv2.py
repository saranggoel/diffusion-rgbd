#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic RGB-D segmentation dataset for smoke tests.")
    parser.add_argument("--root", default="data/dummy_nyuv2")
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    for split, count in [("train", args.train_count), ("val", args.val_count)]:
        rows = []
        for index in range(count):
            rgb, depth, label = make_sample(args.height, args.width, args.classes, rng)
            rgb_path = root / split / "rgb" / f"{index:04d}.png"
            depth_path = root / split / "depth" / f"{index:04d}.png"
            label_path = root / split / "labels" / f"{index:04d}.png"
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(depth).save(depth_path)
            Image.fromarray(label).save(label_path)
            rows.append(
                {
                    "rgb": str(rgb_path.relative_to(root)),
                    "depth": str(depth_path.relative_to(root)),
                    "label": str(label_path.relative_to(root)),
                }
            )
        write_manifest(root / f"{split}.csv", rows)
    print(f"Wrote dummy dataset to {root}")


def make_sample(
    height: int,
    width: int,
    classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    label = ((xx / max(width, 1)) * classes).astype(np.uint8) % classes
    for class_id in range(1, classes):
        center_y = rng.integers(height // 8, max(height // 8 + 1, height - height // 8))
        center_x = rng.integers(width // 8, max(width // 8 + 1, width - width // 8))
        radius = rng.integers(max(4, min(height, width) // 10), max(5, min(height, width) // 4))
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 < radius**2
        label[mask] = class_id

    palette = np.array(
        [
            [35, 35, 40],
            [220, 80, 70],
            [70, 150, 230],
            [90, 190, 120],
            [230, 190, 70],
            [175, 100, 220],
            [80, 210, 210],
            [230, 120, 180],
        ],
        dtype=np.uint8,
    )
    rgb = palette[label % len(palette)]
    texture = rng.normal(0, 12, size=(height, width, 3))
    rgb = np.clip(rgb.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    depth_float = (yy.astype(np.float32) / max(height - 1, 1)) * 4500.0 + label.astype(np.float32) * 250.0
    depth_float += rng.normal(0, 50, size=(height, width))
    depth = np.clip(depth_float, 0, 65535).astype(np.uint16)
    return rgb, depth, label


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rgb", "depth", "label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

