#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_rgbd.config import ensure_dir, load_config
from diffusion_rgbd.data import build_dataset
from diffusion_rgbd.metrics import SegmentationMetrics
from diffusion_rgbd.models import build_model
from diffusion_rgbd.pipeline import prepare_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a baseline across RGB-D robustness conditions.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug: evaluate a subset only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = choose_device()
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    loader = build_loader(cfg)

    conditions = cfg.get("evaluation", {}).get("conditions", ["clean_rgbd"])
    results: dict[str, Any] = {
        "experiment": cfg.get("experiment", {}),
        "model": cfg.get("model", {}),
        "checkpoint": str(args.checkpoint),
        "conditions": {},
    }

    for condition in conditions:
        results["conditions"][condition] = evaluate_condition(
            model=model,
            loader=loader,
            cfg=cfg,
            device=device,
            condition=condition,
            limit_batches=args.limit_batches,
        )
        print(condition, json.dumps(results["conditions"][condition], sort_keys=True))

    out_path = Path(args.out) if args.out else Path(cfg["experiment"]["output_dir"]) / "eval_matrix.json"
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def build_loader(cfg: dict[str, Any]) -> DataLoader:
    eval_cfg = cfg.get("evaluation", {})
    dataset = build_dataset(cfg, split="val")
    return DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    condition: str,
    limit_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=int(cfg["data"]["num_classes"]),
        ignore_index=int(cfg["data"].get("ignore_index", 255)),
        device=device,
    )
    for batch_idx, batch in enumerate(tqdm(loader, desc=condition, leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        torch.manual_seed(int(cfg.get("experiment", {}).get("seed", 13)) + batch_idx)
        batch = move_batch(batch, device)
        inputs, labels = prepare_inputs(batch, cfg, condition=condition)
        logits = model(inputs)
        metrics.update(logits, labels)

    computed = metrics.compute()
    computed["per_class_iou"] = metrics.per_class_iou()
    return computed


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
