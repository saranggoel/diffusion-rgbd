#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_rgbd.config import ensure_dir, load_config
from diffusion_rgbd.data import build_dataset
from diffusion_rgbd.metrics import SegmentationMetrics
from diffusion_rgbd.models import build_model
from diffusion_rgbd.pipeline import prepare_inputs
from diffusion_rgbd.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RGB-D segmentation baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug: stop each epoch early.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs

    seed_everything(int(cfg.get("experiment", {}).get("seed", 13)))
    device = choose_device()
    output_dir = ensure_dir(cfg["experiment"]["output_dir"])

    train_loader = build_loader(cfg, split="train")
    val_loader = build_loader(cfg, split="val")
    model = build_model(cfg).to(device)

    training_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=int(cfg["data"].get("ignore_index", 255)))
    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_miou = -1.0
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 20))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
            limit_batches=args.limit_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            cfg=cfg,
            device=device,
            condition="clean_rgbd",
            limit_batches=args.limit_batches,
        )
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(checkpoint, output_dir / "best.pt")

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def build_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    if split == "train":
        loader_cfg = cfg.get("training", {})
        shuffle = True
    elif split == "val":
        loader_cfg = cfg.get("evaluation", {})
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    dataset = build_dataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(progress):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        inputs, labels = prepare_inputs(batch, cfg, condition="clean_rgbd")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item())
        batches += 1
        progress.set_postfix(loss=total_loss / max(batches, 1))
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    condition: str,
    limit_batches: int | None,
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=int(cfg["data"]["num_classes"]),
        ignore_index=int(cfg["data"].get("ignore_index", 255)),
        device=device,
    )
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"val:{condition}", leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        torch.manual_seed(int(cfg.get("experiment", {}).get("seed", 13)) + batch_idx)
        batch = move_batch(batch, device)
        inputs, labels = prepare_inputs(batch, cfg, condition=condition)
        logits = model(inputs)
        metrics.update(logits, labels)
    return metrics.compute()


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
