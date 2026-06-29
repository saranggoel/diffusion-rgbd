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
from diffusion_rgbd.losses import compute_training_loss
from diffusion_rgbd.metrics import SegmentationMetrics
from diffusion_rgbd.models import build_model
from diffusion_rgbd.pipeline import forward_for_condition, prepare_inputs
from diffusion_rgbd.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
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
    load_initial_weights(model, cfg, device)
    teacher_model = build_teacher_model(cfg, device)

    training_cfg = cfg.get("training", {})
    epochs = int(training_cfg.get("epochs", 20))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = build_scheduler(optimizer, training_cfg, epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=int(cfg["data"].get("ignore_index", 255)))
    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = float(
            checkpoint.get(
                "best_score",
                checkpoint.get("best_miou", checkpoint.get("metrics", {}).get("miou", -1.0)),
            )
        )
        history_path = output_dir / "history.json"
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)

    for epoch in range(start_epoch, epochs + 1):
        train_logs = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
            limit_batches=args.limit_batches,
            teacher_model=teacher_model,
            epoch=epoch,
            total_epochs=epochs,
        )
        if scheduler is not None:
            scheduler.step()
        val_metrics = evaluate_validation(
            model=model,
            loader=val_loader,
            cfg=cfg,
            device=device,
            limit_batches=args.limit_batches,
        )
        metric_name = checkpoint_metric(cfg)
        score = metric_value(val_metrics, metric_name)
        row = {
            "epoch": epoch,
            "validation_condition": validation_condition(cfg),
            "validation_conditions": validation_conditions(cfg),
            "checkpoint_metric": metric_name,
            "checkpoint_score": score,
            "train_loss": train_logs["loss_total"],
            **train_logs,
            **val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        save_json(history, output_dir / "history.json")
        save_json(row, output_dir / "last_metrics.json")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "metrics": val_metrics,
            "best_metric": metric_name,
            "best_score": max(best_score, score),
            "best_miou": max(best_score, score),
        }
        save_checkpoint(checkpoint, output_dir / "last.pt")
        if score > best_score:
            best_score = score
            checkpoint["best_score"] = best_score
            checkpoint["best_miou"] = best_score
            save_checkpoint(checkpoint, output_dir / "best.pt")
            save_json(
                {
                    "epoch": epoch,
                    "best_metric": metric_name,
                    "best_score": best_score,
                    "best_miou": best_score,
                    "metrics": val_metrics,
                    "experiment": cfg.get("experiment", {}),
                    "model": cfg.get("model", {}),
                },
                output_dir / "best_metrics.json",
            )

    save_json(history, output_dir / "history.json")


def save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def save_json(payload: Any, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "none")
    if scheduler_type in {None, "none"}:
        return None
    if scheduler_type == "cosine":
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 30)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    return None


def validation_condition(cfg: dict[str, Any]) -> str:
    training_cfg = cfg.get("training", {})
    evaluation_cfg = cfg.get("evaluation", {})
    return str(
        training_cfg.get(
            "validation_condition",
            evaluation_cfg.get("validation_condition", "clean_rgbd"),
        )
    )


def validation_conditions(cfg: dict[str, Any]) -> list[str]:
    training_cfg = cfg.get("training", {})
    conditions = training_cfg.get("validation_conditions")
    if not conditions:
        return [validation_condition(cfg)]
    return [str(condition) for condition in conditions]


def robust_validation_conditions(cfg: dict[str, Any], conditions: list[str]) -> list[str]:
    training_cfg = cfg.get("training", {})
    configured = training_cfg.get("robust_validation_conditions")
    if configured:
        return [str(condition) for condition in configured if str(condition) in conditions]
    robust = [condition for condition in conditions if condition != "clean_rgbd"]
    return robust or conditions


def checkpoint_metric(cfg: dict[str, Any]) -> str:
    training_cfg = cfg.get("training", {})
    default_metric = "avg_robust_miou" if training_cfg.get("validation_conditions") else "miou"
    return str(training_cfg.get("checkpoint_metric", default_metric))


def metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name in metrics:
        return float(metrics[metric_name])
    if metric_name.startswith("condition_metrics."):
        _, condition, key = metric_name.split(".", maxsplit=2)
        return float(metrics["condition_metrics"][condition][key])
    return float(metrics[metric_name])


def build_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    if split == "train":
        loader_cfg = cfg.get("training", {})
        shuffle = True
    elif split == "val":
        loader_cfg = cfg.get("evaluation", {})
        shuffle = False
    else:
        loader_cfg = cfg.get("evaluation", {})
        shuffle = False

    dataset = build_dataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )


def build_teacher_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module | None:
    teacher_cfg = cfg.get("training", {}).get("teacher")
    if not teacher_cfg:
        return None

    teacher_config_path = teacher_cfg.get("config")
    teacher_checkpoint_path = teacher_cfg.get("checkpoint")

    teacher_model_cfg = load_config(teacher_config_path)
    teacher = build_model(teacher_model_cfg).to(device)
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    teacher.load_state_dict(checkpoint["model"], strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"Loaded frozen teacher from {teacher_checkpoint_path}")
    return teacher


def load_initial_weights(model: torch.nn.Module, cfg: dict[str, Any], device: torch.device) -> None:
    checkpoint_path = cfg.get("training", {}).get("init_from_checkpoint")
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint["model"]
    model_state = model.state_dict()
    adapted_state = {}
    for key, value in checkpoint_state.items():
        if key in model_state:
            adapted_state[key] = value
        elif f"fusion.{key}" in model_state:
            adapted_state[f"fusion.{key}"] = value

    incompatible = model.load_state_dict(adapted_state, strict=False)
    print(
        f"Warm-started {len(adapted_state)} tensors from {checkpoint_path}; "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
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
    teacher_model: torch.nn.Module | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(progress):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, loss_logs = compute_training_loss(
                model,
                batch,
                cfg,
                criterion,
                teacher_model=teacher_model,
                epoch=epoch,
                total_epochs=total_epochs,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for key, value in loss_logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
        progress.set_postfix(loss=totals.get("loss_total", 0.0) / max(batches, 1))
    return {key: value / max(batches, 1) for key, value in totals.items()}


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
        logits = forward_for_condition(model, inputs, cfg, condition)
        metrics.update(logits, labels)
    return metrics.compute()


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, Any]:
    conditions = validation_conditions(cfg)
    if len(conditions) == 1:
        metrics = evaluate(
            model=model,
            loader=loader,
            cfg=cfg,
            device=device,
            condition=conditions[0],
            limit_batches=limit_batches,
        )
        condition_metrics = {conditions[0]: dict(metrics)}
        metrics["validation_conditions"] = conditions
        metrics["condition_metrics"] = condition_metrics
        return metrics

    condition_metrics: dict[str, dict[str, float]] = {}
    flat_metrics: dict[str, Any] = {}
    for condition in conditions:
        metrics = evaluate(
            model=model,
            loader=loader,
            cfg=cfg,
            device=device,
            condition=condition,
            limit_batches=limit_batches,
        )
        condition_metrics[condition] = metrics
        for key, value in metrics.items():
            flat_metrics[f"val_{condition}_{key}"] = float(value)

    robust_conditions = robust_validation_conditions(cfg, conditions)
    missing_conditions = [condition for condition in robust_conditions if condition.endswith("_only")]
    corrupt_conditions = [condition for condition in robust_conditions if condition.endswith("_corrupt")]

    flat_metrics["condition_metrics"] = condition_metrics
    flat_metrics["validation_conditions"] = conditions
    flat_metrics["robust_validation_conditions"] = robust_conditions
    flat_metrics["avg_validation_miou"] = average_miou(condition_metrics, conditions)
    flat_metrics["avg_robust_miou"] = average_miou(condition_metrics, robust_conditions)
    flat_metrics["avg_missing_miou"] = average_miou(condition_metrics, missing_conditions)
    flat_metrics["avg_corrupt_miou"] = average_miou(condition_metrics, corrupt_conditions)
    if "clean_rgbd" in condition_metrics:
        clean_miou = float(condition_metrics["clean_rgbd"]["miou"])
        flat_metrics["clean_miou"] = clean_miou
        flat_metrics["max_robust_miou_drop"] = max(
            clean_miou - float(condition_metrics[condition]["miou"])
            for condition in robust_conditions
        )
    flat_metrics["miou"] = metric_value(flat_metrics, checkpoint_metric(cfg))
    return flat_metrics


def average_miou(condition_metrics: dict[str, dict[str, float]], conditions: list[str]) -> float:
    values = [float(condition_metrics[condition]["miou"]) for condition in conditions if condition in condition_metrics]
    if not values:
        return 0.0
    return sum(values) / len(values)


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
