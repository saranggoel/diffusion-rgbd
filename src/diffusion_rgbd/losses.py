import random
from typing import Any

import torch
import torch.nn.functional as F

from diffusion_rgbd.pipeline import extract_logits, prepare_inputs


DEFAULT_STUDENT_CONDITIONS = [
    "rgb_only",
    "depth_only",
    "rgb_corrupt",
    "depth_corrupt",
]


def compute_training_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    criterion: torch.nn.Module,
    teacher_model: torch.nn.Module | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    model_name = cfg.get("model", {}).get("name", "segformer_fusion")
    if model_name == "segformer_latent_restoration":
        return compute_restoration_training_loss(
            model,
            batch,
            cfg,
            criterion,
            teacher_model=teacher_model,
            epoch=epoch,
            total_epochs=total_epochs,
        )
    return compute_supervised_training_loss(model, batch, cfg, criterion)


def compute_supervised_training_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    criterion: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    condition = sample_condition(cfg.get("training", {}), default="clean_rgbd")
    inputs, labels = prepare_inputs(batch, cfg, condition=condition)
    output = model(inputs)
    logits = extract_logits(output)
    loss = criterion(logits, labels)
    return loss, {"loss_total": float(loss.detach().item()), "loss_seg": float(loss.detach().item())}


def compute_restoration_training_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    criterion: torch.nn.Module,
    teacher_model: torch.nn.Module | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    training_cfg = cfg.get("training", {})
    loss_cfg = training_cfg.get("losses", {})

    clean_inputs, labels = prepare_inputs(batch, cfg, condition="clean_rgbd")
    if teacher_model is None:
        teacher_output = model(clean_inputs, return_dict=True, use_restoration=False)
    else:
        teacher_output = teacher_model(clean_inputs, return_dict=True)
    teacher_logits = teacher_output["logits"]
    teacher_latent = teacher_output["latent"].detach()
    teacher_logits_target = teacher_logits.detach()

    if teacher_model is None:
        seg_loss = criterion(teacher_logits, labels)
    else:
        clean_student_output = model(clean_inputs, return_dict=True, use_restoration=False)
        seg_loss = criterion(clean_student_output["logits"], labels)
    total = float(loss_cfg.get("seg_loss_weight", 1.0)) * seg_loss
    logs: dict[str, float] = {"loss_seg": float(seg_loss.detach().item())}

    student_conditions = training_cfg.get("student_conditions", DEFAULT_STUDENT_CONDITIONS)
    condition_weights = training_cfg.get("student_condition_weights")
    num_student_views = int(training_cfg.get("num_student_views", 1))

    restoration_losses: list[torch.Tensor] = []
    student_ce_losses: list[torch.Tensor] = []
    latent_losses: list[torch.Tensor] = []
    pred_kl_losses: list[torch.Tensor] = []

    for _ in range(max(1, num_student_views)):
        condition = sample_from_list(student_conditions, condition_weights)
        student_inputs, _ = prepare_inputs(batch, cfg, condition=condition)
        student_output = model(student_inputs, return_dict=True, teacher_latent=teacher_latent)

        distance = str(loss_cfg.get("restoration_distance", "mse"))
        if distance == "l1":
            restoration_losses.append(F.l1_loss(student_output["restored_latent"], teacher_latent))
        elif distance == "smooth_l1":
            restoration_losses.append(F.smooth_l1_loss(student_output["restored_latent"], teacher_latent))
        else:
            restoration_losses.append(F.mse_loss(student_output["restored_latent"], teacher_latent))

        if float(loss_cfg.get("student_ce_loss_weight", 0.0)) > 0:
            student_ce_losses.append(criterion(student_output["logits"], labels))

        if float(loss_cfg.get("latent_consistency_weight", 0.0)) > 0:
            latent_losses.append(F.mse_loss(student_output["restored_latent"], teacher_latent))

        if float(loss_cfg.get("prediction_consistency_weight", 0.0)) > 0:
            temperature = float(loss_cfg.get("consistency_temperature", 2.0))
            _, _, height, width = student_output["logits"].shape
            pred_kl_losses.append(
                F.kl_div(
                    F.log_softmax(student_output["logits"] / temperature, dim=1),
                    F.softmax(teacher_logits_target / temperature, dim=1),
                    reduction="batchmean",
                )
                * (temperature**2)
                / float(height * width)
            )

    restoration_loss = mean_or_zero(restoration_losses, seg_loss)
    student_ce_loss = mean_or_zero(student_ce_losses, seg_loss)
    latent_loss = mean_or_zero(latent_losses, seg_loss)
    pred_kl_loss = mean_or_zero(pred_kl_losses, seg_loss)

    restoration_weight = scheduled_loss_weight(
        loss_cfg,
        "restoration_loss_weight",
        epoch=epoch,
        total_epochs=total_epochs,
    )
    student_ce_weight = float(loss_cfg.get("student_ce_loss_weight", 0.0))
    latent_weight = scheduled_loss_weight(
        loss_cfg,
        "latent_consistency_weight",
        epoch=epoch,
        total_epochs=total_epochs,
    )
    pred_kl_weight = scheduled_loss_weight(
        loss_cfg,
        "prediction_consistency_weight",
        epoch=epoch,
        total_epochs=total_epochs,
    )

    total = total + restoration_weight * restoration_loss
    total = total + student_ce_weight * student_ce_loss
    total = total + latent_weight * latent_loss
    total = total + pred_kl_weight * pred_kl_loss

    logs.update(
        {
            "loss_restoration": float(restoration_loss.detach().item()),
            "loss_student_ce": float(student_ce_loss.detach().item()),
            "loss_latent_consistency": float(latent_loss.detach().item()),
            "loss_prediction_consistency": float(pred_kl_loss.detach().item()),
            "weight_restoration": restoration_weight,
            "weight_student_ce": student_ce_weight,
            "weight_latent_consistency": latent_weight,
            "weight_prediction_consistency": pred_kl_weight,
            "loss_total": float(total.detach().item()),
        }
    )
    return total, logs


def sample_condition(training_cfg: dict[str, Any], default: str) -> str:
    conditions = training_cfg.get("train_conditions")
    if not conditions:
        return default
    return sample_from_list(conditions, training_cfg.get("train_condition_weights"))


def sample_from_list(values: list[str], weights: list[float] | None = None) -> str:
    if weights is None:
        return random.choice(values)
    return random.choices(values, weights=weights, k=1)[0]


def mean_or_zero(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if not values:
        return reference.new_zeros(())
    return torch.stack(values).mean()


def scheduled_loss_weight(
    loss_cfg: dict[str, Any],
    weight_name: str,
    epoch: int | None,
    total_epochs: int | None,
) -> float:
    base_weight = float(loss_cfg.get(weight_name, 0.0))
    schedule_cfg = loss_cfg.get("teacher_loss_anneal", {})
    if not schedule_cfg or not bool(schedule_cfg.get("enabled", False)):
        return base_weight

    apply_to = schedule_cfg.get(
        "apply_to",
        ["latent_consistency_weight", "prediction_consistency_weight"],
    )
    if weight_name not in apply_to:
        return base_weight
    if epoch is None or total_epochs is None or total_epochs <= 1:
        return base_weight

    start_fraction = float(schedule_cfg.get("start_fraction", 0.2))
    end_fraction = float(schedule_cfg.get("end_fraction", 0.8))
    final_scale = float(schedule_cfg.get("final_scale", 0.1))
    progress = max(0.0, min(1.0, (float(epoch) - 1.0) / max(float(total_epochs) - 1.0, 1.0)))

    if progress <= start_fraction:
        scale = 1.0
    elif progress >= end_fraction:
        scale = final_scale
    else:
        span = max(end_fraction - start_fraction, 1e-6)
        alpha = (progress - start_fraction) / span
        scale = 1.0 + alpha * (final_scale - 1.0)
    return base_weight * scale
