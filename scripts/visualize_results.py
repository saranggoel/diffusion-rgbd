#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREFERRED_TRAIN_METRICS = [
    "train_loss",
    "miou",
    "pixel_accuracy",
    "mean_class_accuracy",
]

PREFERRED_EVAL_METRICS = [
    "miou",
    "pixel_accuracy",
    "mean_class_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize baseline training logs or robustness evaluation files."
    )
    parser.add_argument("paths", nargs="+", help="Result file(s): JSONL logs, history JSON, eval JSON, or summary CSV.")
    parser.add_argument("--out-dir", default="results/plots", help="Directory where plots will be written.")
    parser.add_argument("--title", default=None, help="Optional plot title prefix.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Optional metric names to plot.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = [load_result(Path(path)) for path in args.paths]
    histories = [item for item in loaded if item["kind"] == "history"]
    evals = [item for item in loaded if item["kind"] == "eval_matrix"]
    summaries = [item for item in loaded if item["kind"] == "summary_csv"]

    written: list[Path] = []
    for item in histories:
        written.append(plot_history(item, out_dir, args.metrics, args.title, args.dpi))
    if len(histories) > 1:
        written.extend(plot_history_comparison(histories, out_dir, args.metrics, args.title, args.dpi))

    for item in evals:
        written.append(plot_eval_matrix(item, out_dir, args.metrics, args.title, args.dpi))

    for item in summaries:
        written.append(plot_summary_csv(item, out_dir, args.title, args.dpi))

    if not written:
        raise SystemExit("No supported result files were found.")

    for path in written:
        print(path)


def load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return {
            "kind": "summary_csv",
            "path": path,
            "name": path.stem,
            "rows": load_csv(path),
        }

    payload = try_load_json(path)
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return {"kind": "history", "path": path, "name": path.stem, "records": payload}
    if isinstance(payload, dict):
        if "conditions" in payload:
            name = payload.get("experiment", {}).get("name") or path.stem
            return {"kind": "eval_matrix", "path": path, "name": name, "payload": payload}
        if "epoch" in payload:
            return {"kind": "history", "path": path, "name": path.stem, "records": [payload]}

    records = load_json_lines(path)
    if records:
        return {"kind": "history", "path": path, "name": path.stem, "records": records}

    raise ValueError(f"Unsupported result file format: {path}")


def try_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pattern = re.compile(r"\{.*\}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_history(
    item: dict[str, Any],
    out_dir: Path,
    metric_names: list[str] | None,
    title_prefix: str | None,
    dpi: int,
) -> Path:
    records = sorted(item["records"], key=lambda row: row.get("epoch", 0))
    metrics = select_history_metrics(records, metric_names)
    if not metrics:
        raise ValueError(f"No numeric metrics found in {item['path']}")

    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.7 * rows), squeeze=False)
    fig.suptitle(make_title(title_prefix, item["name"]), fontsize=14, fontweight="bold")

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(records)]
    for ax, metric in zip(axes.ravel(), metrics):
        values = [float(row[metric]) for row in records if metric in row and is_number(row[metric])]
        metric_epochs = [epochs[idx] for idx, row in enumerate(records) if metric in row and is_number(row[metric])]
        ax.plot(metric_epochs, values, marker="o", linewidth=2.0, markersize=4)
        ax.set_title(pretty_metric(metric))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(pretty_metric(metric))
        ax.grid(True, alpha=0.25)
        if metric != "train_loss":
            ax.set_ylim(bottom=0.0)
        annotate_last(ax, metric_epochs, values)

    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / f"{safe_name(item['name'])}_training_curves.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_history_comparison(
    histories: list[dict[str, Any]],
    out_dir: Path,
    metric_names: list[str] | None,
    title_prefix: str | None,
    dpi: int,
) -> list[Path]:
    all_records = [row for item in histories for row in item["records"]]
    metrics = select_history_metrics(all_records, metric_names)
    written: list[Path] = []
    if not metrics:
        return written

    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.7 * rows), squeeze=False)
    fig.suptitle(make_title(title_prefix, "Baseline comparison"), fontsize=14, fontweight="bold")

    for ax, metric in zip(axes.ravel(), metrics):
        for item in histories:
            records = sorted(item["records"], key=lambda row: row.get("epoch", 0))
            epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(records)]
            values = [float(row[metric]) for row in records if metric in row and is_number(row[metric])]
            metric_epochs = [epochs[idx] for idx, row in enumerate(records) if metric in row and is_number(row[metric])]
            if values:
                ax.plot(metric_epochs, values, marker="o", linewidth=2.0, markersize=3, label=item["name"])
        ax.set_title(pretty_metric(metric))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(pretty_metric(metric))
        ax.grid(True, alpha=0.25)
        if metric != "train_loss":
            ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False)

    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / "training_comparison.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    written.append(out_path)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for item in histories:
            records = sorted(item["records"], key=lambda row: row.get("epoch", 0))
            epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(records)]
            values = [float(row[metric]) for row in records if metric in row and is_number(row[metric])]
            metric_epochs = [epochs[idx] for idx, row in enumerate(records) if metric in row and is_number(row[metric])]
            if values:
                ax.plot(metric_epochs, values, marker="o", linewidth=2.0, markersize=4, label=item["name"])
                annotate_last(ax, metric_epochs, values)
        ax.set_title(make_title(title_prefix, pretty_metric(metric)))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(pretty_metric(metric))
        ax.grid(True, alpha=0.25)
        if metric != "train_loss":
            ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = out_dir / f"comparison_{safe_name(metric)}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        written.append(out_path)

    return written


def plot_eval_matrix(
    item: dict[str, Any],
    out_dir: Path,
    metric_names: list[str] | None,
    title_prefix: str | None,
    dpi: int,
) -> Path:
    payload = item["payload"]
    conditions = payload["conditions"]
    metrics = metric_names or [metric for metric in PREFERRED_EVAL_METRICS if any(metric in row for row in conditions.values())]
    if not metrics:
        raise ValueError(f"No supported eval metrics found in {item['path']}")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.6 * len(metrics)), squeeze=False)
    condition_names = list(conditions.keys())
    for ax, metric in zip(axes.ravel(), metrics):
        values = [float(conditions[name].get(metric, 0.0)) for name in condition_names]
        ax.bar(condition_names, values)
        ax.set_title(pretty_metric(metric))
        ax.set_ylabel(pretty_metric(metric))
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(make_title(title_prefix, f"{item['name']} robustness eval"), fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / f"{safe_name(item['name'])}_eval_matrix.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_summary_csv(item: dict[str, Any], out_dir: Path, title_prefix: str | None, dpi: int) -> Path:
    rows = item["rows"]
    if not rows:
        raise ValueError(f"CSV has no rows: {item['path']}")

    metric_columns = [column for column in rows[0].keys() if column != "Model"]
    models = [row.get("Model", f"row_{idx}") for idx, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(metric_columns)), 4.8))
    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(metric_columns)))

    for model_idx, row in enumerate(rows):
        values = [try_float(row.get(column)) for column in metric_columns]
        offsets = [x + (model_idx - (len(models) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=models[model_idx])

    ax.set_title(make_title(title_prefix, item["name"]))
    ax.set_xticks(x_positions, metric_columns, rotation=25, ha="right")
    ax.set_ylabel("Metric")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"{safe_name(item['name'])}_summary.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def select_history_metrics(records: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    numeric = {
        key
        for row in records
        for key, value in row.items()
        if key != "epoch" and is_number(value)
    }
    if requested:
        return [metric for metric in requested if metric in numeric]

    ordered = [metric for metric in PREFERRED_TRAIN_METRICS if metric in numeric]
    ordered.extend(sorted(numeric.difference(ordered)))
    return ordered


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def try_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def pretty_metric(metric: str) -> str:
    names = {
        "train_loss": "Train Loss",
        "miou": "mIoU",
        "pixel_accuracy": "Pixel Accuracy",
        "mean_class_accuracy": "Mean Class Accuracy",
    }
    return names.get(metric, metric.replace("_", " ").title())


def make_title(prefix: str | None, title: str) -> str:
    return f"{prefix}: {title}" if prefix else title


def annotate_last(ax: plt.Axes, x_values: list[int], y_values: list[float]) -> None:
    if not x_values or not y_values:
        return
    ax.annotate(
        f"{y_values[-1]:.3f}",
        xy=(x_values[-1], y_values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
