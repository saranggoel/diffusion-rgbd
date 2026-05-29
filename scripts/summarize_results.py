#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONDITION_COLUMNS = [
    ("clean_rgbd", "Clean RGB-D"),
    ("rgb_only", "RGB only"),
    ("depth_only", "Depth only"),
    ("rgb_corrupt", "RGB corrupt"),
    ("depth_corrupt", "Depth corrupt"),
    ("both_corrupt", "Both corrupt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize robustness evaluation JSON files into a CSV table.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Evaluation JSON files.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--metric", default="miou", choices=["miou", "pixel_accuracy", "mean_class_accuracy"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [load_row(Path(path), args.metric) for path in args.inputs]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Model"] + [label for _, label in CONDITION_COLUMNS] + ["Avg Robust", "Max Drop"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


def load_row(path: Path, metric: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_name = payload.get("experiment", {}).get("name") or payload.get("model", {}).get("modality") or path.stem
    conditions = payload.get("conditions", {})

    values: dict[str, float | None] = {}
    row: dict[str, Any] = {"Model": model_name}
    for condition, label in CONDITION_COLUMNS:
        value = extract_metric(conditions, condition, metric)
        values[condition] = value
        row[label] = format_value(value)

    robust_values = [value for key, value in values.items() if key != "clean_rgbd" and value is not None]
    row["Avg Robust"] = format_value(sum(robust_values) / len(robust_values) if robust_values else None)

    clean = values.get("clean_rgbd")
    drops = [clean - value for key, value in values.items() if clean is not None and key != "clean_rgbd" and value is not None]
    row["Max Drop"] = format_value(max(drops) if drops else None)
    return row


def extract_metric(conditions: dict[str, Any], condition: str, metric: str) -> float | None:
    if condition not in conditions or metric not in conditions[condition]:
        return None
    return float(conditions[condition][metric])


def format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()

