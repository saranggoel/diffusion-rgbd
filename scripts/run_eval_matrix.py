import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/transformer_fusion_rgbd.yaml",
    "configs/segformer_b2_fusion_rgbd.yaml",
    "configs/segformer_b2_latent_restoration_no_consistency.yaml",
    "configs/segformer_b2_latent_restoration_full_consistency.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--summary-out", default="outputs/final_results.csv")
    parser.add_argument("--limit-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_files: list[str] = []
    for config_path in args.configs:
        output_dir = read_output_dir(Path(config_path))
        checkpoint = output_dir / args.checkpoint_name
        out_json = output_dir / "eval_matrix.json"

        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_json),
        ]
        if args.limit_batches is not None:
            command.extend(["--limit-batches", str(args.limit_batches)])
        subprocess.run(command)
        eval_files.append(str(out_json))

    summary_command = [
        sys.executable,
        "scripts/summarize_results.py",
        "--inputs",
        *eval_files,
        "--out",
        args.summary_out,
    ]
    subprocess.run(summary_command)


def read_output_dir(config_path: Path) -> Path:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return Path(cfg["experiment"]["output_dir"])


if __name__ == "__main__":
    main()
