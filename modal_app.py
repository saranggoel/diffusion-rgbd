# AI assistance used for modal deployment code.

import os
import shlex
from pathlib import Path
from typing import Any

import modal


APP_NAME = "diffusion-rgbd"
VOLUME_NAME = os.environ.get("DIFFUSION_RGBD_MODAL_VOLUME", "diffusion-rgbd-data")
VOLUME_MOUNT = Path("/vol")
WORKDIR = Path("/workspace")
DATASET_NAME = "nyu_depth_v2_labeled.mat"
DEFAULT_GPU_FALLBACKS = ["A100", "L40S", "A10"]

FINAL_CONFIGS = [
    "configs/transformer_fusion_rgbd.yaml",
    "configs/segformer_b2_fusion_rgbd.yaml",
    "configs/segformer_b2_latent_restoration_no_consistency.yaml",
    "configs/segformer_b2_latent_restoration_full_consistency.yaml",
]

REMOTE_REQUIREMENTS = [
    "torch>=2.2",
    "torchvision>=0.17",
    "numpy>=1.24",
    "pillow>=10",
    "pyyaml>=6",
    "h5py>=3.10",
    "scipy>=1.11",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "transformers>=4.40",
]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_pip_install(*REMOTE_REQUIREMENTS)
    .add_local_dir("src", remote_path=str(WORKDIR / "src"))
    .add_local_dir("scripts", remote_path=str(WORKDIR / "scripts"))
    .add_local_dir("configs", remote_path=str(WORKDIR / "configs"))
    .add_local_file("requirements.txt", remote_path=str(WORKDIR / "requirements.txt"))
    .add_local_file("pyproject.toml", remote_path=str(WORKDIR / "pyproject.toml"))
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def run_commands_impl(
    commands: list[list[str]],
    require_dataset: bool = True,
    commit_interval_sec: int = 300,
) -> None:
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    def force_symlink(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        os.symlink(src, dst)

    def prepare_workspace() -> None:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        VOLUME_MOUNT.mkdir(parents=True, exist_ok=True)
        (VOLUME_MOUNT / "outputs").mkdir(parents=True, exist_ok=True)
        (VOLUME_MOUNT / "results").mkdir(parents=True, exist_ok=True)
        (VOLUME_MOUNT / "cache").mkdir(parents=True, exist_ok=True)

        force_symlink(VOLUME_MOUNT / "outputs", WORKDIR / "outputs")
        force_symlink(VOLUME_MOUNT / "results", WORKDIR / "results")

        if not require_dataset:
            return

        dataset_on_volume = VOLUME_MOUNT / DATASET_NAME
        dataset_for_run = dataset_on_volume
        if os.environ.get("DIFFUSION_RGBD_COPY_DATA", "1") != "0":
            local_dataset = Path("/tmp") / DATASET_NAME
            if (
                not local_dataset.exists()
                or local_dataset.stat().st_size != dataset_on_volume.stat().st_size
            ):
                print(f"Copying {dataset_on_volume} to {local_dataset} for faster HDF5 reads")
                shutil.copyfile(dataset_on_volume, local_dataset)
            dataset_for_run = local_dataset

        force_symlink(dataset_for_run, WORKDIR / DATASET_NAME)
        subprocess.run(
            [sys.executable, "scripts/download_nyuv2_meta.py"],
            cwd=WORKDIR,
        )

    def run_one(command: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = f"{WORKDIR / 'src'}:{env.get('PYTHONPATH', '')}"
        env["XDG_CACHE_HOME"] = str(VOLUME_MOUNT / "cache")
        env["TORCH_HOME"] = str(VOLUME_MOUNT / "cache" / "torch")
        env["HF_HOME"] = str(VOLUME_MOUNT / "cache" / "huggingface")
        env["TRANSFORMERS_CACHE"] = str(VOLUME_MOUNT / "cache" / "huggingface")

        print(f"$ {shlex.join(command)}")
        process = subprocess.Popen(command, cwd=WORKDIR, env=env)
        last_commit = time.monotonic()
        while process.poll() is None:
            time.sleep(20)
            if time.monotonic() - last_commit >= commit_interval_sec:
                print("Committing Modal volume snapshot")
                volume.commit()
                last_commit = time.monotonic()

    prepare_workspace()
    try:
        for command in commands:
            run_one(command)
            volume.commit()
    finally:
        volume.commit()


@app.function(volumes={str(VOLUME_MOUNT): volume}, timeout=24 * 60 * 60)
def run_commands_remote(
    commands: list[list[str]],
    require_dataset: bool = True,
    commit_interval_sec: int = 600,
) -> None:
    run_commands_impl(commands, require_dataset=require_dataset, commit_interval_sec=commit_interval_sec)


@app.function(volumes={str(VOLUME_MOUNT): volume}, gpu=DEFAULT_GPU_FALLBACKS, timeout=24 * 60 * 60)
def run_commands_gpu_fallback_remote(
    commands: list[list[str]],
    require_dataset: bool = True,
    commit_interval_sec: int = 600,
) -> None:
    run_commands_impl(commands, require_dataset=require_dataset, commit_interval_sec=commit_interval_sec)


@app.function(volumes={str(VOLUME_MOUNT): volume}, timeout=30 * 60)
def inspect_volume_remote() -> None:
    import subprocess

    subprocess.run(
        [
            "bash",
            "-lc",
            "printf 'Volume: /vol\\n'; find /vol -maxdepth 5 -type f | sort | sed -n '1,200p'",
        ],
    )


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/segformer_b2_fusion_rgbd.yaml",
    gpu: str = "A100",
    epochs: int = 0,
    limit_batches: int = 0,
    resume: str = "",
    checkpoint: str = "",
    out: str = "",
    condition: str = "clean_rgbd",
    count: int = 6,
    timeout_hours: int = 24,
    commit_interval_sec: int = 300,
) -> None:
    if action == "list":
        inspect_volume_remote.remote()
        return

    commands, require_dataset, use_gpu = build_action(
        action=action,
        config=config,
        epochs=epochs,
        limit_batches=limit_batches,
        resume=resume,
        checkpoint=checkpoint,
        out=out,
        condition=condition,
        count=count,
    )

    options: dict[str, Any] = {"timeout": timeout_hours * 60 * 60}
    runner = run_commands_remote
    if use_gpu:
        if is_fallback_gpu(gpu):
            runner = run_commands_gpu_fallback_remote
        else:
            gpu_config = parse_gpu(gpu)
            if gpu_config is not None:
                options["gpu"] = gpu_config

    runner.with_options(**options).remote(
        commands,
        require_dataset=require_dataset,
        commit_interval_sec=commit_interval_sec,
    )


def build_action(
    action: str,
    config: str,
    epochs: int,
    limit_batches: int,
    resume: str,
    checkpoint: str,
    out: str,
    condition: str,
    count: int,
) -> tuple[list[list[str]], bool, bool]:
    if action == "train":
        return ([train_command(config, epochs, limit_batches, resume)], True, True)

    if action == "train-all":
        return (
            [train_command(path, epochs, limit_batches, "") for path in FINAL_CONFIGS],
            True,
            True,
        )

    if action == "eval":
        output_dir = read_output_dir(config)
        ckpt = checkpoint or str(output_dir / "best.pt")
        out_path = out or str(output_dir / "eval_matrix.json")
        command = [
            "python",
            "scripts/evaluate.py",
            "--config",
            config,
            "--checkpoint",
            ckpt,
            "--out",
            out_path,
        ]
        add_limit_batches(command, limit_batches)
        return ([command], True, True)

    if action == "eval-all":
        command = [
            "python",
            "scripts/run_eval_matrix.py",
            "--configs",
            *FINAL_CONFIGS,
            "--summary-out",
            out or "outputs/segformer_b2_results.csv",
        ]
        add_limit_batches(command, limit_batches)
        return ([command], True, True)

    if action == "qualitative":
        output_dir = read_output_dir(config)
        ckpt = checkpoint or str(output_dir / "best.pt")
        out_dir = out or str(output_dir / "qualitative" / condition)
        return (
            [
                [
                    "python",
                    "scripts/export_qualitative.py",
                    "--config",
                    config,
                    "--checkpoint",
                    ckpt,
                    "--condition",
                    condition,
                    "--count",
                    str(count),
                    "--out-dir",
                    out_dir,
                ]
            ],
            True,
            True,
        )

    if action == "qualitative-all":
        config = "configs/segformer_b2_latent_restoration_full_consistency.yaml"
        output_dir = read_output_dir(config)
        ckpt = checkpoint or str(output_dir / "best.pt")
        commands = []
        for condition_name in ["clean_rgbd", "rgb_only", "depth_only", "rgb_corrupt", "depth_corrupt"]:
            commands.append(
                [
                    "python",
                    "scripts/export_qualitative.py",
                    "--config",
                    config,
                    "--checkpoint",
                    ckpt,
                    "--condition",
                    condition_name,
                    "--count",
                    str(count),
                    "--out-dir",
                    f"outputs/final_qualitative/{condition_name}",
                ]
            )
        return commands, True, True

    if action == "plots":
        history_files = [str(read_output_dir(path) / "history.json") for path in FINAL_CONFIGS]
        return (
            [
                [
                    "python",
                    "scripts/visualize_results.py",
                    *history_files,
                    "--out-dir",
                    "outputs/final_plots",
                    "--title",
                    "NYUv2 Final Models",
                ],
                [
                    "python",
                    "scripts/visualize_results.py",
                    "outputs/segformer_b2_results.csv",
                    "--out-dir",
                    "outputs/final_plots",
                ],
            ],
            False,
            False,
        )

    if action == "archive-results":
        return (
            [
                [
                    "python",
                    "-c",
                    (
                        "import shutil; "
                        "from pathlib import Path; "
                        "archive = shutil.make_archive('/vol/rgbd_segmentation_outputs', 'gztar', '/vol', 'outputs'); "
                        "print(f'Wrote {archive}')"
                    ),
                ]
            ],
            False,
            False,
        )

    return [], False, False


def train_command(config: str, epochs: int, limit_batches: int, resume: str) -> list[str]:
    command = ["python", "scripts/train.py", "--config", config]
    if epochs > 0:
        command.extend(["--epochs", str(epochs)])
    add_limit_batches(command, limit_batches)
    if resume:
        command.extend(["--resume", resume])
    return command


def add_limit_batches(command: list[str], limit_batches: int) -> None:
    if limit_batches > 0:
        command.extend(["--limit-batches", str(limit_batches)])


def is_fallback_gpu(gpu: str) -> bool:
    return "," in gpu


def parse_gpu(gpu: str) -> str | None:
    cleaned = gpu.strip()
    if cleaned.lower() in {"", "none", "cpu"}:
        return None
    return cleaned


def read_output_dir(config_path: str) -> Path:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return Path(cfg["experiment"]["output_dir"])
