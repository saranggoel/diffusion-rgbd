# AI assistance used for debugging visualization code.

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_rgbd.config import ensure_dir, load_config
from diffusion_rgbd.data import build_dataset
from diffusion_rgbd.models import build_model
from diffusion_rgbd.pipeline import forward_for_condition, prepare_inputs


PALETTE = np.array(
    [
        [35, 35, 40],
        [220, 80, 70],
        [70, 150, 230],
        [90, 190, 120],
        [230, 190, 70],
        [175, 100, 220],
        [80, 210, 210],
        [230, 120, 180],
        [160, 160, 160],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", default="clean_rgbd")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = choose_device()
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    dataset = build_dataset(cfg, split="val")
    out_dir = ensure_dir(args.out_dir or Path(cfg["experiment"]["output_dir"]) / "qualitative")

    for index in range(min(args.count, len(dataset))):
        sample = dataset[index]
        batch = make_batch(sample, device)
        torch.manual_seed(int(cfg.get("experiment", {}).get("seed", 13)) + index)
        with torch.no_grad():
            inputs, _labels = prepare_inputs(batch, cfg, condition=args.condition)
            pred = (
                forward_for_condition(model, inputs, cfg, args.condition)
                .argmax(dim=1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        grid = make_grid(sample, pred, args.condition)
        sample_id = str(sample["id"]).replace("/", "_")
        grid.save(out_dir / f"{index:03d}_{sample_id}_{args.condition}.png")
    print(f"Wrote qualitative grids to {out_dir}")


def make_batch(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "rgb": sample["rgb"].unsqueeze(0).to(device),
        "depth": sample["depth"].unsqueeze(0).to(device),
        "label": sample["label"].unsqueeze(0).to(device),
        "id": [sample["id"]],
    }


def make_grid(sample: dict[str, Any], pred: np.ndarray, condition: str) -> Image.Image:
    rgb = tensor_rgb_to_image(sample["rgb"])
    depth = tensor_depth_to_image(sample["depth"])
    label = colorize(sample["label"].numpy().astype(np.uint8))
    prediction = colorize(pred)
    tiles = [
        ("RGB", rgb),
        ("Depth", depth),
        ("Ground truth", label),
        ("Prediction", prediction),
    ]
    tile_w, tile_h = rgb.size
    header_h = 24
    canvas = Image.new("RGB", (tile_w * len(tiles), tile_h + header_h), color=(20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, image) in enumerate(tiles):
        left = idx * tile_w
        canvas.paste(image, (left, header_h))
        draw.text((left + 6, 5), title, fill=(245, 245, 245))
    draw.text((6, tile_h + header_h - 18), condition, fill=(245, 245, 245))
    return canvas


def tensor_rgb_to_image(rgb: torch.Tensor) -> Image.Image:
    array = (rgb.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(array)


def tensor_depth_to_image(depth: torch.Tensor) -> Image.Image:
    array = depth[0].numpy().clip(0.0, 1.0)
    image = (array * 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def colorize(label: np.ndarray) -> Image.Image:
    colors = PALETTE[label % len(PALETTE)]
    return Image.fromarray(colors.astype(np.uint8))


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
