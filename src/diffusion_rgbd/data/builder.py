from typing import Any

from torch.utils.data import Dataset

from diffusion_rgbd.data.nyuv2_mat import NYUv2MatDataset


def build_dataset(cfg: dict[str, Any], split: str) -> Dataset:
    data_cfg = cfg["data"]
    mat_split = "train" if split == "train" else data_cfg.get("val_split_name", "test")
    return NYUv2MatDataset(
        mat_path=data_cfg["mat_path"],
        split=mat_split,
        splits_path=data_cfg.get("splits_path"),
        class_mapping_path=data_cfg.get("class_mapping_path"),
        image_size=data_cfg.get("image_size"),
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        label_mode=data_cfg.get("label_mode", "nyu40"),
    )
