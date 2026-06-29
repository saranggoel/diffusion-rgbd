from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


class NYUv2MatDataset(Dataset):
    def __init__(
        self,
        mat_path: str | Path,
        split: str,
        splits_path: str | Path | None,
        class_mapping_path: str | Path | None,
        image_size: tuple[int, int] | list[int] | None,
        ignore_index: int = 255,
        label_mode: str = "nyu40",
    ) -> None:
        self.mat_path = Path(mat_path)
        self.split = split
        self.splits_path = Path(splits_path) if splits_path else None
        self.class_mapping_path = Path(class_mapping_path) if class_mapping_path else None
        self.image_size = tuple(image_size) if image_size else None
        self.ignore_index = ignore_index
        self.label_mode = label_mode
        self._mat: h5py.File | None = None

        with h5py.File(self.mat_path, "r") as handle:
            self.length = int(handle["images"].shape[0])

        self.indices = self._load_indices()
        self.label_lookup = self._load_label_lookup()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        mat_index = int(self.indices[index])
        handle = self._handle

        rgb = self._load_rgb(handle, mat_index)
        depth = self._load_depth(handle, mat_index)
        label = self._load_label(handle, mat_index)

        return {
            "rgb": rgb,
            "depth": depth,
            "label": label,
            "id": f"{mat_index + 1:05d}",
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mat"] = None
        return state

    def __del__(self) -> None:
        if getattr(self, "_mat", None) is not None:
            self._mat.close()

    @property
    def _handle(self) -> h5py.File:
        if self._mat is None:
            self._mat = h5py.File(self.mat_path, "r")
        return self._mat

    def _load_indices(self) -> np.ndarray:
        if self.splits_path is not None and self.splits_path.exists():
            splits = loadmat(self.splits_path)
            key = "trainNdxs" if self.split == "train" else "testNdxs"
            return splits[key].reshape(-1).astype(np.int64) - 1

        train_count = min(795, self.length)
        if self.split == "train":
            return np.arange(train_count, dtype=np.int64)
        return np.arange(train_count, self.length, dtype=np.int64)

    def _load_label_lookup(self) -> np.ndarray | None:
        if self.label_mode == "raw":
            return None

        mapping = loadmat(self.class_mapping_path)["mapClass"].reshape(-1).astype(np.int64)
        lookup = np.full((mapping.shape[0] + 1,), self.ignore_index, dtype=np.int64)
        lookup[1:] = mapping - 1
        lookup[lookup < 0] = self.ignore_index
        return lookup

    def _load_rgb(self, handle: h5py.File, mat_index: int) -> torch.Tensor:
        array = np.asarray(handle["images"][mat_index]).transpose(2, 1, 0).copy()
        image = Image.fromarray(array)
        if self.image_size is not None:
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _load_depth(self, handle: h5py.File, mat_index: int) -> torch.Tensor:
        array = np.asarray(handle["depths"][mat_index]).T.astype(np.float32, copy=True)
        image = Image.fromarray(array)
        if self.image_size is not None:
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        depth = np.asarray(image, dtype=np.float32)
        valid = depth > 0
        if valid.any():
            max_value = max(float(np.percentile(depth[valid], 99.5)), 1e-6)
            depth = np.clip(depth / max_value, 0.0, 1.0)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)
        return torch.from_numpy(depth[None, ...]).contiguous()

    def _load_label(self, handle: h5py.File, mat_index: int) -> torch.Tensor:
        raw = np.asarray(handle["labels"][mat_index]).T.astype(np.int64, copy=True)
        if self.label_lookup is not None:
            label = self.label_lookup[raw]
        else:
            label = raw

        label = label.astype(np.int64, copy=False)
        image = Image.fromarray(label.astype(np.uint16))
        if self.image_size is not None:
            image = image.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        label = np.asarray(image, dtype=np.int64)
        return torch.from_numpy(label).long()
