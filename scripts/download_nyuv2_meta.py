#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


FILES = {
    "splits.mat": "https://raw.githubusercontent.com/VainF/nyuv2-python-toolkit/master/splits.mat",
    "classMapping40.mat": "https://raw.githubusercontent.com/VainF/nyuv2-python-toolkit/master/classMapping40.mat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small NYUv2 split and class-mapping metadata files.")
    parser.add_argument("--out-dir", default="data/nyuv2_meta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        out_path = out_dir / filename
        if out_path.exists():
            print(f"exists {out_path}")
            continue
        print(f"download {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)


if __name__ == "__main__":
    main()

