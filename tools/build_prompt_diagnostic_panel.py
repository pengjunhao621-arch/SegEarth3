#!/usr/bin/env python3
"""Build a deterministic class/region-balanced validation diagnostic panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument(
        "--ann-dir", default="ann_dir/val", help="Annotation directory below data_root"
    )
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def region_name(filename: str) -> str:
    stem = Path(filename).stem
    head, separator, tail = stem.rpartition("_")
    return head if separator and tail.isdigit() else stem


def main():
    args = parse_args()
    ann_dir = os.path.join(args.data_root, args.ann_dir)
    paths = sorted(Path(ann_dir).glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif annotations found in {ann_dir}")

    candidates = []
    global_pixels = np.zeros(args.num_classes, dtype=np.int64)
    for path in paths:
        label = np.asarray(Image.open(path))
        if label.ndim == 3:
            label = label[..., 0]
        counts = np.bincount(
            label[label != args.ignore_index].reshape(-1),
            minlength=args.num_classes,
        )[: args.num_classes]
        global_pixels += counts
        candidates.append(
            dict(
                filename=path.name,
                region=region_name(path.name),
                counts=counts,
                classes=set(np.flatnonzero(counts > 0).tolist()),
            )
        )

    class_image_coverage = np.zeros(args.num_classes, dtype=np.int64)
    selected = []
    used_regions = set()
    remaining = list(candidates)
    rare_weight = 1.0 / np.sqrt(np.maximum(global_pixels, 1))
    rare_weight = rare_weight / rare_weight.max()
    while remaining and len(selected) < args.size:
        best_index = None
        best_key = None
        for index, candidate in enumerate(remaining):
            balance = sum(
                (1.0 + float(rare_weight[class_id]))
                / (1.0 + float(class_image_coverage[class_id]))
                for class_id in candidate["classes"]
            )
            region_bonus = 1.0 if candidate["region"] not in used_regions else 0.0
            key = (balance + region_bonus, -len(selected), candidate["filename"])
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        used_regions.add(chosen["region"])
        for class_id in chosen["classes"]:
            class_image_coverage[class_id] += 1

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("# Deterministic OpenEarthMap Prompt-SAM3 diagnostic panel\n")
        for item in selected:
            handle.write(item["filename"] + "\n")
    report = {
        "annotation_directory": ann_dir,
        "available_images": len(paths),
        "selected_images": len(selected),
        "selected_regions": len(used_regions),
        "class_image_coverage": class_image_coverage.tolist(),
        "class_global_pixels": global_pixels.tolist(),
        "files": [item["filename"] for item in selected],
    }
    with open(output + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

