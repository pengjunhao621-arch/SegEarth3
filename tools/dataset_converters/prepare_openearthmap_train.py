#!/usr/bin/env python3
"""Materialize the OpenEarthMap_wo_xBD training split for MMSegmentation.

The official OpenEarthMap ``train.txt`` contains image names.  The distributed
data keep samples under ``<region>/images`` and ``<region>/labels``.  This tool
accepts either ``OpenEarthMap/OpenEarthMap_wo_xBD/<region>`` or the flattened
``OpenEarthMap/<region>`` layout, resolves the manifest, validates image/label
pairs, and creates flat ``img_dir/train`` and ``ann_dir/train`` directories
that mirror the existing validation layout without duplicating data by default.

No label remapping is performed.  OpenEarthMap masks are expected to contain
the original integer ids 0..8, matching ``OpenEarthMapDataset.METAINFO``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)


TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class Pair:
    """One indexed OpenEarthMap image/annotation pair."""

    key: str
    region: str
    image: Path
    label: Path


def _normalise_token(value: str) -> str:
    value = value.strip().strip("'\"").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def _looks_like_subset_root(path: Path) -> bool:
    """Return whether ``path`` directly contains raw region directories."""

    if not path.is_dir():
        return False
    return any(
        child.is_dir()
        and (child / "images").is_dir()
        and (child / "labels").is_dir()
        for child in path.iterdir()
    )


def _resolve_subset_root(data_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        candidates = [explicit.expanduser().resolve()]
    else:
        # Prefer the original wrapped distribution when it exists.  The final
        # candidate supports servers where OpenEarthMap_wo_xBD was extracted
        # directly into data_root alongside img_dir/ and ann_dir/.
        candidates = [data_root / "OpenEarthMap_wo_xBD", data_root]

    for candidate in candidates:
        if _looks_like_subset_root(candidate):
            return candidate.resolve()

    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find raw OpenEarthMap region directories. Checked: "
        f"{joined}. Expected <region>/images and <region>/labels directly "
        "below one of these paths. Pass --subset-root when stored elsewhere."
    )


def _resolve_split_file(
    data_root: Path, subset_root: Path, explicit: Optional[Path]
) -> Path:
    if explicit is not None:
        split_file = explicit.expanduser().resolve()
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        return split_file

    candidates = (
        subset_root / "train.txt",
        data_root / "train.txt",
        data_root / "splits" / "train.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate train.txt. Checked: {joined}. "
        "Pass --split-file explicitly."
    )


def _label_for_image(
    image: Path, images_dir: Path, labels_dir: Path
) -> Optional[Path]:
    relative = image.relative_to(images_dir)
    direct = labels_dir / relative
    if direct.is_file():
        return direct.resolve()

    label_parent = labels_dir / relative.parent
    if not label_parent.is_dir():
        return None
    matches = [
        path
        for path in label_parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in TIFF_SUFFIXES
        and path.stem == image.stem
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def discover_pairs(subset_root: Path) -> Tuple[List[Pair], List[str]]:
    """Discover all ``images``/``labels`` pairs while preserving region paths."""

    pairs: List[Pair] = []
    missing_labels: List[str] = []
    seen_keys = set()

    image_files = sorted(
        path
        for path in subset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TIFF_SUFFIXES
        and "images" in path.relative_to(subset_root).parts
    )
    for image in image_files:
        relative = image.relative_to(subset_root)
        image_index = max(
            index for index, part in enumerate(relative.parts) if part == "images"
        )
        prefix_parts = relative.parts[:image_index]
        if not prefix_parts:
            missing_labels.append(
                f"{relative.as_posix()} (cannot infer a region before images/)"
            )
            continue

        prefix = Path(*prefix_parts)
        inner = Path(*relative.parts[image_index + 1 :])
        images_dir = subset_root / prefix / "images"
        labels_dir = subset_root / prefix / "labels"
        label = _label_for_image(image, images_dir, labels_dir)
        key = (prefix / inner.with_suffix("")).as_posix()
        if label is None:
            missing_labels.append(key)
            continue
        if key in seen_keys:
            raise RuntimeError(f"Duplicate canonical sample key: {key}")
        seen_keys.add(key)
        pairs.append(
            Pair(
                key=key,
                region=prefix.as_posix(),
                image=image.resolve(),
                label=label,
            )
        )

    if not pairs:
        raise RuntimeError(
            f"No image/label pairs were found below {subset_root}. Expected "
            "<region>/images/*.tif and <region>/labels/*.tif."
        )
    return pairs, missing_labels


def _add_alias(
    aliases: MutableMapping[str, List[Pair]], alias: str, pair: Pair
) -> None:
    normalised = _normalise_token(alias)
    if normalised and pair not in aliases[normalised]:
        aliases[normalised].append(pair)


def build_aliases(
    pairs: Sequence[Pair], subset_root: Path
) -> Tuple[Mapping[str, List[Pair]], Mapping[str, List[Pair]]]:
    """Build sample and region lookup tables for common train.txt formats."""

    aliases: Dict[str, List[Pair]] = defaultdict(list)
    regions: Dict[str, List[Pair]] = defaultdict(list)
    for pair in pairs:
        image_relative = pair.image.relative_to(subset_root).as_posix()
        _add_alias(aliases, pair.key, pair)
        _add_alias(aliases, f"{pair.key}{pair.image.suffix}", pair)
        _add_alias(aliases, image_relative, pair)
        _add_alias(aliases, str(Path(image_relative).with_suffix("")), pair)
        _add_alias(aliases, pair.image.name, pair)
        _add_alias(aliases, pair.image.stem, pair)
        _add_alias(aliases, str(pair.image), pair)

        regions[pair.region].append(pair)
        region_name = Path(pair.region).name
        if pair not in regions[region_name]:
            regions[region_name].append(pair)
    return aliases, regions


def _manifest_tokens(split_file: Path) -> Iterable[Tuple[int, str]]:
    for line_number, raw_line in enumerate(
        split_file.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        # Official OpenEarthMap manifests contain one image basename per line.
        # Taking the first field also supports two-column image/label manifests.
        token = _normalise_token(line.replace(",", " ").split()[0])
        if token:
            yield line_number, token


def resolve_manifest(
    split_file: Path,
    aliases: Mapping[str, List[Pair]],
    regions: Mapping[str, List[Pair]],
    allow_missing: bool = False,
) -> Tuple[List[Pair], dict]:
    selected: List[Pair] = []
    selected_keys = set()
    missing_entries: List[dict] = []
    fatal_errors: List[str] = []
    manifest_entries = 0
    resolved_entries = 0

    for line_number, token in _manifest_tokens(split_file):
        manifest_entries += 1
        candidates = aliases.get(token, [])
        if not candidates and token in regions:
            candidates = regions[token]
        if not candidates:
            missing_entries.append({"line": line_number, "token": token})
            continue
        if len(candidates) > 1 and token not in regions:
            choices = ", ".join(pair.key for pair in candidates[:5])
            suffix = " ..." if len(candidates) > 5 else ""
            fatal_errors.append(
                f"line {line_number}: ambiguous: {token} -> {choices}{suffix}"
            )
            continue
        resolved_entries += 1
        for pair in candidates:
            if pair.key not in selected_keys:
                selected.append(pair)
                selected_keys.add(pair.key)

    errors = list(fatal_errors)
    if missing_entries and not allow_missing:
        errors.extend(
            f"line {entry['line']}: not found: {entry['token']}"
            for entry in missing_entries
        )
    if errors:
        preview = "\n  ".join(errors[:30])
        suffix = f"\n  ... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise RuntimeError(
            f"Could not resolve {len(errors)} train.txt entries:\n  {preview}{suffix}"
        )
    if not selected:
        raise RuntimeError(f"No samples were selected by {split_file}")

    manifest_report = {
        "manifest_entries": manifest_entries,
        "resolved_entries": resolved_entries,
        "resolved_ratio": (
            resolved_entries / manifest_entries if manifest_entries else 0.0
        ),
        "skipped_missing_count": len(missing_entries),
        "skipped_missing_entries": missing_entries,
        "allow_missing": allow_missing,
    }
    return sorted(selected, key=lambda pair: pair.key), manifest_report


def _same_source(destination: Path, source: Path) -> bool:
    try:
        return destination.exists() and destination.samefile(source)
    except OSError:
        return False


def materialize_file(
    source: Path,
    destination: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if destination.exists() or destination.is_symlink():
        if _same_source(destination, source):
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                f"Destination exists with different content: {destination}. "
                "Use --overwrite to replace it."
            )
        if not dry_run:
            destination.unlink()

    if dry_run:
        return "planned"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(str(source.resolve()), str(destination))
    elif mode == "hardlink":
        os.link(str(source), str(destination))
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode}")
    return "created"


def validate_flat_output_names(pairs: Sequence[Pair]) -> None:
    """Reject selections that cannot be represented in flat train folders."""

    collisions: List[str] = []
    for kind, names in (
        ("image", [(pair.image.name, pair.key) for pair in pairs]),
        ("label", [(pair.label.name, pair.key) for pair in pairs]),
    ):
        owners: Dict[str, List[str]] = defaultdict(list)
        for name, key in names:
            owners[name].append(key)
        for name, keys in sorted(owners.items()):
            if len(keys) > 1:
                collisions.append(f"{kind} {name}: {', '.join(keys)}")

    if collisions:
        preview = "\n  ".join(collisions[:30])
        suffix = (
            f"\n  ... and {len(collisions) - 30} more"
            if len(collisions) > 30
            else ""
        )
        raise RuntimeError(
            "Selected samples contain duplicate basenames and cannot be placed "
            f"in flat train directories:\n  {preview}{suffix}"
        )


def inspect_labels(pairs: Sequence[Pair], count: int) -> dict:
    if count == 0:
        return {"enabled": False}
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "--inspect-label-count requires Pillow in the active environment."
        ) from exc

    chosen = pairs if count < 0 else pairs[:count]
    value_counts: Counter[object] = Counter()
    modes = Counter()
    for pair in chosen:
        with Image.open(pair.label) as image:
            modes[image.mode] += 1
            value_counts.update(image.getdata())
    non_scalar = [value for value in value_counts if not isinstance(value, int)]
    if non_scalar:
        raise RuntimeError(
            "Labels are not single-channel indexed masks. Run the original "
            "OpenEarthMap converter before using this preparation script."
        )
    sorted_values = sorted(int(value) for value in value_counts)
    if any(value < 0 or value > 8 for value in sorted_values):
        raise RuntimeError(
            f"Unexpected OpenEarthMap label ids: {sorted_values}. Expected 0..8."
        )
    scalar_counts = {
        str(int(value)): int(count)
        for value, count in sorted(value_counts.items())
    }
    total_pixels = sum(scalar_counts.values())
    pixel_ratios = {
        value: count / total_pixels if total_pixels else 0.0
        for value, count in scalar_counts.items()
    }
    return {
        "enabled": True,
        "files_checked": len(chosen),
        "modes": dict(sorted(modes.items())),
        "unique_values": sorted_values,
        "pixel_counts_by_id": scalar_counts,
        "pixel_ratios_by_id": pixel_ratios,
        "expected_values": list(range(9)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenEarthMap_wo_xBD official train split for SegEarth-OV3."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help=(
            "OpenEarthMap root. Raw regions may be directly below it or below "
            "OpenEarthMap_wo_xBD/."
        ),
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=None,
        help="Explicit directory that directly contains raw region folders.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Official train.txt. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="MMSeg data root. Defaults to data_root.",
    )
    parser.add_argument("--split-name", default="train")
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize files. Symlink avoids another full data copy.",
    )
    parser.add_argument(
        "--inspect-label-count",
        type=int,
        default=0,
        help="Inspect N indexed masks (0 disables; -1 checks all).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Skip train.txt entries for which no complete image/label pair was "
            "found. Ambiguous entries and flat-output name collisions still fail."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    subset_root = _resolve_subset_root(data_root, args.subset_root)
    split_file = _resolve_split_file(data_root, subset_root, args.split_file)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root
    )

    pairs, missing_labels = discover_pairs(subset_root)
    aliases, regions = build_aliases(pairs, subset_root)
    selected, manifest_report = resolve_manifest(
        split_file,
        aliases,
        regions,
        allow_missing=args.allow_missing,
    )
    validate_flat_output_names(selected)
    label_inspection = inspect_labels(selected, args.inspect_label_count)

    operation_counts: Counter[str] = Counter()
    for pair in selected:
        image_destination = (
            output_root / "img_dir" / args.split_name / pair.image.name
        )
        label_destination = (
            output_root / "ann_dir" / args.split_name / pair.label.name
        )
        image_status = materialize_file(
            pair.image,
            image_destination,
            args.mode,
            args.overwrite,
            args.dry_run,
        )
        label_status = materialize_file(
            pair.label,
            label_destination,
            args.mode,
            args.overwrite,
            args.dry_run,
        )
        operation_counts[f"image_{image_status}"] += 1
        operation_counts[f"label_{label_status}"] += 1

    region_counts = Counter(pair.region for pair in selected)
    report = {
        "data_root": str(data_root),
        "subset_root": str(subset_root),
        "source_layout": (
            "flattened_data_root"
            if subset_root == data_root
            else "nested_subset_root"
        ),
        "source_split_file": str(split_file),
        "output_root": str(output_root),
        "output_layout": "flat_img_dir_and_ann_dir",
        "split_name": args.split_name,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "discovered_pairs": len(pairs),
        "selected_pairs": len(selected),
        "manifest": manifest_report,
        "selected_regions": len(region_counts),
        "per_region": dict(sorted(region_counts.items())),
        "unpaired_images_not_selected": missing_labels,
        "runtime_manifest": None,
        "runtime_loading": "scan img_dir/train and ann_dir/train directly",
        "operations": dict(sorted(operation_counts.items())),
        "label_inspection": label_inspection,
        "label_policy": "preserve original ids 0..8; reduce_zero_label=False",
    }
    report_path = output_root / f"{args.split_name}_prepare_report.json"
    if not args.dry_run:
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    missing_count = manifest_report["skipped_missing_count"]
    if missing_count:
        total = manifest_report["manifest_entries"]
        ratio = manifest_report["resolved_ratio"]
        print(
            f"WARNING: skipped {missing_count}/{total} missing manifest entries; "
            f"resolved ratio={ratio:.4f}.",
            file=sys.stderr,
        )
    if args.dry_run:
        print("Dry run only: no files were created.", file=sys.stderr)
    else:
        print(f"Preparation report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
