#!/usr/bin/env python3
"""Prepare the official UAVid-2020 test split for SegEarth-OV3.

This follows the SegEarth-OV-2 baseline protocol: merge Moving Car into Static
Car and crop each 4K frame into 1080x1280 non-overlapping patches.  Labels are
saved as mode-L uint8 masks because Pillow 11.3 corrupts the original
``Image.fromarray(..., mode='P')`` implementation.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from PIL import Image


UAVID_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (128, 64, 128),
    3: (192, 0, 192),  # Static Car
    4: (0, 128, 0),
    5: (128, 128, 0),
    6: (64, 64, 0),
    7: (64, 0, 128),  # Moving Car; remapped to class 3 below.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare UAVid-2020 test images and public test GT.')
    parser.add_argument(
        'dataset_path', type=Path,
        help='Raw root containing test_gt/<sequence>/{Images,Labels}.')
    parser.add_argument(
        '-o', '--out-dir', type=Path, default=Path('data/UAVid'),
        help='Output root. Defaults to data/UAVid.')
    parser.add_argument('--patch-width', type=int, default=1280)
    parser.add_argument('--patch-height', type=int, default=1080)
    parser.add_argument('--overlap-area', type=int, default=0)
    parser.add_argument(
        '--check-only', action='store_true',
        help='Fully decode every paired source PNG and exit without writing.')
    return parser.parse_args()


def crop_starts(length: int, patch: int, stride: int) -> Iterable[int]:
    for start in range(0, length, stride):
        end = start + patch
        if end > length:
            start -= end - length
        yield start


def pad(array: np.ndarray, height: int, width: int, value: int) -> np.ndarray:
    old_height, old_width = array.shape[:2]
    target_height = max(old_height, height)
    target_width = max(old_width, width)
    if (target_height, target_width) == (old_height, old_width):
        return array
    shape = (target_height, target_width) + array.shape[2:]
    output = np.full(shape, value, dtype=array.dtype)
    output[:old_height, :old_width] = array
    return output


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0].astype(np.uint32) << 16
        | rgb[..., 1].astype(np.uint32) << 8
        | rgb[..., 2].astype(np.uint32)
    )


def rgb_to_train_ids(rgb: np.ndarray) -> np.ndarray:
    """Match the official converter, including unknown-color handling.

    The official SegEarth converter initializes its output to class 0 and only
    overwrites pixels matching the eight UAVid colors.  Unknown source colors
    consequently remain background; the preflight scan below reports them.
    """
    packed = pack_rgb(rgb)
    labels = np.zeros(packed.shape, dtype=np.uint8)
    for label_id, (red, green, blue) in UAVID_PALETTE.items():
        color = (red << 16) | (green << 8) | blue
        train_id = 3 if label_id == 7 else label_id
        labels[packed == color] = train_id
    return labels


def output_name(sequence: str, frame: Path, y: int, x: int,
                height: int, width: int) -> str:
    return (
        f'{sequence}_{frame.stem}_{y}_{y + height}_{x}_{x + width}.png')


def find_broken_pngs(paths: Iterable[Path]) -> List[Tuple[Path, str]]:
    """Fully decode PNGs so truncated downloads are reported before cropping."""
    broken = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, SyntaxError, ValueError) as error:
            broken.append((path, f'{type(error).__name__}: {error}'))
    return broken


def decode_packed_color(color: int) -> Tuple[int, int, int]:
    return (
        (int(color) >> 16) & 255,
        (int(color) >> 8) & 255,
        int(color) & 255,
    )


def scan_unknown_label_colors(
        paths: Iterable[Path]
) -> Tuple[Dict[Tuple[int, int, int], int],
           Dict[Tuple[int, int, int], Set[Path]]]:
    known = {
        (red << 16) | (green << 8) | blue
        for red, green, blue in UAVID_PALETTE.values()
    }
    pixel_counts: Dict[Tuple[int, int, int], int] = {}
    source_files: Dict[Tuple[int, int, int], Set[Path]] = {}
    for path in paths:
        with Image.open(path) as label:
            packed = pack_rgb(np.asarray(label.convert('RGB')))
        colors, counts = np.unique(packed, return_counts=True)
        for color, count in zip(colors.tolist(), counts.tolist()):
            if color in known:
                continue
            decoded = decode_packed_color(color)
            pixel_counts[decoded] = pixel_counts.get(decoded, 0) + int(count)
            source_files.setdefault(decoded, set()).add(path)
    return pixel_counts, source_files


def main() -> int:
    args = parse_args()
    raw_root = args.dataset_path.resolve() / 'test_gt'
    output_root = args.out_dir.resolve()
    image_output = output_root / 'img_dir' / 'test'
    label_output = output_root / 'ann_dir' / 'test'

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f'Expected raw test directory: {raw_root}')
    if args.overlap_area < 0 or args.overlap_area >= min(
            args.patch_width, args.patch_height):
        raise ValueError('Invalid --overlap-area.')
    image_sources = sorted(raw_root.glob('*/Images/*.png'))
    label_sources = sorted(raw_root.glob('*/Labels/*.png'))
    image_keys = {(p.parent.parent.name, p.name): p for p in image_sources}
    label_keys = {(p.parent.parent.name, p.name): p for p in label_sources}
    if image_keys.keys() != label_keys.keys():
        missing_labels = sorted(image_keys.keys() - label_keys.keys())
        missing_images = sorted(label_keys.keys() - image_keys.keys())
        raise RuntimeError(
            'Raw test image/GT mismatch. '
            f'Missing labels: {missing_labels[:5]}; '
            f'missing images: {missing_images[:5]}')
    if not image_keys:
        raise RuntimeError(
            f'No paired PNGs found below {raw_root}/<seq>/Images and Labels.')

    print(f'Found {len(image_keys)} paired UAVid test frames.')
    broken = find_broken_pngs(list(image_sources) + list(label_sources))
    if broken:
        details = '\n'.join(
            f'  {path}: {reason}' for path, reason in broken)
        raise RuntimeError(
            f'Found {len(broken)} corrupt or truncated source PNG(s):\n'
            f'{details}\nRe-extract or re-download only the listed source files.')
    print(f'PASSED: fully decoded {len(image_sources)} images and '
          f'{len(label_sources)} labels.')
    unknown_counts, unknown_files = scan_unknown_label_colors(label_sources)
    if unknown_counts:
        print('WARNING: source labels contain colors outside the official '
              'eight-color UAVid palette.')
        print('Matching the official SegEarth converter, these pixels will '
              'be mapped to class 0 (background):')
        for color in sorted(unknown_counts):
            examples = sorted(str(path) for path in unknown_files[color])[:3]
            print(
                f'  color={color}, pixels={unknown_counts[color]}, '
                f'files={len(unknown_files[color])}, examples={examples}')
    else:
        print('PASSED: all label pixels use the official UAVid palette.')
    if args.check_only:
        return 0

    for directory in (image_output, label_output):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f'Output directory must be empty: {directory}')
        directory.mkdir(parents=True, exist_ok=True)

    stride_x = args.patch_width - args.overlap_area
    stride_y = args.patch_height - args.overlap_area
    histogram = np.zeros(256, dtype=np.uint64)
    generated = 0

    for index, key in enumerate(sorted(image_keys), 1):
        sequence, _ = key
        image_path = image_keys[key]
        label_path = label_keys[key]
        try:
            with Image.open(image_path) as image:
                image_array = np.asarray(image.convert('RGB'))
        except (OSError, SyntaxError, ValueError) as error:
            raise RuntimeError(
                f'Failed to decode source image: {image_path}') from error
        try:
            with Image.open(label_path) as label:
                label_array = rgb_to_train_ids(
                    np.asarray(label.convert('RGB')))
        except (OSError, SyntaxError, ValueError) as error:
            raise RuntimeError(
                f'Failed to decode source label: {label_path}') from error
        if image_array.shape[:2] != label_array.shape:
            raise RuntimeError(
                f'Image/label shape mismatch: {image_path} '
                f'{image_array.shape[:2]} vs {label_array.shape}.')

        image_array = pad(
            image_array, args.patch_height, args.patch_width, 0)
        label_array = pad(
            label_array, args.patch_height, args.patch_width, 255)
        full_height, full_width = label_array.shape

        for x in crop_starts(full_width, args.patch_width, stride_x):
            for y in crop_starts(full_height, args.patch_height, stride_y):
                image_patch = image_array[
                    y:y + args.patch_height, x:x + args.patch_width]
                label_patch = label_array[
                    y:y + args.patch_height, x:x + args.patch_width]
                name = output_name(
                    sequence, image_path, y, x,
                    args.patch_height, args.patch_width)
                Image.fromarray(image_patch).save(image_output / name)
                Image.fromarray(label_patch).save(label_output / name)
                histogram += np.bincount(
                    label_patch.reshape(-1), minlength=256).astype(np.uint64)
                generated += 1

        if index % 10 == 0 or index == len(image_keys):
            print(
                f'Processed {index}/{len(image_keys)} frames; '
                f'generated {generated} patches.')

    output_images = {p.name for p in image_output.glob('*.png')}
    output_labels = {p.name for p in label_output.glob('*.png')}
    present_ids = np.flatnonzero(histogram).tolist()
    missing_classes = sorted(set(range(7)) - set(present_ids))

    print('\nValidation summary')
    print(f'  raw frame pairs: {len(image_keys)}')
    print(f'  image patches: {len(output_images)}')
    print(f'  label patches: {len(output_labels)}')
    print(f'  present label ids: {present_ids}')
    print(f'  missing merged classes: {missing_classes}')
    if output_images != output_labels:
        raise RuntimeError('Processed image and annotation names differ.')
    if missing_classes:
        raise RuntimeError(
            f'Processed test GT misses merged classes: {missing_classes}')
    first_label = next(label_output.glob('*.png'))
    with Image.open(first_label) as label:
        if label.mode != 'L':
            raise RuntimeError(
                f'Expected mode-L labels, observed {label.mode}.')

    print(f'PASSED: UAVid test data are ready in {output_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
