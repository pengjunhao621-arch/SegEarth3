#!/usr/bin/env python3
"""Rebuild iSAID validation label tiles without reprocessing RGB images.

MMSegmentation v1.2.2 saves the uint8 class-id array with an explicit Pillow
``mode='P'`` argument.  Pillow 11.3 can silently collapse that array to binary
indices.  This converter follows the official 896/384 crop geometry but saves
the label tiles as ordinary single-channel uint8 (Pillow mode ``L``).
"""

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image


ISAID_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 0, 63),
    2: (0, 63, 63),
    3: (0, 63, 0),
    4: (0, 63, 127),
    5: (0, 63, 191),
    6: (0, 63, 255),
    7: (0, 127, 63),
    8: (0, 127, 127),
    9: (0, 0, 127),
    10: (0, 0, 191),
    11: (0, 0, 255),
    12: (0, 191, 127),
    13: (0, 127, 191),
    14: (0, 127, 255),
    15: (0, 100, 155),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Safely rebuild iSAID validation annotation tiles.')
    parser.add_argument(
        'semantic_masks_zip', type=Path,
        help='Downloaded iSAID val/Semantic_masks/images.zip.')
    parser.add_argument(
        '--image-dir', type=Path, required=True,
        help='Existing processed img_dir/val containing 11,644 PNG tiles.')
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Fresh directory in which rebuilt annotation tiles are written.')
    parser.add_argument('--patch-size', type=int, default=896)
    parser.add_argument('--overlap', type=int, default=384)
    parser.add_argument('--expected-source-count', type=int, default=458)
    parser.add_argument('--expected-tile-count', type=int, default=11644)
    return parser.parse_args()


def convert_rgb_to_ids(rgb: np.ndarray, path: Path) -> np.ndarray:
    packed = (
        rgb[..., 0].astype(np.uint32) << 16
        | rgb[..., 1].astype(np.uint32) << 8
        | rgb[..., 2].astype(np.uint32)
    )
    labels = np.full(packed.shape, 255, dtype=np.uint8)
    for label_id, (red, green, blue) in ISAID_PALETTE.items():
        color = (red << 16) | (green << 8) | blue
        labels[packed == color] = label_id

    unknown = labels == 255
    if np.any(unknown):
        colors, counts = np.unique(packed[unknown], return_counts=True)
        preview = []
        for color, count in zip(colors[:10], counts[:10]):
            value = int(color)
            preview.append((
                ((value >> 16) & 255, (value >> 8) & 255, value & 255),
                int(count),
            ))
        raise ValueError(
            f'{path} contains colors outside the official iSAID palette: '
            f'{preview}')
    return labels


def pad_to_patch(labels: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = labels.shape
    target_height = max(height, patch_size)
    target_width = max(width, patch_size)
    if (target_height, target_width) == (height, width):
        return labels
    padded = np.full((target_height, target_width), 255, dtype=np.uint8)
    padded[:height, :width] = labels
    return padded


def crop_starts(length: int, patch_size: int, stride: int) -> Iterable[int]:
    for start in range(0, length, stride):
        end = start + patch_size
        if end > length:
            start -= end - length
        yield start


def source_stem(path: Path) -> str:
    suffix = '_instance_color_RGB'
    stem = path.stem
    if stem.endswith(suffix):
        return stem[:-len(suffix)]
    return stem.split('_')[0]


def main() -> int:
    args = parse_args()
    archive = args.semantic_masks_zip.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not archive.is_file():
        raise FileNotFoundError(f'Archive not found: {archive}')
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image tile directory not found: {image_dir}')
    if args.overlap < 0 or args.overlap >= args.patch_size:
        raise ValueError('--overlap must satisfy 0 <= overlap < patch-size.')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f'Output directory must be empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = args.patch_size - args.overlap
    tile_histogram = np.zeros(256, dtype=np.uint64)
    generated_names = set()

    with tempfile.TemporaryDirectory(prefix='isaid_val_masks_') as temp_name:
        temp_dir = Path(temp_name)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(temp_dir)

        sources = sorted(
            path for path in temp_dir.rglob('*.png')
            if not path.name.startswith('._'))
        if len(sources) != args.expected_source_count:
            raise RuntimeError(
                f'Expected {args.expected_source_count} source masks, '
                f'found {len(sources)} in {archive}.')

        print(f'Found {len(sources)} source semantic masks.')
        for source_index, path in enumerate(sources, 1):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            labels = pad_to_patch(
                convert_rgb_to_ids(rgb, path), args.patch_size)
            height, width = labels.shape
            base = source_stem(path)

            for x_start in crop_starts(width, args.patch_size, stride):
                for y_start in crop_starts(height, args.patch_size, stride):
                    y_end = y_start + args.patch_size
                    x_end = x_start + args.patch_size
                    patch = labels[y_start:y_end, x_start:x_end]
                    if patch.shape != (args.patch_size, args.patch_size):
                        raise RuntimeError(
                            f'Unexpected patch shape {patch.shape} from {path}.')

                    name = (
                        f'{base}_{y_start}_{y_end}_{x_start}_{x_end}'
                        '_instance_color_RGB.png')
                    Image.fromarray(patch).save(output_dir / name)
                    generated_names.add(name)
                    tile_histogram += np.bincount(
                        patch.reshape(-1), minlength=256).astype(np.uint64)

            if source_index % 25 == 0 or source_index == len(sources):
                print(
                    f'Processed {source_index}/{len(sources)} source masks; '
                    f'generated {len(generated_names)} unique tiles.')

    image_paths = sorted(image_dir.glob('*.png'))
    expected_annotation_names = {
        f'{path.stem}_instance_color_RGB.png' for path in image_paths
    }
    missing_annotations = sorted(expected_annotation_names - generated_names)
    extra_annotations = sorted(generated_names - expected_annotation_names)
    present_labels = np.flatnonzero(tile_histogram).tolist()
    missing_classes = sorted(set(range(16)) - set(present_labels))

    print('\nValidation summary')
    print(f'  existing image tiles: {len(image_paths)}')
    print(f'  generated annotations: {len(generated_names)}')
    print(f'  missing annotation pairs: {len(missing_annotations)}')
    print(f'  extra annotation pairs: {len(extra_annotations)}')
    print(f'  present label ids: {present_labels}')
    print(f'  missing semantic classes: {missing_classes}')

    if len(image_paths) != args.expected_tile_count:
        raise RuntimeError(
            f'Expected {args.expected_tile_count} image tiles, '
            f'found {len(image_paths)}.')
    if len(generated_names) != args.expected_tile_count:
        raise RuntimeError(
            f'Expected {args.expected_tile_count} annotation tiles, '
            f'generated {len(generated_names)}.')
    if missing_annotations or extra_annotations:
        raise RuntimeError(
            'Generated annotation names do not exactly match existing images. '
            f'Missing examples: {missing_annotations[:5]}; '
            f'extra examples: {extra_annotations[:5]}')
    if missing_classes:
        raise RuntimeError(
            f'Rebuilt annotations still miss classes: {missing_classes}')

    first_output = output_dir / next(iter(generated_names))
    with Image.open(first_output) as image:
        output_mode = image.mode
    if output_mode != 'L':
        raise RuntimeError(
            f'Expected Pillow mode L, observed {output_mode}.')

    print(f'PASSED: safe iSAID labels are ready in {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
