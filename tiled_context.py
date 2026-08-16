"""Pixel-exact virtual mosaics for coordinate-named remote-sensing tiles."""

import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image


_TILE_RE = re.compile(
    r'^(?P<prefix>.+)_(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)$')
_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def parse_coordinate_tile(path):
    """Return ``(prefix, x1, y1, x2, y2)`` from a coordinate tile name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _TILE_RE.match(stem)
    if match is None:
        raise ValueError(f'Not a coordinate tile name: {path}')
    return (
        match.group('prefix'),
        int(match.group('x1')),
        int(match.group('y1')),
        int(match.group('x2')),
        int(match.group('y2')),
    )


class CoordinateTileIndex:
    """Index a directory of overlapping coordinate tiles by source mosaic."""

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.groups = defaultdict(list)
        for name in sorted(os.listdir(self.directory)):
            path = os.path.join(self.directory, name)
            if (not os.path.isfile(path)
                    or os.path.splitext(name)[1].lower()
                    not in _IMAGE_SUFFIXES):
                continue
            try:
                prefix, x1, y1, x2, y2 = parse_coordinate_tile(path)
            except ValueError:
                continue
            self.groups[prefix].append(dict(
                path=path, x1=x1, y1=y1, x2=x2, y2=y2))
        if not self.groups:
            raise ValueError(f'No coordinate tiles found in {self.directory}.')
        self.canvases = {
            prefix: (
                max(tile['x2'] for tile in tiles),
                max(tile['y2'] for tile in tiles),
            )
            for prefix, tiles in self.groups.items()
        }

    def source_and_box(self, path):
        prefix, x1, y1, x2, y2 = parse_coordinate_tile(path)
        if prefix not in self.groups:
            raise ValueError(f'{prefix!r} is absent from {self.directory}.')
        return prefix, (x1, y1, x2, y2)

    @staticmethod
    def _centered_box(target_box, requested_size, canvas_size):
        x1, y1, x2, y2 = target_box
        canvas_width, canvas_height = canvas_size
        width = min(int(requested_size), int(canvas_width))
        height = min(int(requested_size), int(canvas_height))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = min(max(int(round(center_x - width / 2.0)), 0),
                   canvas_width - width)
        top = min(max(int(round(center_y - height / 2.0)), 0),
                  canvas_height - height)
        return left, top, left + width, top + height

    def read_context(self, target_path, requested_size):
        """Assemble a real context crop and locate the target tile within it."""
        prefix, target_box = self.source_and_box(target_path)
        context_box = self._centered_box(
            target_box, requested_size, self.canvases[prefix])
        cx1, cy1, cx2, cy2 = context_box
        canvas = np.empty((cy2 - cy1, cx2 - cx1, 3), dtype=np.uint8)
        filled = np.zeros((cy2 - cy1, cx2 - cx1), dtype=bool)
        contributors = 0
        for tile in self.groups[prefix]:
            ix1, iy1 = max(cx1, tile['x1']), max(cy1, tile['y1'])
            ix2, iy2 = min(cx2, tile['x2']), min(cy2, tile['y2'])
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            image = np.asarray(Image.open(tile['path']).convert('RGB'))
            expected = (tile['y2'] - tile['y1'], tile['x2'] - tile['x1'])
            if image.shape[:2] != expected:
                raise ValueError(
                    f'Tile size disagrees with coordinates: {tile["path"]}.')
            source = image[
                iy1 - tile['y1']:iy2 - tile['y1'],
                ix1 - tile['x1']:ix2 - tile['x1']]
            target = np.s_[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1]
            if filled[target].any():
                overlap = filled[target]
                if not np.array_equal(canvas[target][overlap], source[overlap]):
                    raise ValueError(
                        f'RGB overlap mismatch while reading {target_path}.')
            canvas[target] = source
            filled[target] = True
            contributors += 1
        if not filled.all():
            raise ValueError(
                f'Virtual context has uncovered pixels for {target_path}.')
        x1, y1, x2, y2 = target_box
        roi = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        return Image.fromarray(canvas), roi, dict(
            source_prefix=prefix,
            source_canvas=list(self.canvases[prefix]),
            target_box=list(target_box),
            context_box=list(context_box),
            context_size=[cx2 - cx1, cy2 - cy1],
            contributor_tiles=int(contributors),
        )

    def verify_rgb_overlaps(self):
        """Check coverage and every positive-area RGB overlap exactly."""
        checked_pairs = 0
        checked_pixels = 0
        for prefix, tiles in self.groups.items():
            coordinates = {
                (tile['x1'], tile['y1'], tile['x2'], tile['y2'])
                for tile in tiles
            }
            if len(coordinates) != len(tiles):
                raise ValueError(f'{prefix}: duplicate coordinate boxes.')
            xs = sorted({(tile['x1'], tile['x2']) for tile in tiles})
            ys = sorted({(tile['y1'], tile['y2']) for tile in tiles})
            expected = {(x1, y1, x2, y2)
                        for x1, x2 in xs for y1, y2 in ys}
            if coordinates != expected:
                raise ValueError(f'{prefix}: coordinate grid is incomplete.')
            for intervals, axis in ((xs, 'x'), (ys, 'y')):
                if intervals[0][0] != 0:
                    raise ValueError(f'{prefix}: {axis}-grid does not start at 0.')
                for previous, current in zip(intervals, intervals[1:]):
                    if current[0] > previous[1]:
                        raise ValueError(f'{prefix}: gap in {axis}-grid.')
            arrays = {}
            for index, left in enumerate(tiles):
                for right in tiles[index + 1:]:
                    ix1 = max(left['x1'], right['x1'])
                    iy1 = max(left['y1'], right['y1'])
                    ix2 = min(left['x2'], right['x2'])
                    iy2 = min(left['y2'], right['y2'])
                    if ix1 >= ix2 or iy1 >= iy2:
                        continue
                    for tile in (left, right):
                        if tile['path'] not in arrays:
                            arrays[tile['path']] = np.asarray(
                                Image.open(tile['path']).convert('RGB'))
                            expected_shape = (
                                tile['y2'] - tile['y1'],
                                tile['x2'] - tile['x1'])
                            if arrays[tile['path']].shape[:2] != expected_shape:
                                raise ValueError(
                                    'Tile size disagrees with coordinates: '
                                    f'{tile["path"]}.')
                    a = arrays[left['path']][
                        iy1 - left['y1']:iy2 - left['y1'],
                        ix1 - left['x1']:ix2 - left['x1']]
                    b = arrays[right['path']][
                        iy1 - right['y1']:iy2 - right['y1'],
                        ix1 - right['x1']:ix2 - right['x1']]
                    if not np.array_equal(a, b):
                        raise ValueError(
                            f'{prefix}: RGB mismatch between '
                            f'{left["path"]} and {right["path"]}.')
                    checked_pairs += 1
                    checked_pixels += int((ix2 - ix1) * (iy2 - iy1))
        return dict(
            directory=self.directory,
            source_groups=len(self.groups),
            tiles=sum(len(value) for value in self.groups.values()),
            overlap_pairs=checked_pairs,
            overlap_pixels=checked_pixels,
            rgb_exact=True,
        )
