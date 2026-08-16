#!/usr/bin/env python3
"""Read-only RGB overlap check for virtual context reconstruction."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tiled_context import CoordinateTileIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dirs', nargs='+')
    parser.add_argument('--output')
    args = parser.parse_args()
    results = [CoordinateTileIndex(path).verify_rgb_overlaps()
               for path in args.image_dirs]
    payload = {'schema_version': 1, 'results': results, 'all_passed': True}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
