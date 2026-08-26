#!/usr/bin/env python3
"""Plan a safe tail-only resume for an interrupted iSAID screen."""

import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def ordered_images(image_dir):
    root = os.path.abspath(image_dir)
    paths = []
    for directory, _, names in os.walk(root):
        for name in sorted(names):
            if name.lower().endswith('.png'):
                paths.append(os.path.abspath(os.path.join(directory, name)))
    return sorted(paths)


def completed_images(patterns):
    paths = sorted(set(
        path for pattern in patterns
        for path in (glob.glob(pattern) or [pattern])
        if os.path.isfile(path)))
    completed = set()
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f'{path}:{line_number}: incomplete JSONL record; '
                        'remove only this final partial line before resuming.') \
                        from error
                if str(record.get('dataset_name', '')).lower() != 'isaid':
                    continue
                image_path = record.get('img_path')
                if image_path:
                    completed.add(os.path.abspath(str(image_path)))
    return completed, paths


def plan_resume(images, completed):
    if not images:
        raise ValueError('No iSAID PNG images were found.')
    expected = set(images)
    unknown = sorted(completed - expected)
    if unknown:
        raise ValueError(
            f'{len(unknown)} logged image paths are outside the current '
            f'iSAID image directory; first example: {unknown[0]}')
    first_missing = next(
        (index for index, path in enumerate(images) if path not in completed),
        len(images))
    remaining = len(images) - first_missing
    overlap = sum(path in completed for path in images[first_missing:])
    return dict(
        total_images=len(images),
        completed_unique=len(completed),
        first_missing_index=first_missing,
        first_missing_path=(images[first_missing]
                            if first_missing < len(images) else None),
        resume_tail_images=remaining,
        dataset_indices=(-remaining if remaining else None),
        expected_overlap=overlap,
        complete=(remaining == 0),
    )


def main():
    args = parse_args()
    images = ordered_images(args.image_dir)
    completed, inputs = completed_images(args.inputs)
    result = plan_resume(images, completed)
    result['input_files'] = inputs
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
