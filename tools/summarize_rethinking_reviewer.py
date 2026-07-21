#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize learned reviewer prediction changes.')
    parser.add_argument(
        '--inputs', nargs='+', required=True,
        help='JSONL paths or glob patterns.')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    args = parse_args()
    paths = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        paths.extend(matched or [pattern])
    buckets = defaultdict(lambda: defaultdict(float))
    for path in sorted(set(paths)):
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (
                    record.get('dataset_name') or 'unknown',
                    record.get('reviewer_variant') or 'unknown',
                    record.get('reviewer_checkpoint') or '',
                )
                bucket = buckets[key]
                bucket['images'] += 1
                for field in (
                        'valid_pixels',
                        'changed_pixels',
                        'improved_pixels',
                        'harmed_pixels',
                        'wrong_to_wrong_pixels'):
                    bucket[field] += int(record.get(field) or 0)
                context = record.get('reviewer_context') or {}
                for field in (
                        'grid_pixels',
                        'changed_grid_pixels'):
                    bucket[field] += int(context.get(field) or 0)
                bucket['mean_risk_sum'] += float(
                    context.get('mean_risk') or 0.0)
                bucket['mean_recoverable_sum'] += float(
                    context.get('mean_recoverable') or 0.0)
                if bucket.get('gate_threshold') is None:
                    bucket['gate_threshold'] = context.get(
                        'gate_threshold')
                    bucket['action_margin'] = context.get(
                        'action_margin')

    rows = []
    for (dataset, variant, checkpoint), bucket in sorted(
            buckets.items()):
        images = int(bucket['images'])
        changed = int(bucket['changed_pixels'])
        improved = int(bucket['improved_pixels'])
        harmed = int(bucket['harmed_pixels'])
        rows.append({
            'dataset': dataset,
            'variant': variant,
            'checkpoint': checkpoint,
            'images': images,
            'valid_pixels': int(bucket['valid_pixels']),
            'changed_pixels': changed,
            'changed_ratio': safe_div(
                changed, int(bucket['valid_pixels'])),
            'improved_pixels': improved,
            'harmed_pixels': harmed,
            'wrong_to_wrong_pixels': int(
                bucket['wrong_to_wrong_pixels']),
            'net_correct_pixels': improved - harmed,
            'correction_precision': safe_div(
                improved, improved + harmed),
            'harmful_change_rate': safe_div(harmed, changed),
            'grid_changed_ratio': safe_div(
                int(bucket['changed_grid_pixels']),
                int(bucket['grid_pixels'])),
            'mean_risk': safe_div(
                bucket['mean_risk_sum'], images),
            'mean_recoverable': safe_div(
                bucket['mean_recoverable_sum'], images),
            'gate_threshold': bucket.get('gate_threshold'),
            'action_margin': bucket.get('action_margin'),
        })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        'dataset', 'variant', 'checkpoint', 'images']
    with open(
            args.output, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output}.')


if __name__ == '__main__':
    main()
