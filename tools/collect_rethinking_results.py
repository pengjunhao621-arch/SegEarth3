#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect exact MMEngine metrics for reviewer variants.')
    parser.add_argument('--root', required=True)
    parser.add_argument(
        '--eval-root',
        default=None,
        help='Optional eval directory. Defaults to ROOT/eval.')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--datasets', nargs='+',
        default=(
            'udd5',
            'vdd',
            'vaihingen',
            'openearthmap',
            'loveda',
            'potsdam',
        ),
    )
    return parser.parse_args()


def parse_results(path):
    metrics = {}
    if not path.is_file():
        return metrics
    with open(path, encoding='utf-8') as file:
        for line in file:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            try:
                metrics[key] = float(value.strip())
            except ValueError:
                continue
    return metrics


def metric_value(metrics, suffix):
    if suffix in metrics:
        return metrics[suffix]
    matches = [
        value for key, value in metrics.items()
        if key.endswith(suffix)
    ]
    return matches[-1] if matches else None


def main():
    args = parse_args()
    root = Path(args.root)
    eval_root = Path(args.eval_root) if args.eval_root else root / 'eval'
    datasets = tuple(args.datasets)
    variants = (
        'baseline',
        'output_only',
        'three_head',
        'internal_trajectory',
    )
    baseline = {}
    parsed = {}
    for variant in variants:
        for dataset in datasets:
            metrics = parse_results(
                eval_root / variant / dataset / 'results.txt')
            parsed[(variant, dataset)] = metrics
            if variant == 'baseline':
                baseline[dataset] = metric_value(metrics, 'mIoU')

    rows = []
    for variant in variants:
        for dataset in datasets:
            metrics = parsed[(variant, dataset)]
            miou = metric_value(metrics, 'mIoU')
            rows.append({
                'dataset': dataset,
                'variant': variant,
                'mIoU': miou,
                'delta_mIoU': (
                    None
                    if miou is None or baseline.get(dataset) is None
                    else miou - baseline[dataset]
                ),
                'mAcc': metric_value(metrics, 'mAcc'),
                'aAcc': metric_value(metrics, 'aAcc'),
                'results_path': str(
                    eval_root / variant / dataset / 'results.txt'),
            })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(
            args.output, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output}.')


if __name__ == '__main__':
    main()
