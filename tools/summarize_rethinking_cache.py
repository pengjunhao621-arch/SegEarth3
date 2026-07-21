#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rethinking_reviewer import REVIEWER_SOURCE_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize reviewer cache coverage and availability.')
    parser.add_argument('--cache', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    args = parse_args()
    buckets = defaultdict(lambda: {
        'shards': 0,
        'samples': 0,
        'wrong': 0,
        'recoverable': 0,
        'unrecoverable': 0,
        'hard_keep': 0,
        'weighted_samples': 0.0,
        'weighted_wrong': 0.0,
        'weighted_recoverable': 0.0,
        'source_available': [0] * len(REVIEWER_SOURCE_NAMES),
        'source_total': [0] * len(REVIEWER_SOURCE_NAMES),
    })
    files = []
    for root in args.cache:
        files.extend(Path(root).rglob('*.pt'))
    for path in sorted(set(files)):
        record = torch.load(path, map_location='cpu')
        dataset = record.get('dataset_name') or 'unknown'
        bucket = buckets[dataset]
        risk = record['risk_target'] > 0.5
        recoverable = record['recoverable_target'] > 0.5
        hard_keep = record.get('hard_keep_target')
        if hard_keep is None:
            hard_keep = torch.zeros_like(risk, dtype=torch.bool)
        else:
            hard_keep = hard_keep.bool()
        sample_weight = record.get(
            'sample_weight', torch.ones_like(risk, dtype=torch.float32))
        source_mask = record['source_mask'].bool()
        bucket['shards'] += 1
        bucket['samples'] += int(risk.numel())
        bucket['wrong'] += int(risk.sum().item())
        bucket['recoverable'] += int(recoverable.sum().item())
        bucket['unrecoverable'] += int(
            (risk & ~recoverable).sum().item())
        bucket['hard_keep'] += int(hard_keep.sum().item())
        bucket['weighted_samples'] += float(sample_weight.sum().item())
        bucket['weighted_wrong'] += float(
            sample_weight[risk].sum().item())
        bucket['weighted_recoverable'] += float(
            sample_weight[recoverable].sum().item())
        available = source_mask.any(dim=1)
        bucket['source_available'] = [
            old + int(value)
            for old, value in zip(
                bucket['source_available'],
                available.sum(dim=0).tolist(),
            )
        ]
        bucket['source_total'] = [
            old + int(available.shape[0])
            for old in bucket['source_total']
        ]

    rows = []
    for dataset, bucket in sorted(buckets.items()):
        row = {
            'dataset': dataset,
            'shards': bucket['shards'],
            'samples': bucket['samples'],
            'wrong_samples': bucket['wrong'],
            'wrong_ratio': safe_div(
                bucket['wrong'], bucket['samples']),
            'recoverable_wrong_samples': bucket['recoverable'],
            'recoverable_given_wrong': safe_div(
                bucket['recoverable'], bucket['wrong']),
            'unrecoverable_wrong_samples': bucket['unrecoverable'],
            'hard_keep_samples': bucket['hard_keep'],
            'hard_keep_ratio': safe_div(
                bucket['hard_keep'], bucket['samples']),
            'estimated_full_pixels': bucket['weighted_samples'],
            'estimated_wrong_ratio': safe_div(
                bucket['weighted_wrong'], bucket['weighted_samples']),
            'estimated_recoverable_given_wrong': safe_div(
                bucket['weighted_recoverable'],
                bucket['weighted_wrong'],
            ),
        }
        for source_id, source_name in enumerate(
                REVIEWER_SOURCE_NAMES):
            row[f'{source_name}_availability'] = safe_div(
                bucket['source_available'][source_id],
                bucket['source_total'][source_id],
            )
        rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ['dataset']
    with open(
            args.output, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output}.')


if __name__ == '__main__':
    main()
