import argparse
import csv
import glob
import json
import os
from collections import defaultdict


SUM_KEYS = [
    'pair_pixels',
    'base_aggregate_margin_sum',
    'base_direct_margin_sum',
    'probe_margin_sum',
    'margin_gain_sum',
    'target_drop_sum',
    'competitor_drop_sum',
    'relative_stability_sum',
    'base_target_score_sum',
    'base_competitor_score_sum',
    'probe_target_score_sum',
    'probe_competitor_score_sum',
    'base_target_wins_pixels',
    'probe_target_wins_pixels',
]

SUPPORT_KEYS = [
    'base_target_support_pixels',
    'base_competitor_support_pixels',
    'probe_target_support_pixels',
    'probe_competitor_support_pixels',
    'probe_target_only_support_pixels',
    'probe_competitor_only_support_pixels',
    'probe_both_support_pixels',
    'probe_neither_support_pixels',
    'target_gain_support_pixels',
    'competitor_lost_support_pixels',
]

PRESENCE_KEYS = [
    'target_presence_base',
    'competitor_presence_base',
    'target_presence_probe',
    'competitor_presence_probe',
    'target_presence_drop',
    'competitor_presence_drop',
    'presence_relative_stability',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SAM3 evidence-bias JSONL diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(set(paths))


def iter_records(paths):
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_div(num, den):
    return None if den == 0 else num / den


def new_bucket():
    bucket = defaultdict(float)
    bucket['rows'] = 0
    return bucket


def add_row(bucket, row, include_support=False):
    bucket['rows'] += 1
    for key in SUM_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    if include_support:
        for key in SUPPORT_KEYS:
            value = row.get(key)
            if isinstance(value, (int, float)):
                bucket[key] += value
    weight = row.get('pair_pixels') or 0
    if isinstance(weight, (int, float)) and weight > 0:
        for key in PRESENCE_KEYS:
            value = row.get(key)
            if isinstance(value, (int, float)):
                bucket[f'{key}_weighted_sum'] += value * weight
                bucket[f'{key}_weight'] += weight


def finalize(meta, bucket, include_support=False):
    pixels = bucket.get('pair_pixels', 0)
    out = dict(meta)
    out['rows'] = int(bucket.get('rows', 0))
    out['pair_pixels'] = int(pixels)
    for key in SUM_KEYS:
        if key.endswith('_sum'):
            out[key] = bucket.get(key, 0.0)
        elif key != 'pair_pixels':
            out[key] = int(bucket.get(key, 0))
    out['mean_base_aggregate_margin'] = safe_div(
        bucket.get('base_aggregate_margin_sum', 0.0), pixels)
    out['mean_base_direct_margin'] = safe_div(
        bucket.get('base_direct_margin_sum', 0.0), pixels)
    out['mean_probe_margin'] = safe_div(
        bucket.get('probe_margin_sum', 0.0), pixels)
    out['mean_margin_gain'] = safe_div(
        bucket.get('margin_gain_sum', 0.0), pixels)
    out['mean_target_drop'] = safe_div(
        bucket.get('target_drop_sum', 0.0), pixels)
    out['mean_competitor_drop'] = safe_div(
        bucket.get('competitor_drop_sum', 0.0), pixels)
    out['mean_relative_stability'] = safe_div(
        bucket.get('relative_stability_sum', 0.0), pixels)
    out['mean_base_target_score'] = safe_div(
        bucket.get('base_target_score_sum', 0.0), pixels)
    out['mean_base_competitor_score'] = safe_div(
        bucket.get('base_competitor_score_sum', 0.0), pixels)
    out['mean_probe_target_score'] = safe_div(
        bucket.get('probe_target_score_sum', 0.0), pixels)
    out['mean_probe_competitor_score'] = safe_div(
        bucket.get('probe_competitor_score_sum', 0.0), pixels)
    out['base_target_wins_ratio'] = safe_div(
        bucket.get('base_target_wins_pixels', 0), pixels)
    out['probe_target_wins_ratio'] = safe_div(
        bucket.get('probe_target_wins_pixels', 0), pixels)
    for key in PRESENCE_KEYS:
        out[f'mean_{key}'] = safe_div(
            bucket.get(f'{key}_weighted_sum', 0.0),
            bucket.get(f'{key}_weight', 0.0))

    if include_support:
        for key in SUPPORT_KEYS:
            out[key] = int(bucket.get(key, 0))
        out['base_target_support_ratio'] = safe_div(
            bucket.get('base_target_support_pixels', 0), pixels)
        out['base_competitor_support_ratio'] = safe_div(
            bucket.get('base_competitor_support_pixels', 0), pixels)
        out['probe_target_support_ratio'] = safe_div(
            bucket.get('probe_target_support_pixels', 0), pixels)
        out['probe_competitor_support_ratio'] = safe_div(
            bucket.get('probe_competitor_support_pixels', 0), pixels)
        out['probe_target_only_support_ratio'] = safe_div(
            bucket.get('probe_target_only_support_pixels', 0), pixels)
        out['probe_competitor_only_support_ratio'] = safe_div(
            bucket.get('probe_competitor_only_support_pixels', 0), pixels)
        out['target_gain_support_ratio'] = safe_div(
            bucket.get('target_gain_support_pixels', 0), pixels)
        out['competitor_lost_support_ratio'] = safe_div(
            bucket.get('competitor_lost_support_pixels', 0), pixels)
    return out


def summarize(records):
    pair_buckets = {}
    probe_buckets = {}
    support_buckets = {}
    for record in records:
        stats = record.get('evidence_bias_stats') or {}
        dataset = record.get('dataset_name') or stats.get('dataset_name') or ''
        for row in stats.get('pair_stats') or []:
            support_threshold = row.get('support_threshold')
            common = dict(
                dataset_name=dataset,
                target_class_name=row.get('target_class_name'),
                competitor_class_name=row.get('competitor_class_name'),
                probe_name=row.get('probe_name'),
                head=row.get('head'),
                pair_mode=row.get('pair_mode'),
            )
            pair_key = (
                dataset,
                row.get('target_class_name'),
                row.get('competitor_class_name'),
                row.get('probe_name'),
                row.get('head'),
                row.get('pair_mode'),
            )
            probe_key = (
                dataset,
                row.get('probe_name'),
                row.get('head'),
                row.get('pair_mode'),
            )
            if support_threshold is None:
                pair_buckets.setdefault(pair_key, (common, new_bucket()))
                add_row(pair_buckets[pair_key][1], row)
                probe_meta = dict(
                    dataset_name=dataset,
                    probe_name=row.get('probe_name'),
                    head=row.get('head'),
                    pair_mode=row.get('pair_mode'),
                )
                probe_buckets.setdefault(probe_key, (probe_meta, new_bucket()))
                add_row(probe_buckets[probe_key][1], row)
            else:
                support_meta = dict(common)
                support_meta['support_threshold'] = support_threshold
                support_key = pair_key + (support_threshold,)
                support_buckets.setdefault(
                    support_key, (support_meta, new_bucket()))
                add_row(support_buckets[support_key][1], row, include_support=True)

    pair_rows = [
        finalize(meta, bucket) for meta, bucket in pair_buckets.values()
    ]
    probe_rows = [
        finalize(meta, bucket) for meta, bucket in probe_buckets.values()
    ]
    support_rows = [
        finalize(meta, bucket, include_support=True)
        for meta, bucket in support_buckets.values()
    ]
    return pair_rows, probe_rows, support_rows


def write_csv(path, rows, preferred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    keys = []
    for key in preferred:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    pair_rows, probe_rows, support_rows = summarize(iter_records(paths))
    pair_rows.sort(key=lambda row: (
        str(row.get('dataset_name')),
        str(row.get('target_class_name')),
        str(row.get('competitor_class_name')),
        str(row.get('head')),
        -float(row.get('mean_relative_stability') or 0),
    ))
    probe_rows.sort(key=lambda row: (
        str(row.get('dataset_name')),
        str(row.get('probe_name')),
        str(row.get('head')),
    ))
    support_rows.sort(key=lambda row: (
        str(row.get('dataset_name')),
        str(row.get('target_class_name')),
        str(row.get('competitor_class_name')),
        str(row.get('probe_name')),
        str(row.get('head')),
        float(row.get('support_threshold') or 0),
    ))

    preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'probe_name',
        'head',
        'support_threshold',
        'pair_pixels',
        'mean_base_aggregate_margin',
        'mean_base_direct_margin',
        'mean_probe_margin',
        'mean_margin_gain',
        'mean_target_drop',
        'mean_competitor_drop',
        'mean_relative_stability',
        'probe_target_wins_ratio',
        'base_target_wins_ratio',
        'mean_presence_relative_stability',
        'probe_target_support_ratio',
        'probe_competitor_support_ratio',
        'target_gain_support_ratio',
        'competitor_lost_support_ratio',
        'pair_mode',
        'rows',
    ]
    write_csv(
        os.path.join(args.out_dir, 'bias_pair_summary.csv'),
        pair_rows,
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'bias_probe_summary.csv'),
        probe_rows,
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'bias_support_summary.csv'),
        support_rows,
        preferred)


if __name__ == '__main__':
    main()
