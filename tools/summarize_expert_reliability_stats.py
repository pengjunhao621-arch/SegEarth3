import argparse
import csv
import glob
import json
import os
from collections import defaultdict


SUM_KEYS = [
    'valid_pixels',
    'expert_pixels',
    'pair_error_pixels',
    'pair_error_expert_pixels',
    'gt_target_pixels',
    'gt_competitor_pixels',
    'gt_other_pixels',
    'counterfactual_net_pixels',
    'scale_gate_pixels',
    'scale_candidate_pixels',
    'scale_target_in_topk_pixels',
    'scale_scaled_choice_is_target_pixels',
    'final_top1_target_pixels',
    'final_top1_competitor_pixels',
    'semantic_top1_target_pixels',
    'semantic_top1_competitor_pixels',
    'instance_top1_target_pixels',
    'instance_top1_competitor_pixels',
]

MEAN_KEYS = [
    'mean_final_margin',
    'mean_semantic_margin',
    'mean_instance_margin',
    'mean_local_consistency',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize expert reliability JSONL diagnostics.')
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
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_div(num, den):
    return None if den == 0 else num / den


def new_bucket():
    return defaultdict(float)


def add_row(bucket, row):
    bucket['rows'] += 1
    for key in SUM_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    weight = row.get('expert_pixels') or 0
    if isinstance(weight, (int, float)) and weight > 0:
        for key in MEAN_KEYS:
            value = row.get(key)
            if isinstance(value, (int, float)):
                bucket[f'{key}_weighted_sum'] += value * weight
                bucket[f'{key}_weight'] += weight
    for key in ['target_seed_pixels', 'competitor_seed_pixels']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] = max(bucket.get(key, 0), value)


def finalize(meta, bucket):
    out = dict(meta)
    out['rows'] = int(bucket.get('rows', 0))
    for key in SUM_KEYS:
        out[key] = int(bucket.get(key, 0))
    expert_pixels = bucket.get('expert_pixels', 0)
    decided = (
        bucket.get('gt_target_pixels', 0)
        + bucket.get('gt_competitor_pixels', 0))
    out['gt_target_ratio'] = safe_div(
        bucket.get('gt_target_pixels', 0), expert_pixels)
    out['gt_competitor_ratio'] = safe_div(
        bucket.get('gt_competitor_pixels', 0), expert_pixels)
    out['gt_other_ratio'] = safe_div(
        bucket.get('gt_other_pixels', 0), expert_pixels)
    out['counterfactual_precision'] = safe_div(
        bucket.get('gt_target_pixels', 0), decided)
    out['counterfactual_net_ratio'] = safe_div(
        bucket.get('counterfactual_net_pixels', 0), expert_pixels)
    out['pair_error_expert_coverage'] = safe_div(
        bucket.get('pair_error_expert_pixels', 0),
        bucket.get('pair_error_pixels', 0))
    out['target_seed_pixels'] = int(bucket.get('target_seed_pixels', 0))
    out['competitor_seed_pixels'] = int(bucket.get('competitor_seed_pixels', 0))
    for key in MEAN_KEYS:
        out[key] = safe_div(
            bucket.get(f'{key}_weighted_sum', 0),
            bucket.get(f'{key}_weight', 0))
    return out


def summarize(records):
    expert_buckets = {}
    pair_buckets = {}
    for record in records:
        stats = record.get('expert_reliability_stats') or {}
        dataset = record.get('dataset_name') or stats.get('dataset_name') or ''
        for row in stats.get('pair_expert_stats') or []:
            target = row.get('target_class_name')
            competitor = row.get('competitor_class_name')
            expert = row.get('expert_name')
            common = dict(
                dataset_name=dataset,
                target_class_name=target,
                competitor_class_name=competitor,
                requested_scale=row.get('requested_scale'),
                gate_name=row.get('gate_name'),
                drop_threshold=row.get('drop_threshold'),
                pair_mode=row.get('pair_mode'),
                query_mode=row.get('query_mode'),
            )
            expert_key = (
                dataset, target, competitor, expert,
                row.get('requested_scale'), row.get('gate_name'),
                row.get('drop_threshold'), row.get('pair_mode'),
                row.get('query_mode'))
            expert_meta = dict(common)
            expert_meta['expert_name'] = expert
            expert_buckets.setdefault(expert_key, (expert_meta, new_bucket()))
            add_row(expert_buckets[expert_key][1], row)

            pair_key = (
                dataset, target, competitor, row.get('requested_scale'),
                row.get('gate_name'), row.get('drop_threshold'),
                row.get('pair_mode'), row.get('query_mode'))
            pair_buckets.setdefault(pair_key, (common, new_bucket()))
            if expert == 'scale_gate':
                add_row(pair_buckets[pair_key][1], row)

    expert_rows = [
        finalize(meta, bucket) for meta, bucket in expert_buckets.values()
    ]
    pair_rows = [
        finalize(meta, bucket) for meta, bucket in pair_buckets.values()
    ]
    return expert_rows, pair_rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w') as f:
            f.write('')
        return
    preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'expert_name',
        'expert_pixels',
        'gt_target_ratio',
        'gt_competitor_ratio',
        'gt_other_ratio',
        'counterfactual_precision',
        'counterfactual_net_pixels',
        'counterfactual_net_ratio',
        'pair_error_expert_coverage',
        'mean_final_margin',
        'mean_semantic_margin',
        'mean_instance_margin',
        'mean_local_consistency',
        'target_seed_pixels',
        'competitor_seed_pixels',
        'requested_scale',
        'gate_name',
        'drop_threshold',
        'pair_mode',
        'query_mode',
    ]
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
    expert_rows, pair_rows = summarize(iter_records(paths))
    expert_rows.sort(key=lambda item: (
        str(item.get('dataset_name')),
        str(item.get('target_class_name')),
        str(item.get('competitor_class_name')),
        -float(item.get('counterfactual_net_pixels') or 0),
    ))
    pair_rows.sort(key=lambda item: (
        str(item.get('dataset_name')),
        -float(item.get('counterfactual_net_pixels') or 0),
    ))
    write_csv(os.path.join(args.out_dir, 'expert_summary.csv'), expert_rows)
    write_csv(os.path.join(args.out_dir, 'pair_summary.csv'), pair_rows)


if __name__ == '__main__':
    main()
