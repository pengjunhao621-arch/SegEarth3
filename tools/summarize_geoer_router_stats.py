import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize GeoER router JSONL diagnostics.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/geoer_router/vdd/*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for geoer_router_*_summary.csv files.')
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


SUM_KEYS = [
    'pixels',
    'valid_pixels',
    'changed_pixels',
    'improved_pixels',
    'harmed_pixels',
    'net_improved_pixels',
    'base_correct_pixels',
    'routed_correct_pixels',
    'threshold_reject_pixels',
    'gate_pixels',
    'candidate_pixels',
]


MEAN_SUM_KEYS = [
    'gate_base_margin_sum',
    'gate_scaled_margin_sum',
    'gate_margin_gain_sum',
]


MEAN_KEYS = [
    'mean_top1_score',
    'mean_final_margin',
    'mean_top1_semantic',
    'mean_top1_instance',
    'mean_semantic_bg_margin',
    'mean_instance_bg_margin',
    'mean_local_consistency',
]


def new_bucket():
    bucket = defaultdict(float)
    bucket['rows'] = 0
    return bucket


def add_row(bucket, row):
    bucket['rows'] += 1
    for key in SUM_KEYS + MEAN_SUM_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    for key in MEAN_KEYS:
        value = row.get(key)
        pixels = row.get('pixels') or row.get('gate_pixels') or 0
        if isinstance(value, (int, float)) and pixels:
            bucket[f'{key}_weighted_sum'] += value * pixels
            bucket[f'{key}_weight'] += pixels


def finalize(meta, bucket):
    pixels = bucket.get('pixels', 0)
    valid = bucket.get('valid_pixels', 0) or pixels
    out = dict(meta)
    out['rows'] = int(bucket.get('rows', 0))
    for key in SUM_KEYS:
        if key in bucket:
            out[key] = int(bucket.get(key, 0))
    out['changed_ratio'] = safe_div(bucket.get('changed_pixels', 0), pixels)
    out['improved_ratio'] = safe_div(bucket.get('improved_pixels', 0), pixels)
    out['harmed_ratio'] = safe_div(bucket.get('harmed_pixels', 0), pixels)
    out['net_improved_ratio'] = safe_div(
        bucket.get('net_improved_pixels', 0), pixels)
    out['base_accuracy'] = safe_div(
        bucket.get('base_correct_pixels', 0), valid)
    out['routed_accuracy'] = safe_div(
        bucket.get('routed_correct_pixels', 0), valid)
    out['accuracy_delta'] = safe_div(
        bucket.get('net_improved_pixels', 0), valid)
    for src, dst in [
            ('gate_base_margin_sum', 'mean_gate_base_margin'),
            ('gate_scaled_margin_sum', 'mean_gate_scaled_margin'),
            ('gate_margin_gain_sum', 'mean_gate_margin_gain')]:
        out[dst] = safe_div(bucket.get(src, 0), pixels)
    for key in MEAN_KEYS:
        out[key] = safe_div(
            bucket.get(f'{key}_weighted_sum', 0),
            bucket.get(f'{key}_weight', 0))
    return out


def write_csv(path, rows, preferred=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    keys = []
    for key in preferred or []:
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


def summarize(records):
    overall_buckets = {}
    route_buckets = {}
    pair_buckets = {}
    for record in records:
        stats = record.get('geoer_router_stats') or {}
        dataset = record.get('dataset_name') or stats.get('dataset_name') or ''
        scale_on = bool(record.get('geoer_use_scale_stability'))
        reject_on = bool(record.get('geoer_use_reject_recovery'))
        use_router = bool(record.get('use_geoer_router'))
        variant = (
            'combined' if scale_on and reject_on else
            'scale_only' if scale_on else
            'reject_only' if reject_on else
            'disabled')
        meta_base = dict(
            dataset_name=dataset,
            variant=variant,
            use_geoer_router=use_router,
            geoer_use_scale_stability=scale_on,
            geoer_use_reject_recovery=reject_on,
            scale_stability_rerank_pairs=record.get(
                'scale_stability_rerank_pairs'),
            scale_stability_rerank_scale=record.get(
                'scale_stability_rerank_scale'),
            scale_stability_rerank_gate=record.get(
                'scale_stability_rerank_gate'),
            scale_stability_rerank_drop_threshold=record.get(
                'scale_stability_rerank_drop_threshold'),
            geoer_reject_recovery_score_thd=record.get(
                'geoer_reject_recovery_score_thd'),
            geoer_reject_recovery_local_thd=record.get(
                'geoer_reject_recovery_local_thd'),
        )

        overall = stats.get('overall') or {}
        overall_key = (dataset, variant, use_router)
        overall_buckets.setdefault(overall_key, (dict(meta_base), new_bucket()))
        add_row(overall_buckets[overall_key][1], overall)

        for row in stats.get('route_stats') or []:
            route_name = row.get('route_name')
            route_key = (dataset, variant, use_router, route_name)
            route_meta = dict(meta_base)
            route_meta.update(dict(
                route_name=route_name,
                route_type=row.get('route_type'),
                target_class_name=row.get('target_class_name'),
                competitor_class_name=row.get('competitor_class_name'),
            ))
            route_buckets.setdefault(route_key, (route_meta, new_bucket()))
            add_row(route_buckets[route_key][1], row)

        for row in stats.get('pair_stats') or []:
            pair_key = (
                dataset,
                variant,
                use_router,
                row.get('gt_class_name'),
                row.get('base_pred_class_name'),
            )
            pair_meta = dict(meta_base)
            pair_meta.update(dict(
                gt_class_name=row.get('gt_class_name'),
                base_pred_class_name=row.get('base_pred_class_name'),
            ))
            pair_buckets.setdefault(pair_key, (pair_meta, new_bucket()))
            add_row(pair_buckets[pair_key][1], row)

    overall_rows = [
        finalize(meta, bucket)
        for meta, bucket in overall_buckets.values()
    ]
    route_rows = [
        finalize(meta, bucket)
        for meta, bucket in route_buckets.values()
    ]
    pair_rows = [
        finalize(meta, bucket)
        for meta, bucket in pair_buckets.values()
    ]
    return overall_rows, route_rows, pair_rows


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    overall_rows, route_rows, pair_rows = summarize(iter_records(paths))
    common = [
        'dataset_name',
        'variant',
        'use_geoer_router',
        'geoer_use_scale_stability',
        'geoer_use_reject_recovery',
        'rows',
        'pixels',
        'base_accuracy',
        'routed_accuracy',
        'accuracy_delta',
        'changed_pixels',
        'changed_ratio',
        'improved_pixels',
        'harmed_pixels',
        'net_improved_pixels',
        'net_improved_ratio',
    ]
    write_csv(
        os.path.join(args.out_dir, 'geoer_router_overall_summary.csv'),
        sorted(overall_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('variant')),
        )),
        common)
    write_csv(
        os.path.join(args.out_dir, 'geoer_router_route_summary.csv'),
        sorted(route_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('variant')),
            str(item.get('route_name')),
        )),
        common + [
            'route_name',
            'route_type',
            'target_class_name',
            'competitor_class_name',
            'candidate_pixels',
            'gate_pixels',
            'threshold_reject_pixels',
            'mean_gate_base_margin',
            'mean_gate_scaled_margin',
            'mean_gate_margin_gain',
            'mean_top1_score',
            'mean_semantic_bg_margin',
            'mean_instance_bg_margin',
            'mean_local_consistency',
        ])
    write_csv(
        os.path.join(args.out_dir, 'geoer_router_pair_summary.csv'),
        sorted(pair_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('variant')),
            -int(item.get('pixels') or 0),
        )),
        common + [
            'gt_class_name',
            'base_pred_class_name',
        ])


if __name__ == '__main__':
    main()
