import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize weak-support evidence sweep JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/weak_support/vdd/ws_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for weak_support_pair_summary.csv and related tables.')
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


def first(values):
    for value in values:
        if value not in (None, ''):
            return value
    return None


def class_tokens(name):
    tokens = set()
    raw = (name or '').lower().replace('/', ',').replace('_', ',')
    for item in raw.split(','):
        item = item.strip()
        if item:
            tokens.add(item)
    if name:
        tokens.add(name.strip().lower())
    return tokens


def infer_role(name):
    tokens = class_tokens(name)
    if tokens & {'background', 'other', 'clutter'}:
        return 'reject_background'
    if tokens & {'vehicle', 'car'}:
        return 'object_vehicle'
    if tokens & {'building', 'house'}:
        return 'object_building'
    if tokens & {'roof'}:
        return 'built_horizontal_surface'
    if tokens & {'facade', 'wall'}:
        return 'built_vertical_surface'
    if tokens & {'road', 'pavement', 'impervious', 'impervious surface'}:
        return 'transport_surface'
    if tokens & {'grass', 'low vegetation', 'low_vegetation', 'agricultural', 'agriculture'}:
        return 'low_vegetation_material'
    if tokens & {'tree', 'forest'}:
        return 'high_vegetation_canopy'
    if tokens & {'vegetation'}:
        return 'vegetation_generic'
    if tokens & {'water'}:
        return 'water_surface'
    if tokens & {'bareland', 'barren'}:
        return 'bare_surface'
    return 'unknown'


def focus_pair_name(dataset_name, gt_name, pred_name):
    dataset = (dataset_name or '').lower()
    gt = class_tokens(gt_name)
    pred = class_tokens(pred_name)
    if dataset == 'vdd' and 'roof' in gt and 'facade' in pred:
        return 'VDD roof -> facade'
    if dataset == 'potsdam' and 'tree' in gt and 'grass' in pred:
        return 'Potsdam tree -> grass'
    if dataset == 'vaihingen' and 'grass' in gt and 'tree' in pred:
        return 'Vaihingen grass -> tree'
    if dataset == 'openearthmap' and 'pavement' in gt and 'building' in pred:
        return 'OpenEarthMap pavement -> building'
    if (dataset == 'loveda'
            and bool({'forest', 'agricultural', 'agriculture'} & gt)
            and 'background' in pred):
        return f'LoveDA {gt_name} -> background'
    if dataset == 'udd5' and {'road', 'vegetation', 'building'} & gt and 'background' in pred:
        return f'UDD5 {gt_name} -> background'
    return None


PAIR_COUNT_KEYS = [
    'pixels',
    'gt_support_pixels',
    'pred_support_pixels',
    'both_support_pixels',
    'gt_only_support_pixels',
    'pred_only_support_pixels',
    'neither_support_pixels',
    'gt_score_sum',
    'pred_score_sum',
    'margin_sum',
    'gt_beats_pred_pixels',
]

CLASS_COUNT_KEYS = [
    'gt_pixels',
    'support_pixels',
    'support_correct_pixels',
]


def add_sum(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value


def add_mean(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_count'] += 1


def finalize_means(row):
    keys = [key[:-4] for key in list(row.keys()) if key.endswith('_sum')]
    for key in keys:
        count_key = f'{key}_count'
        sum_key = f'{key}_sum'
        if count_key in row:
            row[key] = safe_div(row.get(sum_key, 0.0), row.get(count_key, 0))
    return row


def finalize_pair(base, counts):
    row = dict(base, **counts)
    pixels = row.get('pixels', 0)
    row['gt_support_ratio'] = safe_div(row.get('gt_support_pixels', 0), pixels)
    row['pred_support_ratio'] = safe_div(row.get('pred_support_pixels', 0), pixels)
    row['both_support_ratio'] = safe_div(row.get('both_support_pixels', 0), pixels)
    row['gt_only_support_ratio'] = safe_div(row.get('gt_only_support_pixels', 0), pixels)
    row['pred_only_support_ratio'] = safe_div(row.get('pred_only_support_pixels', 0), pixels)
    row['neither_support_ratio'] = safe_div(row.get('neither_support_pixels', 0), pixels)
    row['mean_gt_score'] = safe_div(row.get('gt_score_sum', 0.0), pixels)
    row['mean_pred_score'] = safe_div(row.get('pred_score_sum', 0.0), pixels)
    row['mean_margin'] = safe_div(row.get('margin_sum', 0.0), pixels)
    row['gt_beats_pred_ratio'] = safe_div(row.get('gt_beats_pred_pixels', 0), pixels)
    focus = focus_pair_name(
        row.get('dataset_name'),
        row.get('gt_class_name'),
        row.get('base_pred_class_name'))
    row['focus_pair'] = bool(focus)
    row['focus_pair_name'] = focus
    return row


def finalize_class(base, counts):
    row = dict(base, **counts)
    row['support_precision'] = safe_div(
        row.get('support_correct_pixels', 0),
        row.get('support_pixels', 0))
    row['support_recall'] = safe_div(
        row.get('support_correct_pixels', 0),
        row.get('gt_pixels', 0))
    row['support_ratio_over_gt'] = safe_div(
        row.get('support_pixels', 0),
        row.get('gt_pixels', 0))
    return row


def add_pair_row(bucket, row):
    bucket['rows'] += 1
    add_sum(bucket, row, PAIR_COUNT_KEYS)


def add_class_row(bucket, row):
    bucket['rows'] += 1
    add_sum(bucket, row, CLASS_COUNT_KEYS)
    add_mean(bucket, row, ['mean_gt_score', 'support_ratio'])


def summarize(records):
    pair_counts = defaultdict(lambda: defaultdict(float))
    pair_meta = {}
    class_counts = defaultdict(lambda: defaultdict(float))
    class_meta = {}
    curve_counts = defaultdict(lambda: defaultdict(float))
    curve_meta = {}
    role_counts = defaultdict(lambda: defaultdict(float))
    role_meta = {}

    for record in records:
        stats = record.get('weak_support_stats') or {}
        dataset = stats.get('dataset_name') or record.get('dataset_name') or 'unknown'
        min_pair_pixels = first([
            record.get('weak_support_min_pair_pixels'),
            stats.get('weak_support_min_pair_pixels'),
        ])

        for row in stats.get('pair_stats') or []:
            source = row.get('source')
            threshold = row.get('threshold')
            gt_name = row.get('gt_class_name')
            pred_name = row.get('base_pred_class_name')
            gt_role = infer_role(gt_name)
            pred_role = infer_role(pred_name)

            key = (
                dataset, source, threshold,
                row.get('gt_class_index'), row.get('base_pred_class_index'))
            add_pair_row(pair_counts[key], row)
            pair_meta[key] = dict(
                dataset_name=dataset,
                source=source,
                threshold=threshold,
                weak_support_min_pair_pixels=min_pair_pixels,
                gt_class_index=row.get('gt_class_index'),
                gt_class_name=gt_name,
                gt_role=gt_role,
                base_pred_class_index=row.get('base_pred_class_index'),
                base_pred_class_name=pred_name,
                pred_role=pred_role,
            )

            curve_key = (dataset, source, threshold)
            add_pair_row(curve_counts[curve_key], row)
            curve_meta[curve_key] = dict(
                dataset_name=dataset,
                source=source,
                threshold=threshold,
                weak_support_min_pair_pixels=min_pair_pixels,
            )

            role_key = (dataset, source, threshold, gt_role, pred_role)
            add_pair_row(role_counts[role_key], row)
            role_meta[role_key] = dict(
                dataset_name=dataset,
                source=source,
                threshold=threshold,
                weak_support_min_pair_pixels=min_pair_pixels,
                gt_role=gt_role,
                pred_role=pred_role,
            )

        for row in stats.get('class_stats') or []:
            source = row.get('source')
            threshold = row.get('threshold')
            class_name = row.get('class_name')
            key = (dataset, source, threshold, row.get('class_index'))
            add_class_row(class_counts[key], row)
            class_meta[key] = dict(
                dataset_name=dataset,
                source=source,
                threshold=threshold,
                weak_support_min_pair_pixels=min_pair_pixels,
                class_index=row.get('class_index'),
                class_name=class_name,
                role=infer_role(class_name),
            )

    pair_rows = [finalize_pair(pair_meta[key], counts) for key, counts in pair_counts.items()]
    class_rows = [finalize_class(class_meta[key], counts) for key, counts in class_counts.items()]
    curve_rows = [finalize_pair(curve_meta[key], counts) for key, counts in curve_counts.items()]
    role_rows = [finalize_pair(role_meta[key], counts) for key, counts in role_counts.items()]

    pair_rows.sort(key=lambda row: (
        not row.get('focus_pair', False),
        row.get('dataset_name'), row.get('source'), row.get('threshold'),
        -(row.get('pixels') or 0)))
    class_rows.sort(key=lambda row: (
        row.get('dataset_name'), row.get('source'), row.get('threshold'),
        row.get('class_index')))
    curve_rows.sort(key=lambda row: (
        row.get('dataset_name'), row.get('source'), row.get('threshold')))
    role_rows.sort(key=lambda row: (
        row.get('dataset_name'), row.get('source'), row.get('threshold'),
        row.get('gt_role'), row.get('pred_role')))
    return pair_rows, class_rows, curve_rows, role_rows


def write_csv(path, rows, preferred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    fieldnames = [key for key in preferred if key in fieldnames or not rows] + [
        key for key in fieldnames if key not in preferred]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input JSONL files matched: {args.inputs}')
    pair_rows, class_rows, curve_rows, role_rows = summarize(iter_records(paths))

    pair_preferred = [
        'dataset_name', 'source', 'threshold',
        'gt_class_index', 'gt_class_name', 'gt_role',
        'base_pred_class_index', 'base_pred_class_name', 'pred_role',
        'focus_pair', 'focus_pair_name', 'rows', 'pixels',
        'gt_support_ratio', 'pred_support_ratio',
        'gt_only_support_ratio', 'pred_only_support_ratio',
        'both_support_ratio', 'neither_support_ratio',
        'mean_margin', 'mean_gt_score', 'mean_pred_score',
        'gt_beats_pred_ratio',
    ]
    class_preferred = [
        'dataset_name', 'source', 'threshold',
        'class_index', 'class_name', 'role',
        'rows', 'gt_pixels', 'support_pixels', 'support_correct_pixels',
        'support_precision', 'support_recall', 'support_ratio_over_gt',
        'support_ratio', 'mean_gt_score',
    ]
    curve_preferred = [
        'dataset_name', 'source', 'threshold', 'rows', 'pixels',
        'gt_support_ratio', 'pred_support_ratio',
        'gt_only_support_ratio', 'pred_only_support_ratio',
        'both_support_ratio', 'neither_support_ratio',
        'mean_margin', 'gt_beats_pred_ratio',
    ]
    role_preferred = [
        'dataset_name', 'source', 'threshold',
        'gt_role', 'pred_role', 'rows', 'pixels',
        'gt_support_ratio', 'pred_support_ratio',
        'gt_only_support_ratio', 'pred_only_support_ratio',
        'both_support_ratio', 'neither_support_ratio',
        'mean_margin', 'gt_beats_pred_ratio',
    ]

    write_csv(os.path.join(args.out_dir, 'weak_support_pair_summary.csv'), pair_rows, pair_preferred)
    write_csv(os.path.join(args.out_dir, 'weak_support_class_summary.csv'), class_rows, class_preferred)
    write_csv(os.path.join(args.out_dir, 'weak_support_curve.csv'), curve_rows, curve_preferred)
    write_csv(os.path.join(args.out_dir, 'weak_support_role_summary.csv'), role_rows, role_preferred)
    print(f'Wrote {len(pair_rows)} pair rows, {len(class_rows)} class rows, '
          f'{len(curve_rows)} curve rows, and {len(role_rows)} role rows to {args.out_dir}')


if __name__ == '__main__':
    main()
