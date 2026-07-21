import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize GeoER scale-normalized re-query diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/scale_requery/potsdam/sr_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for scale_requery_*_summary.csv files.')
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


def focus_pair_name(dataset_name, gt_name, pred_name):
    dataset = (dataset_name or '').lower()
    gt = class_tokens(gt_name)
    pred = class_tokens(pred_name)
    if dataset == 'vdd' and 'roof' in gt and 'facade' in pred:
        return 'VDD roof -> facade'
    if dataset == 'vdd' and 'roof' in gt and 'background' in pred:
        return 'VDD roof -> background'
    if dataset == 'potsdam' and 'tree' in gt and 'grass' in pred:
        return 'Potsdam tree -> grass'
    if dataset == 'potsdam' and 'grass' in gt and 'clutter' in pred:
        return 'Potsdam grass -> clutter'
    if dataset == 'vaihingen' and 'grass' in gt and 'tree' in pred:
        return 'Vaihingen grass -> tree'
    if dataset == 'vaihingen' and 'grass' in gt and 'clutter' in pred:
        return 'Vaihingen grass -> clutter'
    if dataset == 'openearthmap' and 'pavement' in gt and 'building' in pred:
        return 'OpenEarthMap pavement -> building'
    if (dataset == 'loveda'
            and bool({'forest', 'agricultural', 'agriculture'} & gt)
            and 'background' in pred):
        return f'LoveDA {gt_name} -> background'
    if dataset == 'udd5' and {'road', 'vegetation', 'building'} & gt and 'background' in pred:
        return f'UDD5 {gt_name} -> background'
    return None


COUNT_KEYS = [
    'covered_pair_pixels',
    'full_pair_pixels',
    'base_gt_score_sum',
    'base_pred_score_sum',
    'base_margin_sum',
    'requery_gt_score_sum',
    'requery_pred_score_sum',
    'requery_margin_sum',
    'margin_gain_sum',
    'base_gt_beats_pred_pixels',
    'requery_gt_beats_pred_pixels',
    'gt_support_pixels',
    'pred_support_pixels',
    'gt_only_support_pixels',
    'pred_only_support_pixels',
    'both_support_pixels',
    'neither_support_pixels',
    'gt_semantic_sum',
    'pred_semantic_sum',
    'gt_instance_sum',
    'pred_instance_sum',
]


def new_bucket():
    bucket = defaultdict(float)
    bucket['rows'] = 0
    return bucket


def add_row(bucket, row):
    bucket['rows'] += 1
    for key in COUNT_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    for key in ['effective_scale_x', 'effective_scale_y']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_count'] += 1


def finalize_bucket(meta, bucket):
    pixels = bucket.get('covered_pair_pixels', 0)
    out = dict(meta)
    out['rows'] = int(bucket.get('rows', 0))
    out['covered_pair_pixels'] = int(pixels)
    out['full_pair_pixels_sum'] = int(bucket.get('full_pair_pixels', 0))
    out['mean_effective_scale_x'] = safe_div(
        bucket.get('effective_scale_x_sum', 0),
        bucket.get('effective_scale_x_count', 0))
    out['mean_effective_scale_y'] = safe_div(
        bucket.get('effective_scale_y_sum', 0),
        bucket.get('effective_scale_y_count', 0))

    for src, dst in [
            ('base_gt_score_sum', 'mean_base_gt_score'),
            ('base_pred_score_sum', 'mean_base_pred_score'),
            ('base_margin_sum', 'mean_base_margin'),
            ('requery_gt_score_sum', 'mean_requery_gt_score'),
            ('requery_pred_score_sum', 'mean_requery_pred_score'),
            ('requery_margin_sum', 'mean_requery_margin'),
            ('margin_gain_sum', 'mean_margin_gain'),
            ('gt_semantic_sum', 'mean_gt_semantic'),
            ('pred_semantic_sum', 'mean_pred_semantic'),
            ('gt_instance_sum', 'mean_gt_instance'),
            ('pred_instance_sum', 'mean_pred_instance')]:
        out[dst] = safe_div(bucket.get(src, 0), pixels)

    for src, dst in [
            ('base_gt_beats_pred_pixels', 'base_gt_beats_pred_ratio'),
            ('requery_gt_beats_pred_pixels', 'requery_gt_beats_pred_ratio'),
            ('gt_support_pixels', 'gt_support_ratio'),
            ('pred_support_pixels', 'pred_support_ratio'),
            ('gt_only_support_pixels', 'gt_only_support_ratio'),
            ('pred_only_support_pixels', 'pred_only_support_ratio'),
            ('both_support_pixels', 'both_support_ratio'),
            ('neither_support_pixels', 'neither_support_ratio')]:
        out[dst] = safe_div(bucket.get(src, 0), pixels)
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


def select_best(rows):
    grouped = {}
    for row in rows:
        key = (
            row.get('dataset_name'),
            row.get('gt_class_name'),
            row.get('base_pred_class_name'),
            row.get('threshold'),
        )
        score = (
            row.get('mean_requery_margin') if row.get('mean_requery_margin') is not None else -999,
            row.get('mean_margin_gain') if row.get('mean_margin_gain') is not None else -999,
            row.get('requery_gt_beats_pred_ratio') if row.get('requery_gt_beats_pred_ratio') is not None else -999,
            row.get('gt_only_support_ratio') if row.get('gt_only_support_ratio') is not None else -999,
        )
        if key not in grouped or score > grouped[key][0]:
            grouped[key] = (score, row)
    best = []
    for _, row in grouped.values():
        out = dict(row)
        out['selection_rule'] = 'max_requery_margin_then_gain_beats_gt_only'
        best.append(out)
    return sorted(
        best,
        key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('gt_class_name')),
            str(item.get('base_pred_class_name')),
            float(item.get('threshold') or 0),
        ))


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')

    pair_buckets = {}
    scale_buckets = {}
    focus_buckets = {}

    for record in iter_records(paths):
        stats = record.get('scale_requery_stats') or {}
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or ''
        )
        for row in stats.get('pair_stats') or []:
            gt_name = row.get('gt_class_name')
            pred_name = row.get('base_pred_class_name')
            scale = row.get('requested_scale')
            threshold = row.get('threshold')
            pair_key = (dataset, gt_name, pred_name, scale, threshold)
            focus_name = focus_pair_name(dataset, gt_name, pred_name)
            pair_meta = dict(
                dataset_name=dataset,
                focus_pair=focus_name,
                gt_class_name=gt_name,
                base_pred_class_name=pred_name,
                requested_scale=scale,
                threshold=threshold,
            )
            pair_buckets.setdefault(pair_key, (pair_meta, new_bucket()))
            add_row(pair_buckets[pair_key][1], row)

            scale_key = (dataset, scale, threshold)
            scale_meta = dict(
                dataset_name=dataset,
                requested_scale=scale,
                threshold=threshold,
            )
            scale_buckets.setdefault(scale_key, (scale_meta, new_bucket()))
            add_row(scale_buckets[scale_key][1], row)

            if focus_name:
                focus_key = (dataset, focus_name, scale, threshold)
                focus_meta = dict(
                    dataset_name=dataset,
                    focus_pair=focus_name,
                    gt_class_name=gt_name,
                    base_pred_class_name=pred_name,
                    requested_scale=scale,
                    threshold=threshold,
                )
                focus_buckets.setdefault(focus_key, (focus_meta, new_bucket()))
                add_row(focus_buckets[focus_key][1], row)

    pair_rows = [
        finalize_bucket(meta, bucket)
        for meta, bucket in pair_buckets.values()
    ]
    scale_rows = [
        finalize_bucket(meta, bucket)
        for meta, bucket in scale_buckets.values()
    ]
    focus_rows = [
        finalize_bucket(meta, bucket)
        for meta, bucket in focus_buckets.values()
    ]
    best_rows = select_best(pair_rows)

    preferred = [
        'selection_rule',
        'dataset_name',
        'focus_pair',
        'gt_class_name',
        'base_pred_class_name',
        'requested_scale',
        'threshold',
        'rows',
        'covered_pair_pixels',
        'mean_effective_scale_x',
        'mean_effective_scale_y',
        'mean_base_margin',
        'mean_requery_margin',
        'mean_margin_gain',
        'base_gt_beats_pred_ratio',
        'requery_gt_beats_pred_ratio',
        'gt_support_ratio',
        'pred_support_ratio',
        'gt_only_support_ratio',
        'pred_only_support_ratio',
        'both_support_ratio',
        'neither_support_ratio',
        'mean_gt_semantic',
        'mean_pred_semantic',
        'mean_gt_instance',
        'mean_pred_instance',
    ]
    write_csv(
        os.path.join(args.out_dir, 'scale_requery_pair_summary.csv'),
        sorted(pair_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('gt_class_name')),
            str(item.get('base_pred_class_name')),
            float(item.get('requested_scale') or 0),
            float(item.get('threshold') or 0),
        )),
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'scale_requery_focus_summary.csv'),
        sorted(focus_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('focus_pair')),
            float(item.get('requested_scale') or 0),
            float(item.get('threshold') or 0),
        )),
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'scale_requery_scale_summary.csv'),
        sorted(scale_rows, key=lambda item: (
            str(item.get('dataset_name')),
            float(item.get('requested_scale') or 0),
            float(item.get('threshold') or 0),
        )),
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'scale_requery_best_summary.csv'),
        best_rows,
        preferred)


if __name__ == '__main__':
    main()
