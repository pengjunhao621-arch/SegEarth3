import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize GeoER scale-stability oracle diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/scale_stability/vaihingen/ss_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for scale_stability_*_summary.csv files.')
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


SUM_KEYS = [
    'covered_pair_pixels',
    'full_pair_pixels',
    'base_margin_sum',
    'scaled_margin_sum',
    'margin_gain_sum',
    'base_gt_score_sum',
    'base_pred_score_sum',
    'scaled_gt_score_sum',
    'scaled_pred_score_sum',
    'pred_drop_sum',
    'gt_drop_sum',
    'relative_stability_sum',
    'scaled_choice_score_sum',
    'scaled_choice_margin_sum',
    'scaled_choice_drop_sum',
    'choice_relative_stability_sum',
    'gt_semantic_sum',
    'pred_semantic_sum',
    'gt_instance_sum',
    'pred_instance_sum',
    'topk_contains_gt_pixels',
    'scaled_gt_beats_pred_pixels',
    'scaled_choice_is_gt_pixels',
    'scaled_choice_is_pred_pixels',
    'scaled_choice_not_pred_pixels',
    'top1_drop_gate_pixels',
    'top1_drop_gate_gt_pixels',
    'top1_drop_gate_gt_beats_pred_pixels',
    'stability_gate_pixels',
    'stability_gate_gt_pixels',
    'stability_gate_gt_beats_pred_pixels',
]


def new_bucket():
    bucket = defaultdict(float)
    bucket['rows'] = 0
    return bucket


def add_row(bucket, row):
    bucket['rows'] += 1
    for key in SUM_KEYS:
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
            ('base_margin_sum', 'mean_base_margin'),
            ('scaled_margin_sum', 'mean_scaled_margin'),
            ('margin_gain_sum', 'mean_margin_gain'),
            ('base_gt_score_sum', 'mean_base_gt_score'),
            ('base_pred_score_sum', 'mean_base_pred_score'),
            ('scaled_gt_score_sum', 'mean_scaled_gt_score'),
            ('scaled_pred_score_sum', 'mean_scaled_pred_score'),
            ('pred_drop_sum', 'mean_pred_drop'),
            ('gt_drop_sum', 'mean_gt_drop'),
            ('relative_stability_sum', 'mean_relative_stability'),
            ('scaled_choice_score_sum', 'mean_scaled_choice_score'),
            ('scaled_choice_margin_sum', 'mean_scaled_choice_margin'),
            ('scaled_choice_drop_sum', 'mean_scaled_choice_drop'),
            ('choice_relative_stability_sum', 'mean_choice_relative_stability'),
            ('gt_semantic_sum', 'mean_gt_semantic'),
            ('pred_semantic_sum', 'mean_pred_semantic'),
            ('gt_instance_sum', 'mean_gt_instance'),
            ('pred_instance_sum', 'mean_pred_instance')]:
        out[dst] = safe_div(bucket.get(src, 0), pixels)

    for src, dst in [
            ('topk_contains_gt_pixels', 'topk_contains_gt_ratio'),
            ('scaled_gt_beats_pred_pixels', 'scaled_gt_beats_pred_ratio'),
            ('scaled_choice_is_gt_pixels', 'scaled_choice_is_gt_ratio'),
            ('scaled_choice_is_pred_pixels', 'scaled_choice_is_pred_ratio'),
            ('scaled_choice_not_pred_pixels', 'scaled_choice_not_pred_ratio'),
            ('top1_drop_gate_pixels', 'top1_drop_gate_ratio'),
            ('top1_drop_gate_gt_pixels', 'top1_drop_gate_precision'),
            ('top1_drop_gate_gt_beats_pred_pixels', 'top1_drop_gate_gt_beats_pred_ratio'),
            ('stability_gate_pixels', 'stability_gate_ratio'),
            ('stability_gate_gt_pixels', 'stability_gate_precision'),
            ('stability_gate_gt_beats_pred_pixels', 'stability_gate_gt_beats_pred_ratio')]:
        den = pixels
        if src in {
                'top1_drop_gate_gt_pixels',
                'top1_drop_gate_gt_beats_pred_pixels'}:
            den = bucket.get('top1_drop_gate_pixels', 0)
        elif src in {
                'stability_gate_gt_pixels',
                'stability_gate_gt_beats_pred_pixels'}:
            den = bucket.get('stability_gate_pixels', 0)
        out[dst] = safe_div(bucket.get(src, 0), den)
    out['top1_drop_gate_pixels'] = int(bucket.get('top1_drop_gate_pixels', 0))
    out['stability_gate_pixels'] = int(bucket.get('stability_gate_pixels', 0))
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


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')

    pair_buckets = {}
    focus_buckets = {}
    threshold_buckets = {}

    for record in iter_records(paths):
        stats = record.get('scale_stability_stats') or {}
        dataset = record.get('dataset_name') or stats.get('dataset_name') or ''
        for row in stats.get('pair_stats') or []:
            gt_name = row.get('gt_class_name')
            pred_name = row.get('base_pred_class_name')
            threshold = row.get('drop_threshold')
            scale = row.get('requested_scale')
            topk = row.get('scale_stability_topk')
            focus_name = focus_pair_name(dataset, gt_name, pred_name)

            pair_key = (dataset, gt_name, pred_name, scale, topk, threshold)
            pair_meta = dict(
                dataset_name=dataset,
                focus_pair=focus_name,
                gt_class_name=gt_name,
                base_pred_class_name=pred_name,
                requested_scale=scale,
                scale_stability_topk=topk,
                drop_threshold=threshold,
            )
            pair_buckets.setdefault(pair_key, (pair_meta, new_bucket()))
            add_row(pair_buckets[pair_key][1], row)

            threshold_key = (dataset, scale, topk, threshold)
            threshold_meta = dict(
                dataset_name=dataset,
                requested_scale=scale,
                scale_stability_topk=topk,
                drop_threshold=threshold,
            )
            threshold_buckets.setdefault(threshold_key, (threshold_meta, new_bucket()))
            add_row(threshold_buckets[threshold_key][1], row)

            if focus_name:
                focus_key = (dataset, focus_name, scale, topk, threshold)
                focus_meta = dict(
                    dataset_name=dataset,
                    focus_pair=focus_name,
                    gt_class_name=gt_name,
                    base_pred_class_name=pred_name,
                    requested_scale=scale,
                    scale_stability_topk=topk,
                    drop_threshold=threshold,
                )
                focus_buckets.setdefault(focus_key, (focus_meta, new_bucket()))
                add_row(focus_buckets[focus_key][1], row)

    pair_rows = [finalize_bucket(meta, bucket) for meta, bucket in pair_buckets.values()]
    focus_rows = [finalize_bucket(meta, bucket) for meta, bucket in focus_buckets.values()]
    threshold_rows = [
        finalize_bucket(meta, bucket)
        for meta, bucket in threshold_buckets.values()
    ]

    preferred = [
        'dataset_name',
        'focus_pair',
        'gt_class_name',
        'base_pred_class_name',
        'requested_scale',
        'scale_stability_topk',
        'drop_threshold',
        'rows',
        'covered_pair_pixels',
        'mean_effective_scale_x',
        'mean_effective_scale_y',
        'mean_base_margin',
        'mean_scaled_margin',
        'mean_margin_gain',
        'mean_pred_drop',
        'mean_gt_drop',
        'mean_relative_stability',
        'topk_contains_gt_ratio',
        'scaled_gt_beats_pred_ratio',
        'scaled_choice_is_gt_ratio',
        'scaled_choice_is_pred_ratio',
        'scaled_choice_not_pred_ratio',
        'mean_scaled_choice_margin',
        'mean_choice_relative_stability',
        'top1_drop_gate_pixels',
        'top1_drop_gate_ratio',
        'top1_drop_gate_precision',
        'top1_drop_gate_gt_beats_pred_ratio',
        'stability_gate_pixels',
        'stability_gate_ratio',
        'stability_gate_precision',
        'stability_gate_gt_beats_pred_ratio',
        'mean_gt_semantic',
        'mean_pred_semantic',
        'mean_gt_instance',
        'mean_pred_instance',
    ]
    write_csv(
        os.path.join(args.out_dir, 'scale_stability_pair_summary.csv'),
        sorted(pair_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('gt_class_name')),
            str(item.get('base_pred_class_name')),
            float(item.get('drop_threshold') or 0),
        )),
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'scale_stability_focus_summary.csv'),
        sorted(focus_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('focus_pair')),
            float(item.get('drop_threshold') or 0),
        )),
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'scale_stability_threshold_summary.csv'),
        sorted(threshold_rows, key=lambda item: (
            str(item.get('dataset_name')),
            float(item.get('drop_threshold') or 0),
        )),
        preferred)


if __name__ == '__main__':
    main()
