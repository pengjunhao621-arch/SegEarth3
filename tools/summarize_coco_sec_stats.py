import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize CoCo-style SEC diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/coco_sec/vdd/sec_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for coco_sec_overall_summary.csv, '
             'coco_sec_class_summary.csv, and coco_sec_pair_summary.csv')
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
    if den == 0:
        return None
    return num / den


def add_sum(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value


def add_overall(bucket, stats):
    bucket['images'] += 1
    add_sum(bucket, stats, [
        'valid_pixels',
        'base_correct_pixels',
        'fused_correct_pixels',
        'improved_pixels',
        'harmed_pixels',
    ])


def finalize_overall(base, counts):
    row = dict(base)
    row.update(counts)
    valid = row.get('valid_pixels', 0)
    row['base_accuracy'] = safe_div(row.get('base_correct_pixels', 0), valid)
    row['fused_accuracy'] = safe_div(row.get('fused_correct_pixels', 0), valid)
    row['net_improved_pixels'] = row.get('improved_pixels', 0) - row.get('harmed_pixels', 0)
    row['net_improved_ratio'] = safe_div(row.get('net_improved_pixels', 0), valid)
    row['improved_ratio'] = safe_div(row.get('improved_pixels', 0), valid)
    row['harmed_ratio'] = safe_div(row.get('harmed_pixels', 0), valid)
    return row


def add_class(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'gt_pixels',
        'base_correct_pixels',
        'fused_correct_pixels',
        'improved_pixels',
        'harmed_pixels',
    ])
    for key in ['mean_prior', 'mean_base_score', 'mean_fused_score']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_count'] += 1


def finalize_class(base, counts):
    row = dict(base)
    row.update(counts)
    gt_pixels = row.get('gt_pixels', 0)
    row['base_recall'] = safe_div(row.get('base_correct_pixels', 0), gt_pixels)
    row['fused_recall'] = safe_div(row.get('fused_correct_pixels', 0), gt_pixels)
    row['recall_delta'] = (
        None if row['base_recall'] is None or row['fused_recall'] is None
        else row['fused_recall'] - row['base_recall'])
    row['net_improved_pixels'] = row.get('improved_pixels', 0) - row.get('harmed_pixels', 0)
    for key in ['mean_prior', 'mean_base_score', 'mean_fused_score']:
        row[key] = safe_div(row.get(f'{key}_sum', 0.0), row.get(f'{key}_count', 0))
    return row


def add_pair(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'pixels',
        'improved_pixels',
        'fused_beats_pred_pixels',
    ])
    for key in [
            'mean_prior_gt',
            'mean_prior_pred',
            'mean_prior_margin',
            'mean_base_margin',
            'mean_fused_margin']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value * row.get('pixels', 0)
            bucket[f'{key}_pixels'] += row.get('pixels', 0)


def finalize_pair(base, counts):
    row = dict(base)
    row.update(counts)
    pixels = row.get('pixels', 0)
    row['improved_ratio'] = safe_div(row.get('improved_pixels', 0), pixels)
    row['fused_beats_pred_ratio'] = safe_div(row.get('fused_beats_pred_pixels', 0), pixels)
    for key in [
            'mean_prior_gt',
            'mean_prior_pred',
            'mean_prior_margin',
            'mean_base_margin',
            'mean_fused_margin']:
        row[key] = safe_div(row.get(f'{key}_sum', 0.0), row.get(f'{key}_pixels', 0))
    focus_name = focus_pair_name(
        row.get('dataset_name'),
        row.get('gt_class_name'),
        row.get('base_pred_class_name'))
    row['focus_pair'] = bool(focus_name)
    row['focus_pair_name'] = focus_name
    return row


def focus_pair_name(dataset_name, gt_name, pred_name):
    dataset = (dataset_name or '').lower()
    gt = (gt_name or '').lower()
    pred = (pred_name or '').lower()
    gt_tokens = class_name_tokens(gt)
    pred_tokens = class_name_tokens(pred)
    if dataset == 'vdd' and 'roof' in gt_tokens and 'facade' in pred_tokens:
        return 'VDD roof -> facade'
    if dataset == 'potsdam' and 'tree' in gt_tokens and 'grass' in pred_tokens:
        return 'Potsdam tree -> grass'
    if dataset == 'vaihingen' and 'grass' in gt_tokens and 'tree' in pred_tokens:
        return 'Vaihingen grass -> tree'
    if dataset == 'udd5' and 'road' in gt_tokens and 'background' in pred_tokens:
        return 'UDD5 road -> background'
    if dataset == 'udd5' and 'vegetation' in gt_tokens and 'background' in pred_tokens:
        return 'UDD5 vegetation -> background'
    if dataset == 'udd5' and 'building' in gt_tokens and 'background' in pred_tokens:
        return 'UDD5 building -> background'
    if dataset == 'openearthmap' and 'pavement' in gt_tokens and 'building' in pred_tokens:
        return 'OpenEarthMap pavement -> building'
    if (dataset == 'loveda'
            and bool({'forest', 'agricultural', 'agriculture'} & gt_tokens)
            and 'background' in pred_tokens):
        return f'LoveDA {gt} -> background'
    return None


def class_name_tokens(name):
    tokens = set()
    normalized = str(name or '').lower().replace('/', ',').replace('-', ' ')
    for item in normalized.split(','):
        item = item.strip()
        if item:
            tokens.add(item)
            tokens.update(part for part in item.split() if part)
    return tokens


def summarize(records):
    overall_buckets = defaultdict(lambda: defaultdict(int))
    class_buckets = defaultdict(lambda: defaultdict(int))
    pair_buckets = defaultdict(lambda: defaultdict(int))

    for record in records:
        stats = record.get('coco_sec_stats')
        if not stats:
            continue
        dataset_name = record.get('dataset_name') or stats.get('dataset_name')
        base = (
            dataset_name,
            record.get('use_coco_sec_fusion'),
            record.get('coco_sec_lambda'),
            record.get('coco_sec_temperature'),
            record.get('coco_sec_center_prior'),
            record.get('coco_sec_synonym_reduce'),
        )
        add_overall(overall_buckets[base], stats)
        for row in stats.get('class_stats', []):
            key = base + (row.get('class_index'), row.get('class_name'))
            add_class(class_buckets[key], row)
        for row in stats.get('pair_stats', []):
            key = base + (
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('base_pred_class_index'),
                row.get('base_pred_class_name'),
            )
            add_pair(pair_buckets[key], row)

    overall_rows = [
        finalize_overall(
            dict(
                dataset_name=key[0],
                use_coco_sec_fusion=key[1],
                coco_sec_lambda=key[2],
                coco_sec_temperature=key[3],
                coco_sec_center_prior=key[4],
                coco_sec_synonym_reduce=key[5]),
            counts)
        for key, counts in overall_buckets.items()
    ]
    class_rows = [
        finalize_class(
            dict(
                dataset_name=key[0],
                use_coco_sec_fusion=key[1],
                coco_sec_lambda=key[2],
                coco_sec_temperature=key[3],
                coco_sec_center_prior=key[4],
                coco_sec_synonym_reduce=key[5],
                class_index=key[6],
                class_name=key[7]),
            counts)
        for key, counts in class_buckets.items()
    ]
    pair_rows = [
        finalize_pair(
            dict(
                dataset_name=key[0],
                use_coco_sec_fusion=key[1],
                coco_sec_lambda=key[2],
                coco_sec_temperature=key[3],
                coco_sec_center_prior=key[4],
                coco_sec_synonym_reduce=key[5],
                gt_class_index=key[6],
                gt_class_name=key[7],
                base_pred_class_index=key[8],
                base_pred_class_name=key[9]),
            counts)
        for key, counts in pair_buckets.items()
    ]

    overall_rows.sort(key=lambda item: item.get('dataset_name') or '')
    class_rows.sort(key=lambda item: (item.get('dataset_name') or '', item.get('class_index') or -1))
    pair_rows.sort(key=lambda item: (
        item.get('focus_pair') is not True,
        item.get('dataset_name') or '',
        -(item.get('pixels') or 0)))
    return overall_rows, class_rows, pair_rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    overall_rows, class_rows, pair_rows = summarize(iter_records(paths))
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'coco_sec_overall_summary.csv'), overall_rows)
    write_csv(os.path.join(args.out_dir, 'coco_sec_class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'coco_sec_pair_summary.csv'), pair_rows)


if __name__ == '__main__':
    main()
