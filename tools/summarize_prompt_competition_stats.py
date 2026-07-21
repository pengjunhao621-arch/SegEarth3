import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SegEarth-OV3 prompt competition diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/prompt_competition/vdd/prompt_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for prompt_variant_class_summary.csv, '
             'prompt_variant_pair_summary.csv, and prompt_variant_best_pair_summary.csv')
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


def add_class_row(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'gt_pixels',
        'baseline_correct_pixels',
        'class_wrong_pixels',
        'base_class_score_sum',
        'variant_score_sum',
        'variant_gain_sum',
        'variant_gain_pixels',
        'variant_beats_base_pixels',
        'variant_beats_all_pixels',
    ])


def finalize_class_row(base, counts):
    row = dict(base)
    row.update(counts)
    gt_pixels = row.get('gt_pixels', 0)
    gain_pixels = row.get('variant_gain_pixels', 0)
    row['baseline_recall'] = safe_div(
        row.get('baseline_correct_pixels', 0), gt_pixels)
    row['mean_base_class_score'] = safe_div(
        row.get('base_class_score_sum', 0.0), gain_pixels)
    row['mean_variant_score'] = safe_div(
        row.get('variant_score_sum', 0.0), gain_pixels)
    row['mean_variant_gain'] = safe_div(
        row.get('variant_gain_sum', 0.0), gain_pixels)
    row['variant_beats_base_ratio'] = safe_div(
        row.get('variant_beats_base_pixels', 0), gt_pixels)
    row['variant_beats_all_ratio'] = safe_div(
        row.get('variant_beats_all_pixels', 0), gt_pixels)
    return row


def add_pair_row(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'pixels',
        'base_gt_score_sum',
        'base_pred_score_sum',
        'variant_score_sum',
        'variant_gain_sum',
        'variant_vs_pred_margin_sum',
        'variant_beats_base_pixels',
        'variant_beats_pred_pixels',
        'variant_beats_all_pixels',
    ])


def finalize_pair_row(base, counts):
    row = dict(base)
    row.update(counts)
    pixels = row.get('pixels', 0)
    row['mean_base_gt_score'] = safe_div(
        row.get('base_gt_score_sum', 0.0), pixels)
    row['mean_base_pred_score'] = safe_div(
        row.get('base_pred_score_sum', 0.0), pixels)
    row['mean_variant_score'] = safe_div(
        row.get('variant_score_sum', 0.0), pixels)
    row['mean_variant_gain'] = safe_div(
        row.get('variant_gain_sum', 0.0), pixels)
    row['mean_variant_vs_pred_margin'] = safe_div(
        row.get('variant_vs_pred_margin_sum', 0.0), pixels)
    row['variant_beats_base_ratio'] = safe_div(
        row.get('variant_beats_base_pixels', 0), pixels)
    row['variant_beats_pred_ratio'] = safe_div(
        row.get('variant_beats_pred_pixels', 0), pixels)
    row['variant_beats_all_ratio'] = safe_div(
        row.get('variant_beats_all_pixels', 0), pixels)
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
    class_buckets = defaultdict(lambda: defaultdict(int))
    pair_buckets = defaultdict(lambda: defaultdict(int))

    for record in records:
        stats = record.get('prompt_competition_stats')
        if not stats:
            continue
        dataset_name = stats.get('dataset_name') or record.get('dataset_name')
        for row in stats.get('class_variant_stats', []):
            key = (
                dataset_name,
                row.get('class_index'),
                row.get('class_name'),
                row.get('variant_order'),
                row.get('variant_prompt'),
            )
            add_class_row(class_buckets[key], row)
        for row in stats.get('pair_variant_stats', []):
            key = (
                dataset_name,
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('base_pred_class_index'),
                row.get('base_pred_class_name'),
                row.get('variant_order'),
                row.get('variant_prompt'),
            )
            add_pair_row(pair_buckets[key], row)

    class_rows = [
        finalize_class_row(
            dict(
                dataset_name=key[0],
                class_index=key[1],
                class_name=key[2],
                variant_order=key[3],
                variant_prompt=key[4]),
            counts)
        for key, counts in class_buckets.items()
    ]
    pair_rows = [
        finalize_pair_row(
            dict(
                dataset_name=key[0],
                gt_class_index=key[1],
                gt_class_name=key[2],
                base_pred_class_index=key[3],
                base_pred_class_name=key[4],
                variant_order=key[5],
                variant_prompt=key[6]),
            counts)
        for key, counts in pair_buckets.items()
    ]

    best = {}
    for row in pair_rows:
        key = (
            row.get('dataset_name'),
            row.get('gt_class_index'),
            row.get('gt_class_name'),
            row.get('base_pred_class_index'),
            row.get('base_pred_class_name'),
        )
        current = best.get(key)
        score = (
            row.get('mean_variant_vs_pred_margin')
            if row.get('mean_variant_vs_pred_margin') is not None
            else -1e9,
            row.get('variant_beats_all_ratio')
            if row.get('variant_beats_all_ratio') is not None
            else -1e9,
        )
        if current is None or score > current[0]:
            best[key] = (score, row)
    best_rows = [item[1] for item in best.values()]

    class_rows.sort(key=lambda item: (
        item.get('dataset_name') or '',
        item.get('class_index') or -1,
        -(item.get('mean_variant_gain') or -1e9)))
    pair_rows.sort(key=lambda item: (
        item.get('focus_pair') is not True,
        item.get('dataset_name') or '',
        -(item.get('pixels') or 0),
        -(item.get('mean_variant_vs_pred_margin') or -1e9)))
    best_rows.sort(key=lambda item: (
        item.get('focus_pair') is not True,
        item.get('dataset_name') or '',
        -(item.get('pixels') or 0)))
    return class_rows, pair_rows, best_rows


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
    class_rows, pair_rows, best_rows = summarize(iter_records(paths))
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'prompt_variant_class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'prompt_variant_pair_summary.csv'), pair_rows)
    write_csv(os.path.join(args.out_dir, 'prompt_variant_best_pair_summary.csv'), best_rows)


if __name__ == '__main__':
    main()
