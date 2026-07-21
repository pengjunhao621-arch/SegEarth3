import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize threshold-reject recovery precision diagnostics.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/reject_recovery/udd5/rr_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for reject_recovery_*_summary.csv files.')
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


COUNT_KEYS = {
    'pixels',
    'correct_pixels',
    'true_foreground_pixels',
    'true_background_pixels',
}

MEAN_KEYS = {
    'mean_top1_score',
    'mean_final_margin',
    'mean_top1_semantic',
    'mean_top1_instance',
    'mean_semantic_bg_margin',
    'mean_instance_bg_margin',
    'mean_head_agreement',
    'mean_local_consistency',
}


def add_row(bucket, row):
    pixels = row.get('pixels') or 0
    bucket['rows'] += 1
    for key, value in row.items():
        if key in {
                'name', 'group_type', 'class_name', 'pe_space'}:
            continue
        if key in {
                'class_index', 'top1_score_thd', 'semantic_thd',
                'instance_thd', 'final_margin_thd', 'local_thd',
                'score_lo', 'score_hi'}:
            continue
        if isinstance(value, (int, float)) and (
                key in COUNT_KEYS
                or key.endswith('_pixels')):
            bucket[key] += value
        elif isinstance(value, (int, float)) and (
                key in MEAN_KEYS
                or key.endswith('_mean_margin')):
            bucket[f'__sum__{key}'] += value * pixels
            bucket[f'__den__{key}'] += pixels


def finalize(base, counts, denominator):
    row = dict(base)
    row.update({key: value for key, value in counts.items()
                if not key.startswith('__')})
    pixels = row.get('pixels', 0)
    row['pixel_ratio_in_candidates'] = safe_div(pixels, denominator)
    row['precision'] = safe_div(row.get('correct_pixels', 0), pixels)
    row['true_foreground_ratio'] = safe_div(
        row.get('true_foreground_pixels', 0), pixels)
    row['true_background_ratio'] = safe_div(
        row.get('true_background_pixels', 0), pixels)
    for key, value in list(row.items()):
        if key.endswith('_pixels') and isinstance(value, (int, float)):
            row[key[:-7] + '_ratio'] = safe_div(value, pixels)
    for key, value in counts.items():
        if key.startswith('__sum__'):
            mean_key = key.replace('__sum__', '')
            den = counts.get(f'__den__{mean_key}', 0)
            row[mean_key] = safe_div(value, den)
    return row


def summarize(records):
    totals = defaultdict(float)
    gate_counts = defaultdict(lambda: defaultdict(float))
    gate_meta = {}
    band_counts = defaultdict(lambda: defaultdict(float))
    band_meta = {}
    pred_class_counts = defaultdict(lambda: defaultdict(float))
    pred_class_meta = {}
    gt_class_counts = defaultdict(lambda: defaultdict(float))
    gt_class_meta = {}

    for record in records:
        stats = record.get('reject_recovery_precision_stats') or {}
        dataset = stats.get('dataset_name') or record.get('dataset_name') or 'unknown'
        candidate_pixels = stats.get('threshold_reject_candidate_pixels') or 0
        totals[dataset] += candidate_pixels

        for row in stats.get('gate_stats') or []:
            key = (dataset, row.get('name'))
            add_row(gate_counts[key], row)
            gate_meta[key] = dict(dataset_name=dataset, name=row.get('name'))
            for meta_key in [
                    'top1_score_thd', 'semantic_thd', 'instance_thd',
                    'final_margin_thd', 'local_thd', 'pe_space']:
                if row.get(meta_key) is not None:
                    gate_meta[key][meta_key] = row.get(meta_key)

        for row in stats.get('band_stats') or []:
            key = (dataset, row.get('name'), row.get('score_lo'), row.get('score_hi'))
            add_row(band_counts[key], row)
            band_meta[key] = dict(
                dataset_name=dataset,
                name=row.get('name'),
                score_lo=row.get('score_lo'),
                score_hi=row.get('score_hi'),
            )

        for row in stats.get('pred_class_stats') or []:
            key = (dataset, row.get('class_index'))
            add_row(pred_class_counts[key], row)
            pred_class_meta[key] = dict(
                dataset_name=dataset,
                class_index=row.get('class_index'),
                class_name=row.get('class_name'),
            )

        for row in stats.get('gt_class_stats') or []:
            key = (dataset, row.get('class_index'))
            add_row(gt_class_counts[key], row)
            gt_class_meta[key] = dict(
                dataset_name=dataset,
                class_index=row.get('class_index'),
                class_name=row.get('class_name'),
            )

    gate_rows = [
        finalize(gate_meta[key], counts, totals.get(key[0], 0))
        for key, counts in gate_counts.items()
    ]
    band_rows = [
        finalize(band_meta[key], counts, totals.get(key[0], 0))
        for key, counts in band_counts.items()
    ]
    pred_class_rows = [
        finalize(pred_class_meta[key], counts, totals.get(key[0], 0))
        for key, counts in pred_class_counts.items()
    ]
    gt_class_rows = [
        finalize(gt_class_meta[key], counts, totals.get(key[0], 0))
        for key, counts in gt_class_counts.items()
    ]

    gate_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        -(row.get('pixels') or 0),
        row.get('name')))
    band_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('score_lo') if row.get('score_lo') is not None else 999,
        row.get('score_hi') if row.get('score_hi') is not None else 999))
    pred_class_rows.sort(key=lambda row: (
        row.get('dataset_name'), row.get('class_index')))
    gt_class_rows.sort(key=lambda row: (
        row.get('dataset_name'), row.get('class_index')))
    return gate_rows, band_rows, pred_class_rows, gt_class_rows


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

    gate_rows, band_rows, pred_class_rows, gt_class_rows = summarize(
        iter_records(paths))

    preferred = [
        'dataset_name', 'name', 'class_index', 'class_name',
        'top1_score_thd', 'semantic_thd', 'instance_thd',
        'final_margin_thd', 'local_thd', 'score_lo', 'score_hi',
        'pe_space', 'rows', 'pixels', 'pixel_ratio_in_candidates',
        'correct_pixels', 'precision',
        'true_foreground_pixels', 'true_foreground_ratio',
        'true_background_pixels', 'true_background_ratio',
        'mean_top1_score', 'mean_final_margin',
        'mean_top1_semantic', 'mean_top1_instance',
        'mean_semantic_bg_margin', 'mean_instance_bg_margin',
        'mean_head_agreement', 'mean_local_consistency',
    ]
    write_csv(
        os.path.join(args.out_dir, 'reject_recovery_gate_summary.csv'),
        gate_rows,
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'reject_recovery_score_band_summary.csv'),
        band_rows,
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'reject_recovery_pred_class_summary.csv'),
        pred_class_rows,
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'reject_recovery_gt_class_summary.csv'),
        gt_class_rows,
        preferred)
    print(
        f'Wrote {len(gate_rows)} gate rows, {len(band_rows)} score-band rows, '
        f'{len(pred_class_rows)} pred-class rows, and {len(gt_class_rows)} '
        f'gt-class rows to {args.out_dir}')


if __name__ == '__main__':
    main()
