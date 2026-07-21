import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SegEarth-OV3 raw instance mask oracle diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/raw_mask_oracle/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for raw_mask_class_summary.csv, raw_mask_pair_summary.csv, '
             'and raw_mask_candidate_summary.csv')
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
        'pred_pixels',
        'wrong_pixels',
        'seed_pixels',
        'candidate_count',
        'kept_candidate_count',
        'best_purity_pixels',
        'best_gt_recall_pixels',
        'best_wrong_cover_pixels',
        'best_seed_cover_pixels',
    ])
    for key in [
            'best_purity',
            'best_gt_recall',
            'best_wrong_coverage',
            'best_seed_coverage']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_images'] += 1
    for key in [
            'best_purity_kept',
            'best_gt_recall_kept',
            'best_wrong_coverage_kept',
            'best_seed_coverage_kept']:
        value = row.get(key)
        if isinstance(value, bool):
            bucket[f'{key}_true'] += int(value)
            bucket[f'{key}_images'] += 1


def finalize_class_row(base, counts):
    row = dict(base)
    row.update(counts)
    row['kept_candidate_ratio'] = safe_div(
        row.get('kept_candidate_count', 0),
        row.get('candidate_count', 0))
    for key in [
            'best_purity',
            'best_gt_recall',
            'best_wrong_coverage',
            'best_seed_coverage']:
        row[f'mean_{key}'] = safe_div(
            row.get(f'{key}_sum', 0.0),
            row.get(f'{key}_images', 0))
    for key in [
            'best_purity_kept',
            'best_gt_recall_kept',
            'best_wrong_coverage_kept',
            'best_seed_coverage_kept']:
        row[f'{key}_ratio'] = safe_div(
            row.get(f'{key}_true', 0),
            row.get(f'{key}_images', 0))
    return row


def add_pair_row(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'pixels',
        'gt_candidate_count',
        'pred_candidate_count',
        'gt_best_pair_cover_pixels',
        'pred_best_pair_cover_pixels',
    ])
    if row.get('raw_support_favors_gt') is True:
        bucket['raw_support_favors_gt_images'] += 1
    for key in [
            'gt_best_raw_score',
            'pred_best_raw_score',
            'gt_best_presence_score',
            'pred_best_presence_score',
            'gt_best_purity',
            'pred_best_purity']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_images'] += 1
    for key in ['gt_best_kept', 'pred_best_kept']:
        value = row.get(key)
        if isinstance(value, bool):
            bucket[f'{key}_true'] += int(value)
            bucket[f'{key}_images'] += 1


def finalize_pair_row(base, counts):
    row = dict(base)
    row.update(counts)
    pixels = row.get('pixels', 0)
    row['gt_best_pair_coverage'] = safe_div(
        row.get('gt_best_pair_cover_pixels', 0), pixels)
    row['pred_best_pair_coverage'] = safe_div(
        row.get('pred_best_pair_cover_pixels', 0), pixels)
    row['pair_coverage_margin'] = safe_div(
        row.get('gt_best_pair_cover_pixels', 0)
        - row.get('pred_best_pair_cover_pixels', 0),
        pixels)
    row['raw_support_favors_gt_image_ratio'] = safe_div(
        row.get('raw_support_favors_gt_images', 0), row.get('images', 0))
    for key in [
            'gt_best_raw_score',
            'pred_best_raw_score',
            'gt_best_presence_score',
            'pred_best_presence_score',
            'gt_best_purity',
            'pred_best_purity']:
        row[f'mean_{key}'] = safe_div(
            row.get(f'{key}_sum', 0.0),
            row.get(f'{key}_images', 0))
    for key in ['gt_best_kept', 'pred_best_kept']:
        row[f'{key}_ratio'] = safe_div(
            row.get(f'{key}_true', 0),
            row.get(f'{key}_images', 0))
    focus_name = focus_pair_name(
        row.get('dataset_name'),
        row.get('gt_class_name'),
        row.get('base_pred_class_name'))
    row['focus_pair'] = bool(focus_name)
    row['focus_pair_name'] = focus_name
    return row


def add_candidate_row(bucket, row):
    bucket['candidates'] += 1
    if row.get('kept') is True:
        bucket['kept_candidates'] += 1
    add_sum(bucket, row, [
        'mask_pixels',
        'correct_pixels',
        'wrong_cover_pixels',
        'seed_cover_pixels',
        'raw_score',
        'raw_presence_score',
    ])
    for key in ['purity', 'gt_recall', 'wrong_coverage', 'seed_coverage']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_count'] += 1


def finalize_candidate_row(base, counts):
    row = dict(base)
    row.update(counts)
    row['kept_candidate_ratio'] = safe_div(
        row.get('kept_candidates', 0),
        row.get('candidates', 0))
    row['mean_raw_score'] = safe_div(
        row.get('raw_score', 0.0),
        row.get('candidates', 0))
    row['mean_raw_presence_score'] = safe_div(
        row.get('raw_presence_score', 0.0),
        row.get('candidates', 0))
    row['candidate_purity'] = safe_div(
        row.get('correct_pixels', 0),
        row.get('mask_pixels', 0))
    for key in ['purity', 'gt_recall', 'wrong_coverage', 'seed_coverage']:
        row[f'mean_{key}'] = safe_div(
            row.get(f'{key}_sum', 0.0),
            row.get(f'{key}_count', 0))
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
    for item in (name or '').replace('/', ',').split(','):
        item = item.strip().lower()
        if item:
            tokens.add(item)
    if name:
        tokens.add(name.strip().lower())
    return tokens


def summarize(records):
    class_buckets = defaultdict(lambda: defaultdict(int))
    pair_buckets = defaultdict(lambda: defaultdict(int))
    candidate_buckets = defaultdict(lambda: defaultdict(int))

    for record in records:
        stats = record.get('raw_mask_oracle_stats')
        if not stats:
            continue
        dataset_name = stats.get('dataset_name') or record.get('dataset_name')
        for row in stats.get('class_stats', []):
            key = (dataset_name, row.get('class_index'), row.get('class_name'))
            add_class_row(class_buckets[key], row)
        for row in stats.get('pair_stats', []):
            key = (
                dataset_name,
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('base_pred_class_index'),
                row.get('base_pred_class_name'),
            )
            add_pair_row(pair_buckets[key], row)
        for row in stats.get('candidate_stats', []):
            key = (dataset_name, row.get('class_index'), row.get('class_name'))
            add_candidate_row(candidate_buckets[key], row)

    class_rows = [
        finalize_class_row(
            dict(dataset_name=key[0], class_index=key[1], class_name=key[2]),
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
                base_pred_class_name=key[4]),
            counts)
        for key, counts in pair_buckets.items()
    ]
    candidate_rows = [
        finalize_candidate_row(
            dict(dataset_name=key[0], class_index=key[1], class_name=key[2]),
            counts)
        for key, counts in candidate_buckets.items()
    ]

    class_rows.sort(key=lambda item: (item.get('dataset_name') or '', item.get('class_index') or -1))
    pair_rows.sort(key=lambda item: (item.get('focus_pair') is not True, -(item.get('pixels') or 0)))
    candidate_rows.sort(key=lambda item: (item.get('dataset_name') or '', item.get('class_index') or -1))
    return class_rows, pair_rows, candidate_rows


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
    class_rows, pair_rows, candidate_rows = summarize(iter_records(paths))
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'raw_mask_class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'raw_mask_pair_summary.csv'), pair_rows)
    write_csv(os.path.join(args.out_dir, 'raw_mask_candidate_summary.csv'), candidate_rows)


if __name__ == '__main__':
    main()
