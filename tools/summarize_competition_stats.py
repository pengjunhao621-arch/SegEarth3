import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize SegEarth-OV3 competition JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/competition/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for gt_class_competition_summary.csv and pair_competition_summary.csv')
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


def add(bucket, key, value):
    if value is not None:
        bucket[key].append(value)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_div(num, den):
    if den == 0:
        return None
    return num / den


def summarize(records):
    gt_buckets = defaultdict(lambda: defaultdict(list))
    gt_counts = defaultdict(lambda: dict(images=0, gt_pixels=0, correct_pixels=0, fn_pixels=0))
    pair_buckets = defaultdict(lambda: defaultdict(list))
    pair_counts = defaultdict(lambda: dict(images=0, pixels=0, gt_pixels=0))

    for record in records:
        for gt_record in record.get('competition_stats') or []:
            gt_key = gt_record['gt_class_index']
            gt_bucket = gt_buckets[gt_key]
            gt_count = gt_counts[gt_key]
            gt_count['images'] += 1
            gt_count['gt_pixels'] += gt_record.get('gt_pixels') or 0
            gt_count['correct_pixels'] += gt_record.get('correct_pixels') or 0
            gt_count['fn_pixels'] += gt_record.get('fn_pixels') or 0
            gt_bucket['gt_class_name'].append(gt_record.get('gt_class_name'))
            for key in [
                'gt_area_ratio',
                'recall',
                'target_final_mean_on_gt',
                'target_final_mean_on_correct',
                'target_final_mean_on_fn',
                'top1_score_mean_on_fn',
                'top2_score_mean_on_fn',
                'top1_top2_margin_mean_on_fn',
                'target_gap_mean_on_fn',
                'threshold_reject_ratio_on_fn',
                'target_semantic_mean_on_fn',
                'target_instance_mean_on_fn',
            ]:
                add(gt_bucket, key, gt_record.get(key))

            for competitor in gt_record.get('competitors', []):
                pair_key = (gt_record['gt_class_index'], competitor['pred_class_index'])
                pair_bucket = pair_buckets[pair_key]
                pair_count = pair_counts[pair_key]
                pair_count['images'] += 1
                pair_count['pixels'] += competitor.get('pixels') or 0
                pair_count['gt_pixels'] += gt_record.get('gt_pixels') or 0
                pair_bucket['gt_class_name'].append(gt_record.get('gt_class_name'))
                pair_bucket['pred_class_name'].append(competitor.get('pred_class_name'))
                for key in [
                    'pixel_ratio_in_gt',
                    'target_final_mean',
                    'competitor_final_mean',
                    'target_gap_mean',
                    'top1_top2_margin_mean',
                    'threshold_reject_ratio',
                    'target_semantic_mean',
                    'competitor_semantic_mean',
                    'target_instance_mean',
                    'competitor_instance_mean',
                ]:
                    add(pair_bucket, key, competitor.get(key))

    gt_rows = []
    for gt_key, bucket in sorted(gt_buckets.items()):
        count = gt_counts[gt_key]
        gt_rows.append(dict(
            gt_class_index=gt_key,
            gt_class_name=first(bucket['gt_class_name']),
            images=count['images'],
            gt_pixels=count['gt_pixels'],
            correct_pixels=count['correct_pixels'],
            fn_pixels=count['fn_pixels'],
            dataset_recall=safe_div(count['correct_pixels'], count['gt_pixels']),
            dataset_fn_ratio=safe_div(count['fn_pixels'], count['gt_pixels']),
            **{f'mean_{key}': mean(values)
               for key, values in bucket.items()
               if key != 'gt_class_name'}
        ))

    pair_rows = []
    for (gt_class, pred_class), bucket in sorted(pair_buckets.items()):
        count = pair_counts[(gt_class, pred_class)]
        pair_rows.append(dict(
            gt_class_index=gt_class,
            gt_class_name=first(bucket['gt_class_name']),
            pred_class_index=pred_class,
            pred_class_name=first(bucket['pred_class_name']),
            images=count['images'],
            pixels=count['pixels'],
            dataset_pixel_ratio_in_seen_gt=safe_div(count['pixels'], count['gt_pixels']),
            **{f'mean_{key}': mean(values)
               for key, values in bucket.items()
               if key not in ('gt_class_name', 'pred_class_name')}
        ))

    pair_rows.sort(key=lambda row: row['pixels'], reverse=True)
    return gt_rows, pair_rows


def first(values):
    for value in values:
        if value is not None:
            return value
    return None


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        'gt_class_index',
        'gt_class_name',
        'pred_class_index',
        'pred_class_name',
        'images',
        'gt_pixels',
        'correct_pixels',
        'fn_pixels',
        'pixels',
        'dataset_recall',
        'dataset_fn_ratio',
        'dataset_pixel_ratio_in_seen_gt',
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    os.makedirs(args.out_dir, exist_ok=True)
    gt_rows, pair_rows = summarize(iter_records(paths))
    write_csv(os.path.join(args.out_dir, 'gt_class_competition_summary.csv'), gt_rows)
    write_csv(os.path.join(args.out_dir, 'pair_competition_summary.csv'), pair_rows)
    print(f'Read {len(paths)} files')
    print(f'Wrote {len(gt_rows)} GT-class rows and {len(pair_rows)} pair rows to {args.out_dir}')


if __name__ == '__main__':
    main()
