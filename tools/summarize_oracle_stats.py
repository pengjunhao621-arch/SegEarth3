import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize SegEarth-OV3 oracle JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/oracle/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for oracle_summary.csv, class_oracle_summary.csv, and pair_oracle_summary.csv')
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


def first(values):
    for value in values:
        if value is not None:
            return value
    return None


def add_pixel_keys(bucket, item, skip=None):
    skip = skip or set()
    for key, value in item.items():
        if key in skip:
            continue
        if key.endswith('_pixels') and isinstance(value, int):
            bucket[key] += value


def ratio_name(pixel_key):
    if pixel_key.endswith('_pixels'):
        return pixel_key[:-7] + '_ratio'
    return pixel_key + '_ratio'


def summarize(records):
    oracle_counts = defaultdict(int)
    head_counts = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(lambda: defaultdict(int))
    class_names = defaultdict(list)
    pair_counts = defaultdict(lambda: defaultdict(int))
    pair_names = defaultdict(lambda: defaultdict(list))

    for record in records:
        stats = record.get('oracle_stats') or {}
        valid_pixels = stats.get('valid_pixels') or 0
        oracle_counts['images'] += 1
        oracle_counts['valid_pixels'] += valid_pixels

        oracle = stats.get('oracle') or {}
        add_pixel_keys(oracle_counts, oracle, skip={'valid_pixels'})

        for head in stats.get('heads') or []:
            head_name = head.get('head')
            if head_name is None:
                continue
            bucket = head_counts[head_name]
            bucket['images'] += 1
            bucket['valid_pixels'] += valid_pixels
            add_pixel_keys(bucket, head, skip={'valid_pixels'})

        for class_stat in stats.get('class_stats') or []:
            class_idx = class_stat.get('class_index')
            if class_idx is None:
                continue
            bucket = class_counts[class_idx]
            bucket['images'] += 1
            bucket['gt_pixels'] += class_stat.get('gt_pixels') or 0
            class_names[class_idx].append(class_stat.get('class_name'))
            add_pixel_keys(bucket, class_stat, skip={'gt_pixels'})

        for pair_stat in stats.get('pair_stats') or []:
            gt_class = pair_stat.get('gt_class_index')
            pred_class = pair_stat.get('pred_class_index')
            if gt_class is None or pred_class is None:
                continue
            pair_key = (gt_class, pred_class)
            bucket = pair_counts[pair_key]
            bucket['images'] += 1
            bucket['pixels'] += pair_stat.get('pixels') or 0
            pair_names[pair_key]['gt_class_name'].append(pair_stat.get('gt_class_name'))
            pair_names[pair_key]['pred_class_name'].append(pair_stat.get('pred_class_name'))
            add_pixel_keys(bucket, pair_stat, skip={'pixels'})

    oracle_rows = [build_global_row('any_head_oracle', oracle_counts, oracle_counts['valid_pixels'])]
    for head_name, counts in sorted(head_counts.items()):
        oracle_rows.append(build_global_row(head_name, counts, counts['valid_pixels']))

    class_rows = []
    for class_idx, counts in sorted(class_counts.items()):
        gt_pixels = counts['gt_pixels']
        row = dict(
            class_index=class_idx,
            class_name=first(class_names[class_idx]),
            images=counts['images'],
            gt_pixels=gt_pixels,
        )
        row.update(pixel_ratios(counts, gt_pixels, skip={'gt_pixels'}))
        class_rows.append(row)

    pair_rows = []
    for (gt_class, pred_class), counts in sorted(pair_counts.items()):
        pixels = counts['pixels']
        row = dict(
            gt_class_index=gt_class,
            gt_class_name=first(pair_names[(gt_class, pred_class)]['gt_class_name']),
            pred_class_index=pred_class,
            pred_class_name=first(pair_names[(gt_class, pred_class)]['pred_class_name']),
            images=counts['images'],
            pixels=pixels,
        )
        row.update(pixel_ratios(counts, pixels, skip={'pixels'}))
        pair_rows.append(row)
    pair_rows.sort(key=lambda row: row['pixels'], reverse=True)

    return oracle_rows, class_rows, pair_rows


def build_global_row(name, counts, denominator):
    row = dict(
        name=name,
        images=counts['images'],
        valid_pixels=counts['valid_pixels'],
    )
    row.update(pixel_ratios(counts, denominator, skip={'valid_pixels'}))
    return row


def pixel_ratios(counts, denominator, skip=None):
    skip = skip or set()
    row = {}
    for key, value in sorted(counts.items()):
        if not key.endswith('_pixels') or key in skip:
            continue
        row[key] = value
        row[ratio_name(key)] = safe_div(value, denominator)
    return row


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        'name',
        'class_index',
        'class_name',
        'gt_class_index',
        'gt_class_name',
        'pred_class_index',
        'pred_class_name',
        'images',
        'valid_pixels',
        'gt_pixels',
        'pixels',
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
    oracle_rows, class_rows, pair_rows = summarize(iter_records(paths))
    write_csv(os.path.join(args.out_dir, 'oracle_summary.csv'), oracle_rows)
    write_csv(os.path.join(args.out_dir, 'class_oracle_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'pair_oracle_summary.csv'), pair_rows)
    print(f'Read {len(paths)} files')
    print(
        f'Wrote {len(oracle_rows)} oracle rows, {len(class_rows)} class rows, '
        f'and {len(pair_rows)} pair rows to {args.out_dir}')


if __name__ == '__main__':
    main()
