import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize SegEarth-OV3 top-k verifier JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/topk_verifier/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for overall_summary.csv, class_summary.csv, and pair_summary.csv')
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


def add_pixels(bucket, item, skip=None):
    skip = skip or set()
    for key, value in item.items():
        if key in skip:
            continue
        if key.endswith('_pixels') and isinstance(value, int):
            bucket[key] += value
        elif key == 'net_improved_pixels' and isinstance(value, int):
            bucket[key] += value


def summarize(records):
    overall = defaultdict(int)
    class_counts = defaultdict(lambda: defaultdict(int))
    class_names = defaultdict(list)
    regime_counts = defaultdict(lambda: defaultdict(int))
    pair_counts = defaultdict(lambda: defaultdict(int))
    pair_names = defaultdict(lambda: defaultdict(list))

    for record in records:
        stats = record.get('topk_verifier_stats') or {}
        valid_pixels = stats.get('valid_pixels') or 0
        overall['images'] += 1
        overall['valid_pixels'] += valid_pixels
        add_pixels(overall, stats.get('overall') or {}, skip={'valid_pixels'})

        for class_stat in stats.get('class_stats') or []:
            class_idx = class_stat.get('class_index')
            if class_idx is None:
                continue
            bucket = class_counts[class_idx]
            bucket['images'] += 1
            bucket['gt_pixels'] += class_stat.get('gt_pixels') or 0
            class_names[class_idx].append(class_stat.get('class_name'))
            add_pixels(bucket, class_stat, skip={'gt_pixels'})

        for regime_stat in stats.get('regime_stats') or []:
            regime = regime_stat.get('regime')
            if regime is None:
                continue
            bucket = regime_counts[regime]
            bucket['images'] += 1
            bucket['pixels'] += regime_stat.get('pixels') or 0
            add_pixels(bucket, regime_stat, skip={'pixels'})

        for pair_stat in stats.get('pair_stats') or []:
            gt_class = pair_stat.get('gt_class_index')
            pred_class = pair_stat.get('base_pred_class_index')
            if gt_class is None or pred_class is None:
                continue
            pair_key = (gt_class, pred_class)
            bucket = pair_counts[pair_key]
            bucket['images'] += 1
            bucket['pixels'] += pair_stat.get('pixels') or 0
            pair_names[pair_key]['gt_class_name'].append(pair_stat.get('gt_class_name'))
            pair_names[pair_key]['base_pred_class_name'].append(pair_stat.get('base_pred_class_name'))
            add_pixels(bucket, pair_stat, skip={'pixels'})

    overall_rows = [build_row(dict(name='topk_verifier', **overall), overall.get('valid_pixels', 0))]

    class_rows = []
    for class_idx, counts in sorted(class_counts.items()):
        gt_pixels = counts.get('gt_pixels', 0)
        row = build_row(dict(
            class_index=class_idx,
            class_name=first(class_names[class_idx]),
            **counts,
        ), gt_pixels, skip={'gt_pixels'})
        class_rows.append(row)

    pair_rows = []
    for (gt_class, pred_class), counts in sorted(pair_counts.items()):
        pixels = counts.get('pixels', 0)
        row = build_row(dict(
            gt_class_index=gt_class,
            gt_class_name=first(pair_names[(gt_class, pred_class)]['gt_class_name']),
            base_pred_class_index=pred_class,
            base_pred_class_name=first(pair_names[(gt_class, pred_class)]['base_pred_class_name']),
            **counts,
        ), pixels, skip={'pixels'})
        pair_rows.append(row)
    pair_rows.sort(key=lambda item: item.get('pixels') or 0, reverse=True)

    regime_rows = []
    for regime, counts in sorted(regime_counts.items()):
        pixels = counts.get('pixels', 0)
        row = build_row(dict(regime=regime, **counts), pixels, skip={'pixels'})
        regime_rows.append(row)

    return overall_rows, class_rows, regime_rows, pair_rows


def build_row(values, denominator, skip=None):
    skip = skip or set()
    row = dict(values)
    for key, value in list(values.items()):
        if key in skip:
            continue
        if key.endswith('_pixels') and isinstance(value, int):
            row[key[:-7] + '_ratio'] = safe_div(value, denominator)
    if 'net_improved_pixels' in values:
        row['net_improved_ratio'] = safe_div(values['net_improved_pixels'], denominator)
    return row


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        'name',
        'class_index',
        'class_name',
        'regime',
        'gt_class_index',
        'gt_class_name',
        'base_pred_class_index',
        'base_pred_class_name',
        'images',
        'valid_pixels',
        'gt_pixels',
        'pixels',
        'base_correct_pixels',
        'base_correct_ratio',
        'verifier_correct_pixels',
        'verifier_correct_ratio',
        'changed_pixels',
        'changed_ratio',
        'improved_pixels',
        'improved_ratio',
        'harmed_pixels',
        'harmed_ratio',
        'net_improved_pixels',
        'net_improved_ratio',
        'recovered_pixels',
        'recovered_ratio',
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
    overall_rows, class_rows, regime_rows, pair_rows = summarize(iter_records(paths))
    write_csv(os.path.join(args.out_dir, 'overall_summary.csv'), overall_rows)
    write_csv(os.path.join(args.out_dir, 'class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'regime_summary.csv'), regime_rows)
    write_csv(os.path.join(args.out_dir, 'pair_summary.csv'), pair_rows)
    print(f'Read {len(paths)} files')
    print(
        f'Wrote {len(overall_rows)} overall rows, {len(class_rows)} class rows, '
        f'{len(regime_rows)} regime rows, and {len(pair_rows)} pair rows to {args.out_dir}')


if __name__ == '__main__':
    main()
