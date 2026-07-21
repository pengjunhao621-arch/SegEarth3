import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SegEarth-OV3 top2 risk/arbitration JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/top2_risk/udd5_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for overall_summary.csv, regime_summary.csv, class_summary.csv, and pair_summary.csv')
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


def add_row(bucket, row):
    pixels = row.get('pixels') or 0
    bucket['images'] += 1
    bucket['pixels'] += pixels
    for key, value in row.items():
        if key in {
                'name', 'regime', 'class_name', 'gt_class_name',
                'base_pred_class_name'}:
            continue
        if key in {'class_index', 'gt_class_index', 'base_pred_class_index'}:
            continue
        if key == 'pixels':
            continue
        if key.endswith('_pixels') and isinstance(value, int):
            bucket[key] += value
        elif key == 'net_improved_pixels' and isinstance(value, int):
            bucket[key] += value
        elif key == 'oracle_flip_net_pixels' and isinstance(value, int):
            bucket[key] += value
        elif key.startswith('mean_') and value is not None:
            bucket[f'__sum__{key}'] += value * pixels
            bucket[f'__den__{key}'] += pixels


def finalize_row(base, counts, denominator):
    row = dict(base)
    row.update({k: v for k, v in counts.items() if not k.startswith('__')})
    pixels = row.get('pixels') or 0
    row['pixel_ratio'] = safe_div(pixels, denominator)
    for key, value in list(row.items()):
        if key.endswith('_pixels') and isinstance(value, (int, float)):
            row[key[:-7] + '_ratio'] = safe_div(value, pixels)
    if 'net_improved_pixels' in row:
        row['net_improved_ratio'] = safe_div(row['net_improved_pixels'], pixels)
    if 'oracle_flip_net_pixels' in row:
        row['oracle_flip_net_ratio'] = safe_div(row['oracle_flip_net_pixels'], pixels)
    if 'apply_pixels' in row and 'top2_is_gt_pixels' in row:
        # This is not the same as precision for arbitrary rows, because
        # top2_is_gt_pixels counts all pixels in the row. Applied precision is
        # directly dumped by the model for each image, so aggregate it via counts.
        pass
    if 'apply_pixels' in row and 'improved_pixels' in row and 'harmed_pixels' in row:
        row['applied_net_precision_proxy'] = safe_div(
            row['improved_pixels'] - row['harmed_pixels'], row['apply_pixels'])
    for key, value in counts.items():
        if key.startswith('__sum__'):
            mean_key = key.replace('__sum__', '')
            den = counts.get(f'__den__{mean_key}', 0)
            row[mean_key] = safe_div(value, den)
    return row


def summarize(records):
    total = defaultdict(int)
    overall_counts = defaultdict(float)
    regime_counts = defaultdict(lambda: defaultdict(float))
    class_counts = defaultdict(lambda: defaultdict(float))
    class_names = defaultdict(list)
    pair_counts = defaultdict(lambda: defaultdict(float))
    pair_names = defaultdict(lambda: defaultdict(list))

    for record in records:
        stats = record.get('top2_risk_stats') or {}
        valid_pixels = stats.get('valid_pixels') or 0
        total['images'] += 1
        total['valid_pixels'] += valid_pixels

        overall = stats.get('overall')
        if overall:
            add_row(overall_counts, overall)

        for row in stats.get('regime_stats') or []:
            regime = row.get('regime')
            if regime is None:
                continue
            add_row(regime_counts[regime], row)

        for row in stats.get('class_stats') or []:
            class_idx = row.get('class_index')
            if class_idx is None:
                continue
            add_row(class_counts[class_idx], row)
            class_names[class_idx].append(row.get('class_name'))

        for row in stats.get('pair_stats') or []:
            gt_class = row.get('gt_class_index')
            pred_class = row.get('base_pred_class_index')
            if gt_class is None or pred_class is None:
                continue
            key = (gt_class, pred_class)
            add_row(pair_counts[key], row)
            pair_names[key]['gt_class_name'].append(row.get('gt_class_name'))
            pair_names[key]['base_pred_class_name'].append(row.get('base_pred_class_name'))

    valid_pixels = total.get('valid_pixels', 0)
    overall_rows = [finalize_row(
        dict(name='all_pixels',
             images=total.get('images', 0),
             valid_pixels=valid_pixels),
        overall_counts,
        valid_pixels)]

    regime_rows = []
    for regime, counts in sorted(regime_counts.items()):
        regime_rows.append(finalize_row(dict(regime=regime), counts, valid_pixels))
    regime_rows.sort(key=lambda item: item.get('pixels') or 0, reverse=True)

    class_rows = []
    for class_idx, counts in sorted(class_counts.items()):
        class_rows.append(finalize_row(
            dict(class_index=class_idx, class_name=first(class_names[class_idx])),
            counts,
            valid_pixels))

    pair_rows = []
    for (gt_class, pred_class), counts in sorted(pair_counts.items()):
        pair_rows.append(finalize_row(
            dict(
                gt_class_index=gt_class,
                gt_class_name=first(pair_names[(gt_class, pred_class)]['gt_class_name']),
                base_pred_class_index=pred_class,
                base_pred_class_name=first(pair_names[(gt_class, pred_class)]['base_pred_class_name']),
            ),
            counts,
            valid_pixels))
    pair_rows.sort(key=lambda item: item.get('pixels') or 0, reverse=True)

    return overall_rows, regime_rows, class_rows, pair_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        'name',
        'regime',
        'class_index',
        'class_name',
        'gt_class_index',
        'gt_class_name',
        'base_pred_class_index',
        'base_pred_class_name',
        'images',
        'valid_pixels',
        'pixels',
        'pixel_ratio',
        'base_correct_pixels',
        'base_correct_ratio',
        'top2_is_gt_pixels',
        'top2_is_gt_ratio',
        'apply_pixels',
        'apply_ratio',
        'applied_top2_is_gt_pixels',
        'applied_top2_is_gt_ratio',
        'changed_pixels',
        'changed_ratio',
        'improved_pixels',
        'improved_ratio',
        'harmed_pixels',
        'harmed_ratio',
        'net_improved_pixels',
        'net_improved_ratio',
        'applied_net_precision_proxy',
        'oracle_flip_improve_pixels',
        'oracle_flip_improve_ratio',
        'oracle_flip_harm_pixels',
        'oracle_flip_harm_ratio',
        'oracle_flip_net_pixels',
        'oracle_flip_net_ratio',
        'mean_top1_gt_gap',
        'mean_final_margin',
        'mean_semantic_advantage',
        'mean_instance_advantage',
        'mean_agreement_advantage',
        'mean_top1_sem_only',
        'threshold_reject_ratio',
        'low_margin_ratio',
        'confident_margin_ratio',
        'head_disagree_ratio',
        'sem_support_ratio',
        'inst_support_ratio',
        'agreement_support_ratio',
        'top1_sem_overexpands_ratio',
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
    overall_rows, regime_rows, class_rows, pair_rows = summarize(iter_records(paths))
    write_csv(os.path.join(args.out_dir, 'overall_summary.csv'), overall_rows)
    write_csv(os.path.join(args.out_dir, 'regime_summary.csv'), regime_rows)
    write_csv(os.path.join(args.out_dir, 'class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'pair_summary.csv'), pair_rows)
    print(f'Read {len(paths)} files')
    print(
        f'Wrote {len(overall_rows)} overall rows, {len(regime_rows)} regime rows, '
        f'{len(class_rows)} class rows, and {len(pair_rows)} pair rows to {args.out_dir}')


if __name__ == '__main__':
    main()
