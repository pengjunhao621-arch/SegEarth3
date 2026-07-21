import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SegEarth-OV3 multi-view evidence consistency oracle JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/multiview_oracle/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for view_summary.csv, regime_summary.csv, class_summary.csv, and pair_summary.csv')
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
                'name', 'view_name', 'regime', 'class_name',
                'gt_class_name', 'base_pred_class_name'}:
            continue
        if key in {'class_index', 'gt_class_index', 'base_pred_class_index'}:
            continue
        if key == 'pixels':
            continue
        if key.endswith('_pixels') and isinstance(value, int):
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
    for key, value in counts.items():
        if key.startswith('__sum__'):
            mean_key = key.replace('__sum__', '')
            den = counts.get(f'__den__{mean_key}', 0)
            row[mean_key] = safe_div(value, den)
    return row


def summarize(records):
    total = defaultdict(int)
    view_counts = defaultdict(lambda: defaultdict(float))
    regime_counts = defaultdict(lambda: defaultdict(float))
    class_counts = defaultdict(lambda: defaultdict(float))
    class_names = defaultdict(list)
    pair_counts = defaultdict(lambda: defaultdict(float))
    pair_names = defaultdict(lambda: defaultdict(list))
    view_name_sets = defaultdict(set)

    for record in records:
        stats = record.get('multiview_oracle_stats') or {}
        valid_pixels = stats.get('valid_pixels') or 0
        total['images'] += 1
        total['valid_pixels'] += valid_pixels
        for view_name in stats.get('view_names') or []:
            view_name_sets['all'].add(view_name)

        for row in stats.get('view_stats') or []:
            view_name = row.get('view_name')
            if view_name is None:
                continue
            add_row(view_counts[view_name], row)

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

    view_rows = []
    for view_name, counts in sorted(view_counts.items()):
        view_rows.append(finalize_row(dict(view_name=view_name), counts, valid_pixels))

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

    meta_rows = [dict(
        images=total.get('images', 0),
        valid_pixels=valid_pixels,
        view_names=','.join(sorted(view_name_sets['all'])),
    )]
    return meta_rows, view_rows, regime_rows, class_rows, pair_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        'view_name',
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
        'base_wrong_pixels',
        'base_wrong_ratio',
        'threshold_reject_ratio',
        'low_margin_ratio',
        'confident_margin_ratio',
        'base_final_topk_contains_gt_ratio',
        'base_any_head_topk_contains_gt_ratio',
        'any_view_final_topk_contains_gt_ratio',
        'any_view_semantic_topk_contains_gt_ratio',
        'any_view_instance_topk_contains_gt_ratio',
        'any_view_any_head_topk_contains_gt_ratio',
        'any_view_final_top1_gt_ratio',
        'stable_final_topk_ratio',
        'stable_any_head_topk_ratio',
        'base_large_gap_ratio',
        'large_gap_resolved_ratio',
        'gap_reduced_0p10_ratio',
        'mean_base_top1_gt_gap',
        'mean_best_final_top1_gt_gap',
        'mean_gap_reduction',
        'mean_base_final_margin',
        'mean_final_topk_view_count',
        'mean_any_head_topk_view_count',
        'final_topk_contains_gt_ratio',
        'semantic_topk_contains_gt_ratio',
        'instance_topk_contains_gt_ratio',
        'any_head_topk_contains_gt_ratio',
        'final_top1_gt_ratio',
        'mean_top1_gt_gap',
        'mean_final_margin',
        'view_names',
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
    meta_rows, view_rows, regime_rows, class_rows, pair_rows = summarize(iter_records(paths))
    write_csv(os.path.join(args.out_dir, 'meta_summary.csv'), meta_rows)
    write_csv(os.path.join(args.out_dir, 'view_summary.csv'), view_rows)
    write_csv(os.path.join(args.out_dir, 'regime_summary.csv'), regime_rows)
    write_csv(os.path.join(args.out_dir, 'class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'pair_summary.csv'), pair_rows)
    print(f'Read {len(paths)} files')
    print(
        f'Wrote {len(view_rows)} view rows, {len(regime_rows)} regime rows, '
        f'{len(class_rows)} class rows, and {len(pair_rows)} pair rows to {args.out_dir}')


if __name__ == '__main__':
    main()
