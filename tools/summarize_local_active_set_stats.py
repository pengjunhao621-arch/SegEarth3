#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    return [path for path in paths if os.path.exists(path)]


def iter_records(paths):
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_bucket():
    return defaultdict(float)


def add_weighted_mean(bucket, key, value, weight):
    if value is None:
        return
    bucket[f'{key}_sum'] += float(value) * weight
    bucket[f'{key}_weight'] += weight


def weighted_mean(bucket, key):
    return safe_div(bucket.get(f'{key}_sum', 0.0),
                    bucket.get(f'{key}_weight', 0.0))


def main():
    parser = argparse.ArgumentParser(
        description='Summarize local active-set diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No local active-set JSONL files matched.')

    overall = defaultdict(new_bucket)
    classes = defaultdict(new_bucket)
    pairs = defaultdict(new_bucket)
    inventory = defaultdict(lambda: dict(images=set(), records=0))

    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get('local_active_set_stats') or {}
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{record_index}")
        inventory[dataset]['images'].add(image_id)
        inventory[dataset]['records'] += 1

        for row in stats.get('overall') or []:
            key = (
                dataset,
                row.get('window'),
                row.get('area_threshold'),
                row.get('sources'),
            )
            bucket = overall[key]
            bucket['images'] += 1
            for name in [
                    'valid_pixels', 'wrong_pixels',
                    'wrong_gt_kept_pixels',
                    'wrong_pred_removed_pixels',
                    'desired_keep_gt_remove_pred_pixels',
                    'both_gt_and_pred_kept_pixels',
                    'gt_removed_pixels',
                    'pred_removed_but_gt_removed_pixels',
            ]:
                bucket[name] += int(row.get(name, 0) or 0)
            add_weighted_mean(
                bucket, 'mean_active_count_on_wrong',
                row.get('mean_active_count_on_wrong'),
                int(row.get('wrong_pixels', 0) or 0))
            add_weighted_mean(
                bucket, 'mean_active_count_on_valid',
                row.get('mean_active_count_on_valid'),
                int(row.get('valid_pixels', 0) or 0))

        for row in stats.get('class_stats') or []:
            key = (
                dataset,
                row.get('window'),
                row.get('area_threshold'),
                row.get('class_index'),
                row.get('class_name'),
                row.get('role'),
            )
            bucket = classes[key]
            bucket['images'] += 1
            for name in [
                    'wrong_pixels', 'gt_kept_pixels',
                    'pred_removed_pixels', 'desired_pixels']:
                bucket[name] += int(row.get(name, 0) or 0)
            add_weighted_mean(
                bucket, 'mean_active_count',
                row.get('mean_active_count'),
                int(row.get('wrong_pixels', 0) or 0))

        for row in stats.get('pair_stats') or []:
            key = (
                dataset,
                row.get('window'),
                row.get('area_threshold'),
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('gt_role'),
                row.get('pred_class_index'),
                row.get('pred_class_name'),
                row.get('pred_role'),
            )
            bucket = pairs[key]
            bucket['images'] += 1
            for name in [
                    'pair_pixels', 'gt_kept_pixels',
                    'pred_removed_pixels', 'desired_pixels',
                    'both_kept_pixels', 'gt_removed_pixels']:
                bucket[name] += int(row.get(name, 0) or 0)
            add_weighted_mean(
                bucket, 'mean_active_count',
                row.get('mean_active_count'),
                int(row.get('pair_pixels', 0) or 0))

    overall_rows = []
    for key, bucket in sorted(overall.items()):
        dataset, window, area_threshold, sources = key
        wrong = int(bucket['wrong_pixels'])
        row = dict(
            dataset=dataset,
            window=window,
            area_threshold=area_threshold,
            sources=sources,
            images=int(bucket['images']),
            valid_pixels=int(bucket['valid_pixels']),
            wrong_pixels=wrong,
            wrong_gt_kept_ratio=safe_div(
                bucket['wrong_gt_kept_pixels'], wrong),
            wrong_pred_removed_ratio=safe_div(
                bucket['wrong_pred_removed_pixels'], wrong),
            desired_keep_gt_remove_pred_ratio=safe_div(
                bucket['desired_keep_gt_remove_pred_pixels'], wrong),
            both_gt_and_pred_kept_ratio=safe_div(
                bucket['both_gt_and_pred_kept_pixels'], wrong),
            gt_removed_ratio=safe_div(
                bucket['gt_removed_pixels'], wrong),
            pred_removed_but_gt_removed_ratio=safe_div(
                bucket['pred_removed_but_gt_removed_pixels'], wrong),
            mean_active_count_on_wrong=weighted_mean(
                bucket, 'mean_active_count_on_wrong'),
            mean_active_count_on_valid=weighted_mean(
                bucket, 'mean_active_count_on_valid'),
            desired_pixels=int(
                bucket['desired_keep_gt_remove_pred_pixels']),
            gt_removed_pixels=int(bucket['gt_removed_pixels']),
        )
        overall_rows.append(row)

    class_rows = []
    for key, bucket in sorted(classes.items()):
        (
            dataset, window, area_threshold, class_index, class_name, role,
        ) = key
        wrong = int(bucket['wrong_pixels'])
        class_rows.append(dict(
            dataset=dataset,
            window=window,
            area_threshold=area_threshold,
            class_index=class_index,
            class_name=class_name,
            role=role,
            images=int(bucket['images']),
            wrong_pixels=wrong,
            gt_kept_ratio=safe_div(bucket['gt_kept_pixels'], wrong),
            pred_removed_ratio=safe_div(
                bucket['pred_removed_pixels'], wrong),
            desired_ratio=safe_div(bucket['desired_pixels'], wrong),
            mean_active_count=weighted_mean(
                bucket, 'mean_active_count'),
        ))

    pair_rows = []
    for key, bucket in sorted(pairs.items()):
        (
            dataset, window, area_threshold, gt_class_index, gt_class_name,
            gt_role, pred_class_index, pred_class_name, pred_role,
        ) = key
        pixels = int(bucket['pair_pixels'])
        pair_rows.append(dict(
            dataset=dataset,
            window=window,
            area_threshold=area_threshold,
            gt_class_index=gt_class_index,
            gt_class_name=gt_class_name,
            gt_role=gt_role,
            pred_class_index=pred_class_index,
            pred_class_name=pred_class_name,
            pred_role=pred_role,
            images=int(bucket['images']),
            pair_pixels=pixels,
            gt_kept_ratio=safe_div(bucket['gt_kept_pixels'], pixels),
            pred_removed_ratio=safe_div(
                bucket['pred_removed_pixels'], pixels),
            desired_ratio=safe_div(bucket['desired_pixels'], pixels),
            both_kept_ratio=safe_div(bucket['both_kept_pixels'], pixels),
            gt_removed_ratio=safe_div(bucket['gt_removed_pixels'], pixels),
            mean_active_count=weighted_mean(bucket, 'mean_active_count'),
        ))
    pair_rows.sort(
        key=lambda row: (
            row['dataset'],
            -int(row['pair_pixels']),
            row['window'],
            float(row['area_threshold']),
        ))

    inventory_rows = [
        dict(dataset=dataset,
             images=len(bucket['images']),
             records=int(bucket['records']))
        for dataset, bucket in sorted(inventory.items())
    ]

    write_csv(
        os.path.join(args.output_dir, 'local_active_overall.csv'),
        overall_rows)
    write_csv(
        os.path.join(args.output_dir, 'local_active_class_summary.csv'),
        class_rows)
    write_csv(
        os.path.join(args.output_dir, 'local_active_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(args.output_dir, 'local_active_inventory.csv'),
        inventory_rows)


if __name__ == '__main__':
    main()
