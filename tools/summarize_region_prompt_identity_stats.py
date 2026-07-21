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


def add_weighted(bucket, key, value, weight):
    if value is None:
        return
    bucket[f'{key}_sum'] += float(value) * weight
    bucket[f'{key}_weight'] += weight


def weighted(bucket, key):
    return safe_div(bucket.get(f'{key}_sum', 0.0),
                    bucket.get(f'{key}_weight', 0.0))


def main():
    parser = argparse.ArgumentParser(
        description='Summarize region-prompt identity diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No region identity JSONL files matched.')

    overall = defaultdict(lambda: defaultdict(float))
    pairs = defaultdict(lambda: defaultdict(float))
    oracle = defaultdict(lambda: defaultdict(float))
    groups = defaultdict(lambda: defaultdict(float))
    inventory = defaultdict(lambda: dict(images=set(), records=0))

    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get('region_prompt_identity_stats') or {}
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{record_index}")
        inventory[dataset]['images'].add(image_id)
        inventory[dataset]['records'] += 1
        inventory[dataset]['raw_candidates'] = (
            inventory[dataset].get('raw_candidates', 0)
            + int(stats.get('raw_candidate_count', 0) or 0))
        inventory[dataset]['region_groups'] = (
            inventory[dataset].get('region_groups', 0)
            + int(stats.get('region_group_count', 0) or 0))

        for row in stats.get('group_stats') or []:
            key = (
                dataset,
                row.get('scorer_name'),
                row.get('top1_class_index'),
                row.get('top1_class_name'),
                row.get('top1_role'),
            )
            b = groups[key]
            pixels = int(row.get('mask_pixels', 0) or 0)
            b['groups'] += 1
            b['mask_pixels'] += pixels
            b['top1_is_gt_mode_pixels'] += (
                pixels if row.get('top1_is_gt_mode') is True else 0)
            b['top1_is_pred_mode_pixels'] += (
                pixels if row.get('top1_is_pred_mode') is True else 0)
            add_weighted(
                b, 'baseline_correct_ratio',
                row.get('baseline_correct_ratio'), pixels)
            add_weighted(
                b, 'gt_mode_purity', row.get('gt_mode_purity'), pixels)
            add_weighted(
                b, 'member_class_count',
                row.get('member_class_count'), pixels)
            add_weighted(
                b, 'member_count', row.get('member_count'), pixels)

        for row in stats.get('pair_scorer_stats') or []:
            scorer = row.get('scorer_name')
            pair_key = (
                dataset,
                scorer,
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('gt_role'),
                row.get('pred_class_index'),
                row.get('pred_class_name'),
                row.get('pred_role'),
            )
            overall_key = (dataset, scorer)
            pixels = int(row.get('pair_pixels', 0) or 0)
            for key in (pair_key, overall_key):
                b = pairs[key] if len(key) > 2 else overall[key]
                b['regions'] += 1
                b['pair_pixels'] += pixels
                b['gt_beats_pred_pixels'] += (
                    pixels if row.get('gt_beats_pred') is True else 0)
                b['top1_is_gt_pixels'] += (
                    pixels if row.get('top1_is_gt') is True else 0)
                b['top1_is_pred_pixels'] += (
                    pixels if row.get('top1_is_pred') is True else 0)
                add_weighted(
                    b, 'gt_minus_pred', row.get('gt_minus_pred'), pixels)
                add_weighted(b, 'gt_rank', row.get('gt_rank'), pixels)
                add_weighted(b, 'pred_rank', row.get('pred_rank'), pixels)

        for row in stats.get('pair_oracle_stats') or []:
            pair_key = (
                dataset,
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('gt_role'),
                row.get('pred_class_index'),
                row.get('pred_class_name'),
                row.get('pred_role'),
            )
            overall_key = (dataset, 'any_scorer_oracle')
            pixels = int(row.get('pair_pixels', 0) or 0)
            for key in (pair_key, overall_key):
                b = oracle[key]
                b['regions'] += 1
                b['pair_pixels'] += pixels
                b['any_scorer_favors_gt_pixels'] += (
                    pixels if row.get('any_scorer_favors_gt') is True else 0)
                add_weighted(
                    b, 'best_gt_minus_pred',
                    row.get('best_gt_minus_pred'), pixels)
                add_weighted(b, 'best_gt_rank', row.get('best_gt_rank'), pixels)
                scorer = row.get('best_scorer')
                if scorer:
                    b[f'best_scorer::{scorer}'] += pixels

    overall_rows = []
    for key, b in sorted(overall.items()):
        dataset, scorer = key
        pixels = int(b['pair_pixels'])
        overall_rows.append(dict(
            dataset=dataset,
            scorer_name=scorer,
            regions=int(b['regions']),
            pair_pixels=pixels,
            gt_beats_pred_ratio=safe_div(
                b['gt_beats_pred_pixels'], pixels),
            top1_is_gt_ratio=safe_div(b['top1_is_gt_pixels'], pixels),
            top1_is_pred_ratio=safe_div(b['top1_is_pred_pixels'], pixels),
            mean_gt_minus_pred=weighted(b, 'gt_minus_pred'),
            mean_gt_rank=weighted(b, 'gt_rank'),
            mean_pred_rank=weighted(b, 'pred_rank'),
        ))

    pair_rows = []
    for key, b in sorted(pairs.items()):
        (
            dataset, scorer, gt_class_index, gt_class_name, gt_role,
            pred_class_index, pred_class_name, pred_role,
        ) = key
        pixels = int(b['pair_pixels'])
        pair_rows.append(dict(
            dataset=dataset,
            scorer_name=scorer,
            gt_class_index=gt_class_index,
            gt_class_name=gt_class_name,
            gt_role=gt_role,
            pred_class_index=pred_class_index,
            pred_class_name=pred_class_name,
            pred_role=pred_role,
            regions=int(b['regions']),
            pair_pixels=pixels,
            gt_beats_pred_ratio=safe_div(
                b['gt_beats_pred_pixels'], pixels),
            top1_is_gt_ratio=safe_div(b['top1_is_gt_pixels'], pixels),
            top1_is_pred_ratio=safe_div(b['top1_is_pred_pixels'], pixels),
            mean_gt_minus_pred=weighted(b, 'gt_minus_pred'),
            mean_gt_rank=weighted(b, 'gt_rank'),
            mean_pred_rank=weighted(b, 'pred_rank'),
        ))
    pair_rows.sort(
        key=lambda row: (row['dataset'], -row['pair_pixels'], row['scorer_name']))

    def oracle_sort_key(item):
        key, _ = item
        dataset = str(key[0]) if key else ''
        if len(key) == 2:
            return (dataset, 0, str(key[1]), '', '')
        return (
            dataset,
            1,
            str(key[2]),
            str(key[5]),
            str(key[1]),
        )

    oracle_rows = []
    for key, b in sorted(oracle.items(), key=oracle_sort_key):
        pixels = int(b['pair_pixels'])
        row = dict(pair_pixels=pixels,
                   regions=int(b['regions']),
                   any_scorer_favors_gt_ratio=safe_div(
                       b['any_scorer_favors_gt_pixels'], pixels),
                   mean_best_gt_minus_pred=weighted(
                       b, 'best_gt_minus_pred'),
                   mean_best_gt_rank=weighted(b, 'best_gt_rank'))
        if len(key) == 2:
            dataset, name = key
            row.update(dataset=dataset, oracle_name=name)
        else:
            (
                dataset, gt_class_index, gt_class_name, gt_role,
                pred_class_index, pred_class_name, pred_role,
            ) = key
            row.update(
                dataset=dataset,
                gt_class_index=gt_class_index,
                gt_class_name=gt_class_name,
                gt_role=gt_role,
                pred_class_index=pred_class_index,
                pred_class_name=pred_class_name,
                pred_role=pred_role,
            )
        scorer_items = [
            (name.split('::', 1)[1], count)
            for name, count in b.items()
            if name.startswith('best_scorer::')
        ]
        if scorer_items:
            scorer, count = max(scorer_items, key=lambda item: item[1])
            row['top_best_scorer'] = scorer
            row['top_best_scorer_ratio'] = safe_div(count, pixels)
        oracle_rows.append(row)
    oracle_rows.sort(
        key=lambda row: (
            row['dataset'],
            -row['pair_pixels'],
            row.get('gt_class_name') or row.get('oracle_name') or '',
        ))

    group_rows = []
    for key, b in sorted(groups.items()):
        dataset, scorer, top1_class_index, top1_class_name, top1_role = key
        pixels = int(b['mask_pixels'])
        group_rows.append(dict(
            dataset=dataset,
            scorer_name=scorer,
            top1_class_index=top1_class_index,
            top1_class_name=top1_class_name,
            top1_role=top1_role,
            groups=int(b['groups']),
            mask_pixels=pixels,
            top1_is_gt_mode_ratio=safe_div(
                b['top1_is_gt_mode_pixels'], pixels),
            top1_is_pred_mode_ratio=safe_div(
                b['top1_is_pred_mode_pixels'], pixels),
            baseline_correct_ratio=weighted(
                b, 'baseline_correct_ratio'),
            gt_mode_purity=weighted(b, 'gt_mode_purity'),
            mean_member_class_count=weighted(b, 'member_class_count'),
            mean_member_count=weighted(b, 'member_count'),
        ))
    group_rows.sort(
        key=lambda row: (row['dataset'], row['scorer_name'], -row['mask_pixels']))

    inventory_rows = [
        dict(dataset=dataset,
             images=len(bucket['images']),
             records=int(bucket['records']),
             raw_candidates=int(bucket.get('raw_candidates', 0)),
             region_groups=int(bucket.get('region_groups', 0)))
        for dataset, bucket in sorted(inventory.items())
    ]

    write_csv(
        os.path.join(args.output_dir, 'region_identity_overall.csv'),
        overall_rows)
    write_csv(
        os.path.join(args.output_dir, 'region_identity_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(args.output_dir, 'region_identity_oracle_summary.csv'),
        oracle_rows)
    write_csv(
        os.path.join(args.output_dir, 'region_identity_group_summary.csv'),
        group_rows)
    write_csv(
        os.path.join(args.output_dir, 'region_identity_inventory.csv'),
        inventory_rows)


if __name__ == '__main__':
    main()
