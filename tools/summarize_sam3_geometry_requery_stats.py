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
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def new_acc(key_values=None):
    acc = defaultdict(float)
    if key_values:
        acc.update(key_values)
    acc['images'] = set()
    acc['pairs_seen'] = set()
    return acc


def add_weighted(acc, sum_key, mean_value, weight):
    if mean_value is None or weight is None or weight <= 0:
        return
    acc[sum_key] += float(mean_value) * float(weight)


def update_acc(acc, record, pair, mode):
    image_id = record.get('img_path') or record.get('image_id') or 'unknown'
    pair_key = (
        pair.get('pred_class_name'),
        pair.get('target_class_name'),
        mode.get('mode'),
    )
    acc['images'].add(image_id)
    acc['pairs_seen'].add(pair_key)
    support_pixels = int(pair.get('support_pixels') or 0)
    target_gt = int(mode.get('target_gt_pixels') or 0)
    pred_gt = int(mode.get('pred_gt_pixels') or 0)
    other_gt = int(mode.get('other_gt_pixels') or 0)
    help_pixels = int(
        mode.get('requery_beats_base_pred_on_target_gt_pixels') or 0)
    harm_pixels = int(
        mode.get('harmful_requery_beats_base_pred_on_pred_gt_pixels') or 0)
    pair_overtake = int(mode.get('pair_requery_beats_base_pred_pixels') or 0)
    pair_support = int(pair.get('support_pixels') or 0)

    acc['support_pixels'] += support_pixels
    acc['target_gt_pixels'] += target_gt
    acc['pred_gt_pixels'] += pred_gt
    acc['other_gt_pixels'] += other_gt
    acc['target_help_pixels'] += help_pixels
    acc['harmful_pixels'] += harm_pixels
    acc['pair_overtake_pixels'] += pair_overtake
    acc['pair_support_for_overtake'] += pair_support
    acc['target_seed_pixels'] += int(pair.get('target_seed_pixels') or 0)
    acc['pred_seed_pixels'] += int(pair.get('pred_seed_pixels') or 0)
    acc['presence_sum'] += float(mode.get('presence_score') or 0.0)
    acc['num_masks_sum'] += float(mode.get('num_masks') or 0.0)
    acc['mode_rows'] += 1

    add_weighted(
        acc, 'base_gap_target_sum',
        mode.get('mean_base_gap_on_target_gt'), target_gt)
    add_weighted(
        acc, 'requery_gain_target_sum',
        mode.get('mean_requery_gain_on_target_gt'), target_gt)
    add_weighted(
        acc, 'requery_margin_target_sum',
        mode.get('mean_requery_margin_on_target_gt'), target_gt)
    add_weighted(
        acc, 'requery_gain_pred_sum',
        mode.get('mean_requery_gain_on_pred_gt'), pred_gt)
    add_weighted(
        acc, 'requery_margin_pred_sum',
        mode.get('mean_requery_margin_on_pred_gt'), pred_gt)
    add_weighted(
        acc, 'pair_base_gap_sum',
        pair.get('mean_base_gap'), support_pixels)
    add_weighted(
        acc, 'pair_requery_gain_sum',
        mode.get('pair_mean_requery_gain'), pair_support)


def finalize_acc(acc):
    row = dict(acc)
    row['images'] = len(acc['images'])
    row['pairs'] = len(acc['pairs_seen'])
    row['target_gt_ratio'] = safe_div(
        acc['target_gt_pixels'], acc['support_pixels'])
    row['pred_gt_ratio'] = safe_div(
        acc['pred_gt_pixels'], acc['support_pixels'])
    row['target_help_ratio'] = safe_div(
        acc['target_help_pixels'], acc['target_gt_pixels'])
    row['harmful_ratio'] = safe_div(
        acc['harmful_pixels'], acc['pred_gt_pixels'])
    row['pair_overtake_ratio'] = safe_div(
        acc['pair_overtake_pixels'], acc['pair_support_for_overtake'])
    row['net_help_minus_harm_pixels'] = (
        acc['target_help_pixels'] - acc['harmful_pixels'])
    row['mean_base_gap_on_target_gt'] = safe_div(
        acc['base_gap_target_sum'], acc['target_gt_pixels'])
    row['mean_requery_gain_on_target_gt'] = safe_div(
        acc['requery_gain_target_sum'], acc['target_gt_pixels'])
    row['mean_requery_margin_on_target_gt'] = safe_div(
        acc['requery_margin_target_sum'], acc['target_gt_pixels'])
    row['mean_requery_gain_on_pred_gt'] = safe_div(
        acc['requery_gain_pred_sum'], acc['pred_gt_pixels'])
    row['mean_requery_margin_on_pred_gt'] = safe_div(
        acc['requery_margin_pred_sum'], acc['pred_gt_pixels'])
    row['mean_pair_base_gap'] = safe_div(
        acc['pair_base_gap_sum'], acc['support_pixels'])
    row['mean_pair_requery_gain'] = safe_div(
        acc['pair_requery_gain_sum'], acc['pair_support_for_overtake'])
    row['mean_presence_score'] = safe_div(acc['presence_sum'], acc['mode_rows'])
    row['mean_num_masks'] = safe_div(acc['num_masks_sum'], acc['mode_rows'])
    return row


def iter_mode_rows(paths):
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context = record.get('sam3_geometry_requery')
                if not context:
                    continue
                for pair in context.get('pair_stats') or []:
                    for mode in pair.get('mode_stats') or []:
                        if mode.get('error'):
                            continue
                        yield record, context, pair, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No SAM3 geometry re-query JSONL found.')

    dataset_acc = {}
    mode_acc = {}
    role_pair_acc = {}
    pair_acc = {}
    rows = []

    for record, context, pair, mode in iter_mode_rows(paths):
        dataset = (
            record.get('dataset_name')
            or context.get('dataset_name')
            or 'unknown')
        mode_name = mode.get('mode') or 'unknown'
        pred_name = pair.get('pred_class_name')
        target_name = pair.get('target_class_name')
        pred_role = pair.get('pred_role')
        target_role = pair.get('target_role')

        keys = [
            (dataset_acc, (dataset,), dict(dataset=dataset)),
            (mode_acc, (dataset, mode_name),
             dict(dataset=dataset, mode=mode_name)),
            (role_pair_acc, (dataset, mode_name, pred_role, target_role),
             dict(dataset=dataset, mode=mode_name,
                  pred_role=pred_role, target_role=target_role,
                  role_pair=f'{pred_role}->{target_role}')),
            (pair_acc, (dataset, mode_name, pred_name, target_name),
             dict(dataset=dataset, mode=mode_name,
                  pred_class_name=pred_name,
                  target_class_name=target_name,
                  pair=f'{pred_name}->{target_name}',
                  pred_role=pred_role,
                  target_role=target_role)),
        ]
        for table, key, key_values in keys:
            if key not in table:
                table[key] = new_acc(key_values)
            update_acc(table[key], record, pair, mode)

        rows.append(dict(
            dataset=dataset,
            image_id=record.get('img_path'),
            mode=mode_name,
            pred_class_name=pred_name,
            target_class_name=target_name,
            pred_role=pred_role,
            target_role=target_role,
            support_pixels=pair.get('support_pixels'),
            target_gt_pixels=mode.get('target_gt_pixels'),
            pred_gt_pixels=mode.get('pred_gt_pixels'),
            target_help_ratio=mode.get(
                'requery_beats_base_pred_on_target_gt_ratio'),
            harmful_ratio=mode.get(
                'harmful_requery_beats_base_pred_on_pred_gt_ratio'),
            mean_base_gap_on_target_gt=mode.get(
                'mean_base_gap_on_target_gt'),
            mean_requery_gain_on_target_gt=mode.get(
                'mean_requery_gain_on_target_gt'),
            mean_requery_margin_on_target_gt=mode.get(
                'mean_requery_margin_on_target_gt'),
            pair_requery_beats_base_pred_ratio=mode.get(
                'pair_requery_beats_base_pred_ratio'),
            presence_score=mode.get('presence_score'),
            num_masks=mode.get('num_masks'),
            target_prompt=pair.get('target_prompt'),
            pred_prompt=pair.get('pred_prompt'),
        ))

    common_fields = [
        'dataset', 'mode', 'role_pair', 'pair', 'pred_role', 'target_role',
        'pred_class_name', 'target_class_name', 'images', 'pairs',
        'mode_rows', 'support_pixels', 'target_seed_pixels',
        'pred_seed_pixels', 'target_gt_pixels', 'pred_gt_pixels',
        'other_gt_pixels', 'target_gt_ratio', 'pred_gt_ratio',
        'target_help_pixels', 'harmful_pixels', 'target_help_ratio',
        'harmful_ratio', 'net_help_minus_harm_pixels',
        'pair_overtake_pixels', 'pair_overtake_ratio',
        'mean_base_gap_on_target_gt', 'mean_requery_gain_on_target_gt',
        'mean_requery_margin_on_target_gt',
        'mean_requery_gain_on_pred_gt',
        'mean_requery_margin_on_pred_gt',
        'mean_pair_base_gap', 'mean_pair_requery_gain',
        'mean_presence_score', 'mean_num_masks',
    ]

    dataset_rows = [finalize_acc(acc) for acc in dataset_acc.values()]
    mode_rows = [finalize_acc(acc) for acc in mode_acc.values()]
    role_rows = [finalize_acc(acc) for acc in role_pair_acc.values()]
    pair_rows = [finalize_acc(acc) for acc in pair_acc.values()]

    dataset_rows.sort(key=lambda row: row.get('dataset') or '')
    mode_rows.sort(key=lambda row: (
        row.get('dataset') or '', row.get('mode') or ''))
    role_rows.sort(key=lambda row: (
        row.get('dataset') or '', row.get('mode') or '',
        row.get('role_pair') or ''))
    pair_rows.sort(key=lambda row: (
        row.get('dataset') or '', row.get('mode') or '',
        row.get('pair') or ''))

    write_csv(
        os.path.join(args.output_dir, 'dataset_summary.csv'),
        dataset_rows,
        [field for field in common_fields if field not in {
            'mode', 'role_pair', 'pair', 'pred_role', 'target_role',
            'pred_class_name', 'target_class_name'}])
    write_csv(
        os.path.join(args.output_dir, 'mode_summary.csv'),
        mode_rows,
        [field for field in common_fields if field not in {
            'role_pair', 'pair', 'pred_role', 'target_role',
            'pred_class_name', 'target_class_name'}])
    write_csv(
        os.path.join(args.output_dir, 'role_pair_summary.csv'),
        role_rows,
        [field for field in common_fields if field not in {
            'pair', 'pred_class_name', 'target_class_name'}])
    write_csv(
        os.path.join(args.output_dir, 'pair_summary.csv'),
        pair_rows,
        [field for field in common_fields if field not in {'role_pair'}])
    write_csv(
        os.path.join(args.output_dir, 'image_pair_rows.csv'),
        rows,
        [
            'dataset', 'image_id', 'mode', 'pred_class_name',
            'target_class_name', 'pred_role', 'target_role',
            'support_pixels', 'target_gt_pixels', 'pred_gt_pixels',
            'target_help_ratio', 'harmful_ratio',
            'mean_base_gap_on_target_gt',
            'mean_requery_gain_on_target_gt',
            'mean_requery_margin_on_target_gt',
            'pair_requery_beats_base_pred_ratio',
            'presence_score', 'num_masks', 'target_prompt', 'pred_prompt',
        ])


if __name__ == '__main__':
    main()
