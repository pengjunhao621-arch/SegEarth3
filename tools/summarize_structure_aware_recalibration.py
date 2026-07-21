#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_div(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize structure-aware recalibration diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    result = []
    seen = set()
    for path in paths:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def iter_records(paths):
    for path in paths:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stats = record.get('structure_aware_recalibration_stats')
                if stats:
                    yield record, stats


def write_csv(path, rows, preferred=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    fields = []
    for key in preferred or []:
        if key not in fields:
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_class_counts(bucket, row):
    for key in ('gt_pixels', 'pred_pixels', 'tp_pixels', 'fp_pixels',
                'fn_pixels'):
        bucket[key] += int(row.get(key) or 0)


def class_metric(meta, bucket):
    gt = bucket['gt_pixels']
    pred = bucket['pred_pixels']
    tp = bucket['tp_pixels']
    fp = bucket['fp_pixels']
    fn = bucket['fn_pixels']
    row = dict(meta)
    row.update(dict(
        gt_pixels=int(gt),
        pred_pixels=int(pred),
        tp_pixels=int(tp),
        fp_pixels=int(fp),
        fn_pixels=int(fn),
        iou=safe_div(tp, tp + fp + fn),
        precision=safe_div(tp, pred),
        recall=safe_div(tp, gt),
        pred_gt_area_ratio=safe_div(pred, gt),
    ))
    return row


def summarize(records):
    dataset = defaultdict(lambda: defaultdict(float))
    dataset_meta = {}
    role = defaultdict(lambda: defaultdict(int))
    role_meta = {}
    classes = defaultdict(lambda: defaultdict(int))
    class_meta = {}
    pairs = defaultdict(lambda: defaultdict(int))
    pair_meta = {}

    for record, stats in records:
        name = stats.get('dataset_name') or record.get('dataset_name') or 'unknown'
        variant = record.get('structure_recalibration_variant') or 'unknown'
        key = (name, variant, bool(record.get('use_structure_aware_recalibration')))
        dataset_meta[key] = dict(
            dataset_name=name,
            variant=variant,
            use_structure_aware_recalibration=bool(
                record.get('use_structure_aware_recalibration')),
        )
        bucket = dataset[key]
        for field in (
                'valid_pixels', 'base_correct_pixels', 'base_wrong_pixels',
                'high_conf_wrong_pixels', 'changed_pixels',
                'improved_pixels', 'harmed_pixels',
                'net_improved_pixels'):
            bucket[field] += float(stats.get(field) or 0)
        bucket['images'] += 1

        for row in stats.get('readout_role_stats') or []:
            rkey = (name, variant, row.get('readout_name'), row.get('role'))
            role_meta[rkey] = dict(
                dataset_name=name,
                variant=variant,
                readout_name=row.get('readout_name'),
                role=row.get('role'),
            )
            rb = role[rkey]
            for field in (
                    'gt_pixels', 'base_wrong_pixels',
                    'high_conf_wrong_pixels', 'top1_correct_pixels',
                    'topk_contains_gt_pixels',
                    'wrong_top1_recovers_gt_pixels',
                    'wrong_topk_contains_gt_pixels'):
                if field in row:
                    rb[field] += int(row.get(field) or 0)
            # Older/newer records may store only ratios. Keep pixel counts when
            # present; ratios are intentionally recomputed from counts.

        for row in stats.get('class_stats') or []:
            ckey = (
                name,
                variant,
                row.get('metric_prefix'),
                int(row.get('class_index')),
            )
            class_meta[ckey] = dict(
                dataset_name=name,
                variant=variant,
                metric_prefix=row.get('metric_prefix'),
                class_index=int(row.get('class_index')),
                class_name=row.get('class_name'),
                role=row.get('role'),
            )
            add_class_counts(classes[ckey], row)

        for row in stats.get('pair_stats') or []:
            pkey = (
                name,
                variant,
                int(row.get('gt_class_index')),
                int(row.get('base_pred_class_index')),
            )
            pair_meta[pkey] = dict(
                dataset_name=name,
                variant=variant,
                gt_class_index=int(row.get('gt_class_index')),
                gt_class_name=row.get('gt_class_name'),
                gt_role=row.get('gt_role'),
                base_pred_class_index=int(row.get('base_pred_class_index')),
                base_pred_class_name=row.get('base_pred_class_name'),
                pred_role=row.get('pred_role'),
            )
            pb = pairs[pkey]
            for field in (
                    'pixels', 'high_conf_wrong_pixels',
                    'recal_recovers_gt_pixels',
                    'recal_harms_to_other_pixels'):
                pb[field] += int(row.get(field) or 0)

    dataset_rows = []
    for key, bucket in sorted(dataset.items()):
        row = dict(dataset_meta[key])
        changed = bucket['changed_pixels']
        base_wrong = bucket['base_wrong_pixels']
        row.update(dict(
            images=int(bucket['images']),
            valid_pixels=int(bucket['valid_pixels']),
            base_correct_pixels=int(bucket['base_correct_pixels']),
            base_wrong_pixels=int(base_wrong),
            high_conf_wrong_pixels=int(bucket['high_conf_wrong_pixels']),
            high_conf_wrong_ratio=safe_div(
                bucket['high_conf_wrong_pixels'], base_wrong),
            changed_pixels=int(changed),
            changed_ratio=safe_div(changed, bucket['valid_pixels']),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            net_improved_pixels=int(bucket['net_improved_pixels']),
            correction_precision=safe_div(
                bucket['improved_pixels'], changed),
            harmful_rate=safe_div(bucket['harmed_pixels'], changed),
        ))
        dataset_rows.append(row)

    role_rows = []
    for key, bucket in sorted(role.items()):
        gt = bucket['gt_pixels']
        wrong = bucket['base_wrong_pixels']
        row = dict(role_meta[key])
        row.update(dict(
            gt_pixels=int(gt),
            base_wrong_pixels=int(wrong),
            high_conf_wrong_pixels=int(bucket['high_conf_wrong_pixels']),
            high_conf_wrong_ratio=safe_div(
                bucket['high_conf_wrong_pixels'], wrong),
            top1_accuracy=safe_div(
                bucket['top1_correct_pixels'], gt),
            topk_contains_gt_ratio=safe_div(
                bucket['topk_contains_gt_pixels'], gt),
            wrong_top1_recovers_gt_ratio=safe_div(
                bucket['wrong_top1_recovers_gt_pixels'], wrong),
            wrong_topk_contains_gt_ratio=safe_div(
                bucket['wrong_topk_contains_gt_pixels'], wrong),
        ))
        role_rows.append(row)

    class_rows = [
        class_metric(class_meta[key], bucket)
        for key, bucket in sorted(classes.items())
    ]
    by_dataset_variant_class = defaultdict(dict)
    for row in class_rows:
        by_dataset_variant_class[(
            row['dataset_name'], row['variant'], row['class_index'])][
                row['metric_prefix']] = row
    for row in class_rows:
        base = by_dataset_variant_class[(
            row['dataset_name'], row['variant'], row['class_index'])].get(
                'base')
        if base is not None:
            row['base_iou'] = base.get('iou')
            row['delta_iou_vs_base'] = (
                None if row.get('iou') is None or base.get('iou') is None
                else row['iou'] - base['iou'])

    pair_rows = []
    for key, bucket in sorted(pairs.items()):
        pixels = bucket['pixels']
        row = dict(pair_meta[key])
        row.update(dict(
            pixels=int(pixels),
            high_conf_wrong_pixels=int(bucket['high_conf_wrong_pixels']),
            high_conf_wrong_ratio=safe_div(
                bucket['high_conf_wrong_pixels'], pixels),
            recal_recovers_gt_pixels=int(
                bucket['recal_recovers_gt_pixels']),
            recal_recovers_gt_ratio=safe_div(
                bucket['recal_recovers_gt_pixels'], pixels),
            recal_harms_to_other_pixels=int(
                bucket['recal_harms_to_other_pixels']),
            recal_harms_to_other_ratio=safe_div(
                bucket['recal_harms_to_other_pixels'], pixels),
        ))
        pair_rows.append(row)

    return dataset_rows, role_rows, class_rows, pair_rows


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No structure-aware JSONL files found.')
    rows = summarize(iter_records(paths))
    dataset_rows, role_rows, class_rows, pair_rows = rows
    write_csv(
        os.path.join(args.out_dir, 'structure_dataset_summary.csv'),
        dataset_rows,
        preferred=[
            'dataset_name', 'variant', 'use_structure_aware_recalibration',
            'images', 'valid_pixels', 'base_wrong_pixels',
            'high_conf_wrong_ratio', 'changed_ratio',
            'correction_precision', 'harmful_rate',
            'net_improved_pixels',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'structure_role_summary.csv'),
        role_rows,
        preferred=[
            'dataset_name', 'variant', 'readout_name', 'role',
            'gt_pixels', 'base_wrong_pixels', 'high_conf_wrong_ratio',
            'top1_accuracy', 'topk_contains_gt_ratio',
            'wrong_top1_recovers_gt_ratio',
            'wrong_topk_contains_gt_ratio',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'structure_class_summary.csv'),
        class_rows,
        preferred=[
            'dataset_name', 'variant', 'metric_prefix', 'class_index',
            'class_name', 'role', 'gt_pixels', 'pred_pixels', 'iou',
            'base_iou', 'delta_iou_vs_base', 'precision', 'recall',
            'pred_gt_area_ratio',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'structure_pair_summary.csv'),
        pair_rows,
        preferred=[
            'dataset_name', 'variant', 'gt_class_name', 'gt_role',
            'base_pred_class_name', 'pred_role', 'pixels',
            'high_conf_wrong_ratio', 'recal_recovers_gt_ratio',
            'recal_harms_to_other_ratio',
        ],
    )


if __name__ == '__main__':
    main()
