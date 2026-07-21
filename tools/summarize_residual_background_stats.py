#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def load_records(inputs):
    paths = []
    for pattern in inputs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def add_matrix(dst, src):
    if not src:
        return
    if not dst:
        dst.extend([[0 for _ in row] for row in src])
    for i, row in enumerate(src):
        for j, value in enumerate(row):
            dst[i][j] += int(value)


def confusion_metrics(matrix):
    class_count = len(matrix)
    if class_count == 0:
        return dict(mIoU=0.0, aAcc=0.0, IoU=[], gt=[], pred=[], tp=[])
    tp = [int(matrix[i][i]) for i in range(class_count)]
    gt = [int(sum(matrix[i])) for i in range(class_count)]
    pred = [
        int(sum(matrix[i][j] for i in range(class_count)))
        for j in range(class_count)
    ]
    ious = []
    valid_ious = []
    for i in range(class_count):
        union = gt[i] + pred[i] - tp[i]
        iou = safe_div(tp[i], union)
        ious.append(iou)
        if gt[i] > 0:
            valid_ious.append(iou)
    return dict(
        mIoU=safe_div(sum(valid_ious), len(valid_ious)) * 100.0,
        aAcc=safe_div(sum(tp), sum(gt)) * 100.0,
        IoU=[value * 100.0 for value in ious],
        gt=gt,
        pred=pred,
        tp=tp,
    )


def setting_name(record):
    return '|'.join([
        str(record.get('residual_background_route', 'unknown')),
        f"sf={record.get('residual_background_score_factor')}",
        f"loc={record.get('residual_background_min_local')}",
        f"rel={record.get('residual_background_min_reliability_margin')}",
        f"fgbg={record.get('residual_background_min_fg_bg_margin')}",
    ])


def main():
    parser = argparse.ArgumentParser(
        description='Summarize residual-background modeling diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = {}
    for record in load_records(args.inputs):
        stats = record.get('residual_background_stats') or {}
        if not stats:
            continue
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown'
        )
        setting = setting_name(record)
        key = (dataset, setting)
        info = datasets.setdefault(key, dict(
            dataset=dataset,
            setting=setting,
            route=record.get('residual_background_route', ''),
            use_method=bool(record.get('use_residual_background_modeling')),
            class_names=stats.get('class_names') or [],
            class_roles=stats.get('class_roles') or [],
            baseline=[],
            residual=[],
            images=0,
            valid_pixels=0,
            baseline_bg_pixels=0,
            baseline_true_bg_pixels=0,
            foreground_gt_to_bg_pixels=0,
            route_pixels=0,
            threshold_route_pixels=0,
            weak_bg_route_pixels=0,
            recovered_pixels=0,
            changed_pixels=0,
            improved_pixels=0,
            harmed_pixels=0,
            wrong_to_wrong_pixels=0,
            changed_true_bg_pixels=0,
            recovered_foreground_gt_pixels=0,
            recovered_true_bg_pixels=0,
            class_stats=defaultdict(lambda: defaultdict(int)),
            pair_stats=defaultdict(lambda: defaultdict(int)),
        ))
        if not info['class_names'] and stats.get('class_names'):
            info['class_names'] = stats.get('class_names')
        if not info['class_roles'] and stats.get('class_roles'):
            info['class_roles'] = stats.get('class_roles')
        add_matrix(info['baseline'], stats.get('baseline_confusion'))
        add_matrix(info['residual'], stats.get('residual_confusion'))
        info['images'] += 1
        for key_name in [
                'valid_pixels',
                'baseline_bg_pixels',
                'baseline_true_bg_pixels',
                'foreground_gt_to_bg_pixels',
                'route_pixels',
                'threshold_route_pixels',
                'weak_bg_route_pixels',
                'recovered_pixels',
                'changed_pixels',
                'improved_pixels',
                'harmed_pixels',
                'wrong_to_wrong_pixels',
                'changed_true_bg_pixels',
                'recovered_foreground_gt_pixels',
                'recovered_true_bg_pixels',
        ]:
            info[key_name] += int(stats.get(key_name, 0) or 0)
        for row in stats.get('class_stats') or []:
            class_idx = int(row.get('class_index', -1))
            if class_idx < 0:
                continue
            bucket = info['class_stats'][class_idx]
            for key_name, value in row.items():
                if key_name in ('class_index', 'class_name', 'role'):
                    continue
                bucket[key_name] += int(value or 0)
        for row in stats.get('pair_stats') or []:
            pair_key = (
                int(row.get('gt_class_index', -1)),
                int(row.get('base_pred_class_index', -1)),
            )
            bucket = info['pair_stats'][pair_key]
            for key_name, value in row.items():
                if key_name.endswith('name') or key_name.endswith('role'):
                    continue
                if key_name in ('gt_class_index', 'base_pred_class_index'):
                    continue
                bucket[key_name] += int(value or 0)

    summary_path = os.path.join(args.output_dir, 'residual_background_summary.csv')
    class_path = os.path.join(args.output_dir, 'residual_background_class_summary.csv')
    pair_path = os.path.join(args.output_dir, 'residual_background_pair_summary.csv')

    with open(summary_path, 'w', newline='') as f:
        fields = [
            'dataset', 'setting', 'route', 'images',
            'baseline_mIoU', 'mIoU', 'delta_mIoU',
            'baseline_aAcc', 'aAcc', 'delta_aAcc',
            'baseline_bg_ratio', 'foreground_gt_to_bg_ratio',
            'route_ratio', 'threshold_route_ratio', 'weak_bg_route_ratio',
            'changed_ratio', 'recovered_ratio',
            'correction_precision', 'harmful_change_rate',
            'recovered_foreground_gt_ratio', 'recovered_true_bg_ratio',
            'changed_true_bg_ratio',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for _, info in sorted(datasets.items()):
            base = confusion_metrics(info['baseline'])
            residual = confusion_metrics(info['residual'])
            writer.writerow(dict(
                dataset=info['dataset'],
                setting=info['setting'],
                route=info['route'],
                images=info['images'],
                baseline_mIoU=base['mIoU'],
                mIoU=residual['mIoU'],
                delta_mIoU=residual['mIoU'] - base['mIoU'],
                baseline_aAcc=base['aAcc'],
                aAcc=residual['aAcc'],
                delta_aAcc=residual['aAcc'] - base['aAcc'],
                baseline_bg_ratio=safe_div(
                    info['baseline_bg_pixels'], info['valid_pixels']),
                foreground_gt_to_bg_ratio=safe_div(
                    info['foreground_gt_to_bg_pixels'],
                    info['valid_pixels']),
                route_ratio=safe_div(
                    info['route_pixels'], info['valid_pixels']),
                threshold_route_ratio=safe_div(
                    info['threshold_route_pixels'], info['valid_pixels']),
                weak_bg_route_ratio=safe_div(
                    info['weak_bg_route_pixels'], info['valid_pixels']),
                changed_ratio=safe_div(
                    info['changed_pixels'], info['valid_pixels']),
                recovered_ratio=safe_div(
                    info['recovered_pixels'], info['valid_pixels']),
                correction_precision=safe_div(
                    info['improved_pixels'],
                    info['improved_pixels'] + info['harmed_pixels']),
                harmful_change_rate=safe_div(
                    info['harmed_pixels'], info['changed_pixels']),
                recovered_foreground_gt_ratio=safe_div(
                    info['recovered_foreground_gt_pixels'],
                    info['recovered_pixels']),
                recovered_true_bg_ratio=safe_div(
                    info['recovered_true_bg_pixels'],
                    info['recovered_pixels']),
                changed_true_bg_ratio=safe_div(
                    info['changed_true_bg_pixels'],
                    info['changed_pixels']),
            ))

    with open(class_path, 'w', newline='') as f:
        fields = [
            'dataset', 'setting', 'class_index', 'class_name', 'role',
            'gt_pixels', 'baseline_iou', 'iou', 'delta_iou',
            'base_bg_pixels', 'base_bg_ratio',
            'recovered_pixels', 'recovered_ratio',
            'changed_pixels', 'improved_pixels', 'harmed_pixels',
            'wrong_to_wrong_pixels',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for _, info in sorted(datasets.items()):
            base = confusion_metrics(info['baseline'])
            residual = confusion_metrics(info['residual'])
            for class_idx, bucket in sorted(info['class_stats'].items()):
                name = (
                    info['class_names'][class_idx]
                    if class_idx < len(info['class_names'])
                    else f'class_{class_idx}'
                )
                role = (
                    info['class_roles'][class_idx]
                    if class_idx < len(info['class_roles'])
                    else ''
                )
                gt_pixels = int(bucket.get('gt_pixels', 0))
                writer.writerow(dict(
                    dataset=info['dataset'],
                    setting=info['setting'],
                    class_index=class_idx,
                    class_name=name,
                    role=role,
                    gt_pixels=gt_pixels,
                    baseline_iou=base['IoU'][class_idx],
                    iou=residual['IoU'][class_idx],
                    delta_iou=(
                        residual['IoU'][class_idx]
                        - base['IoU'][class_idx]),
                    base_bg_pixels=int(bucket.get('base_bg_pixels', 0)),
                    base_bg_ratio=safe_div(
                        bucket.get('base_bg_pixels', 0), gt_pixels),
                    recovered_pixels=int(bucket.get('recovered_pixels', 0)),
                    recovered_ratio=safe_div(
                        bucket.get('recovered_pixels', 0), gt_pixels),
                    changed_pixels=int(bucket.get('changed_pixels', 0)),
                    improved_pixels=int(bucket.get('improved_pixels', 0)),
                    harmed_pixels=int(bucket.get('harmed_pixels', 0)),
                    wrong_to_wrong_pixels=int(
                        bucket.get('wrong_to_wrong_pixels', 0)),
                ))

    with open(pair_path, 'w', newline='') as f:
        fields = [
            'dataset', 'setting',
            'gt_class_index', 'gt_class_name',
            'base_pred_class_index', 'base_pred_class_name',
            'pair_pixels', 'changed_pixels',
            'changed_ratio', 'improved_pixels', 'harmed_pixels',
            'wrong_to_wrong_pixels', 'correction_precision',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for _, info in sorted(datasets.items()):
            for (gt_idx, pred_idx), bucket in sorted(
                    info['pair_stats'].items(),
                    key=lambda item: item[1].get('changed_pixels', 0),
                    reverse=True):
                gt_name = (
                    info['class_names'][gt_idx]
                    if 0 <= gt_idx < len(info['class_names'])
                    else f'class_{gt_idx}'
                )
                pred_name = (
                    info['class_names'][pred_idx]
                    if 0 <= pred_idx < len(info['class_names'])
                    else f'class_{pred_idx}'
                )
                pair_pixels = int(bucket.get('pair_pixels', 0))
                improved = int(bucket.get('improved_pixels', 0))
                harmed = int(bucket.get('harmed_pixels', 0))
                writer.writerow(dict(
                    dataset=info['dataset'],
                    setting=info['setting'],
                    gt_class_index=gt_idx,
                    gt_class_name=gt_name,
                    base_pred_class_index=pred_idx,
                    base_pred_class_name=pred_name,
                    pair_pixels=pair_pixels,
                    changed_pixels=int(bucket.get('changed_pixels', 0)),
                    changed_ratio=safe_div(
                        bucket.get('changed_pixels', 0), pair_pixels),
                    improved_pixels=improved,
                    harmed_pixels=harmed,
                    wrong_to_wrong_pixels=int(
                        bucket.get('wrong_to_wrong_pixels', 0)),
                    correction_precision=safe_div(
                        improved, improved + harmed),
                ))

    print(f'Wrote {summary_path}')
    print(f'Wrote {class_path}')
    print(f'Wrote {pair_path}')


if __name__ == '__main__':
    main()
