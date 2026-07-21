#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


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
    for idx in range(class_count):
        union = gt[idx] + pred[idx] - tp[idx]
        iou = safe_div(tp[idx], union)
        ious.append(iou)
        if gt[idx] > 0:
            valid_ious.append(iou)
    return dict(
        mIoU=sum(valid_ious) / max(1, len(valid_ious)) * 100.0,
        aAcc=safe_div(sum(tp), sum(gt)) * 100.0,
        IoU=[value * 100.0 for value in ious],
        gt=gt,
        pred=pred,
        tp=tp,
    )


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


def binary_auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
            total += 1
    return safe_div(wins, total)


def mean_feature(row, key):
    return safe_div(row.get(f'{key}_sum', 0.0), row.get('images', 0))


def main():
    parser = argparse.ArgumentParser(
        description='Summarize concept-specificity diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = {}
    auc_data = defaultdict(lambda: dict(scores=[], labels=[]))
    feature_keys = [
        'presence_score',
        'support_area',
        'scene_contrast',
        'common_overlap',
        'mass_concentration',
        'peakiness',
        'head_iou',
        'agreement_area',
        'prompt_stability',
        'prompt_count',
        'mean',
        'std',
        'p90',
        'p95',
        'max',
    ]

    for record in load_records(args.inputs):
        stats = record.get('concept_specificity_stats') or {}
        if not stats:
            continue
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        info = datasets.setdefault(dataset, dict(
            class_names=stats.get('class_names') or [],
            baseline=[],
            rankers=defaultdict(lambda: dict(
                confusion=[],
                images=0,
                valid_pixels=0,
                changed=0,
                improved=0,
                harmed=0,
                wrong_to_wrong=0,
                active_count=0,
                gt_present_count=0,
                active_gt_count=0,
                inactive_gt_images=0,
            )),
            class_features=defaultdict(lambda: dict(
                images=0,
                gt_present_images=0,
                pred_present_images=0,
                gt_pixels=0,
                pred_pixels=0,
                selected=defaultdict(int),
            )),
        ))
        if not info['class_names'] and stats.get('class_names'):
            info['class_names'] = stats.get('class_names')
        add_matrix(info['baseline'], stats.get('baseline_confusion'))

        class_rows = stats.get('class_features') or []
        class_labels = [bool(row.get('gt_present')) for row in class_rows]
        for class_row in class_rows:
            class_idx = int(class_row.get('class_index', -1))
            if class_idx < 0:
                continue
            bucket = info['class_features'][class_idx]
            bucket['images'] += 1
            if class_row.get('gt_present'):
                bucket['gt_present_images'] += 1
            if class_row.get('pred_present'):
                bucket['pred_present_images'] += 1
            bucket['gt_pixels'] += int(class_row.get('gt_pixels', 0) or 0)
            bucket['pred_pixels'] += int(class_row.get('pred_pixels', 0) or 0)
            bucket['class_name'] = class_row.get('class_name', '')
            bucket['role'] = class_row.get('role', '')
            for key in feature_keys:
                bucket[f'{key}_sum'] = (
                    bucket.get(f'{key}_sum', 0.0)
                    + float(class_row.get(key, 0.0) or 0.0)
                )

        for ranker_row in stats.get('ranker_results') or []:
            ranker = str(ranker_row.get('ranker'))
            result_specs = [
                dict(
                    budget_type='gt_budget',
                    confusion_key='confusion',
                    changed_key='changed_pixels',
                    improved_key='improved_pixels',
                    harmed_key='harmed_pixels',
                    wrong_key='wrong_to_wrong_pixels',
                    active_key='active_count',
                    active_gt_key='active_gt_count',
                    inactive_key='inactive_gt_classes',
                    mask_key='active_mask',
                ),
                dict(
                    budget_type='no_gt_budget',
                    confusion_key='no_gt_prune_confusion',
                    changed_key='no_gt_prune_changed_pixels',
                    improved_key='no_gt_prune_improved_pixels',
                    harmed_key='no_gt_prune_harmed_pixels',
                    wrong_key='no_gt_prune_wrong_to_wrong_pixels',
                    active_key='no_gt_prune_active_count',
                    active_gt_key='no_gt_prune_active_gt_count',
                    inactive_key='no_gt_prune_inactive_gt_classes',
                    mask_key='no_gt_prune_active_mask',
                ),
            ]
            for spec in result_specs:
                if not ranker_row.get(spec['confusion_key']):
                    continue
                bucket = info['rankers'][(ranker, spec['budget_type'])]
                add_matrix(
                    bucket['confusion'],
                    ranker_row.get(spec['confusion_key']))
                bucket['images'] += 1
                bucket['valid_pixels'] += int(
                    stats.get('valid_pixels', 0) or 0)
                bucket['changed'] += int(
                    ranker_row.get(spec['changed_key'], 0) or 0)
                bucket['improved'] += int(
                    ranker_row.get(spec['improved_key'], 0) or 0)
                bucket['harmed'] += int(
                    ranker_row.get(spec['harmed_key'], 0) or 0)
                bucket['wrong_to_wrong'] += int(
                    ranker_row.get(spec['wrong_key'], 0) or 0)
                bucket['active_count'] += int(
                    ranker_row.get(spec['active_key'], 0) or 0)
                bucket['gt_present_count'] += int(
                    ranker_row.get('gt_present_count', 0) or 0)
                bucket['active_gt_count'] += int(
                    ranker_row.get(spec['active_gt_key'], 0) or 0)
                if ranker_row.get(spec['inactive_key']):
                    bucket['inactive_gt_images'] += 1
                active_mask = ranker_row.get(spec['mask_key']) or []
                selected_name = f"{ranker}:{spec['budget_type']}"
                for class_idx, selected in enumerate(active_mask):
                    if selected:
                        info['class_features'][class_idx]['selected'][
                            selected_name] += 1
            scores = ranker_row.get('scores') or []
            if len(scores) == len(class_labels):
                auc_data[(dataset, ranker)]['scores'].extend(
                    float(value) for value in scores)
                auc_data[(dataset, ranker)]['labels'].extend(class_labels)

    summary_path = os.path.join(
        args.output_dir, 'concept_specificity_summary.csv')
    class_path = os.path.join(
        args.output_dir, 'concept_specificity_class_summary.csv')
    feature_path = os.path.join(
        args.output_dir, 'concept_specificity_feature_summary.csv')
    auc_path = os.path.join(
        args.output_dir, 'concept_specificity_ranker_auc.csv')

    with open(summary_path, 'w', newline='') as f:
        fields = [
            'dataset', 'ranker', 'budget_type', 'images',
            'baseline_mIoU', 'mIoU',
            'delta_mIoU', 'baseline_aAcc', 'aAcc', 'delta_aAcc',
            'changed_ratio', 'correction_precision', 'harmful_change_rate',
            'active_recall', 'active_precision', 'avg_active_count',
            'inactive_gt_image_ratio',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            base = confusion_metrics(info['baseline'])
            for key, bucket in sorted(info['rankers'].items()):
                ranker, budget_type = key
                metrics = confusion_metrics(bucket['confusion'])
                writer.writerow(dict(
                    dataset=dataset,
                    ranker=ranker,
                    budget_type=budget_type,
                    images=bucket['images'],
                    baseline_mIoU=base['mIoU'],
                    mIoU=metrics['mIoU'],
                    delta_mIoU=metrics['mIoU'] - base['mIoU'],
                    baseline_aAcc=base['aAcc'],
                    aAcc=metrics['aAcc'],
                    delta_aAcc=metrics['aAcc'] - base['aAcc'],
                    changed_ratio=safe_div(
                        bucket['changed'], bucket['valid_pixels']),
                    correction_precision=safe_div(
                        bucket['improved'],
                        bucket['improved'] + bucket['harmed']),
                    harmful_change_rate=safe_div(
                        bucket['harmed'], bucket['changed']),
                    active_recall=safe_div(
                        bucket['active_gt_count'],
                        bucket['gt_present_count']),
                    active_precision=safe_div(
                        bucket['active_gt_count'],
                        bucket['active_count']),
                    avg_active_count=safe_div(
                        bucket['active_count'], bucket['images']),
                    inactive_gt_image_ratio=safe_div(
                        bucket['inactive_gt_images'], bucket['images']),
                ))

    with open(class_path, 'w', newline='') as f:
        fields = [
            'dataset', 'ranker', 'budget_type', 'class_index', 'class_name',
            'baseline_iou', 'iou', 'delta_iou',
            'baseline_gt_pixels', 'pred_pixels', 'tp_pixels',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            class_names = info['class_names']
            base = confusion_metrics(info['baseline'])
            for key, bucket in sorted(info['rankers'].items()):
                ranker, budget_type = key
                metrics = confusion_metrics(bucket['confusion'])
                for idx, name in enumerate(class_names):
                    writer.writerow(dict(
                        dataset=dataset,
                        ranker=ranker,
                        budget_type=budget_type,
                        class_index=idx,
                        class_name=name,
                        baseline_iou=base['IoU'][idx],
                        iou=metrics['IoU'][idx],
                        delta_iou=metrics['IoU'][idx] - base['IoU'][idx],
                        baseline_gt_pixels=base['gt'][idx],
                        pred_pixels=metrics['pred'][idx],
                        tp_pixels=metrics['tp'][idx],
                    ))

    with open(feature_path, 'w', newline='') as f:
        fields = [
            'dataset', 'class_index', 'class_name', 'role', 'images',
            'gt_present_ratio', 'pred_present_ratio',
            'gt_pixels', 'pred_pixels',
        ] + [f'mean_{key}' for key in feature_keys]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            for class_idx, bucket in sorted(info['class_features'].items()):
                row = dict(
                    dataset=dataset,
                    class_index=class_idx,
                    class_name=bucket.get('class_name', ''),
                    role=bucket.get('role', ''),
                    images=bucket['images'],
                    gt_present_ratio=safe_div(
                        bucket['gt_present_images'], bucket['images']),
                    pred_present_ratio=safe_div(
                        bucket['pred_present_images'], bucket['images']),
                    gt_pixels=bucket['gt_pixels'],
                    pred_pixels=bucket['pred_pixels'],
                )
                for key in feature_keys:
                    row[f'mean_{key}'] = mean_feature(bucket, key)
                writer.writerow(row)

    with open(auc_path, 'w', newline='') as f:
        fields = ['dataset', 'ranker', 'auroc', 'positives', 'negatives']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (dataset, ranker), values in sorted(auc_data.items()):
            labels = values['labels']
            writer.writerow(dict(
                dataset=dataset,
                ranker=ranker,
                auroc=binary_auc(values['scores'], labels),
                positives=sum(1 for label in labels if label),
                negatives=sum(1 for label in labels if not label),
            ))

    print(f'Wrote {summary_path}')
    print(f'Wrote {class_path}')
    print(f'Wrote {feature_path}')
    print(f'Wrote {auc_path}')


if __name__ == '__main__':
    main()
