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
    pred = [int(sum(matrix[i][j] for i in range(class_count)))
            for j in range(class_count)]
    ious = []
    valid_ious = []
    for i in range(class_count):
        union = gt[i] + pred[i] - tp[i]
        iou = safe_div(tp[i], union)
        ious.append(iou)
        if gt[i] > 0:
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
                if not line:
                    continue
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize active-concept competition diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = {}
    for record in load_records(args.inputs):
        stats = record.get('active_concept_stats') or {}
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        info = datasets.setdefault(dataset, dict(
            class_names=stats.get('class_names') or [],
            baseline=[],
            variants=defaultdict(lambda: dict(
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
                non_bg_active_count=0,
                non_bg_gt_present_count=0,
                non_bg_active_gt_count=0,
                inactive_gt_images=0,
            )),
            class_feature=defaultdict(lambda: dict(
                images=0,
                gt_present_images=0,
                presence_score_sum=0.0,
                final_area_sum=0.0,
                semantic_area_sum=0.0,
                instance_area_sum=0.0,
                agreement_area_sum=0.0,
                pred_area_sum=0.0,
                topk_area_sum=0.0,
                selected=defaultdict(int),
            )),
        ))
        if not info['class_names'] and stats.get('class_names'):
            info['class_names'] = stats.get('class_names')
        add_matrix(info['baseline'], stats.get('baseline_confusion'))
        for class_row in stats.get('class_features') or []:
            class_idx = int(class_row.get('class_index', -1))
            if class_idx < 0:
                continue
            bucket = info['class_feature'][class_idx]
            bucket['images'] += 1
            if class_row.get('gt_present'):
                bucket['gt_present_images'] += 1
            for key in [
                    'presence_score',
                    'final_area',
                    'semantic_area',
                    'instance_area',
                    'agreement_area',
                    'pred_area',
                    'topk_area',
            ]:
                bucket[f'{key}_sum'] += float(class_row.get(key, 0.0) or 0.0)
        for variant in stats.get('variants') or []:
            name = str(variant.get('variant'))
            bucket = info['variants'][name]
            add_matrix(bucket['confusion'], variant.get('confusion'))
            bucket['images'] += 1
            bucket['valid_pixels'] += int(stats.get('valid_pixels', 0) or 0)
            bucket['changed'] += int(variant.get('changed_pixels', 0) or 0)
            bucket['improved'] += int(variant.get('improved_pixels', 0) or 0)
            bucket['harmed'] += int(variant.get('harmed_pixels', 0) or 0)
            bucket['wrong_to_wrong'] += int(
                variant.get('wrong_to_wrong_pixels', 0) or 0)
            bucket['active_count'] += int(variant.get('active_count', 0) or 0)
            bucket['gt_present_count'] += int(
                variant.get('gt_present_count', 0) or 0)
            bucket['active_gt_count'] += int(
                variant.get('active_gt_count', 0) or 0)
            bucket['non_bg_active_count'] += int(
                variant.get('non_bg_active_count', 0) or 0)
            bucket['non_bg_gt_present_count'] += int(
                variant.get('non_bg_gt_present_count', 0) or 0)
            bucket['non_bg_active_gt_count'] += int(round(
                float(variant.get('non_bg_active_recall', 0.0) or 0.0)
                * int(variant.get('non_bg_gt_present_count', 0) or 0)))
            if variant.get('inactive_gt_classes'):
                bucket['inactive_gt_images'] += 1
            mask = variant.get('active_mask') or []
            for class_idx, selected in enumerate(mask):
                if selected:
                    info['class_feature'][class_idx]['selected'][name] += 1

    overall_path = os.path.join(args.output_dir, 'active_concept_summary.csv')
    class_path = os.path.join(args.output_dir, 'active_concept_class_summary.csv')
    selector_path = os.path.join(
        args.output_dir, 'active_concept_selector_summary.csv')

    with open(overall_path, 'w', newline='') as f:
        fields = [
            'dataset', 'variant', 'images', 'baseline_mIoU', 'mIoU',
            'delta_mIoU', 'baseline_aAcc', 'aAcc', 'delta_aAcc',
            'changed_ratio', 'correction_precision', 'harmful_change_rate',
            'active_recall', 'active_precision', 'non_bg_active_recall',
            'avg_active_count', 'inactive_gt_image_ratio',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            base = confusion_metrics(info['baseline'])
            for variant, bucket in sorted(info['variants'].items()):
                metrics = confusion_metrics(bucket['confusion'])
                writer.writerow(dict(
                    dataset=dataset,
                    variant=variant,
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
                    non_bg_active_recall=safe_div(
                        bucket['non_bg_active_gt_count'],
                        bucket['non_bg_gt_present_count']),
                    avg_active_count=safe_div(
                        bucket['active_count'], bucket['images']),
                    inactive_gt_image_ratio=safe_div(
                        bucket['inactive_gt_images'], bucket['images']),
                ))

    with open(class_path, 'w', newline='') as f:
        fields = [
            'dataset', 'variant', 'class_index', 'class_name',
            'baseline_iou', 'iou', 'delta_iou',
            'baseline_gt_pixels', 'pred_pixels', 'tp_pixels',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            class_names = info['class_names']
            base = confusion_metrics(info['baseline'])
            for variant, bucket in sorted(info['variants'].items()):
                metrics = confusion_metrics(bucket['confusion'])
                for idx, name in enumerate(class_names):
                    writer.writerow(dict(
                        dataset=dataset,
                        variant=variant,
                        class_index=idx,
                        class_name=name,
                        baseline_iou=base['IoU'][idx],
                        iou=metrics['IoU'][idx],
                        delta_iou=metrics['IoU'][idx] - base['IoU'][idx],
                        baseline_gt_pixels=base['gt'][idx],
                        pred_pixels=metrics['pred'][idx],
                        tp_pixels=metrics['tp'][idx],
                    ))

    with open(selector_path, 'w', newline='') as f:
        variants = sorted({
            variant
            for info in datasets.values()
            for variant in info['variants']
        })
        fields = [
            'dataset', 'class_index', 'class_name', 'images',
            'gt_present_ratio', 'presence_mean', 'final_area_mean',
            'semantic_area_mean', 'instance_area_mean',
            'agreement_area_mean', 'pred_area_mean', 'topk_area_mean',
        ] + [f'{variant}_selected_ratio' for variant in variants]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            class_names = info['class_names']
            for idx, name in enumerate(class_names):
                bucket = info['class_feature'].get(idx)
                if not bucket:
                    continue
                images = bucket['images']
                row = dict(
                    dataset=dataset,
                    class_index=idx,
                    class_name=name,
                    images=images,
                    gt_present_ratio=safe_div(
                        bucket['gt_present_images'], images),
                    presence_mean=safe_div(
                        bucket['presence_score_sum'], images),
                    final_area_mean=safe_div(
                        bucket['final_area_sum'], images),
                    semantic_area_mean=safe_div(
                        bucket['semantic_area_sum'], images),
                    instance_area_mean=safe_div(
                        bucket['instance_area_sum'], images),
                    agreement_area_mean=safe_div(
                        bucket['agreement_area_sum'], images),
                    pred_area_mean=safe_div(
                        bucket['pred_area_sum'], images),
                    topk_area_mean=safe_div(
                        bucket['topk_area_sum'], images),
                )
                for variant in variants:
                    row[f'{variant}_selected_ratio'] = safe_div(
                        bucket['selected'].get(variant, 0), images)
                writer.writerow(row)

    print(f'Wrote {overall_path}')
    print(f'Wrote {class_path}')
    print(f'Wrote {selector_path}')


if __name__ == '__main__':
    main()
