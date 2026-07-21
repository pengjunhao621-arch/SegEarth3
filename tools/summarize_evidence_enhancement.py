#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def add_matrix(dst, src):
    if src is None:
        return dst
    if not dst:
        dst.extend([[0 for _ in row] for row in src])
    for i, row in enumerate(src):
        for j, value in enumerate(row):
            dst[i][j] += int(value)
    return dst


def metrics_from_confusion(matrix):
    if not matrix:
        return dict(miou=None, acc=None, class_iou=[])
    n = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(n))
    ious = []
    for i in range(n):
        tp = matrix[i][i]
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(n))
        union = row_sum + col_sum - tp
        ious.append(None if union == 0 else tp / union)
    valid_ious = [value for value in ious if value is not None]
    return dict(
        miou=None if not valid_ious else sum(valid_ious) / len(valid_ious),
        acc=None if total == 0 else correct / total,
        class_iou=ious,
    )


def safe_delta(a, b):
    if a is None or b is None:
        return None
    return a - b


def iter_records(paths):
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
                stats = record.get('evidence_enhancement')
                if stats:
                    yield record, stats


def summarize(paths):
    dataset_conf = {}
    class_acc = defaultdict(lambda: defaultdict(float))
    prompt_acc = defaultdict(lambda: defaultdict(float))
    image_rows = []

    for record, stats in iter_records(paths):
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        key = (
            dataset,
            str(stats.get('source')),
            str(stats.get('presence_mode')),
            str(stats.get('reduce')),
            str(stats.get('combine')),
            str(record.get('use_evidence_enhancement')),
        )
        if key not in dataset_conf:
            dataset_conf[key] = dict(
                dataset=dataset,
                source=str(stats.get('source')),
                presence_mode=str(stats.get('presence_mode')),
                reduce=str(stats.get('reduce')),
                combine=str(stats.get('combine')),
                use_evidence_enhancement=str(
                    record.get('use_evidence_enhancement')),
                images=0,
                baseline=[],
                enhanced=[],
            )
        bucket = dataset_conf[key]
        bucket['images'] += 1
        add_matrix(bucket['baseline'], stats.get('baseline_confusion'))
        add_matrix(bucket['enhanced'], stats.get('enhanced_confusion'))

        base_metrics = metrics_from_confusion(stats.get('baseline_confusion'))
        enhanced_metrics = metrics_from_confusion(
            stats.get('enhanced_confusion'))
        image_rows.append(dict(
            dataset=dataset,
            image_id=record.get('img_path'),
            source=str(stats.get('source')),
            presence_mode=str(stats.get('presence_mode')),
            reduce=str(stats.get('reduce')),
            combine=str(stats.get('combine')),
            baseline_miou=base_metrics['miou'],
            enhanced_miou=enhanced_metrics['miou'],
            miou_delta=safe_delta(
                enhanced_metrics['miou'], base_metrics['miou']),
            selected_classes='|'.join(stats.get('selected_class_names') or []),
        ))

        for row in stats.get('class_rows') or []:
            class_key = (
                key,
                int(row.get('class_index')),
                row.get('class_name'),
                row.get('role'),
                str(row.get('enhanced')),
            )
            acc = class_acc[class_key]
            acc['rows'] += 1
            acc['gt_pixels'] += int(row.get('gt_pixels') or 0)
            for name in [
                    'base_iou', 'enhanced_iou', 'iou_delta',
                    'base_recall', 'enhanced_recall',
                    'mean_base_logit_on_gt',
                    'mean_enhanced_logit_on_gt',
                    'mean_logit_gain_on_gt']:
                value = row.get(name)
                if value is None:
                    continue
                weight = max(1, int(row.get('gt_pixels') or 0))
                acc[f'{name}_sum'] += float(value) * weight
                acc[f'{name}_weight'] += weight
            acc['base_pred_pixels'] += int(row.get('base_pred_pixels') or 0)
            acc['enhanced_pred_pixels'] += int(
                row.get('enhanced_pred_pixels') or 0)

        for row in stats.get('class_prompt_rows') or []:
            prompt_key = (
                key,
                int(row.get('class_index')),
                row.get('class_name'),
                row.get('role'),
                row.get('prompt'),
            )
            acc = prompt_acc[prompt_key]
            acc['rows'] += 1
            for name in [
                    'presence_score', 'score_mean', 'score_max',
                    'semantic_mean', 'instance_mean',
                    'reduced_score_mean', 'reduced_score_max',
                    'applied_gain_mean', 'applied_gain_max']:
                value = row.get(name)
                if value is None:
                    continue
                acc[f'{name}_sum'] += float(value)

    dataset_rows = []
    for key, bucket in dataset_conf.items():
        base = metrics_from_confusion(bucket['baseline'])
        enhanced = metrics_from_confusion(bucket['enhanced'])
        dataset_rows.append(dict(
            dataset=bucket['dataset'],
            source=bucket['source'],
            presence_mode=bucket['presence_mode'],
            reduce=bucket['reduce'],
            combine=bucket['combine'],
            use_evidence_enhancement=bucket['use_evidence_enhancement'],
            images=bucket['images'],
            baseline_miou=base['miou'],
            enhanced_miou=enhanced['miou'],
            miou_delta=safe_delta(enhanced['miou'], base['miou']),
            baseline_acc=base['acc'],
            enhanced_acc=enhanced['acc'],
            acc_delta=safe_delta(enhanced['acc'], base['acc']),
        ))

    class_rows = []
    for class_key, acc in class_acc.items():
        key, class_index, class_name, role, enhanced_flag = class_key
        dataset, source, presence_mode, reduce, combine, use_apply = key
        row = dict(
            dataset=dataset,
            source=source,
            presence_mode=presence_mode,
            reduce=reduce,
            combine=combine,
            use_evidence_enhancement=use_apply,
            class_index=class_index,
            class_name=class_name,
            role=role,
            enhanced=enhanced_flag,
            rows=int(acc['rows']),
            gt_pixels=int(acc['gt_pixels']),
            base_pred_pixels=int(acc['base_pred_pixels']),
            enhanced_pred_pixels=int(acc['enhanced_pred_pixels']),
        )
        for name in [
                'base_iou', 'enhanced_iou', 'iou_delta',
                'base_recall', 'enhanced_recall',
                'mean_base_logit_on_gt',
                'mean_enhanced_logit_on_gt',
                'mean_logit_gain_on_gt']:
            weight = acc.get(f'{name}_weight', 0)
            row[name] = (
                None if weight == 0
                else acc.get(f'{name}_sum', 0.0) / weight)
        class_rows.append(row)

    prompt_rows = []
    for prompt_key, acc in prompt_acc.items():
        key, class_index, class_name, role, prompt = prompt_key
        dataset, source, presence_mode, reduce, combine, use_apply = key
        row = dict(
            dataset=dataset,
            source=source,
            presence_mode=presence_mode,
            reduce=reduce,
            combine=combine,
            use_evidence_enhancement=use_apply,
            class_index=class_index,
            class_name=class_name,
            role=role,
            prompt=prompt,
            rows=int(acc['rows']),
        )
        for name in [
                'presence_score', 'score_mean', 'score_max',
                'semantic_mean', 'instance_mean',
                'reduced_score_mean', 'reduced_score_max',
                'applied_gain_mean', 'applied_gain_max']:
            row[name] = (
                None if acc['rows'] == 0
                else acc.get(f'{name}_sum', 0.0) / acc['rows'])
        prompt_rows.append(row)

    return dataset_rows, class_rows, prompt_rows, image_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No evidence enhancement JSONL found.')
    dataset_rows, class_rows, prompt_rows, image_rows = summarize(paths)
    write_csv(os.path.join(args.out_dir, 'evidence_enhancement_dataset_summary.csv'), dataset_rows)
    write_csv(os.path.join(args.out_dir, 'evidence_enhancement_class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'evidence_enhancement_prompt_summary.csv'), prompt_rows)
    write_csv(os.path.join(args.out_dir, 'evidence_enhancement_image_summary.csv'), image_rows)
    print(f'Wrote evidence enhancement summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
