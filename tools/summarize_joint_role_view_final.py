#!/usr/bin/env python3
"""Summarize final Role--View accuracy diagnostics and isolated costs."""

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict


def _load_jsonl(paths, field=None):
    rows = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if field is None or field in value:
                    rows.append(value)
    return rows


def _sum_matrix(values):
    size = len(values[0])
    result = [[0 for _ in range(size)] for _ in range(size)]
    for value in values:
        for row in range(size):
            for column in range(size):
                result[row][column] += int(value[row][column])
    return result


def _metrics(confusion):
    size = len(confusion)
    gt = [sum(confusion[row]) for row in range(size)]
    pred = [sum(confusion[row][column] for row in range(size))
            for column in range(size)]
    true = [confusion[index][index] for index in range(size)]
    ious = []
    for index in range(size):
        union = gt[index] + pred[index] - true[index]
        ious.append(true[index] / union if union else None)
    valid = [value for value in ious if value is not None]
    return dict(
        miou=100.0 * sum(valid) / len(valid),
        aacc=100.0 * sum(true) / max(1, sum(gt)),
        ious=ious, gt_pixels=gt, pred_pixels=pred)


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, math.ceil(
        fraction * len(values)) - 1))
    return values[index]


def summarize_accuracy(records):
    deduplicated = {}
    duplicates = defaultdict(int)
    for record in records:
        key = (record['dataset_name'], record['img_path'])
        payload = record['joint_role_view_final']
        if key in deduplicated:
            if deduplicated[key] != payload:
                raise ValueError(f'Conflicting duplicate image record: {key}.')
            duplicates[record['dataset_name']] += 1
        else:
            deduplicated[key] = payload

    grouped = defaultdict(list)
    class_names = {}
    source_records = {}
    for record in records:
        key = (record['dataset_name'], record['img_path'])
        if key in source_records:
            continue
        source_records[key] = record
        dataset = record['dataset_name']
        class_names[dataset] = record['class_names']
        payload = record['joint_role_view_final']
        grouped[(dataset, 'official')].append(payload['official'])
        for name, value in payload['profiles'].items():
            grouped[(dataset, name)].append(value)

    profile_rows, class_rows = [], []
    metrics_by_key = {}
    for (dataset, profile), values in sorted(grouped.items()):
        confusion = _sum_matrix([
            (value['confusion']['matrix']
             if isinstance(value['confusion'], dict)
             else value['confusion'])
            for value in values])
        metrics = _metrics(confusion)
        metrics_by_key[(dataset, profile)] = metrics
        official = metrics_by_key.get((dataset, 'official'))
        row = dict(
            dataset=dataset, profile=profile, images=len(values),
            miou=metrics['miou'], aacc=metrics['aacc'],
            changed_pixels=sum(int(value['changed_pixels']) for value in values),
            improved_pixels=sum(int(value['improved_pixels']) for value in values),
            harmed_pixels=sum(int(value['harmed_pixels']) for value in values),
            foreground_added_pixels=sum(int(
                value['foreground_added_pixels']) for value in values),
            foreground_removed_pixels=sum(int(
                value['foreground_removed_pixels']) for value in values),
            foreground_class_switch_pixels=sum(int(
                value['foreground_class_switch_pixels']) for value in values),
        )
        for field in ('candidate', 'update', 'operator', 'admission', 'slots'):
            row[field] = values[0].get(field, '')
        if official is not None:
            row['delta_to_official'] = metrics['miou'] - official['miou']
        profile_rows.append(row)
        for index, name in enumerate(class_names[dataset]):
            class_rows.append(dict(
                dataset=dataset, profile=profile, class_index=index,
                class_name=name,
                iou=(None if metrics['ious'][index] is None
                     else 100.0 * metrics['ious'][index]),
                gt_pixels=metrics['gt_pixels'][index],
                predicted_pixels=metrics['pred_pixels'][index],
            ))

    # The official row sorts first within each dataset, so fill any missing delta.
    for row in profile_rows:
        official = metrics_by_key[(row['dataset'], 'official')]
        row['delta_to_official'] = row['miou'] - official['miou']
    return profile_rows, class_rows, dict(duplicates)


def summarize_efficiency(records):
    grouped = defaultdict(list)
    for value in records:
        if not value.get('warmup', False):
            grouped[(value.get('dataset', ''), value['profile'])].append(value)
    rows = []
    for (dataset, profile), values in sorted(grouped.items()):
        latencies = [float(value['latency_seconds']) for value in values]
        rows.append(dict(
            dataset=dataset, profile=profile, measured_iterations=len(values),
            latency_mean_seconds=statistics.mean(latencies),
            latency_median_seconds=statistics.median(latencies),
            latency_p90_seconds=_percentile(latencies, 0.90),
            images_per_second=(len(values) / sum(latencies)),
            peak_allocated_gib=max(int(value['peak_allocated_bytes'])
                                   for value in values) / 2 ** 30,
            peak_reserved_gib=max(int(value['peak_reserved_bytes'])
                                  for value in values) / 2 ** 30,
            image_encoder_calls_mean=statistics.mean(
                float(value.get('image_encoder_calls', 0))
                for value in values),
            grounding_calls_mean=statistics.mean(
                float(value.get('grounding_calls', 0)) for value in values),
            unique_cached_text_prompts=max(
                int(value.get('unique_cached_text_prompts', 0))
                for value in values),
        ))
    lookup = {(row['dataset'], row['profile']): row for row in rows}
    for row in rows:
        baseline = lookup.get((row['dataset'], 'baseline'))
        if baseline is not None:
            row['latency_over_baseline'] = (
                row['latency_mean_seconds']
                / baseline['latency_mean_seconds'])
            row['peak_allocated_over_baseline'] = (
                row['peak_allocated_gib']
                / max(baseline['peak_allocated_gib'], 1e-12))
    return rows


def _report(profile_rows, efficiency_rows):
    lines = ['# Joint Role--View Final Experiment', '']
    lines += [
        '| Dataset | Official | Role only | View only | Joint residual | '
        'Joint direct | Fast Joint |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    lookup = {(row['dataset'], row['profile']): row
              for row in profile_rows}
    datasets = sorted({row['dataset'] for row in profile_rows})
    names = ('official', 'role_only_residual', 'view_only',
             'joint_residual', 'joint_direct', 'fast_joint_residual')
    for dataset in datasets:
        values = []
        for name in names:
            row = lookup.get((dataset, name))
            values.append('—' if row is None else f"{row['miou']:.3f}")
        lines.append(f"| {dataset} | " + ' | '.join(values) + ' |')
    lines += ['', '## Residual/direct and matched admission controls', '',
              '| Dataset | Direct − residual | Fast − Full | '
              'Anchor admission − native |',
              '|---|---:|---:|---:|']
    for dataset in datasets:
        def delta(left, right):
            lhs = lookup.get((dataset, left))
            rhs = lookup.get((dataset, right))
            return ('—' if lhs is None or rhs is None
                    else f"{lhs['miou'] - rhs['miou']:+.3f}")
        lines.append(
            f'| {dataset} | {delta("joint_direct", "joint_residual")} | '
            f'{delta("fast_joint_residual", "joint_residual")} | '
            f'{delta("full_anchor_admission", "full_native_admission")} |')
    if efficiency_rows:
        lines += ['', '## Isolated deployment efficiency', '',
                  '| Dataset | Profile | Mean s/image | P90 | images/s | '
                  'Peak allocated GiB | × baseline latency | Image encoders | '
                  'Groundings |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in efficiency_rows:
            lines.append(
                f"| {row['dataset']} | {row['profile']} | "
                f"{row['latency_mean_seconds']:.4f} | "
                f"{row['latency_p90_seconds']:.4f} | "
                f"{row['images_per_second']:.4f} | "
                f"{row['peak_allocated_gib']:.3f} | "
                f"{row.get('latency_over_baseline', float('nan')):.3f} | "
                f"{row['image_encoder_calls_mean']:.2f} | "
                f"{row['grounding_calls_mean']:.2f} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-inputs', nargs='*', default=[])
    parser.add_argument('--benchmark-inputs', nargs='*', default=[])
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    accuracy_records = _load_jsonl(
        args.audit_inputs, field='joint_role_view_final')
    benchmark_records = _load_jsonl(args.benchmark_inputs)
    if not accuracy_records and not benchmark_records:
        raise ValueError('No final audit or benchmark records were found.')
    profiles, classes, duplicates = (
        summarize_accuracy(accuracy_records)
        if accuracy_records else ([], [], {}))
    efficiency = summarize_efficiency(benchmark_records)
    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, 'final_profiles.csv'), profiles)
    _write_csv(os.path.join(args.out_dir, 'final_classes.csv'), classes)
    _write_csv(os.path.join(args.out_dir, 'final_efficiency.csv'), efficiency)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(dict(
            profiles=profiles, classes=classes, efficiency=efficiency,
            duplicate_counts=duplicates), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'joint_role_view_final.md'), 'w',
              encoding='utf-8') as handle:
        handle.write(_report(profiles, efficiency))


if __name__ == '__main__':
    main()
