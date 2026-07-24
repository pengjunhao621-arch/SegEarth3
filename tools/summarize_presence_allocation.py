#!/usr/bin/env python3
"""Summarize the training-free SAM3 presence-allocation audit."""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np


SCHEMA_VERSION = 'presence-allocation-v1'
VARIANTS = (
    'p0_baseline',
    'p1_branch_once',
    'p2_delayed_gate',
    'p3_logit_prior',
    'p4_instance_scope',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--expected-datasets',
        nargs='+',
        default=[
            'udd5', 'vdd', 'vaihingen',
            'potsdam', 'openearthmap', 'loveda',
        ],
    )
    parser.add_argument(
        '--integrity-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(
            matches or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


def iter_records(paths):
    for path in paths:
        with open(path) as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record['_source_file'] = path
                record['_source_line'] = line_number
                yield record


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def safe_mean(values):
    values = [
        float(value) for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return sum(values) / len(values) if values else None


def write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted(set().union(
        *(row.keys() for row in rows))) if rows else []
    with open(path, 'w', newline='') as handle:
        if not fieldnames:
            handle.write('')
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def confusion_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    accuracy = safe_div(true_positive.sum(), confusion.sum())
    return {
        'iou': iou,
        'miou': float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
        'pixel_accuracy': accuracy,
        'gt_count': gt_count,
        'pred_count': pred_count,
        'true_positive': true_positive,
        'union': union,
    }


def add_confusion(target, source):
    source = np.asarray(source, dtype=np.int64)
    if target is None:
        return source.copy()
    if target.shape != source.shape:
        raise ValueError(
            f'Confusion shape mismatch: {target.shape} vs {source.shape}')
    target += source
    return target


def fmt(value, digits=3):
    if value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def summarize(records, args):
    expected = [
        item.strip().lower()
        for item in args.expected_datasets
        if item.strip()
    ]
    by_dataset = defaultdict(list)
    seen_images = {}
    duplicate_images = defaultdict(int)
    for record in records:
        if record.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(
                f"Unexpected schema at {record.get('_source_file')}:"
                f"{record.get('_source_line')}: "
                f"{record.get('schema_version')!r}")
        dataset = str(record.get('dataset_name') or 'unknown').lower()
        image_path = str(record.get('img_path') or '')
        if not image_path:
            raise ValueError(
                'Every presence-allocation record must contain img_path.')
        image_key = (dataset, image_path)
        prior = seen_images.get(image_key)
        if prior is not None:
            if int(prior.get('rank', 0)) == int(record.get('rank', 0)):
                raise ValueError(
                    'Duplicate image record from the same rank; this usually '
                    'means JSONL files were appended or passed twice: '
                    f'{dataset}:{image_path}')
            if (
                    prior.get('variant_stats') != record.get('variant_stats')
                    or prior.get('valid_pixels') != record.get('valid_pixels')):
                raise ValueError(
                    'Cross-rank duplicate image has inconsistent exact '
                    f'statistics: {dataset}:{image_path}')
            duplicate_images[dataset] += 1
            continue
        seen_images[image_key] = record
        by_dataset[dataset].append(record)

    summary_rows = []
    class_rows = []
    contrast_rows = []
    mechanism_rows = []
    integrity_rows = []
    dataset_metrics = {}

    for dataset, dataset_records in sorted(by_dataset.items()):
        class_names = list(dataset_records[0].get('class_names') or [])
        variant_confusions = {name: None for name in VARIANTS}
        variant_counts = {
            name: defaultdict(int) for name in VARIANTS}
        contrast_counts = defaultdict(lambda: defaultdict(int))
        prompt_rows = []
        integrity_max = defaultdict(float)
        artifacts = 0

        for record in dataset_records:
            if list(record.get('class_names') or []) != class_names:
                raise ValueError(
                    f'Class-name mismatch inside dataset {dataset}.')
            if record.get('artifact_path'):
                artifacts += 1
            integrity = record.get('integrity') or {}
            for key in (
                    'class_logit_max_abs',
                    'class_logit_mae',
                    'prompt_reconstruction_max_abs',
                    'baseline_prediction_mismatch_pixels'):
                value = integrity.get(key)
                if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))):
                    raise ValueError(
                        f'Invalid integrity field {key!r} for '
                        f"{dataset}:{record.get('img_path')}")
                integrity_max[key] = max(
                    integrity_max[key], float(value))
            for name in VARIANTS:
                stats = (record.get('variant_stats') or {}).get(name)
                if not isinstance(stats, dict):
                    raise ValueError(
                        f'Missing {name} stats for {dataset}.')
                variant_confusions[name] = add_confusion(
                    variant_confusions[name], stats.get('confusion'))
                for key in (
                        'changed_pixels', 'improved_pixels',
                        'harmed_pixels', 'wrong_to_wrong_pixels',
                        'net_correct_pixels'):
                    variant_counts[name][key] += int(
                        stats.get(key, 0) or 0)
            for contrast in record.get('contrast_stats') or []:
                key = (
                    str(contrast.get('reference')),
                    str(contrast.get('candidate')),
                    str(contrast.get('hypothesis')),
                )
                for field in (
                        'changed_pixels', 'improved_pixels',
                        'harmed_pixels', 'wrong_to_wrong_pixels',
                        'net_correct_pixels'):
                    contrast_counts[key][field] += int(
                        contrast.get(field, 0) or 0)
            prompt_rows.extend(record.get('prompt_stats') or [])

        metrics = {
            name: confusion_metrics(confusion)
            for name, confusion in variant_confusions.items()
        }
        dataset_metrics[dataset] = metrics
        baseline_miou = metrics['p0_baseline']['miou']
        valid_pixels = int(variant_confusions['p0_baseline'].sum())
        for name in VARIANTS:
            current = metrics[name]
            delta_miou = (
                None
                if baseline_miou is None or current['miou'] is None
                else 100.0 * (current['miou'] - baseline_miou)
            )
            counts = variant_counts[name]
            summary_rows.append(dict(
                dataset=dataset,
                variant=name,
                images=len(dataset_records),
                valid_pixels=valid_pixels,
                miou=(
                    None if current['miou'] is None
                    else 100.0 * current['miou']),
                delta_miou=delta_miou,
                pixel_accuracy=(
                    None if current['pixel_accuracy'] is None
                    else 100.0 * current['pixel_accuracy']),
                changed_pixels=counts['changed_pixels'],
                changed_ratio=safe_div(
                    counts['changed_pixels'], valid_pixels),
                improved_pixels=counts['improved_pixels'],
                harmed_pixels=counts['harmed_pixels'],
                wrong_to_wrong_pixels=counts['wrong_to_wrong_pixels'],
                net_correct_pixels=counts['net_correct_pixels'],
            ))
            for class_index, class_name in enumerate(class_names):
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    class_index=class_index,
                    class_name=class_name,
                    iou=(
                        None
                        if not np.isfinite(current['iou'][class_index])
                        else 100.0 * current['iou'][class_index]),
                    gt_pixels=int(current['gt_count'][class_index]),
                    predicted_pixels=int(
                        current['pred_count'][class_index]),
                    true_positive=int(
                        current['true_positive'][class_index]),
                ))

        for (reference, candidate, hypothesis), counts in sorted(
                contrast_counts.items()):
            contrast_rows.append(dict(
                dataset=dataset,
                reference=reference,
                candidate=candidate,
                hypothesis=hypothesis,
                valid_pixels=valid_pixels,
                changed_pixels=counts['changed_pixels'],
                changed_ratio=safe_div(
                    counts['changed_pixels'], valid_pixels),
                improved_pixels=counts['improved_pixels'],
                harmed_pixels=counts['harmed_pixels'],
                wrong_to_wrong_pixels=counts['wrong_to_wrong_pixels'],
                net_correct_pixels=counts['net_correct_pixels'],
                action_precision=safe_div(
                    counts['improved_pixels'],
                    counts['improved_pixels']
                    + counts['harmed_pixels']
                    + counts['wrong_to_wrong_pixels'],
                ),
            ))

        raw_candidates = sum(
            int(row.get('raw_candidate_count', 0) or 0)
            for row in prompt_rows)
        native_kept = sum(
            int(row.get('native_keep_count', 0) or 0)
            for row in prompt_rows)
        raw_gate_kept = sum(
            int(row.get('raw_gate_keep_count', 0) or 0)
            for row in prompt_rows)
        deleted = sum(
            int(row.get('presence_deleted_count', 0) or 0)
            for row in prompt_rows)
        mechanism_rows.append(dict(
            dataset=dataset,
            images=len(dataset_records),
            distributed_padding_duplicates_removed=(
                duplicate_images[dataset]),
            prompt_views=len(prompt_rows),
            artifacts_saved=artifacts,
            raw_candidates=raw_candidates,
            native_kept=native_kept,
            raw_gate_kept=raw_gate_kept,
            presence_deleted_candidates=deleted,
            presence_deleted_candidate_ratio=safe_div(
                deleted, raw_gate_kept),
            mean_presence=safe_mean(
                row.get('presence') for row in prompt_rows),
            mean_deleted_raw_instance_area=safe_mean(
                row.get('deleted_raw_instance_area')
                for row in prompt_rows),
            mean_semantic_threshold_suppressed_area=safe_mean(
                row.get('semantic_threshold_suppressed_area')
                for row in prompt_rows),
            mean_instance_double_suppressed_area=safe_mean(
                row.get('instance_double_suppressed_area')
                for row in prompt_rows),
        ))
        integrity_pass = (
            integrity_max['class_logit_max_abs']
            <= float(args.integrity_tolerance)
            and integrity_max['prompt_reconstruction_max_abs']
            <= float(args.integrity_tolerance)
            and integrity_max[
                'baseline_prediction_mismatch_pixels'] == 0
        )
        integrity_rows.append(dict(
            dataset=dataset,
            images=len(dataset_records),
            integrity_pass=integrity_pass,
            **dict(integrity_max),
        ))

    observed = sorted(by_dataset)
    missing = [dataset for dataset in expected if dataset not in by_dataset]
    extra = [dataset for dataset in observed if dataset not in expected]
    universal_positive = {}
    universal_nonnegative = {}
    for name in VARIANTS[1:]:
        deltas = []
        for dataset in expected:
            metrics = dataset_metrics.get(dataset)
            if metrics is None:
                continue
            baseline = metrics['p0_baseline']['miou']
            current = metrics[name]['miou']
            if baseline is not None and current is not None:
                deltas.append(100.0 * (current - baseline))
        universal_positive[name] = (
            not missing and len(deltas) == len(expected)
            and all(delta > 0.0 for delta in deltas)
        )
        universal_nonnegative[name] = (
            not missing and len(deltas) == len(expected)
            and all(delta >= 0.0 for delta in deltas)
        )

    feasibility = dict(
        schema_version=SCHEMA_VERSION,
        expected_datasets=expected,
        observed_datasets=observed,
        missing_datasets=missing,
        extra_datasets=extra,
        distributed_padding_duplicates_removed={
            dataset: count
            for dataset, count in sorted(duplicate_images.items())
            if count > 0
        },
        complete=not missing,
        integrity_pass=(
            bool(integrity_rows)
            and all(row['integrity_pass'] for row in integrity_rows)
        ),
        universal_positive=universal_positive,
        universal_nonnegative=universal_nonnegative,
        interpretation=(
            'A positive variant is evidence for its predeclared structural '
            'hypothesis, not permission to select pixels or datasets '
            'post hoc. No target-specific threshold is evaluated.'
        ),
    )
    return (
        summary_rows,
        class_rows,
        contrast_rows,
        mechanism_rows,
        integrity_rows,
        feasibility,
    )


def write_report(path, summary_rows, mechanism_rows, integrity_rows,
                 feasibility):
    by_dataset_variant = {
        (row['dataset'], row['variant']): row for row in summary_rows}
    mechanism = {
        row['dataset']: row for row in mechanism_rows}
    integrity = {
        row['dataset']: row for row in integrity_rows}
    lines = [
        '# Presence Allocation Audit',
        '',
        'This report evaluates fixed computation-graph interventions. '
        'It does not route top-1/top-2 predictions and does not change the '
        'returned SegEarth-OV3 baseline output.',
        '',
        '## Integrity',
        '',
        '| Dataset | Reconstruction | Class max error | Prompt max error |',
        '|---|---:|---:|---:|',
    ]
    for dataset in feasibility['observed_datasets']:
        row = integrity[dataset]
        lines.append(
            f"| {dataset} | {'PASS' if row['integrity_pass'] else 'FAIL'} "
            f"| {fmt(row.get('class_logit_max_abs'), 7)} "
            f"| {fmt(row.get('prompt_reconstruction_max_abs'), 7)} |")
    lines.extend([
        '',
        '## Exact counterfactual mIoU',
        '',
        '| Dataset | P0 baseline | P1 branch once | P2 delayed gate | '
        'P3 logit prior | P4 instance scope |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for dataset in feasibility['observed_datasets']:
        cells = []
        for variant in VARIANTS:
            row = by_dataset_variant[(dataset, variant)]
            if variant == 'p0_baseline':
                cells.append(fmt(row['miou']))
            else:
                delta = row['delta_miou']
                cells.append(
                    f"{fmt(row['miou'])} "
                    f"({fmt(delta) if delta is None else f'{delta:+.3f}'})")
        lines.append(
            f'| {dataset} | ' + ' | '.join(cells) + ' |')
    lines.extend([
        '',
        '## Mechanism activation',
        '',
        '| Dataset | Presence-deleted candidates | Deleted ratio | '
        'Mean semantic suppressed area | '
        'Mean instance double-suppressed area |',
        '|---|---:|---:|---:|---:|',
    ])
    for dataset in feasibility['observed_datasets']:
        row = mechanism[dataset]
        lines.append(
            f"| {dataset} | {row['presence_deleted_candidates']} "
            f"| {fmt(row['presence_deleted_candidate_ratio'], 5)} "
            f"| {fmt(row['mean_semantic_threshold_suppressed_area'], 5)} "
            f"| {fmt(row['mean_instance_double_suppressed_area'], 5)} |")
    lines.extend([
        '',
        '## Predeclared interpretation',
        '',
        '- P1 − P0 isolates duplicate presence on the instance branch.',
        '- P2 − P1 isolates presence inside the irreversible candidate gate.',
        '- P3 − P2 isolates probability multiplication versus one logit prior.',
        '- P4 − P1 isolates decoder presence exported into the semantic branch.',
        '- A structural component is universal only when the same fixed '
        'variant is positive on every expected dataset.',
        '',
        'Universal-positive variants: '
        + ', '.join(
            name for name, passed in
            feasibility['universal_positive'].items() if passed)
        if any(feasibility['universal_positive'].values())
        else 'Universal-positive variants: none.',
        '',
    ])
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit('No input JSONL files found.')
    records = list(iter_records(paths))
    if not records:
        raise SystemExit('Input JSONL files contain no records.')
    os.makedirs(args.out_dir, exist_ok=True)
    (
        summary_rows,
        class_rows,
        contrast_rows,
        mechanism_rows,
        integrity_rows,
        feasibility,
    ) = summarize(records, args)
    write_csv(
        os.path.join(args.out_dir, 'presence_allocation_summary.csv'),
        summary_rows)
    write_csv(
        os.path.join(args.out_dir, 'presence_allocation_class.csv'),
        class_rows)
    write_csv(
        os.path.join(args.out_dir, 'presence_allocation_contrasts.csv'),
        contrast_rows)
    write_csv(
        os.path.join(args.out_dir, 'presence_allocation_mechanisms.csv'),
        mechanism_rows)
    write_csv(
        os.path.join(args.out_dir, 'presence_allocation_integrity.csv'),
        integrity_rows)
    with open(
            os.path.join(args.out_dir, 'feasibility.json'), 'w') as handle:
        json.dump(feasibility, handle, indent=2, ensure_ascii=False)
    write_report(
        os.path.join(args.out_dir, 'report.md'),
        summary_rows,
        mechanism_rows,
        integrity_rows,
        feasibility,
    )
    print(json.dumps(feasibility, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
