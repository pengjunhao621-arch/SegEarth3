#!/usr/bin/env python3
"""Summarize exact same-forward semantic/instance fusion diagnostics."""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dual_head_fusion_definitions import (
    SCHEMA_VERSION,
    VARIANT_NAMES,
)


VARIANTS = ('p0_baseline',) + VARIANT_NAMES
EXCLUDED_DATASETS = {'isaid'}


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
    parser.add_argument(
        '--positive-dataset-gate', type=int, default=4)
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


def finite_values(values):
    return [
        float(value) for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]


def safe_mean(values):
    values = finite_values(values)
    return sum(values) / len(values) if values else None


def weighted_mean(pairs):
    numerator = 0.0
    denominator = 0.0
    for value, weight in pairs:
        if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and isinstance(weight, (int, float))
                and float(weight) > 0):
            numerator += float(value) * float(weight)
            denominator += float(weight)
    return numerator / denominator if denominator > 0 else None


def safe_correlation(left, right):
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(float(x))
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
            and math.isfinite(float(y))
        )
    ]
    if len(pairs) < 2:
        return None
    left_array = np.asarray([pair[0] for pair in pairs])
    right_array = np.asarray([pair[1] for pair in pairs])
    if left_array.std() == 0 or right_array.std() == 0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def add_confusion(target, source):
    source = np.asarray(source, dtype=np.int64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError(
            f'Confusion matrix must be square, got {source.shape}.')
    if target is None:
        return source.copy()
    if target.shape != source.shape:
        raise ValueError(
            f'Confusion shape mismatch: {target.shape} vs {source.shape}.')
    target += source
    return target


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
    return dict(
        iou=iou,
        miou=(
            float(np.nanmean(iou))
            if np.isfinite(iou).any() else None),
        pixel_accuracy=safe_div(
            float(true_positive.sum()), float(confusion.sum())),
        true_positive=true_positive,
        gt_count=gt_count,
        pred_count=pred_count,
        union=union,
    )


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


def _validate_and_deduplicate(records):
    grouped = defaultdict(list)
    seen = {}
    duplicates = defaultdict(int)
    for record in records:
        if record.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(
                f"Unexpected schema at {record.get('_source_file')}:"
                f"{record.get('_source_line')}: "
                f"{record.get('schema_version')!r}")
        dataset = str(
            record.get('dataset_name') or 'unknown').strip().lower()
        if dataset in EXCLUDED_DATASETS:
            raise ValueError(
                f'Dataset {dataset!r} is explicitly excluded from this '
                'project evaluation.')
        image_path = str(record.get('img_path') or '')
        if not image_path:
            raise ValueError(
                'Every dual-head fusion record must contain img_path.')
        key = (dataset, image_path)
        prior = seen.get(key)
        if prior is not None:
            if int(prior.get('rank', 0)) == int(record.get('rank', 0)):
                raise ValueError(
                    'Duplicate image record from the same rank; use a fresh '
                    f'output directory: {dataset}:{image_path}')
            if (
                    prior.get('variant_stats') != record.get('variant_stats')
                    or prior.get('valid_pixels') != record.get('valid_pixels')):
                raise ValueError(
                    'Cross-rank padding duplicate has inconsistent exact '
                    f'statistics: {dataset}:{image_path}')
            duplicates[dataset] += 1
            continue
        seen[key] = record
        grouped[dataset].append(record)
    return grouped, dict(duplicates)


def summarize(records, args):
    expected = [
        str(item).strip().lower()
        for item in args.expected_datasets
        if str(item).strip()
    ]
    if EXCLUDED_DATASETS.intersection(expected):
        raise ValueError('iSAID must not be requested by this experiment.')
    grouped, duplicates = _validate_and_deduplicate(records)

    summary_rows = []
    class_rows = []
    contrast_rows = []
    mechanism_rows = []
    agreement_rows = []
    integrity_rows = []
    dataset_deltas = defaultdict(dict)

    for dataset, dataset_records in sorted(grouped.items()):
        class_names = list(dataset_records[0].get('class_names') or [])
        if not class_names:
            raise ValueError(f'Missing class_names for {dataset}.')
        confusions = {name: None for name in VARIANTS}
        changes = {
            name: defaultdict(int) for name in VARIANTS}
        contrasts = defaultdict(lambda: defaultdict(int))
        prompt_stats = []
        integrity_max = defaultdict(float)
        agreement_groups = defaultdict(list)
        artifact_count = 0

        for record in dataset_records:
            if list(record.get('class_names') or []) != class_names:
                raise ValueError(
                    f'Class-name mismatch inside dataset {dataset}.')
            if record.get('artifact_path'):
                artifact_count += 1
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

            variant_stats = record.get('variant_stats') or {}
            for name in VARIANTS:
                stats = variant_stats.get(name)
                if not isinstance(stats, dict):
                    raise ValueError(
                        f'Missing variant {name!r} for {dataset}.')
                confusions[name] = add_confusion(
                    confusions[name], stats.get('confusion'))
                if name != 'p0_baseline':
                    for key in (
                            'changed_pixels', 'improved_pixels',
                            'harmed_pixels', 'wrong_to_wrong_pixels',
                            'net_correct_pixels'):
                        changes[name][key] += int(stats.get(key, 0))

            for row in record.get('contrast_stats') or []:
                key = (
                    str(row.get('reference')),
                    str(row.get('candidate')),
                    str(row.get('hypothesis')),
                )
                for count_name in (
                        'changed_pixels', 'improved_pixels',
                        'harmed_pixels', 'wrong_to_wrong_pixels',
                        'net_correct_pixels'):
                    contrasts[key][count_name] += int(
                        row.get(count_name, 0))
            prompt_stats.extend(record.get('prompt_stats') or [])
            for row in record.get('high_agreement_stats') or []:
                key = (
                    float(row.get('gap_threshold')),
                    float(row.get('evidence_threshold')),
                )
                agreement_groups[key].append(row)

        metrics = {
            name: confusion_metrics(confusions[name])
            for name in VARIANTS
        }
        base_miou = metrics['p0_baseline']['miou']
        base_correct = int(
            metrics['p0_baseline']['true_positive'].sum())
        base_total = int(confusions['p0_baseline'].sum())
        base_wrong = base_total - base_correct

        for name in VARIANTS:
            metric = metrics[name]
            delta = (
                100.0 * (metric['miou'] - base_miou)
                if metric['miou'] is not None
                and base_miou is not None else None)
            row = dict(
                dataset=dataset,
                variant=name,
                images=len(dataset_records),
                valid_pixels=int(confusions[name].sum()),
                miou=metric['miou'],
                miou_percent=(
                    100.0 * metric['miou']
                    if metric['miou'] is not None else None),
                delta_miou_pp=delta,
                pixel_accuracy=metric['pixel_accuracy'],
                pixel_accuracy_percent=(
                    100.0 * metric['pixel_accuracy']
                    if metric['pixel_accuracy'] is not None else None),
                artifacts=artifact_count,
            )
            if name != 'p0_baseline':
                row.update(changes[name])
                row['improved_to_harmed_ratio'] = safe_div(
                    changes[name]['improved_pixels'],
                    changes[name]['harmed_pixels'])
                row['changed_fraction'] = safe_div(
                    changes[name]['changed_pixels'],
                    int(confusions[name].sum()))
            summary_rows.append(row)
            dataset_deltas[name][dataset] = delta

            for class_index, class_name in enumerate(class_names):
                iou = metric['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    class_index=class_index,
                    class_name=class_name,
                    gt_pixels=int(metric['gt_count'][class_index]),
                    predicted_pixels=int(
                        metric['pred_count'][class_index]),
                    intersection=int(
                        metric['true_positive'][class_index]),
                    union=int(metric['union'][class_index]),
                    iou=(
                        float(iou)
                        if math.isfinite(float(iou)) else None),
                    delta_iou_pp=(
                        100.0 * (
                            float(iou)
                            - float(
                                metrics['p0_baseline']['iou'][
                                    class_index])
                        )
                        if (
                            math.isfinite(float(iou))
                            and math.isfinite(float(
                                metrics['p0_baseline']['iou'][
                                    class_index]))
                        ) else None),
                ))

        for (
                reference, candidate, hypothesis), counts in sorted(
                    contrasts.items()):
            row = dict(
                dataset=dataset,
                reference=reference,
                candidate=candidate,
                hypothesis=hypothesis,
            )
            row.update(counts)
            row['improved_to_harmed_ratio'] = safe_div(
                counts['improved_pixels'],
                counts['harmed_pixels'])
            row['net_correct_fraction'] = safe_div(
                counts['net_correct_pixels'], base_total)
            contrast_rows.append(row)

        numeric_prompt_keys = sorted(set().union(*[
            set(row.keys()) for row in prompt_stats
        ])) if prompt_stats else []
        mechanism = dict(
            dataset=dataset,
            images=len(dataset_records),
            prompts=len(prompt_stats),
            no_native_candidate_prompt_fraction=safe_div(
                sum(
                    int(row.get('native_keep_count', 0)) == 0
                    for row in prompt_stats),
                len(prompt_stats)),
            artifacts=artifact_count,
            presence_object_score_correlation=safe_correlation(
                [row.get('presence') for row in prompt_stats],
                [row.get('object_score_max') for row in prompt_stats],
            ),
        )
        sum_keys = {
            'raw_candidate_count', 'native_keep_count'}
        for key in numeric_prompt_keys:
            values = [row.get(key) for row in prompt_stats]
            if key in sum_keys:
                mechanism[f'{key}_sum'] = sum(
                    int(value or 0) for value in values)
            mechanism[f'{key}_mean'] = safe_mean(values)
        mechanism_rows.append(mechanism)

        for (gap, evidence), rows in sorted(agreement_groups.items()):
            selected = sum(
                int(row.get('selected_pixels', 0)) for row in rows)
            wrong = sum(
                int(row.get('wrong_pixels', 0)) for row in rows)
            correct = sum(
                int(row.get('correct_pixels', 0)) for row in rows)
            agreement_rows.append(dict(
                dataset=dataset,
                gap_threshold=gap,
                evidence_threshold=evidence,
                selected_pixels=selected,
                wrong_pixels=wrong,
                correct_pixels=correct,
                wrong_fraction=safe_div(wrong, selected),
                baseline_error_coverage=safe_div(wrong, base_wrong),
                mean_wrong_presence=weighted_mean([
                    (row.get('mean_wrong_presence'),
                     row.get('wrong_pixels', 0))
                    for row in rows
                ]),
                mean_wrong_object_score=weighted_mean([
                    (row.get('mean_wrong_object_score'),
                     row.get('wrong_pixels', 0))
                    for row in rows
                ]),
                mean_wrong_margin=weighted_mean([
                    (row.get('mean_wrong_margin'),
                     row.get('wrong_pixels', 0))
                    for row in rows
                ]),
                low_presence_wrong_pixels=sum(
                    int(row.get('low_presence_wrong_pixels', 0))
                    for row in rows),
                low_object_wrong_pixels=sum(
                    int(row.get('low_object_wrong_pixels', 0))
                    for row in rows),
            ))

        tolerance = float(args.integrity_tolerance)
        integrity_pass = (
            integrity_max['class_logit_max_abs'] <= tolerance
            and integrity_max['prompt_reconstruction_max_abs'] <= tolerance
            and integrity_max[
                'baseline_prediction_mismatch_pixels'] == 0)
        integrity_rows.append(dict(
            dataset=dataset,
            images=len(dataset_records),
            distributed_padding_duplicates_removed=duplicates.get(
                dataset, 0),
            integrity_pass=integrity_pass,
            **integrity_max,
        ))

    present = sorted(grouped)
    missing = sorted(set(expected) - set(present))
    unexpected = sorted(set(present) - set(expected))
    ranking_rows = []
    for name in VARIANT_NAMES:
        deltas = [
            dataset_deltas[name].get(dataset)
            for dataset in expected
        ]
        finite = finite_values(deltas)
        positive = sum(value > 0 for value in finite)
        nonnegative = sum(value >= 0 for value in finite)
        ranking_rows.append(dict(
            variant=name,
            evaluated_datasets=len(finite),
            expected_datasets=len(expected),
            positive_datasets=positive,
            nonnegative_datasets=nonnegative,
            macro_mean_delta_miou_pp=safe_mean(finite),
            median_delta_miou_pp=(
                float(np.median(finite)) if finite else None),
            worst_delta_miou_pp=min(finite) if finite else None,
            best_delta_miou_pp=max(finite) if finite else None,
            positive_gate_pass=(
                len(finite) == len(expected)
                and positive >= int(args.positive_dataset_gate)),
            all_positive=(
                len(finite) == len(expected)
                and positive == len(expected)),
        ))
    ranking_rows.sort(
        key=lambda row: (
            int(row['positive_gate_pass']),
            int(row['positive_datasets']),
            (
                row['macro_mean_delta_miou_pp']
                if row['macro_mean_delta_miou_pp'] is not None
                else float('-inf')
            ),
            (
                row['worst_delta_miou_pp']
                if row['worst_delta_miou_pp'] is not None
                else float('-inf')
            ),
        ),
        reverse=True,
    )

    decision = dict(
        schema_version=SCHEMA_VERSION,
        complete=(not missing and not unexpected),
        expected_datasets=expected,
        present_datasets=present,
        missing_datasets=missing,
        unexpected_datasets=unexpected,
        excluded_datasets=sorted(EXCLUDED_DATASETS),
        integrity_pass=(
            bool(integrity_rows)
            and all(row['integrity_pass'] for row in integrity_rows)),
        positive_dataset_gate=int(args.positive_dataset_gate),
        variants_passing_positive_gate=[
            row['variant'] for row in ranking_rows
            if row['positive_gate_pass']
        ],
        distributed_padding_duplicates_removed=duplicates,
        ranking=[dict(row) for row in ranking_rows],
        interpretation=(
            'The >=4-dataset gate is a prioritization signal, not proof. '
            'Inspect per-class IoU, improved-versus-harmed transitions, '
            'mechanism diagnostics, and seed stability before selecting a '
            'fusion graph.'
        ),
    )
    return (
        summary_rows,
        class_rows,
        contrast_rows,
        mechanism_rows,
        agreement_rows,
        ranking_rows,
        integrity_rows,
        decision,
    )


def _format(value, digits=3):
    if value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def write_report(path, decision, ranking_rows):
    lines = [
        '# Dual-head Fusion Diagnostic Summary',
        '',
        f"- Complete: `{decision['complete']}`",
        f"- Integrity pass: `{decision['integrity_pass']}`",
        '- Present datasets: '
        + ', '.join(decision['present_datasets']),
        '- Missing datasets: '
        + (', '.join(decision['missing_datasets']) or 'none'),
        '- Variants passing the positive-dataset gate: '
        + (
            ', '.join(decision['variants_passing_positive_gate'])
            or 'none'
        ),
        '',
        '| Rank | Variant | Positive | Mean ΔmIoU (pp) | '
        'Median | Worst | Gate |',
        '|---:|---|---:|---:|---:|---:|---|',
    ]
    for index, row in enumerate(ranking_rows, 1):
        lines.append(
            f"| {index} | {row['variant']} | "
            f"{row['positive_datasets']}/{row['expected_datasets']} | "
            f"{_format(row['macro_mean_delta_miou_pp'])} | "
            f"{_format(row['median_delta_miou_pp'])} | "
            f"{_format(row['worst_delta_miou_pp'])} | "
            f"{row['positive_gate_pass']} |"
        )
    lines.extend([
        '',
        'The ranking is diagnostic. A viable structural claim also requires '
        'the expected mechanism contrast, favorable improved/harmed pixel '
        'transitions, interpretable per-class behavior, and repeated-seed '
        'confirmation.',
        '',
    ])
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit('No input JSONL files were found.')
    result = summarize(list(iter_records(paths)), args)
    (
        summary_rows,
        class_rows,
        contrast_rows,
        mechanism_rows,
        agreement_rows,
        ranking_rows,
        integrity_rows,
        decision,
    ) = result
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(args.out_dir, 'fusion_summary.csv'),
        summary_rows)
    write_csv(
        os.path.join(args.out_dir, 'fusion_class_iou.csv'),
        class_rows)
    write_csv(
        os.path.join(args.out_dir, 'fusion_contrasts.csv'),
        contrast_rows)
    write_csv(
        os.path.join(args.out_dir, 'fusion_mechanisms.csv'),
        mechanism_rows)
    write_csv(
        os.path.join(args.out_dir, 'high_agreement_errors.csv'),
        agreement_rows)
    write_csv(
        os.path.join(args.out_dir, 'cross_dataset_ranking.csv'),
        ranking_rows)
    write_csv(
        os.path.join(args.out_dir, 'integrity.csv'),
        integrity_rows)
    with open(
            os.path.join(args.out_dir, 'decision.json'), 'w') as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False)
    write_report(
        os.path.join(args.out_dir, 'report.md'),
        decision,
        ranking_rows,
    )
    print(json.dumps(
        dict(
            inputs=len(paths),
            datasets=decision['present_datasets'],
            complete=decision['complete'],
            integrity_pass=decision['integrity_pass'],
            passing_variants=decision[
                'variants_passing_positive_gate'],
        ),
        ensure_ascii=False,
    ))


if __name__ == '__main__':
    main()
