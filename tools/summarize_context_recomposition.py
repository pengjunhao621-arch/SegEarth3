#!/usr/bin/env python3
"""Summarize the training-free R0--R14 Context recomposition screen."""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    CONTEXT_RECOMPOSITION_PROTOCOL as PROTOCOL,
    CONTEXT_RECOMPOSITION_SCHEMA_VERSION as SCHEMA_VERSION,
    CONTEXT_RECOMPOSITION_VARIANT_NAMES as VARIANTS,
    CONTEXT_RECOMPOSITION_VARIANT_SPECS as VARIANT_SPECS,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')
SPEC_BY_NAME = {
    name: dict(research_id=research_id, mechanism=mechanism)
    for name, research_id, mechanism in VARIANT_SPECS
}


def _paths(patterns):
    return sorted(set(
        path for pattern in patterns
        for path in (glob.glob(pattern) or [pattern])
        if os.path.isfile(path)))


def load_records(patterns, expected, allow_incomplete):
    records, seen, duplicates = [], {}, defaultdict(int)
    for path in _paths(patterns):
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                dataset = str(record.get('dataset_name', '')).lower()
                payload = record.get('context_recomposition')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1)) != SCHEMA_VERSION
                        or payload.get('protocol') != PROTOCOL
                        or tuple(payload.get('variants', {})) != VARIANTS):
                    raise ValueError(
                        f'{path}:{line_number}: recomposition contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['context_recomposition'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No context_recomposition_v1 records found.')
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    variant_rows, class_rows, mechanism_rows = [], [], []
    by_dataset_metrics = {}

    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in VARIANTS}
        oracle_matrices = {name: None for name in VARIANTS}
        counters = {name: defaultdict(int) for name in VARIANTS}
        mechanism = defaultdict(lambda: defaultdict(float))
        mechanism_count = defaultdict(int)
        first = values[0]['context_recomposition']
        metadata_key = (
            tuple(first['selected_slots']), first['selected_admission'],
            float(first['blend']), float(first['probability_clip']),
            float(first['logit_clip']), float(first['near_ratio']),
            int(first['lowpass_divisor']))

        for record in values:
            payload = record['context_recomposition']
            key = (
                tuple(payload['selected_slots']), payload['selected_admission'],
                float(payload['blend']), float(payload['probability_clip']),
                float(payload['logit_clip']), float(payload['near_ratio']),
                int(payload['lowpass_divisor']))
            if key != metadata_key:
                raise ValueError(f'{dataset}: experiment metadata drifted.')
            for name, row in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], row['confusion']['matrix'])
                oracle_matrices[name] = add_matrix(
                    oracle_matrices[name], row['oracle_confusion']['matrix'])
                for field in (
                        'changed_pixels', 'improved_pixels', 'harmed_pixels',
                        'help_minus_harm', 'background_to_foreground_pixels',
                        'foreground_to_background_pixels',
                        'foreground_to_foreground_pixels'):
                    counters[name][field] += int(row.get(field, 0))
            for name, row in payload.get('mechanism_summaries', {}).items():
                mechanism_count[name] += 1
                for field, value in row.items():
                    mechanism[name][field] += float(value)

        metrics = {
            name: confusion_metrics(matrix) for name, matrix in matrices.items()}
        oracle_metrics = {
            name: confusion_metrics(matrix)
            for name, matrix in oracle_matrices.items()}
        by_dataset_metrics[dataset] = metrics
        for name in VARIANTS:
            spec = SPEC_BY_NAME[name]
            reference_name = values[0]['context_recomposition'][
                'variants'][name]['reference_variant']
            metric, reference = metrics[name], metrics[reference_name]
            variant_rows.append(dict(
                dataset=dataset,
                variant=name,
                research_id=spec['research_id'],
                mechanism=spec['mechanism'],
                images=len(values),
                reference_variant=reference_name,
                miou=metric['miou'],
                aacc=metric['aacc'],
                delta_to_reference=metric['miou'] - reference['miou'],
                delta_to_official=(
                    metric['miou'] - metrics['cr_official']['miou']),
                delta_to_role_text=(
                    metric['miou']
                    - metrics['cr_r0_native_role_text']['miou']),
                oracle_miou=oracle_metrics[name]['miou'],
                oracle_headroom_over_reference=(
                    oracle_metrics[name]['miou'] - reference['miou']),
                **counters[name],
            ))
            for class_index, class_name in enumerate(values[0]['class_names']):
                value = metric['iou'][class_index]
                ref_value = reference['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    research_id=spec['research_id'],
                    mechanism=spec['mechanism'],
                    class_index=class_index,
                    class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_reference=(
                        None if value is None or ref_value is None
                        else 100.0 * (value - ref_value)),
                ))
        for name, fields in mechanism.items():
            row = dict(
                dataset=dataset,
                variant=name,
                research_id=SPEC_BY_NAME[name]['research_id'],
                mechanism=SPEC_BY_NAME[name]['mechanism'],
                images=len(values))
            row.update({
                field: value / mechanism_count[name]
                for field, value in fields.items()})
            mechanism_rows.append(row)

    cross_rows = []
    for name in VARIANTS:
        if name == 'cr_official':
            continue
        rows = [
            row for row in variant_rows if row['variant'] == name]
        deltas = [row['delta_to_reference'] for row in rows]
        role_deltas = [row['delta_to_role_text'] for row in rows]
        official_deltas = [row['delta_to_official'] for row in rows]
        oracle = [row['oracle_headroom_over_reference'] for row in rows]
        spec = SPEC_BY_NAME[name]
        cross_rows.append(dict(
            variant=name,
            research_id=spec['research_id'],
            mechanism=spec['mechanism'],
            datasets=len(rows),
            positive_vs_reference=sum(value > 0 for value in deltas),
            positive_vs_role_text=sum(value > 0 for value in role_deltas),
            macro_delta_to_reference=sum(deltas) / len(deltas),
            macro_delta_to_role_text=sum(role_deltas) / len(role_deltas),
            macro_delta_to_official=sum(official_deltas) / len(official_deltas),
            median_delta_to_role_text=statistics.median(role_deltas),
            min_delta_to_role_text=min(role_deltas),
            max_delta_to_role_text=max(role_deltas),
            macro_oracle_headroom=sum(oracle) / len(oracle),
        ))
    return variant_rows, class_rows, mechanism_rows, cross_rows


def write_report(path, variant_rows, cross_rows, missing, duplicates):
    by_dataset = defaultdict(dict)
    for row in variant_rows:
        by_dataset[row['dataset']][row['variant']] = row
    lines = [
        '# Context Recomposition v1: R0--R14',
        '',
        'All candidates are frozen, class-agnostic fixed rules built from the '
        'current evaluation image/aligned real Context. GT is used only for '
        'evaluation and oracle diagnostics.',
        '',
        '## Protected endpoints',
        '',
        '| Dataset | Official | R0 Native Role-Text | R0 gain |',
        '|---|---:|---:|---:|',
    ]
    for dataset, rows in sorted(by_dataset.items()):
        official = rows['cr_official']['miou']
        role = rows['cr_r0_native_role_text']['miou']
        lines.append(
            f'| {dataset} | {official:.3f} | {role:.3f} | '
            f'{role-official:+.3f} |')
    lines.extend([
        '',
        '## R1--R14 relative to Native Role-Text',
        '',
        '| Variant | ID | Mechanism | Positive | Macro Δ | Range | Oracle |',
        '|---|---|---|---:|---:|---:|---:|',
    ])
    for row in sorted(
            (value for value in cross_rows
             if value['variant'] != 'cr_r0_native_role_text'),
            key=lambda value: -value['macro_delta_to_role_text']):
        lines.append(
            f'| {row["variant"]} | {row["research_id"]} | '
            f'{row["mechanism"]} | '
            f'{row["positive_vs_role_text"]}/{row["datasets"]} | '
            f'{row["macro_delta_to_role_text"]:+.3f} | '
            f'[{row["min_delta_to_role_text"]:+.3f}, '
            f'{row["max_delta_to_role_text"]:+.3f}] | '
            f'{row["macro_oracle_headroom"]:+.3f} |')
    lines.extend([
        '',
        '## Decision boundary',
        '',
        '- Primary reference is R0 Native Role-Text, not the official baseline.',
        '- A method candidate requires positive macro mIoU and at least 4/6 '
        'positive datasets; oracle headroom alone is diagnostic.',
        '- R12/R13 and R11 prior-only are mechanism controls unless their fixed '
        'rule is repeatedly mIoU-positive.',
        '- No dataset-specific weight, GT gate, external exemplar, retrieval '
        'bank, or training is present.',
        f'- Missing datasets: {missing or "none"}',
        f'- Deduplicated DDP records: {duplicates or "none"}',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    expected = tuple(str(value).lower() for value in args.expected_datasets)
    if 'isaid' in expected:
        raise ValueError('iSAID is excluded from project evaluation.')
    records, missing, duplicates = load_records(
        args.inputs, expected, args.allow_incomplete)
    variant_rows, class_rows, mechanism_rows, cross_rows = summarize(records)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'classes.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'mechanisms.csv'), mechanism_rows)
    write_csv(os.path.join(args.out_dir, 'cross_dataset.csv'), cross_rows)
    write_report(
        os.path.join(args.out_dir, 'report.md'), variant_rows, cross_rows,
        missing, duplicates)


if __name__ == '__main__':
    main()
