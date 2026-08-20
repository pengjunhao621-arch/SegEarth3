#!/usr/bin/env python3
"""Summarize the frozen V0--V2/V4--V8 multimodal fusion screen."""

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
    ROLE_MULTIMODAL_FUSION_PROTOCOL as PROTOCOL,
    ROLE_MULTIMODAL_FUSION_SCHEMA_VERSION as SCHEMA_VERSION,
    ROLE_MULTIMODAL_FUSION_VARIANT_NAMES as VARIANTS,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


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
                payload = record.get('role_multimodal_fusion')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1)) != SCHEMA_VERSION
                        or payload.get('protocol') != PROTOCOL
                        or tuple(payload.get('variants', {})) != VARIANTS):
                    raise ValueError(
                        f'{path}:{line_number}: fusion contract mismatch.')
                exclusions = set(payload.get('excluded_methods', ()))
                if not {
                        'text_native_exemplar', 'support_image',
                        'retrieval_reference_bank'}.issubset(exclusions):
                    raise ValueError(
                        f'{path}:{line_number}: excluded-method contract missing.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['role_multimodal_fusion'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No role_multimodal_fusion_v1 records found.')
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def variant_parts(name):
    if name == 'rmf_v0_official':
        return 'v0', 'official'
    if name.startswith('rmf_v1_role_text_'):
        return 'v1_role_text', name.rsplit('_', 1)[-1]
    if name.endswith('_anchor'):
        return name[len('rmf_'):-len('_anchor')], 'anchor'
    for method in (
            'v2_canvas_fine', 'v2_canvas_shared', 'v4_transport',
            'v5_agreement', 'v6_qk', 'v7_value', 'v8_full'):
        prefix = f'rmf_{method}_'
        if name.startswith(prefix):
            return method, name[len(prefix):]
    raise ValueError(f'Cannot parse variant {name!r}.')


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    variant_rows, class_rows, mechanism_rows = [], [], []
    metrics_by_dataset = {}
    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in VARIANTS}
        oracle_matrices = {name: None for name in VARIANTS}
        counters = {name: defaultdict(int) for name in VARIANTS}
        mechanism = defaultdict(lambda: defaultdict(float))
        mechanism_count = defaultdict(int)
        first = values[0]['role_multimodal_fusion']
        metadata_key = (
            tuple(first['selected_slots']), first['selected_admission'],
            float(first['blend']), int(first['local_window']))
        for record in values:
            payload = record['role_multimodal_fusion']
            key = (
                tuple(payload['selected_slots']), payload['selected_admission'],
                float(payload['blend']), int(payload['local_window']))
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
            for method, row in payload.get(
                    'mechanism_summaries', {}).items():
                mechanism_count[method] += 1
                for field, value in row.items():
                    mechanism[method][field] += float(value)

        metrics = {
            name: confusion_metrics(matrix) for name, matrix in matrices.items()}
        oracle_metrics = {
            name: confusion_metrics(matrix)
            for name, matrix in oracle_matrices.items()}
        metrics_by_dataset[dataset] = metrics
        for name in VARIANTS:
            method, scope = variant_parts(name)
            reference_name = values[0]['role_multimodal_fusion'][
                'variants'][name]['reference_variant']
            metric, reference = metrics[name], metrics[reference_name]
            variant_rows.append(dict(
                dataset=dataset,
                variant=name,
                method=method,
                role_scope=scope,
                images=len(values),
                reference_variant=reference_name,
                miou=metric['miou'],
                aacc=metric['aacc'],
                delta_to_reference=metric['miou'] - reference['miou'],
                delta_to_v0=(
                    metric['miou'] - metrics['rmf_v0_official']['miou']),
                delta_to_v1_psi=(
                    metric['miou']
                    - metrics['rmf_v1_role_text_psi']['miou']),
                oracle_miou=oracle_metrics[name]['miou'],
                oracle_headroom_over_reference=(
                    oracle_metrics[name]['miou'] - reference['miou']),
                **counters[name],
            ))
            for class_index, class_name in enumerate(
                    values[0]['class_names']):
                value = metric['iou'][class_index]
                ref_value = reference['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    method=method,
                    role_scope=scope,
                    class_index=class_index,
                    class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_reference=(
                        None if value is None or ref_value is None
                        else 100.0 * (value - ref_value)),
                ))
        for method, fields in mechanism.items():
            row = dict(dataset=dataset, method=method, images=len(values))
            row.update({
                field: value / mechanism_count[method]
                for field, value in fields.items()})
            mechanism_rows.append(row)

    cross_rows = []
    for name in VARIANTS:
        if name == 'rmf_v0_official':
            continue
        method, scope = variant_parts(name)
        deltas, v0_deltas, v1_deltas, oracle = [], [], [], []
        for dataset, metrics in sorted(metrics_by_dataset.items()):
            dataset_row = next(
                row for row in variant_rows
                if row['dataset'] == dataset and row['variant'] == name)
            deltas.append(dataset_row['delta_to_reference'])
            v0_deltas.append(dataset_row['delta_to_v0'])
            v1_deltas.append(dataset_row['delta_to_v1_psi'])
            oracle.append(dataset_row['oracle_headroom_over_reference'])
        cross_rows.append(dict(
            variant=name,
            method=method,
            role_scope=scope,
            datasets=len(deltas),
            positive_vs_reference=sum(value > 0 for value in deltas),
            macro_delta_to_reference=sum(deltas) / len(deltas),
            median_delta_to_reference=statistics.median(deltas),
            min_delta_to_reference=min(deltas),
            max_delta_to_reference=max(deltas),
            macro_delta_to_v0=sum(v0_deltas) / len(v0_deltas),
            macro_delta_to_v1_psi=sum(v1_deltas) / len(v1_deltas),
            macro_oracle_headroom=sum(oracle) / len(oracle),
        ))
    return variant_rows, class_rows, mechanism_rows, cross_rows


def write_report(path, variant_rows, cross_rows, missing, duplicates):
    by_dataset = defaultdict(dict)
    for row in variant_rows:
        by_dataset[row['dataset']][row['variant']] = row
    lines = [
        '# Role Multimodal Fusion v1',
        '',
        'This frozen screen excludes image exemplars, support sets, retrieval '
        'banks, extra labels, and training. V2 uses only Fine and aligned '
        'Context from the current evaluation image/verified adjacent tiles.',
        '',
        '## Protected endpoints',
        '',
        '| Dataset | V0 official | V1 Role-Text PSI | V1 gain |',
        '|---|---:|---:|---:|',
    ]
    for dataset, rows in sorted(by_dataset.items()):
        v0 = rows['rmf_v0_official']['miou']
        v1 = rows['rmf_v1_role_text_psi']['miou']
        lines.append(
            f'| {dataset} | {v0:.3f} | {v1:.3f} | {v1-v0:+.3f} |')
    lines.extend([
        '',
        '## Fixed variants across datasets',
        '',
        '| Variant | Scope | Positive vs own reference | Macro Δ | Range | '
        'Oracle headroom |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for row in sorted(
            cross_rows,
            key=lambda value: -value['macro_delta_to_reference']):
        lines.append(
            f'| {row["variant"]} | {row["role_scope"]} | '
            f'{row["positive_vs_reference"]}/{row["datasets"]} | '
            f'{row["macro_delta_to_reference"]:+.3f} | '
            f'[{row["min_delta_to_reference"]:+.3f}, '
            f'{row["max_delta_to_reference"]:+.3f}] | '
            f'{row["macro_oracle_headroom"]:+.3f} |')
    lines.extend([
        '',
        '## Interpretation boundary',
        '',
        '- A negative row rejects this fixed realization, not the entire idea '
        'family. The Fine-only canvas control and Q/K-only, V-only, full '
        'operators distinguish several structurally different realizations.',
        '- Oracle headroom uses GT only to ask whether the variant introduced '
        'complementary correct pixels; it is never used as an inference gate.',
        '- A candidate for method development should improve final mIoU and '
        'show a repeatable role/mechanism pattern, rather than rely on one '
        'dataset-specific weight.',
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
