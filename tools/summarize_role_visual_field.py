#!/usr/bin/env python3
"""Summarize the two-view role-specific visual-field experiment."""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    ROLE_VISUAL_FIELD_COMPOSITIONS as RVF_COMPOSITIONS,
    ROLE_VISUAL_FIELD_PROTOCOL as RVF_PROTOCOL,
    ROLE_VISUAL_FIELD_SCHEMA_VERSION as RVF_SCHEMA_VERSION,
    ROLE_VISUAL_FIELD_VARIANT_NAMES as RVF_VARIANT_NAMES,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def load_records(patterns, expected, allow_incomplete):
    paths = sorted(set(
        path for pattern in patterns
        for path in (glob.glob(pattern) or [pattern])
        if os.path.isfile(path)))
    records, seen, duplicates = [], {}, defaultdict(int)
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                dataset = str(record.get('dataset_name', '')).lower()
                payload = record.get('role_visual_field')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1))
                        != RVF_SCHEMA_VERSION
                        or payload.get('protocol') != RVF_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != RVF_VARIANT_NAMES):
                    raise ValueError(
                        f'{path}:{line_number}: visual-field contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['role_visual_field'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No role_visual_field_v1 records found.')
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    rows, class_rows, best_rows = [], [], []
    metrics_by_dataset = {}
    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in RVF_VARIANT_NAMES}
        counters = {name: defaultdict(int) for name in RVF_VARIANT_NAMES}
        first = values[0]['role_visual_field']
        selection = tuple(first['selected_slots'])
        admission = first['selected_admission']
        for record in values:
            payload = record['role_visual_field']
            if (tuple(payload['selected_slots']) != selection
                    or payload['selected_admission'] != admission):
                raise ValueError(f'{dataset}: selected Role-Text drifted.')
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in ('changed_pixels', 'improved_pixels',
                              'harmed_pixels', 'help_minus_harm'):
                    counters[name][field] += int(stats.get(field, 0))
        metrics = {
            name: confusion_metrics(matrix) for name, matrix in matrices.items()
        }
        metrics_by_dataset[dataset] = metrics
        official = metrics['rvf_official']
        best_text = metrics['rvf_best_text']
        for name in RVF_VARIANT_NAMES:
            metric = metrics[name]
            family = (
                'anchor' if name.startswith('rvf_anchor_')
                else 'text' if name.startswith('rvf_text_')
                else 'reference')
            composition = (
                name.rsplit('_', 1)[-1] if family != 'reference' else None)
            reference_name = (
                'rvf_best_text' if family == 'text' else 'rvf_official')
            reference = metrics[reference_name]
            counterpart = None
            interaction = None
            if family in ('anchor', 'text'):
                counterpart = metrics[
                    f'rvf_{"anchor" if family == "text" else "text"}_'
                    f'{composition}']
                anchor_gain = (
                    metrics[f'rvf_anchor_{composition}']['miou']
                    - official['miou'])
                text_gain = (
                    metrics[f'rvf_text_{composition}']['miou']
                    - best_text['miou'])
                interaction = text_gain - anchor_gain
            rows.append(dict(
                dataset=dataset,
                variant=name,
                family=family,
                composition=composition,
                images=len(values),
                selected_slots=list(selection),
                selected_admission=admission,
                miou=metric['miou'],
                aacc=metric['aacc'],
                delta_to_official=metric['miou'] - official['miou'],
                delta_to_best_text=metric['miou'] - best_text['miou'],
                delta_to_family_reference=(
                    metric['miou'] - reference['miou']),
                delta_to_counterpart=(
                    None if counterpart is None
                    else metric['miou'] - counterpart['miou']),
                text_visual_interaction=interaction,
                **counters[name],
            ))
            for class_index, class_name in enumerate(
                    values[0]['class_names']):
                value = metric['iou'][class_index]
                ref_value = reference['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    family=family,
                    composition=composition,
                    class_index=class_index,
                    class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_family_reference=(
                        None if value is None or ref_value is None
                        else 100.0 * (value - ref_value)),
                ))
        mixed = tuple(
            value for value in RVF_COMPOSITIONS
            if value not in ('FFF', 'CCC'))
        for family, reference_name in (
                ('anchor', 'rvf_official'), ('text', 'rvf_best_text')):
            best = max(
                mixed,
                key=lambda value: metrics[
                    f'rvf_{family}_{value}']['miou'])
            best_metric = metrics[f'rvf_{family}_{best}']
            fff = metrics[f'rvf_{family}_FFF']
            ccc = metrics[f'rvf_{family}_CCC']
            best_rows.append(dict(
                dataset=dataset,
                family=family,
                best_mixed=best,
                best_mixed_miou=best_metric['miou'],
                family_reference=reference_name,
                family_reference_miou=metrics[reference_name]['miou'],
                fff_miou=fff['miou'],
                ccc_miou=ccc['miou'],
                delta_to_fff=best_metric['miou'] - fff['miou'],
                delta_to_ccc=best_metric['miou'] - ccc['miou'],
                exceeds_both=(
                    best_metric['miou'] > max(fff['miou'], ccc['miou'])),
            ))
    fixed_rows = []
    for family in ('anchor', 'text'):
        for composition in RVF_COMPOSITIONS:
            values = []
            positives = 0
            for dataset, metrics in metrics_by_dataset.items():
                reference = (
                    metrics['rvf_official'] if family == 'anchor'
                    else metrics['rvf_best_text'])
                delta = (
                    metrics[f'rvf_{family}_{composition}']['miou']
                    - reference['miou'])
                values.append(delta)
                positives += delta > 0
            fixed_rows.append(dict(
                family=family,
                composition=composition,
                datasets=len(values),
                positive_datasets=positives,
                macro_delta=sum(values) / len(values),
                min_delta=min(values),
                max_delta=max(values),
            ))
    return rows, class_rows, best_rows, fixed_rows


def write_report(path, best_rows, fixed_rows, missing, duplicates):
    lines = [
        '# Role Visual Field v1',
        '',
        'Primary question: does a mixed P/S/I Fine–Context allocation exceed '
        'both uniform FFF and uniform CCC?',
        '',
        '## Dataset-wise best mixed allocation',
        '',
        '| Dataset | Family | Best mixed | mIoU | Δ vs FFF | Δ vs CCC | > both |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for row in best_rows:
        lines.append(
            f'| {row["dataset"]} | {row["family"]} | '
            f'{row["best_mixed"]} | {row["best_mixed_miou"]:.3f} | '
            f'{row["delta_to_fff"]:+.3f} | {row["delta_to_ccc"]:+.3f} | '
            f'{row["exceeds_both"]} |')
    lines.extend([
        '',
        '## Fixed cross-dataset allocations',
        '',
        '| Family | Composition | Positive | Macro Δ | Min Δ | Max Δ |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in sorted(
            fixed_rows,
            key=lambda item: (item['family'], -item['macro_delta'])):
        lines.append(
            f'| {row["family"]} | {row["composition"]} | '
            f'{row["positive_datasets"]}/{row["datasets"]} | '
            f'{row["macro_delta"]:+.3f} | {row["min_delta"]:+.3f} | '
            f'{row["max_delta"]:+.3f} |')
    lines.extend([
        '',
        f'- Missing datasets: {missing or "none"}',
        f'- Deduplicated DDP records: {duplicates or "none"}',
        '- Dataset-wise best mixed rows are validation diagnostics; the fixed '
        'cross-dataset table is the evidence for a reusable allocation.',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    del args.integrity_tolerance
    expected = tuple(str(value).lower() for value in args.expected_datasets)
    records, missing, duplicates = load_records(
        args.inputs, expected, args.allow_incomplete)
    rows, class_rows, best_rows, fixed_rows = summarize(records)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'variants.csv'), rows)
    write_csv(os.path.join(args.out_dir, 'class_variants.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'best_mixed.csv'), best_rows)
    write_csv(os.path.join(args.out_dir, 'fixed_compositions.csv'), fixed_rows)
    write_report(
        os.path.join(args.out_dir, 'report.md'),
        best_rows, fixed_rows, missing, duplicates)
    print(os.path.join(args.out_dir, 'report.md'))


if __name__ == '__main__':
    main()
