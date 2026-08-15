#!/usr/bin/env python3
"""Summarize CoCo-style class calibration and factorized role evidence."""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    CLASS_ROLE_ALIGNMENT_PROTOCOL,
    CLASS_ROLE_ALIGNMENT_SCHEMA_VERSION,
    CRA_VARIANT_NAMES,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')
COUNTERS = (
    'changed_pixels', 'improved_pixels', 'harmed_pixels',
    'wrong_to_wrong_pixels', 'help_minus_harm',
)


def mean(values):
    values = [float(value) for value in values if value is not None
              and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def weighted_mean(pairs):
    pairs = [(float(value), int(weight)) for value, weight in pairs
             if value is not None and int(weight) > 0]
    weight = sum(item[1] for item in pairs)
    return (sum(value * count for value, count in pairs) / weight
            if weight else None)


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
                payload = record.get('class_role_alignment')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1))
                        != CLASS_ROLE_ALIGNMENT_SCHEMA_VERSION
                        or payload.get('protocol')
                        != CLASS_ROLE_ALIGNMENT_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != CRA_VARIANT_NAMES):
                    raise ValueError(
                        f'{path}:{line_number}: class-role contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['class_role_alignment'] != payload:
                        raise ValueError(
                            f'Inconsistent DDP duplicate for {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No class_role_alignment_v1 records found.')
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def variant_metadata(name):
    if name == 'cra_official':
        return dict(family='reference', base='official', weight=None,
                    evidence=None, temperature=None)
    if name == 'cra_best_full':
        return dict(family='reference', base='best', weight=None,
                    evidence=None, temperature=None)
    if '_coco_' in name:
        prefix, weight = name.rsplit('_', 1)
        base = 'official' if prefix.startswith('cra_official') else 'best'
        return dict(family='coco', base=base,
                    weight=float(weight[1:]) / 100.0,
                    evidence=None, temperature=None)
    prefix, evidence, temperature = name.rsplit('_', 2)
    if prefix != 'cra_best_fcra':
        raise ValueError(f'Unknown class-role variant: {name}.')
    return dict(family='fcra', base='best', weight=None,
                evidence=evidence,
                temperature=float(temperature[1:]) / 100.0)


def summarize_variants(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    rows, class_rows = [], []
    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in CRA_VARIANT_NAMES}
        counters = {name: defaultdict(int) for name in CRA_VARIANT_NAMES}
        margins = {name: [] for name in CRA_VARIANT_NAMES}
        selected = tuple(values[0]['class_role_alignment'][
            'selected_combination'])
        admission = values[0]['class_role_alignment']['selected_admission']
        for record in values:
            payload = record['class_role_alignment']
            if (tuple(payload['selected_combination']) != selected
                    or payload['selected_admission'] != admission):
                raise ValueError(f'{dataset}: role selection drifted.')
            valid_pixels = int(record['valid_pixels'])
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in COUNTERS:
                    counters[name][field] += int(stats.get(field, 0))
                margins[name].append((stats.get('mean_top1_margin'),
                                      valid_pixels))
        metrics = {name: confusion_metrics(matrix)
                   for name, matrix in matrices.items()}
        official = metrics['cra_official']
        best = metrics['cra_best_full']
        for name in CRA_VARIANT_NAMES:
            metadata = variant_metadata(name)
            reference_name = ('cra_official' if metadata['base'] == 'official'
                              else 'cra_best_full')
            if metadata['family'] == 'reference':
                reference_name = name
            metric = metrics[name]
            reference = metrics[reference_name]
            rows.append(dict(
                dataset=dataset, variant=name, images=len(values),
                selected_presence_slot=selected[0],
                selected_semantic_slot=selected[1],
                selected_instance_slot=selected[2],
                selected_admission=admission,
                reference_variant=reference_name,
                miou=metric['miou'], aacc=metric['aacc'],
                delta_to_reference=metric['miou'] - reference['miou'],
                delta_to_official=metric['miou'] - official['miou'],
                delta_to_best_full=metric['miou'] - best['miou'],
                mean_top1_margin=weighted_mean(margins[name]),
                **metadata, **counters[name],
            ))
            for class_index, class_name in enumerate(values[0]['class_names']):
                value = metric['iou'][class_index]
                ref_value = reference['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset, variant=name,
                    class_index=class_index, class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_reference=(
                        None if value is None or ref_value is None
                        else 100.0 * (value - ref_value)),
                    **metadata,
                ))
    return rows, class_rows


def collect_view_rows(records):
    class_rows, gate_rows, action_rows, view_rows = [], [], [], []
    for record in records:
        dataset = str(record['dataset_name']).lower()
        for view in record.get('views', []):
            payload = view.get('class_role_alignment_diagnosis')
            if payload is None:
                continue
            common = dict(
                dataset=dataset, img_path=record.get('img_path'),
                view_id=view.get('view_id'))
            view_rows.append(dict(
                **common,
                prior_entropy_mean=payload.get('prior_entropy_mean'),
                prior_margin_mean=payload.get('prior_margin_mean'),
                selection_max_abs=view.get(
                    'class_role_alignment_selection_max_abs')))
            class_rows.extend(dict(common, **row)
                              for row in payload.get('class_rows', []))
            gate_rows.extend(dict(common, **row)
                             for row in payload.get('gate_rows', []))
            action_rows.extend(dict(common, **row)
                               for row in payload.get('action_rows', []))
    return class_rows, gate_rows, action_rows, view_rows


def summarize_actions(action_rows):
    groups = defaultdict(list)
    for row in action_rows:
        groups[(row['dataset'], row['role'])].append(row)
    rows = []
    for (dataset, role), values in sorted(groups.items()):
        helpful = sum(int(row['helpful_pixels']) for row in values)
        harmful = sum(int(row['harmful_pixels']) for row in values)
        rows.append(dict(
            dataset=dataset, role=role,
            action_unit=values[0]['action_unit'],
            class_views=len(values),
            active=sum(int(row['active_pixels']) for row in values),
            helpful=helpful, harmful=harmful,
            d_helpful_mean=weighted_mean(
                (row.get('d_helpful_mean'), row['helpful_pixels'])
                for row in values),
            d_harmful_mean=weighted_mean(
                (row.get('d_harmful_mean'), row['harmful_pixels'])
                for row in values),
            gamma_helpful_mean=weighted_mean(
                (row.get('gamma_helpful_mean'), row['helpful_pixels'])
                for row in values),
            gamma_harmful_mean=weighted_mean(
                (row.get('gamma_harmful_mean'), row['harmful_pixels'])
                for row in values),
        ))
    return rows


def find(rows, dataset, variant):
    return next(row for row in rows
                if row['dataset'] == dataset and row['variant'] == variant)


def fmt(value, signed=False, digits=3):
    if value is None:
        return 'n/a'
    return f'{float(value):+.{digits}f}' if signed else f'{float(value):.{digits}f}'


def write_markdown(path, variants, actions, views, duplicates):
    datasets = sorted({row['dataset'] for row in variants})
    lines = [
        '# Class-role alignment v1', '',
        f'- DDP padding duplicates removed: {duplicates or "none"}',
        '- CoCo rows add only a block18 class prior; no synonym expansion is used.',
        '- FCRA rows gate the registered best-full P/S/I residuals with D or '
        'class-centered Gamma evidence; hard query admission remains fixed.',
        '- Primary metric: final mIoU. Action separation and margins are '
        'mechanism diagnostics, not selection results.', '',
        '## CoCo-style class calibration', '',
        '| Dataset | Official | Best-full | Official + .3/.5/.7/.9 | '
        'Best + .3/.5/.7/.9 |',
        '|---|---:|---:|---:|---:|',
    ]
    for dataset in datasets:
        official = find(variants, dataset, 'cra_official')
        best = find(variants, dataset, 'cra_best_full')
        official_values = '/'.join(fmt(find(
            variants, dataset, f'cra_official_coco_l{value:03d}'
        )['delta_to_reference'], signed=True) for value in (30, 50, 70, 90))
        best_values = '/'.join(fmt(find(
            variants, dataset, f'cra_best_coco_l{value:03d}'
        )['delta_to_reference'], signed=True) for value in (30, 50, 70, 90))
        lines.append(
            f"| {dataset} | {official['miou']:.3f} | {best['miou']:.3f} | "
            f'{official_values} | {best_values} |')

    lines.extend([
        '', '## Factorized class-role alignment', '',
        '| Dataset | Best-full | D t=.05/.10 | Gamma t=.05/.10 |',
        '|---|---:|---:|---:|',
    ])
    for dataset in datasets:
        best = find(variants, dataset, 'cra_best_full')
        d_values = '/'.join(fmt(find(
            variants, dataset, f'cra_best_fcra_d_t{value:03d}'
        )['delta_to_reference'], signed=True) for value in (5, 10))
        gamma_values = '/'.join(fmt(find(
            variants, dataset, f'cra_best_fcra_gamma_t{value:03d}'
        )['delta_to_reference'], signed=True) for value in (5, 10))
        lines.append(
            f"| {dataset} | {best['miou']:.3f} | {d_values} | "
            f'{gamma_values} |')

    lines.extend([
        '', '## Helpful-vs-harmful role evidence', '',
        '| Dataset/role | unit | helpful/harmful | D help/harm | '
        'Gamma help/harm |',
        '|---|---|---:|---:|---:|',
    ])
    for row in actions:
        lines.append(
            f"| {row['dataset']}/{row['role']} | {row['action_unit']} | "
            f"{row['helpful']}/{row['harmful']} | "
            f"{fmt(row['d_helpful_mean'], digits=5)}/"
            f"{fmt(row['d_harmful_mean'], digits=5)} | "
            f"{fmt(row['gamma_helpful_mean'], digits=5)}/"
            f"{fmt(row['gamma_harmful_mean'], digits=5)} |")

    lines.extend(['', '## Integrity', '',
                  '| Dataset | max best-full selection error |',
                  '|---|---:|'])
    for dataset in datasets:
        value = max(float(row['selection_max_abs'] or 0.0)
                    for row in views if row['dataset'] == dataset)
        lines.append(f'| {dataset} | {value:.8g} |')
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
    expected = tuple(str(value).lower() for value in args.expected_datasets)
    if 'isaid' in expected:
        raise ValueError('iSAID is excluded from project evaluation.')
    records, missing, duplicates = load_records(
        args.inputs, expected, args.allow_incomplete)
    variants, class_variants = summarize_variants(records)
    class_views, gates, actions, views = collect_view_rows(records)
    action_summary = summarize_actions(actions)
    if max(float(row['selection_max_abs'] or 0.0) for row in views) > (
            args.integrity_tolerance):
        raise ValueError('Class-role best-full reconstruction exceeded tolerance.')
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'variants.csv'), variants)
    write_csv(os.path.join(args.out_dir, 'class_variants.csv'), class_variants)
    write_csv(os.path.join(args.out_dir, 'alignment_class_views.csv'), class_views)
    write_csv(os.path.join(args.out_dir, 'gate_views.csv'), gates)
    write_csv(os.path.join(args.out_dir, 'role_action_views.csv'), actions)
    write_csv(os.path.join(args.out_dir, 'role_action_summary.csv'), action_summary)
    write_csv(os.path.join(args.out_dir, 'view_summary.csv'), views)
    payload = dict(
        protocol=CLASS_ROLE_ALIGNMENT_PROTOCOL,
        records=len(records), missing_datasets=missing,
        duplicate_counts=duplicates,
        variants=variants, class_variants=class_variants,
        role_action_summary=action_summary,
        views=views,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(
        os.path.join(args.out_dir, 'summary.md'), variants,
        action_summary, views, duplicates)


if __name__ == '__main__':
    main()
