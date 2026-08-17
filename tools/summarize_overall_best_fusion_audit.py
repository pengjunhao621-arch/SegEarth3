#!/usr/bin/env python3
"""Summarize matched fusion operators around selected Role-Text states."""

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    FUSION_AUDIT_PROTOCOL,
    FUSION_AUDIT_SCHEMA_VERSION,
    FUSION_AUDIT_VARIANT_NAMES,
    FUSION_AUDIT_VARIANT_SPECS,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')
TRANSITION_FIELDS = (
    'changed_pixels', 'improved_pixels', 'harmed_pixels',
    'wrong_to_wrong_pixels', 'help_minus_harm',
    'background_to_foreground_pixels',
    'foreground_to_background_pixels',
    'foreground_to_foreground_pixels',
)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def mean(values):
    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def expand_inputs(patterns):
    return sorted(set(
        path for pattern in patterns
        for path in (glob.glob(pattern) or [pattern])
        if os.path.isfile(path)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def load_records(paths, expected, allow_incomplete):
    expected = tuple(str(value).lower() for value in expected)
    if 'isaid' in expected:
        raise ValueError('iSAID is excluded from project evaluation.')
    records, seen, duplicates = [], {}, defaultdict(int)
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                dataset = str(record.get('dataset_name', '')).lower()
                payload = record.get('overall_best_fusion_audit')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1))
                        != FUSION_AUDIT_SCHEMA_VERSION
                        or payload.get('protocol') != FUSION_AUDIT_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != FUSION_AUDIT_VARIANT_NAMES):
                    raise ValueError(
                        f'{path}:{line_number}: fusion-audit contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['overall_best_fusion_audit'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No overall_best_fusion_audit_v1 records found.')
    observed = {str(record['dataset_name']).lower() for record in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def summarize(records, tolerance):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    family = dict(FUSION_AUDIT_VARIANT_SPECS)
    variant_rows, class_rows = [], []
    mechanism_rows, mechanism_class_rows, margin_rows = [], [], []

    for dataset, values in sorted(grouped.items()):
        class_names = values[0]['class_names']
        matrices = {name: None for name in FUSION_AUDIT_VARIANT_NAMES}
        counters = {
            name: defaultdict(int) for name in FUSION_AUDIT_VARIANT_NAMES}
        selected = tuple(values[0]['overall_best_fusion_audit'][
            'selected_combination'])
        admission = values[0]['overall_best_fusion_audit'][
            'selected_admission']
        component_sums = defaultdict(lambda: defaultdict(int))
        global_sums = defaultdict(int)
        bin_sums = defaultdict(lambda: defaultdict(int))
        class_sums = defaultdict(lambda: defaultdict(int))
        selection_errors = []

        for record in values:
            payload = record['overall_best_fusion_audit']
            if (tuple(payload['selected_combination']) != selected
                    or payload['selected_admission'] != admission):
                raise ValueError(f'{dataset}: selected Role-Text drifted.')
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in TRANSITION_FIELDS:
                    counters[name][field] += int(stats.get(field, 0))
            mechanism = payload['mechanism']
            for field in (
                    'class_entries', 'selected_instance_winner_entries',
                    'anchor_instance_winner_entries', 's_to_i_entries',
                    'i_to_s_entries'):
                global_sums[field] += int(mechanism.get(field, 0))
            for role, row in mechanism['components'].items():
                for field, value in row.items():
                    component_sums[role][field] += int(value)
            for row in mechanism['margin_bins']:
                for field, value in row.items():
                    if field != 'bin':
                        bin_sums[row['bin']][field] += int(value)
            for row in mechanism['class_rows']:
                key = (int(row['class_index']), row['class_name'])
                for field, value in row.items():
                    if field not in ('class_index', 'class_name'):
                        class_sums[key][field] += int(value)
            selection_errors.extend(
                float(view.get('fusion_audit_selection_max_abs', 0.0))
                for view in record.get('views', []))

        max_error = max(selection_errors or [0.0])
        if max_error > tolerance:
            raise ValueError(
                f'{dataset}: native fusion parity {max_error} > {tolerance}.')
        metrics = {
            name: confusion_metrics(matrix) for name, matrix in matrices.items()
        }
        official = metrics['ofa_official']
        native = metrics['ofa_native']
        valid_pixels = int(sum(official['gt']))
        for name in FUSION_AUDIT_VARIANT_NAMES:
            metric = metrics[name]
            reference_name = (
                'ofa_official' if name in ('ofa_official', 'ofa_native')
                else 'ofa_native')
            reference = metrics[reference_name]
            variant_rows.append(dict(
                dataset=dataset,
                variant=name,
                family=family[name],
                images=len(values),
                selected_presence_slot=selected[0],
                selected_semantic_slot=selected[1],
                selected_instance_slot=selected[2],
                selected_admission=admission,
                reference_variant=reference_name,
                miou=metric['miou'],
                aacc=metric['aacc'],
                delta_to_reference=metric['miou'] - reference['miou'],
                delta_to_official=metric['miou'] - official['miou'],
                delta_to_native=metric['miou'] - native['miou'],
                changed_ratio=safe_div(
                    counters[name]['changed_pixels'], valid_pixels),
                fusion_selection_max_abs=max_error,
                **counters[name],
            ))
            for class_index, class_name in enumerate(class_names):
                value = metric['iou'][class_index]
                native_value = native['iou'][class_index]
                official_value = official['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset, variant=name, family=family[name],
                    class_index=class_index, class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_native=(
                        None if value is None or native_value is None
                        else 100.0 * (value - native_value)),
                    delta_iou_to_official=(
                        None if value is None or official_value is None
                        else 100.0 * (value - official_value)),
                ))

        for role, row in sorted(component_sums.items()):
            mechanism_rows.append(dict(
                dataset=dataset, scope='component', role=role,
                **row,
                helpful_losing_ratio=safe_div(
                    row.get('helpful_losing_entries', 0),
                    row.get('helpful_entries', 0)),
                harmful_winning_ratio=safe_div(
                    row.get('harmful_winning_entries', 0),
                    row.get('harmful_entries', 0)),
            ))
        mechanism_rows.append(dict(
            dataset=dataset, scope='head_competition', role='all',
            **global_sums,
            selected_instance_winner_ratio=safe_div(
                global_sums['selected_instance_winner_entries'],
                global_sums['class_entries']),
            anchor_instance_winner_ratio=safe_div(
                global_sums['anchor_instance_winner_entries'],
                global_sums['class_entries']),
        ))
        for label, row in bin_sums.items():
            margin_rows.append(dict(
                dataset=dataset, margin_bin=label, **row,
                accuracy=safe_div(row['correct_pixels'], row['pixels']),
                net_correction=(row['corrected_vs_official']
                                - row['harmed_vs_official']),
            ))
        for (class_index, class_name), row in sorted(class_sums.items()):
            mechanism_class_rows.append(dict(
                dataset=dataset, class_index=class_index,
                class_name=class_name, **row))

    return (variant_rows, class_rows, mechanism_rows,
            mechanism_class_rows, margin_rows)


def fixed_operator_rows(variant_rows):
    rows = []
    for name, family in FUSION_AUDIT_VARIANT_SPECS:
        if family != 'candidate':
            continue
        values = [row for row in variant_rows if row['variant'] == name]
        deltas = [float(row['delta_to_native']) for row in values]
        rows.append(dict(
            variant=name,
            datasets=len(values),
            positive_datasets=sum(value > 0.0 for value in deltas),
            negative_datasets=sum(value < 0.0 for value in deltas),
            macro_delta=mean(deltas),
            median_delta=statistics.median(deltas) if deltas else None,
            min_delta=min(deltas) if deltas else None,
            max_delta=max(deltas) if deltas else None,
        ))
    return sorted(
        rows, key=lambda row: (row['positive_datasets'], row['macro_delta']),
        reverse=True)


def fmt(value, signed=False):
    if value is None:
        return 'n/a'
    return f'{float(value):+.3f}' if signed else f'{float(value):.3f}'


def write_markdown(path, variants, fixed, mechanisms, duplicates, missing):
    datasets = sorted({row['dataset'] for row in variants})
    lookup = {(row['dataset'], row['variant']): row for row in variants}
    lines = [
        '# Overall-best fusion audit v1', '',
        '- Every candidate is operator-matched: `official + G(selected) - '
        'G(anchor)`. A zero Role-Text update therefore remains the exact '
        'official baseline.',
        '- `ofa_native` is the registered overall-best Role-Text path: outer '
        'Presence times element-wise max of Semantic and Instance.',
        '- Oracle/per-dataset best rows diagnose existence only. A method '
        'candidate must use one fixed operator across all datasets.',
        f'- Removed identical DDP padding duplicates: {duplicates or "none"}.',
        f'- Missing datasets: {missing or "none"}.', '',
        '## Registered Role-Text path', '',
        '| Dataset | Official | Native Role-Text | Gain | Selection |',
        '|---|---:|---:|---:|---|',
    ]
    for dataset in datasets:
        official = lookup[(dataset, 'ofa_official')]
        native = lookup[(dataset, 'ofa_native')]
        selection = (
            f"P{native['selected_presence_slot']}+"
            f"S{native['selected_semantic_slot']}+"
            f"I{native['selected_instance_slot']} / "
            f"{native['selected_admission']}")
        lines.append(
            f"| {dataset} | {official['miou']:.3f} | "
            f"{native['miou']:.3f} | {native['delta_to_official']:+.3f} | "
            f"{selection} |")

    lines.extend([
        '', '## Fixed fusion operators', '',
        '| Operator | Positive | Macro delta | Median | Min | Max |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in fixed:
        lines.append(
            f"| {row['variant']} | {row['positive_datasets']}/"
            f"{row['datasets']} | {fmt(row['macro_delta'], True)} | "
            f"{fmt(row['median_delta'], True)} | "
            f"{fmt(row['min_delta'], True)} | "
            f"{fmt(row['max_delta'], True)} |")

    lines.extend([
        '', '## Per-dataset best candidate (diagnostic only)', '',
        '| Dataset | Native | Best candidate | mIoU | Delta | Help-Harm |',
        '|---|---:|---|---:|---:|---:|',
    ])
    for dataset in datasets:
        candidates = [row for row in variants
                      if row['dataset'] == dataset
                      and row['family'] == 'candidate']
        best = max(candidates, key=lambda row: row['miou'])
        native = lookup[(dataset, 'ofa_native')]
        lines.append(
            f"| {dataset} | {native['miou']:.3f} | {best['variant']} | "
            f"{best['miou']:.3f} | {best['delta_to_native']:+.3f} | "
            f"{best['help_minus_harm']} |")

    lines.extend([
        '', '## Does max discard directionally useful role residuals?', '',
        '| Dataset | Semantic helpful-losing | Instance helpful-losing | '
        'Semantic harmful-winning | Instance harmful-winning |',
        '|---|---:|---:|---:|---:|',
    ])
    mechanism_lookup = {
        (row['dataset'], row.get('role')): row
        for row in mechanisms if row['scope'] == 'component'}
    for dataset in datasets:
        semantic = mechanism_lookup[(dataset, 'semantic')]
        instance = mechanism_lookup[(dataset, 'instance')]
        lines.append(
            f"| {dataset} | {semantic.get('helpful_losing_entries', 0)} "
            f"({fmt(semantic.get('helpful_losing_ratio'))}) | "
            f"{instance.get('helpful_losing_entries', 0)} "
            f"({fmt(instance.get('helpful_losing_ratio'))}) | "
            f"{semantic.get('harmful_winning_entries', 0)} "
            f"({fmt(semantic.get('harmful_winning_ratio'))}) | "
            f"{instance.get('harmful_winning_entries', 0)} "
            f"({fmt(instance.get('harmful_winning_ratio'))}) |")

    supported = [row for row in fixed
                 if row['positive_datasets'] >= 4
                 and row['macro_delta'] is not None
                 and row['macro_delta'] > 0.0]
    lines.extend(['', '## Decision rule', ''])
    if supported:
        best = supported[0]
        lines.append(
            f"- H3 receives initial support: `{best['variant']}` is positive "
            f"on {best['positive_datasets']}/{best['datasets']} datasets with "
            f"macro {best['macro_delta']:+.3f} mIoU. Inspect class and "
            'transition tables before promoting it to a method.')
    else:
        lines.append(
            '- No fixed operator currently satisfies the 4/6-positive and '
            'positive-macro criterion. This favors H4 at the output-fusion '
            'level; it does not prove that internal attention/readout is '
            'sufficient, because H1/H2 were not activation-patched here.')
    lines.append(
        '- This experiment distinguishes H3 from output-level H4. It cannot '
        'by itself prove H2 (an internal state contains useful evidence that '
        'native downstream computation loses).')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    records, missing, duplicates = load_records(
        paths, args.expected_datasets, args.allow_incomplete)
    (variants, classes, mechanisms,
     mechanism_classes, margins) = summarize(
        records, args.integrity_tolerance)
    fixed = fixed_operator_rows(variants)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'fusion_variants.csv'), variants)
    write_csv(os.path.join(args.out_dir, 'fusion_classes.csv'), classes)
    write_csv(os.path.join(args.out_dir, 'fusion_fixed_operators.csv'), fixed)
    write_csv(os.path.join(args.out_dir, 'fusion_mechanisms.csv'), mechanisms)
    write_csv(os.path.join(
        args.out_dir, 'fusion_mechanism_classes.csv'), mechanism_classes)
    write_csv(os.path.join(args.out_dir, 'fusion_margin_bins.csv'), margins)
    write_markdown(
        os.path.join(args.out_dir, 'fusion_audit.md'),
        variants, fixed, mechanisms, duplicates, missing)
    print(json.dumps(dict(
        records=len(records), datasets=sorted({
            str(row['dataset_name']).lower() for row in records}),
        missing=missing, duplicates=duplicates,
        best_fixed=fixed[0] if fixed else None,
        out_dir=os.path.abspath(args.out_dir),
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
