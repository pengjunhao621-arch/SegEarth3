#!/usr/bin/env python3
"""Summarize the Joint Role--View Profile screen."""

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
    JOINT_ROLE_VIEW_OPERATOR_NAMES,
    JOINT_ROLE_VIEW_PROTOCOL,
    JOINT_ROLE_VIEW_SCHEMA_VERSION,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    return parser.parse_args()


def _expected_variants(payload):
    candidates = payload['role_candidates']
    anchor = payload['anchor_candidate']
    operators = tuple(
        str(value['name']) for value in payload['operator_specs'])
    if (not operators or len(operators) != len(set(operators))
            or any(value not in JOINT_ROLE_VIEW_OPERATOR_NAMES
                   for value in operators)):
        raise ValueError('Invalid Joint Role--View operator subset.')
    names = ['jrv_official']
    names.extend(
        f'jrv_anchor__{operator}'
        for operator in operators)
    for candidate in candidates:
        if candidate['id'] == anchor:
            continue
        names.extend(
            f'jrv_{candidate["id"]}__{operator}'
            for operator in operators)
    return tuple(names)


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
                payload = record.get('joint_role_view')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1))
                        != JOINT_ROLE_VIEW_SCHEMA_VERSION
                        or payload.get('protocol') != JOINT_ROLE_VIEW_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != _expected_variants(payload)):
                    raise ValueError(
                        f'{path}:{line_number}: Joint Role--View contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['joint_role_view'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No joint_role_view_profile_v1 records found.')
    observed = {str(value['dataset_name']).lower() for value in records}
    missing = [value for value in expected if value not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def _variant_name(candidate, operator, anchor):
    prefix = 'anchor' if candidate == anchor else candidate
    return f'jrv_{prefix}__{operator}'


def _profile_row(dataset, label, name, metrics, rows, official, role_reference):
    source = rows[name]
    metric = metrics[name]
    return dict(
        dataset=dataset,
        profile=label,
        variant=name,
        candidate_id=source['candidate_id'],
        slots='+'.join(str(value) for value in source['slots']),
        admission=source['admission'],
        operator=source['operator'],
        operator_family=source['operator_family'],
        global_weight=source['global_weight'],
        rho=source['rho'],
        miou=metric['miou'],
        aacc=metric['aacc'],
        delta_to_official=metric['miou'] - official['miou'],
        delta_to_role_only=metric['miou'] - role_reference['miou'],
    )


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)

    all_rows, profile_rows, class_rows = [], [], []
    interaction_rows, complement_rows, reproduction_rows = [], [], []
    dataset_payloads = {}
    for dataset, values in sorted(grouped.items()):
        first = values[0]['joint_role_view']
        variant_names = tuple(first['variants'])
        matrices = {name: None for name in variant_names}
        counters = {name: defaultdict(int) for name in variant_names}
        oracle_matrices = {
            name: None for name in first['complementarity']}
        complement_counts = {
            name: defaultdict(int) for name in first['complementarity']}
        contract = dict(
            global_source=first['global_source'],
            reference_endpoint=first['reference_endpoint'],
            role_candidates=first['role_candidates'],
            anchor_candidate=first['anchor_candidate'],
            current_role_candidate=first['current_role_candidate'],
            prior_view_operator=first['prior_view_operator'],
            prior_view_miou=first['prior_view_miou'],
            operator_specs=first['operator_specs'],
        )
        operator_names = tuple(
            str(value['name']) for value in contract['operator_specs'])
        for record in values:
            payload = record['joint_role_view']
            for key, expected in contract.items():
                if payload[key] != expected:
                    raise ValueError(f'{dataset}: metadata drifted at {key}.')
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in ('changed_pixels', 'improved_pixels',
                              'harmed_pixels', 'help_minus_harm'):
                    counters[name][field] += int(stats[field])
            for candidate, stats in payload['complementarity'].items():
                oracle_matrices[candidate] = add_matrix(
                    oracle_matrices[candidate],
                    stats['oracle_confusion']['matrix'])
                for field in ('local_only_correct_pixels',
                              'global_only_correct_pixels',
                              'disagreement_pixels'):
                    complement_counts[candidate][field] += int(stats[field])

        metrics = {
            name: confusion_metrics(matrix) for name, matrix in matrices.items()
        }
        row_metadata = first['variants']
        official_name = 'jrv_official'
        official = metrics[official_name]
        anchor = contract['anchor_candidate']
        current = contract['current_role_candidate']
        endpoint = contract['reference_endpoint']
        role_only_name = _variant_name(current, endpoint, anchor)
        role_only = metrics[role_only_name]
        sequential_name = _variant_name(
            current, contract['prior_view_operator'], anchor)
        candidates = {
            candidate['id']: candidate
            for candidate in contract['role_candidates']}
        expected_role_miou = candidates[current]['source_miou']
        measured_sequential_miou = metrics[sequential_name]['miou']
        expected_sequential_miou = contract['prior_view_miou']
        reproduction_rows.append(dict(
            dataset=dataset,
            current_role_candidate=current,
            reference_endpoint=endpoint,
            expected_role_only_miou=expected_role_miou,
            measured_role_only_miou=role_only['miou'],
            role_only_reproduction_delta=(
                None if expected_role_miou is None
                else role_only['miou'] - expected_role_miou),
            prior_view_operator=contract['prior_view_operator'],
            expected_sequential_miou=expected_sequential_miou,
            measured_sequential_miou=measured_sequential_miou,
            sequential_reproduction_delta=(
                None if expected_sequential_miou is None
                else measured_sequential_miou - expected_sequential_miou),
            reference_identity_max_abs=max(
                float(value['joint_role_view'][
                    'reference_identity_max_abs'])
                for value in values),
            native_prompt_parity_max_abs=max(
                float(value['joint_role_view'][
                    'native_prompt_parity_max_abs'])
                for value in values),
        ))

        anchor_names = [
            _variant_name(anchor, operator, anchor)
            for operator in operator_names]
        current_names = [
            _variant_name(current, operator, anchor)
            for operator in operator_names]
        role_names = [
            _variant_name(candidate['id'], endpoint, anchor)
            for candidate in contract['role_candidates']
            if candidate['id'] != anchor]
        role_best_name = max(role_names, key=lambda name: metrics[name]['miou'])
        role_reference = metrics[role_best_name]
        joint_names = [
            name for name in variant_names
            if name != official_name
            and row_metadata[name]['candidate_id'] != anchor]
        view_only_name = max(anchor_names, key=lambda name: metrics[name]['miou'])
        current_best_view_name = max(
            current_names, key=lambda name: metrics[name]['miou'])
        joint_name = max(joint_names, key=lambda name: metrics[name]['miou'])
        selected = (
            ('official', official_name),
            ('role_only', role_only_name),
            ('role_only_best', role_best_name),
            ('view_only_best', view_only_name),
            ('sequential_role_view', sequential_name),
            ('current_role_best_view', current_best_view_name),
            ('joint_role_view_best', joint_name),
        )
        for label, name in selected:
            profile_rows.append(_profile_row(
                dataset, label, name, metrics, row_metadata,
                official, role_reference))
            for class_index, class_name in enumerate(
                    values[0]['class_names']):
                class_rows.append(dict(
                    dataset=dataset,
                    profile=label,
                    variant=name,
                    class_index=class_index,
                    class_name=class_name,
                    iou=metrics[name]['iou'][class_index],
                    gt_pixels=metrics[name]['gt'][class_index],
                    predicted_pixels=metrics[name]['pred'][class_index],
                ))

        for name in variant_names:
            source = row_metadata[name]
            metric = metrics[name]
            all_rows.append(dict(
                dataset=dataset,
                variant=name,
                candidate_id=source['candidate_id'],
                slots='+'.join(str(value) for value in source['slots']),
                admission=source['admission'],
                operator=source['operator'],
                operator_family=source['operator_family'],
                global_weight=source['global_weight'],
                rho=source['rho'],
                miou=metric['miou'],
                aacc=metric['aacc'],
                delta_to_official=metric['miou'] - official['miou'],
                delta_to_role_only=metric['miou'] - role_reference['miou'],
                images=len(values),
                changed_pixels=counters[name]['changed_pixels'],
                improved_pixels=counters[name]['improved_pixels'],
                harmed_pixels=counters[name]['harmed_pixels'],
                help_minus_harm=counters[name]['help_minus_harm'],
            ))

        anchor_operator_metrics = {
            operator: metrics[_variant_name(anchor, operator, anchor)]
            for operator in operator_names}
        for candidate in contract['role_candidates']:
            identifier = candidate['id']
            if identifier == anchor:
                continue
            candidate_role = metrics[_variant_name(
                identifier, endpoint, anchor)]['miou']
            for operator in operator_names:
                name = _variant_name(identifier, operator, anchor)
                interaction_rows.append(dict(
                    dataset=dataset,
                    candidate_id=identifier,
                    slots='+'.join(str(value) for value in candidate['slots']),
                    admission=candidate['admission'],
                    operator=operator,
                    miou=metrics[name]['miou'],
                    role_main_effect=candidate_role - official['miou'],
                    view_main_effect=(
                        anchor_operator_metrics[operator]['miou']
                        - official['miou']),
                    joint_delta=metrics[name]['miou'] - official['miou'],
                    role_view_interaction=(
                        metrics[name]['miou'] - candidate_role
                        - anchor_operator_metrics[operator]['miou']
                        + official['miou']),
                ))

        for candidate, matrix in oracle_matrices.items():
            if 'local' not in operator_names or 'global' not in operator_names:
                continue
            local_name = _variant_name(candidate, 'local', anchor)
            global_name = _variant_name(candidate, 'global', anchor)
            local_metric = metrics[local_name]
            global_metric = metrics[global_name]
            oracle = confusion_metrics(matrix)
            complement_rows.append(dict(
                dataset=dataset,
                candidate_id=candidate,
                local_miou=local_metric['miou'],
                global_miou=global_metric['miou'],
                oracle_miou=oracle['miou'],
                oracle_headroom=(
                    oracle['miou']
                    - max(local_metric['miou'], global_metric['miou'])),
                local_only_correct_pixels=(
                    complement_counts[candidate]['local_only_correct_pixels']),
                global_only_correct_pixels=(
                    complement_counts[candidate]['global_only_correct_pixels']),
                disagreement_pixels=(
                    complement_counts[candidate]['disagreement_pixels']),
            ))
        dataset_payloads[dataset] = dict(
            metrics=metrics,
            rows=row_metadata,
            anchor=anchor,
            current=current,
            endpoint=endpoint,
            official=official_name,
            role_only=role_only_name,
            role_best=role_best_name,
            sequential=sequential_name,
            current_best_view=current_best_view_name,
            joint=joint_name,
            operators=operator_names,
        )

    operator_rows = []
    datasets = sorted(dataset_payloads)
    available_operators = sorted(set(
        operator for payload in dataset_payloads.values()
        for operator in payload['operators']))
    for operator in available_operators:
        current_values, best_role_values = [], []
        positive_current = positive_best = 0
        selected_candidates = []
        for dataset in datasets:
            payload = dataset_payloads[dataset]
            if operator not in payload['operators']:
                continue
            metrics, rows = payload['metrics'], payload['rows']
            current_name = _variant_name(
                payload['current'], operator, payload['anchor'])
            current_value = metrics[current_name]['miou']
            role_reference = metrics[payload['role_best']]['miou']
            current_values.append(current_value)
            positive_current += int(current_value > role_reference)
            choices = [
                name for name in metrics
                if name != 'jrv_official'
                and rows[name]['candidate_id'] != payload['anchor']
                and rows[name]['operator'] == operator]
            best_name = max(choices, key=lambda name: metrics[name]['miou'])
            best_value = metrics[best_name]['miou']
            best_role_values.append(best_value)
            positive_best += int(best_value > role_reference)
            selected_candidates.append(
                f'{dataset}:{rows[best_name]["candidate_id"]}')
        operator_rows.append(dict(
            operator=operator,
            datasets=len(current_values),
            current_role_macro=sum(current_values) / len(current_values),
            current_role_positive_vs_role_only=positive_current,
            best_role_macro=sum(best_role_values) / len(best_role_values),
            best_role_positive_vs_role_only=positive_best,
            selected_candidates=';'.join(selected_candidates),
        ))
    macro_rows = []
    grouped_profiles = defaultdict(list)
    for row in profile_rows:
        grouped_profiles[row['profile']].append(row)
    for profile, values in grouped_profiles.items():
        macro_rows.append(dict(
            profile=profile,
            datasets=len(values),
            macro_miou=sum(value['miou'] for value in values) / len(values),
            macro_delta_to_official=(
                sum(value['delta_to_official'] for value in values)
                / len(values)),
            macro_delta_to_role_only=(
                sum(value['delta_to_role_only'] for value in values)
                / len(values)),
            positive_vs_official=sum(
                value['delta_to_official'] > 0 for value in values),
            positive_vs_role_only=sum(
                value['delta_to_role_only'] > 0 for value in values),
        ))
    return dict(
        variants=all_rows,
        profiles=profile_rows,
        profile_macros=macro_rows,
        classes=class_rows,
        interactions=interaction_rows,
        complementarity=complement_rows,
        operators=operator_rows,
        reproduction=reproduction_rows,
    )


def _fmt(value):
    return 'n/a' if value is None else f'{float(value):.3f}'


def write_report(path, result, missing, duplicates):
    profiles = defaultdict(dict)
    for row in result['profiles']:
        profiles[row['dataset']][row['profile']] = row
    lines = [
        '# Joint Role--View Profile v1', '',
        '## Core profile comparison', '',
        '| Dataset | Official | Registered Role | Best Role | View only | Sequential | '
        'Current Role + best View | Joint best | Joint profile |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for dataset in sorted(profiles):
        rows = profiles[dataset]
        joint = rows['joint_role_view_best']
        profile = (
            f"P/S/I={joint['slots']}; {joint['admission']}; "
            f"{joint['operator']}")
        lines.append(
            f"| {dataset} | {_fmt(rows['official']['miou'])} | "
            f"{_fmt(rows['role_only']['miou'])} | "
            f"{_fmt(rows['role_only_best']['miou'])} | "
            f"{_fmt(rows['view_only_best']['miou'])} | "
            f"{_fmt(rows['sequential_role_view']['miou'])} | "
            f"{_fmt(rows['current_role_best_view']['miou'])} | "
            f"{_fmt(joint['miou'])} | {profile} |")
    lines.extend(['', '## Reproduction audit', '',
                  '| Dataset | Role expected | Role measured | Delta | '
                  'Sequential expected | Sequential measured | Delta | '
                  'Identity max abs | Prompt parity max abs |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for row in result['reproduction']:
        lines.append(
            f"| {row['dataset']} | "
            f"{_fmt(row['expected_role_only_miou'])} | "
            f"{_fmt(row['measured_role_only_miou'])} | "
            f"{_fmt(row['role_only_reproduction_delta'])} | "
            f"{_fmt(row['expected_sequential_miou'])} | "
            f"{_fmt(row['measured_sequential_miou'])} | "
            f"{_fmt(row['sequential_reproduction_delta'])} | "
            f"{row['reference_identity_max_abs']:.3e} | "
            f"{row['native_prompt_parity_max_abs']:.3e} |")
    lines.extend(['', '## Common View operator comparison', '',
                  '| Operator | Current Role macro | Positive | '
                  'Best Role macro | Positive |',
                  '|---|---:|---:|---:|---:|'])
    for row in sorted(
            result['operators'], key=lambda value: value['best_role_macro'],
            reverse=True):
        lines.append(
            f"| {row['operator']} | {_fmt(row['current_role_macro'])} | "
            f"{row['current_role_positive_vs_role_only']}/"
            f"{row['datasets']} | {_fmt(row['best_role_macro'])} | "
            f"{row['best_role_positive_vs_role_only']}/"
            f"{row['datasets']} |")
    lines.extend(['', '## Macro profile ablation', '',
                  '| Profile | Macro mIoU | Delta to official | '
                  'Delta to Role only | Positive vs Role only |',
                  '|---|---:|---:|---:|---:|'])
    for row in sorted(
            result['profile_macros'],
            key=lambda value: value['macro_miou'], reverse=True):
        lines.append(
            f"| {row['profile']} | {_fmt(row['macro_miou'])} | "
            f"{_fmt(row['macro_delta_to_official'])} | "
            f"{_fmt(row['macro_delta_to_role_only'])} | "
            f"{row['positive_vs_role_only']}/{row['datasets']} |")
    lines.extend([
        '', f'Missing datasets: {missing or "none"}.',
        f'DDP padding duplicates removed: {duplicates or "none"}.',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    expected = tuple(str(value).lower() for value in args.expected_datasets)
    records, missing, duplicates = load_records(
        args.inputs, expected, args.allow_incomplete)
    result = summarize(records)
    os.makedirs(args.out_dir, exist_ok=True)
    for name in ('variants', 'profiles', 'profile_macros', 'classes',
                 'interactions', 'complementarity', 'operators',
                 'reproduction'):
        write_csv(
            os.path.join(args.out_dir, f'joint_role_view_{name}.csv'),
            result[name])
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(dict(
            missing_datasets=missing,
            duplicate_counts=duplicates,
            **result), handle, ensure_ascii=False, indent=2)
    write_report(
        os.path.join(args.out_dir, 'joint_role_view.md'),
        result, missing, duplicates)


if __name__ == '__main__':
    main()
