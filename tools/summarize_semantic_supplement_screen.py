#!/usr/bin/env python3
"""Summarize the six-dataset two-prompt semantic supplement screen."""

import argparse
import csv
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

from semantic_supplement_definitions import (
    CAUSAL_REFERENCE,
    FAMILIES,
    FIXED_FAMILY_VARIANTS,
    MAX_CANDIDATES,
    PROTOCOL,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    aggregate_variant_name,
    candidate_variant_name,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def median(values):
    values = [float(value) for value in values if finite(value)]
    return statistics.median(values) if values else None


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def add_matrix(target, source):
    source = [[int(value) for value in row] for row in source]
    if target is None:
        return source
    return [
        [left + right for left, right in zip(left_row, right_row)]
        for left_row, right_row in zip(target, source)
    ]


def confusion_metrics(matrix):
    count = len(matrix)
    tp = [float(matrix[idx][idx]) for idx in range(count)]
    gt = [float(sum(row)) for row in matrix]
    pred = [float(sum(matrix[row][col] for row in range(count)))
            for col in range(count)]
    union = [gt[idx] + pred[idx] - tp[idx] for idx in range(count)]
    iou = [safe_div(tp[idx], union[idx]) for idx in range(count)]
    valid_iou = [value for value in iou if finite(value)]
    return dict(
        iou=iou,
        miou=mean(valid_iou) * 100.0,
        aacc=safe_div(sum(tp), sum(gt)) * 100.0,
        gt=gt,
        pred=pred,
    )


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def load_records(paths, expected_datasets, allow_incomplete):
    expected = tuple(str(name).lower() for name in expected_datasets)
    if any(name == 'isaid' for name in expected):
        raise ValueError('iSAID is excluded from project evaluation.')
    records = []
    seen = {}
    duplicate_counts = defaultdict(int)
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get('schema_version', -1)) != SCHEMA_VERSION:
                    raise ValueError(f'Unexpected schema at {path}:{line_number}.')
                if record.get('settings', {}).get('protocol') != PROTOCOL:
                    raise ValueError(f'Wrong protocol at {path}:{line_number}.')
                dataset = str(record.get('dataset_name', '')).lower()
                if dataset not in expected:
                    raise ValueError(f'Unexpected dataset {dataset!r}.')
                if tuple(record.get('variants', {})) != VARIANT_NAMES:
                    raise ValueError(
                        f'Variant contract mismatch at {path}:{line_number}.')
                key = (dataset, str(record.get('img_path')))
                rank = int(record.get('rank', 0))
                prior_by_rank = seen.get(key)
                if prior_by_rank is not None:
                    if rank in prior_by_rank:
                        raise ValueError(
                            'Duplicate image record from the same rank; use '
                            f'a fresh ROOT before rerunning: {key}.')
                    prior = next(iter(prior_by_rank.values()))
                    exact_fields = (
                        'class_names', 'valid_pixels', 'prob_thd',
                        'confidence_threshold', 'primary_variant',
                        'prompt_count', 'settings', 'variants',
                    )
                    inconsistent = [
                        field for field in exact_fields
                        if prior.get(field) != record.get(field)
                    ]
                    if inconsistent:
                        raise ValueError(
                            'Cross-rank padding duplicate has inconsistent '
                            f'exact fields {inconsistent}: {key}.')
                    prior_by_rank[rank] = record
                    duplicate_counts[dataset] += 1
                    continue
                seen[key] = {rank: record}
                records.append(record)
    if not records:
        raise ValueError('No semantic-supplement records were found.')
    observed = {str(record['dataset_name']).lower() for record in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing expected datasets: {missing}.')
    return records, missing, dict(duplicate_counts)


def summarize_dataset(dataset, records, tolerance):
    class_names = records[0]['class_names']
    if any(record['class_names'] != class_names for record in records):
        raise ValueError(f'Class order differs within {dataset}.')
    matrices = {name: None for name in VARIANT_NAMES}
    counters = {name: defaultdict(int) for name in VARIANT_NAMES}
    candidate_values = defaultdict(lambda: defaultdict(list))
    family_values = defaultdict(lambda: defaultdict(list))
    controls = defaultdict(list)
    synonym_counts = []

    for record in records:
        for name in VARIANT_NAMES:
            stats = record['variants'][name]
            matrices[name] = add_matrix(
                matrices[name], stats['confusion']['matrix'])
            for key in (
                    'changed_pixels', 'improved_pixels', 'harmed_pixels',
                    'wrong_to_wrong_pixels', 'help_minus_harm'):
                counters[name][key] += int(stats.get(key, 0))
        for view in record.get('views', []):
            for key in (
                    'baseline_reconstruction_max_abs',
                    'raw_recomposition_max_abs',
                    'anchor_vs_official_max_abs',
                    'diagnostic_cpu_bytes',
                    'diagnostic_variant_count'):
                controls[key].append(view.get(key))
            synonym_counts.append(view.get('synonym_query_count'))
            for row in view.get('candidate_rows', []):
                key = (
                    int(row['class_index']), str(row['family']),
                    int(row['slot']))
                for field, value in row.items():
                    candidate_values[key][field].append(value)
            for row in view.get('family_rows', []):
                key = (int(row['class_index']), str(row['family']))
                for field, value in row.items():
                    family_values[key][field].append(value)

    metrics = {name: confusion_metrics(matrix)
               for name, matrix in matrices.items()}
    official = metrics['baseline']
    anchor = metrics[CAUSAL_REFERENCE]
    valid_pixels = int(sum(sum(row) for row in matrices['baseline']))
    variant_rows = []
    class_rows = []
    for name in VARIANT_NAMES:
        metric = metrics[name]
        changed = counters[name]['changed_pixels']
        variant_rows.append(dict(
            dataset=dataset,
            variant=name,
            images=len(records),
            miou=metric['miou'],
            delta_miou_to_official=metric['miou'] - official['miou'],
            delta_miou_to_anchor=metric['miou'] - anchor['miou'],
            aacc=metric['aacc'],
            delta_aacc_to_official=metric['aacc'] - official['aacc'],
            changed_ratio_to_official=safe_div(changed, valid_pixels),
            improved_pixels=counters[name]['improved_pixels'],
            harmed_pixels=counters[name]['harmed_pixels'],
            help_minus_harm=counters[name]['help_minus_harm'],
        ))
        for class_idx, class_name in enumerate(class_names):
            iou = metric['iou'][class_idx]
            official_iou = official['iou'][class_idx]
            anchor_iou = anchor['iou'][class_idx]
            class_rows.append(dict(
                dataset=dataset,
                variant=name,
                class_index=class_idx,
                class_name=class_name,
                iou=(iou * 100.0 if finite(iou) else None),
                delta_iou_to_official=(
                    (iou - official_iou) * 100.0
                    if finite(iou) and finite(official_iou) else None),
                delta_iou_to_anchor=(
                    (iou - anchor_iou) * 100.0
                    if finite(iou) and finite(anchor_iou) else None),
                gt_pixels=int(official['gt'][class_idx]),
                pred_pixels=int(metric['pred'][class_idx]),
            ))

    variant_lookup = {row['variant']: row for row in variant_rows}
    class_lookup = {(row['variant'], row['class_index']): row
                    for row in class_rows}
    candidate_effects = []
    class_candidate_effects = []
    for family, _ in FAMILIES:
        for slot in range(1, MAX_CANDIDATES + 1):
            names = {
                path: candidate_variant_name(family, slot, path)
                for path in (
                    'shared_native', 'semantic_replace', 'semantic_residual')
            }
            values = {path: variant_lookup[name]['miou']
                      for path, name in names.items()}
            candidate_effects.append(dict(
                dataset=dataset,
                family=family,
                slot=slot,
                anchor_miou=anchor['miou'],
                shared_native_delta=(
                    values['shared_native'] - anchor['miou']),
                semantic_replace_delta=(
                    values['semantic_replace'] - anchor['miou']),
                semantic_residual_delta=(
                    values['semantic_residual'] - anchor['miou']),
                role_separation_gain_replace=(
                    values['semantic_replace'] - values['shared_native']),
                role_separation_gain_residual=(
                    values['semantic_residual'] - values['shared_native']),
                residual_protection_gain=(
                    values['semantic_residual']
                    - values['semantic_replace']),
            ))
            for class_idx, class_name in enumerate(class_names):
                anchor_iou = class_lookup[(CAUSAL_REFERENCE, class_idx)]['iou']
                path_iou = {
                    path: class_lookup[(name, class_idx)]['iou']
                    for path, name in names.items()
                }
                if not finite(anchor_iou) or not all(
                        finite(value) for value in path_iou.values()):
                    continue
                class_candidate_effects.append(dict(
                    dataset=dataset,
                    family=family,
                    slot=slot,
                    class_index=class_idx,
                    class_name=class_name,
                    shared_native_delta=(
                        path_iou['shared_native'] - anchor_iou),
                    semantic_replace_delta=(
                        path_iou['semantic_replace'] - anchor_iou),
                    semantic_residual_delta=(
                        path_iou['semantic_residual'] - anchor_iou),
                    role_separation_gain_residual=(
                        path_iou['semantic_residual']
                        - path_iou['shared_native']),
                    residual_protection_gain=(
                        path_iou['semantic_residual']
                        - path_iou['semantic_replace']),
                ))

    family_effects = []
    for family, _ in FAMILIES:
        replace_name = aggregate_variant_name(
            family, 'mean_semantic_replace')
        residual_name = aggregate_variant_name(
            family, 'mean_semantic_residual')
        replace = variant_lookup[replace_name]
        residual = variant_lookup[residual_name]
        family_effects.append(dict(
            dataset=dataset,
            family=family,
            fixed_rule='mean_available_candidates',
            anchor_miou=anchor['miou'],
            mean_replace_miou=replace['miou'],
            mean_replace_delta=replace['delta_miou_to_anchor'],
            mean_residual_miou=residual['miou'],
            mean_residual_delta=residual['delta_miou_to_anchor'],
            residual_protection_gain=(residual['miou'] - replace['miou']),
        ))

    prompt_rows = []
    text_fields = {
        'class_name', 'family', 'anchor_prompt', 'candidate_prompt'}
    bool_fields = {'available'}
    nested_fields = {'path_final_means'}
    for (class_idx, family, slot), values in sorted(candidate_values.items()):
        row = dict(
            dataset=dataset,
            class_index=class_idx,
            family=family,
            slot=slot,
        )
        for field, numbers in values.items():
            if field in text_fields:
                row[field] = numbers[0] if numbers else None
            elif field in bool_fields:
                row[field] = bool(numbers[0]) if numbers else False
            elif field in nested_fields:
                keys = sorted(set().union(*(item.keys() for item in numbers)))
                for key in keys:
                    row[f'{field}_{key}'] = mean(
                        item.get(key) for item in numbers)
            elif field not in ('class_index', 'slot'):
                row[field] = mean(numbers)
        prompt_rows.append(row)

    family_response_rows = []
    for (class_idx, family), values in sorted(family_values.items()):
        row = dict(
            dataset=dataset,
            class_index=class_idx,
            family=family,
        )
        for field, numbers in values.items():
            if field == 'class_name':
                row[field] = numbers[0] if numbers else None
            elif field == 'aggregate':
                row[field] = numbers[0] if numbers else None
            elif field not in ('class_index', 'family'):
                row[field] = mean(numbers)
        family_response_rows.append(row)

    oracle_rows = []
    for family, _ in FAMILIES:
        candidate_names = [
            candidate_variant_name(
                family, slot, 'semantic_residual')
            for slot in range(1, MAX_CANDIDATES + 1)
        ]
        best = max(candidate_names, key=lambda name: variant_lookup[name]['miou'])
        oracle_rows.append(dict(
            dataset=dataset,
            family=family,
            oracle_scope='dataset_best_residual_slot',
            oracle_only=True,
            selected_variant=best,
            miou=variant_lookup[best]['miou'],
            delta_miou_to_anchor=variant_lookup[best]['delta_miou_to_anchor'],
        ))
        for class_idx, class_name in enumerate(class_names):
            candidates = [
                class_lookup[(name, class_idx)]
                for name in candidate_names
                if finite(class_lookup[(name, class_idx)]['iou'])
            ]
            if not candidates:
                continue
            best_class = max(candidates, key=lambda row: row['iou'])
            oracle_rows.append(dict(
                dataset=dataset,
                family=family,
                oracle_scope='class_best_residual_slot_noncoherent',
                oracle_only=True,
                class_index=class_idx,
                class_name=class_name,
                selected_variant=best_class['variant'],
                iou=best_class['iou'],
                delta_iou_to_anchor=best_class['delta_iou_to_anchor'],
            ))

    control_summary = {
        key: max([float(value) for value in values if finite(value)] or [0.0])
        for key, values in controls.items()
    }
    control_summary.update(dict(
        dataset=dataset,
        images=len(records),
        synonym_query_count=max(
            [int(value) for value in synonym_counts if finite(value)] or [0]),
    ))
    control_summary['integrity_passed'] = bool(
        control_summary['baseline_reconstruction_max_abs'] <= tolerance
        and control_summary['raw_recomposition_max_abs'] <= tolerance)
    return dict(
        dataset=dataset,
        records=len(records),
        class_names=class_names,
        variant_rows=variant_rows,
        class_rows=class_rows,
        candidate_effects=candidate_effects,
        class_candidate_effects=class_candidate_effects,
        family_effects=family_effects,
        prompt_rows=prompt_rows,
        family_response_rows=family_response_rows,
        oracle_rows=oracle_rows,
        controls=control_summary,
    )


def cross_dataset_rows(dataset_summaries):
    by_variant = defaultdict(list)
    for summary in dataset_summaries:
        for row in summary['variant_rows']:
            by_variant[row['variant']].append(row)
    rows = []
    for name in VARIANT_NAMES:
        values = by_variant[name]
        anchor_deltas = [row['delta_miou_to_anchor'] for row in values]
        official_deltas = [row['delta_miou_to_official'] for row in values]
        rows.append(dict(
            variant=name,
            datasets=len(values),
            positive_vs_anchor=sum(value > 0.0 for value in anchor_deltas),
            negative_vs_anchor=sum(value < 0.0 for value in anchor_deltas),
            macro_delta_vs_anchor=mean(anchor_deltas),
            median_delta_vs_anchor=median(anchor_deltas),
            worst_delta_vs_anchor=min(anchor_deltas),
            best_delta_vs_anchor=max(anchor_deltas),
            positive_vs_official=sum(value > 0.0 for value in official_deltas),
            macro_delta_vs_official=mean(official_deltas),
            per_dataset_delta_vs_anchor=json.dumps(
                {row['dataset']: row['delta_miou_to_anchor']
                 for row in values}, ensure_ascii=False, sort_keys=True),
        ))
    return rows


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    records, missing, duplicate_counts = load_records(
        paths, args.expected_datasets, args.allow_incomplete)
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    summaries = [
        summarize_dataset(dataset, grouped[dataset], args.integrity_tolerance)
        for dataset in args.expected_datasets if dataset in grouped
    ]
    variant_rows = [row for summary in summaries
                    for row in summary['variant_rows']]
    class_rows = [row for summary in summaries
                  for row in summary['class_rows']]
    candidate_effects = [row for summary in summaries
                         for row in summary['candidate_effects']]
    class_candidate_effects = [row for summary in summaries
                               for row in summary['class_candidate_effects']]
    family_effects = [row for summary in summaries
                      for row in summary['family_effects']]
    prompt_rows = [row for summary in summaries
                   for row in summary['prompt_rows']]
    family_response_rows = [row for summary in summaries
                            for row in summary['family_response_rows']]
    oracle_rows = [row for summary in summaries
                   for row in summary['oracle_rows']]
    controls = [summary['controls'] for summary in summaries]
    cross_rows = cross_dataset_rows(summaries)
    cross_lookup = {row['variant']: row for row in cross_rows}
    fixed_decisions = []
    for name in FIXED_FAMILY_VARIANTS:
        row = dict(cross_lookup[name])
        row['direction_screen_passed'] = bool(
            row['datasets'] >= 6
            and row['positive_vs_anchor'] >= 4
            and row['macro_delta_vs_anchor'] > 0.0
            and row['median_delta_vs_anchor'] > 0.0)
        fixed_decisions.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'dataset_variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'per_class.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'candidate_effects.csv'),
              candidate_effects)
    write_csv(os.path.join(args.out_dir, 'per_class_candidate_effects.csv'),
              class_candidate_effects)
    write_csv(os.path.join(args.out_dir, 'family_effects.csv'), family_effects)
    write_csv(os.path.join(args.out_dir, 'prompt_response.csv'), prompt_rows)
    write_csv(os.path.join(args.out_dir, 'family_response.csv'),
              family_response_rows)
    write_csv(os.path.join(args.out_dir, 'oracle_diagnostics.csv'), oracle_rows)
    write_csv(os.path.join(args.out_dir, 'integrity.csv'), controls)
    write_csv(os.path.join(args.out_dir, 'cross_dataset_variants.csv'),
              cross_rows)

    payload = dict(
        schema_version=SCHEMA_VERSION,
        protocol=PROTOCOL,
        source_files=paths,
        missing_datasets=missing,
        distributed_padding_duplicates_removed=duplicate_counts,
        datasets=summaries,
        cross_dataset_variants=cross_rows,
        fixed_family_decisions=fixed_decisions,
        oracle_diagnostics=oracle_rows,
        all_integrity_passed=all(row['integrity_passed'] for row in controls),
    )
    with open(os.path.join(args.out_dir, 'summary.json'),
              'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'decision_report.md'),
              'w', encoding='utf-8') as handle:
        handle.write('# Two-prompt semantic supplement direction screen\n\n')
        handle.write(
            f'- Datasets present: `{", ".join(grouped)}`; missing: '
            f'`{", ".join(missing) if missing else "none"}`.\n')
        handle.write(
            '- DDP padding duplicates removed after exact validation: '
            f'`{json.dumps(duplicate_counts, sort_keys=True)}`.\n')
        handle.write(
            f'- Exact implementation controls passed: '
            f'`{payload["all_integrity_passed"]}`.\n')
        handle.write(
            '- All causal deltas use `anchor_native`; official synonym '
            'baseline is reported separately.\n')
        handle.write('\n## Frozen family rules\n\n')
        for row in fixed_decisions:
            handle.write(
                f'- `{row["variant"]}`: positive '
                f'{row["positive_vs_anchor"]}/{row["datasets"]}, macro '
                f'{row["macro_delta_vs_anchor"]:.4f}, median '
                f'{row["median_delta_vs_anchor"]:.4f}, worst '
                f'{row["worst_delta_vs_anchor"]:.4f}; direction gate '
                f'passed: `{row["direction_screen_passed"]}`.\n')
        handle.write('\n## Interpretation contract\n\n')
        handle.write(
            '- `shared_native` tests sending the same candidate through all '
            'SAM3 roles. `semantic_replace` fixes anchor Presence/instance '
            'but replaces semantic. `semantic_residual` keeps the anchor '
            'semantic base and applies the predeclared bounded correction.\n')
        handle.write(
            '- `oracle_diagnostics.csv` contains evaluation-label-selected '
            'upper bounds only. Its class-level rows are non-coherent '
            'per-class IoU diagnostics and are never a final segmentation '
            'method. No oracle value may be reported as deployable results.\n')
    print(json.dumps(dict(
        out_dir=os.path.abspath(args.out_dir),
        datasets=list(grouped),
        missing=missing,
        distributed_padding_duplicates_removed=duplicate_counts,
        all_integrity_passed=payload['all_integrity_passed'],
        fixed_family_decisions=fixed_decisions,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
