#!/usr/bin/env python3
"""Summarize the six-dataset SAM3 head-role prompt-conflict audit."""

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

from head_role_prompt_conflict_definitions import (
    CAUSAL_REFERENCE,
    DESCRIPTION_SLOTS,
    HEAD_PATHS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    variant_name,
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
    total = sum(gt)
    return dict(
        iou=iou,
        miou=mean(valid_iou) * 100.0,
        aacc=safe_div(sum(tp), total) * 100.0,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


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
                if record.get('settings', {}).get('protocol') \
                        != 'head_role_prompt_conflict_v1':
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
                        name for name in exact_fields
                        if prior.get(name) != record.get(name)
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
        raise ValueError('No head-role prompt-conflict records were found.')
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
    head_values = defaultdict(lambda: defaultdict(list))
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
            controls['baseline_reconstruction_max_abs'].append(
                view.get('baseline_reconstruction_max_abs'))
            controls['raw_recomposition_max_abs'].append(
                view.get('raw_recomposition_max_abs'))
            controls['literal_vs_official_max_abs'].append(
                view.get('literal_vs_official_max_abs'))
            controls['diagnostic_cpu_bytes'].append(
                view.get('diagnostic_cpu_bytes'))
            controls['diagnostic_variant_count'].append(
                view.get('diagnostic_variant_count'))
            synonym_counts.append(view.get('synonym_query_count'))
            for row in view.get('head_role_rows', []):
                key = (int(row['class_index']), int(row['prompt_slot']))
                target = head_values[key]
                for field, value in row.items():
                    if field in (
                            'class_index', 'class_name', 'prompt_slot',
                            'prompt_role', 'literal_prompt',
                            'description_prompt', 'path_final_means',
                            'path_kept_counts'):
                        if field not in target:
                            target[field] = []
                        target[field].append(value)
                    elif finite(value):
                        target[field].append(value)

    metrics = {name: confusion_metrics(matrix)
               for name, matrix in matrices.items()}
    official = metrics['baseline']
    literal = metrics[CAUSAL_REFERENCE]
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
            delta_miou_to_literal=metric['miou'] - literal['miou'],
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
            literal_iou = literal['iou'][class_idx]
            class_rows.append(dict(
                dataset=dataset,
                variant=name,
                class_index=class_idx,
                class_name=class_name,
                iou=(iou * 100.0 if finite(iou) else None),
                delta_iou_to_official=(
                    (iou - official_iou) * 100.0
                    if finite(iou) and finite(official_iou) else None),
                delta_iou_to_literal=(
                    (iou - literal_iou) * 100.0
                    if finite(iou) and finite(literal_iou) else None),
                gt_pixels=int(official['gt'][class_idx]),
                pred_pixels=int(metric['pred'][class_idx]),
            ))

    variant_lookup = {row['variant']: row for row in variant_rows}
    class_lookup = {(row['variant'], row['class_index']): row
                    for row in class_rows}
    mechanism_rows = []
    class_mechanism_rows = []
    for prompt_slot, prompt_role in DESCRIPTION_SLOTS:
        names = {path_name: variant_name(prompt_slot, path_name)
                 for path_name, _, _, _ in HEAD_PATHS}
        value = {path: variant_lookup[name]['miou']
                 for path, name in names.items()}
        mechanism_rows.append(dict(
            dataset=dataset,
            prompt_slot=prompt_slot,
            prompt_role=prompt_role,
            literal_miou=literal['miou'],
            native_delta=value['native'] - literal['miou'],
            semantic_only_delta=(
                value['semantic_only'] - literal['miou']),
            instance_only_delta=(
                value['instance_only'] - literal['miou']),
            presence_only_delta=(
                value['presence_only'] - literal['miou']),
            semantic_instance_delta=(
                value['semantic_instance_literal_presence']
                - literal['miou']),
            presence_recovery=(
                value['semantic_instance_literal_presence']
                - value['native']),
            instance_effect_given_literal_presence=(
                value['semantic_instance_literal_presence']
                - value['semantic_only']),
            semantic_effect_given_literal_presence=(
                value['semantic_instance_literal_presence']
                - value['instance_only']),
            shared_prompt_penalty_to_best_role=(
                value['native'] - max(
                    value['semantic_only'], value['instance_only'],
                    value['presence_only'],
                    value['semantic_instance_literal_presence'])),
            semantic_positive_native_nonpositive=bool(
                value['semantic_only'] > literal['miou']
                and value['native'] <= literal['miou']),
        ))
        for class_idx, class_name in enumerate(class_names):
            literal_iou = class_lookup[
                (CAUSAL_REFERENCE, class_idx)]['iou']
            path_iou = {path: class_lookup[(name, class_idx)]['iou']
                        for path, name in names.items()}
            if not finite(literal_iou) or not all(
                    finite(number) for number in path_iou.values()):
                continue
            class_mechanism_rows.append(dict(
                dataset=dataset,
                class_index=class_idx,
                class_name=class_name,
                prompt_slot=prompt_slot,
                prompt_role=prompt_role,
                native_delta=path_iou['native'] - literal_iou,
                semantic_only_delta=(
                    path_iou['semantic_only'] - literal_iou),
                instance_only_delta=(
                    path_iou['instance_only'] - literal_iou),
                presence_only_delta=(
                    path_iou['presence_only'] - literal_iou),
                semantic_instance_delta=(
                    path_iou['semantic_instance_literal_presence']
                    - literal_iou),
                presence_recovery=(
                    path_iou['semantic_instance_literal_presence']
                    - path_iou['native']),
                instance_effect_given_literal_presence=(
                    path_iou['semantic_instance_literal_presence']
                    - path_iou['semantic_only']),
                semantic_effect_given_literal_presence=(
                    path_iou['semantic_instance_literal_presence']
                    - path_iou['instance_only']),
                semantic_positive_native_nonpositive=bool(
                    path_iou['semantic_only'] > literal_iou
                    and path_iou['native'] <= literal_iou),
            ))

    head_rows = []
    text_fields = {
        'class_name', 'prompt_role', 'literal_prompt', 'description_prompt'}
    nested_fields = {'path_final_means', 'path_kept_counts'}
    for (class_idx, prompt_slot), values in sorted(head_values.items()):
        row = dict(
            dataset=dataset,
            class_index=class_idx,
            prompt_slot=prompt_slot,
        )
        for field, numbers in values.items():
            if field in text_fields:
                row[field] = numbers[0] if numbers else None
            elif field in nested_fields:
                keys = sorted(set().union(*(item.keys() for item in numbers)))
                for key in keys:
                    row[f'{field}_{key}'] = mean(
                        item.get(key) for item in numbers)
            else:
                row[field] = mean(numbers)
        head_rows.append(row)

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
        mechanism_rows=mechanism_rows,
        class_mechanism_rows=class_mechanism_rows,
        head_rows=head_rows,
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
        literal_deltas = [row['delta_miou_to_literal'] for row in values]
        official_deltas = [row['delta_miou_to_official'] for row in values]
        rows.append(dict(
            variant=name,
            datasets=len(values),
            positive_vs_literal=sum(value > 0.0 for value in literal_deltas),
            negative_vs_literal=sum(value < 0.0 for value in literal_deltas),
            macro_delta_vs_literal=mean(literal_deltas),
            median_delta_vs_literal=median(literal_deltas),
            worst_delta_vs_literal=min(literal_deltas),
            best_delta_vs_literal=max(literal_deltas),
            positive_vs_official=sum(value > 0.0 for value in official_deltas),
            macro_delta_vs_official=mean(official_deltas),
            per_dataset_delta_vs_literal=json.dumps(
                {row['dataset']: row['delta_miou_to_literal']
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
    mechanism_rows = [row for summary in summaries
                      for row in summary['mechanism_rows']]
    class_mechanism_rows = [row for summary in summaries
                            for row in summary['class_mechanism_rows']]
    head_rows = [row for summary in summaries for row in summary['head_rows']]
    controls = [summary['controls'] for summary in summaries]
    cross_rows = cross_dataset_rows(summaries)
    fixed_candidates = [
        row for row in cross_rows
        if row['variant'] not in ('baseline', CAUSAL_REFERENCE)
    ]
    best_fixed = max(
        fixed_candidates,
        key=lambda row: (
            row['positive_vs_literal'], row['macro_delta_vs_literal'],
            row['worst_delta_vs_literal']))
    conflict_counts = dict(
        datasets_with_semantic_positive_native_nonpositive=len({
            row['dataset'] for row in mechanism_rows
            if row['semantic_positive_native_nonpositive']}),
        dataset_slot_cases=sum(
            bool(row['semantic_positive_native_nonpositive'])
            for row in mechanism_rows),
        dataset_slot_presence_recovery_positive=sum(
            row['presence_recovery'] > 0.0 for row in mechanism_rows),
        dataset_slot_instance_effect_negative=sum(
            row['instance_effect_given_literal_presence'] < 0.0
            for row in mechanism_rows),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'dataset_variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'per_class.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'dataset_mechanisms.csv'), mechanism_rows)
    write_csv(os.path.join(args.out_dir, 'per_class_mechanisms.csv'),
              class_mechanism_rows)
    write_csv(os.path.join(args.out_dir, 'head_response.csv'), head_rows)
    write_csv(os.path.join(args.out_dir, 'integrity.csv'), controls)
    write_csv(os.path.join(args.out_dir, 'cross_dataset_variants.csv'),
              cross_rows)
    payload = dict(
        schema_version=SCHEMA_VERSION,
        source_files=paths,
        missing_datasets=missing,
        distributed_padding_duplicates_removed=duplicate_counts,
        datasets=summaries,
        cross_dataset_variants=cross_rows,
        best_fixed_candidate=best_fixed,
        conflict_counts=conflict_counts,
        all_integrity_passed=all(row['integrity_passed'] for row in controls),
    )
    with open(os.path.join(args.out_dir, 'summary.json'),
              'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'decision_report.md'),
              'w', encoding='utf-8') as handle:
        handle.write('# SAM3 head-role prompt-conflict audit\n\n')
        handle.write(
            f'- Datasets present: `{", ".join(grouped)}`; missing: '
            f'`{", ".join(missing) if missing else "none"}`.\n')
        handle.write(
            '- DDP sampler padding duplicates removed after exact-field '
            f'validation: `{json.dumps(duplicate_counts, sort_keys=True)}`.\n')
        handle.write(
            f'- Exact implementation controls passed: '
            f'`{payload["all_integrity_passed"]}`.\n')
        handle.write(
            '- Official baseline and canonical literal are reported '
            'separately; all causal head deltas use `literal_native`.\n')
        handle.write('\n## Fixed cross-dataset candidate\n\n')
        handle.write(
            f'- `{best_fixed["variant"]}`: positive on '
            f'{best_fixed["positive_vs_literal"]}/{best_fixed["datasets"]}, '
            f'macro ΔmIoU {best_fixed["macro_delta_vs_literal"]:.4f}, '
            f'median {best_fixed["median_delta_vs_literal"]:.4f}, '
            f'worst {best_fixed["worst_delta_vs_literal"]:.4f}.\n')
        handle.write('\n## Hypothesis evidence\n\n')
        handle.write(
            '- Dataset-slot cases where semantic-only improves but the same '
            f'description natively does not: '
            f'{conflict_counts["dataset_slot_cases"]}; represented in '
            f'{conflict_counts["datasets_with_semantic_positive_native_nonpositive"]} '
            'datasets.\n')
        handle.write(
            '- Positive recovery after restoring literal Presence while '
            f'keeping descriptive semantic+instance: '
            f'{conflict_counts["dataset_slot_presence_recovery_positive"]} '
            'dataset-slot cases.\n')
        handle.write(
            '- Negative instance contribution under fixed literal Presence: '
            f'{conflict_counts["dataset_slot_instance_effect_negative"]} '
            'dataset-slot cases.\n')
        handle.write(
            '\nThese are causal diagnostic comparisons over a fixed bank, '
            'not a deployable prompt selector. A post-hoc best slot is an '
            'oracle and must not be reported as a method.\n')
    print(json.dumps(dict(
        out_dir=os.path.abspath(args.out_dir),
        datasets=list(grouped),
        missing=missing,
        distributed_padding_duplicates_removed=duplicate_counts,
        all_integrity_passed=payload['all_integrity_passed'],
        best_fixed_candidate=best_fixed,
        conflict_counts=conflict_counts,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
