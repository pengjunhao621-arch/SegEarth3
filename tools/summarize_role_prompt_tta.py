#!/usr/bin/env python3
"""Summarize exact per-image diagnostics from role-aware prompt TTA."""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_prompt_tta_definitions import SCHEMA_VERSION, VARIANT_NAMES


EXCLUDED_DATASETS = {'isaid'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--expected-datasets', nargs='+',
        default=['udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda'])
    parser.add_argument('--positive-dataset-gate', type=int, default=4)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def std(values):
    values = [float(value) for value in values if finite(value)]
    if not values:
        return None
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values)
                     / len(values))


def safe_div(a, b):
    return float(a) / float(b) if b else None


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_records(paths):
    seen = {}
    grouped = defaultdict(list)
    duplicate_counts = defaultdict(int)
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get('schema_version', -1)) != SCHEMA_VERSION:
                    raise ValueError(
                        f'Unexpected schema at {path}:{line_number}.')
                dataset = str(record.get('dataset_name', '')).strip().lower()
                if dataset in EXCLUDED_DATASETS:
                    raise ValueError('iSAID is excluded from project evaluation.')
                if not dataset or not record.get('img_path'):
                    raise ValueError(f'Missing dataset/img_path at {path}:{line_number}.')
                if tuple(record.get('variants', {}).keys()) != VARIANT_NAMES:
                    raise ValueError(
                        f'Variant order/content mismatch at {path}:{line_number}.')
                key = (dataset, str(record['img_path']))
                if key in seen:
                    prior = seen[key]
                    if int(prior.get('rank', 0)) == int(record.get('rank', 0)):
                        raise ValueError(
                            'Duplicate image from the same rank; use a fresh '
                            f'ROOT before rerunning: {key}.')
                    if prior['variants'] != record['variants']:
                        raise ValueError(
                            f'Inconsistent distributed duplicate: {key}.')
                    duplicate_counts[dataset] += 1
                    continue
                seen[key] = record
                grouped[dataset].append(record)
    return grouped, dict(duplicate_counts)


def add_matrix(target, source):
    if (
            not isinstance(source, list)
            or not source
            or any(not isinstance(row, list) for row in source)
            or any(len(row) != len(source) for row in source)):
        raise ValueError('Confusion matrix must be a non-empty square list.')
    source = [[int(value) for value in row] for row in source]
    if target is None:
        return source
    if len(target) != len(source):
        raise ValueError('Confusion matrix class-count mismatch.')
    return [
        [left + right for left, right in zip(target_row, source_row)]
        for target_row, source_row in zip(target, source)
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
    total = sum(sum(row) for row in matrix)
    return dict(
        iou=iou,
        miou=float(sum(valid_iou) / len(valid_iou) * 100.0),
        aacc=float(sum(tp) / max(total, 1) * 100.0),
        gt=gt,
        pred=pred,
        union=union,
    )


def weight_block(view, name):
    for block in view.get('weight_stats', []):
        if block.get('name') == name:
            return block
    return None


def class_weight_row(view, name, class_idx):
    block = weight_block(view, name)
    if block is None:
        return None
    for row in block.get('classes', []):
        if int(row.get('class_index', -1)) == class_idx:
            return row
    return None


def summarize_dataset(dataset, records):
    matrices = {name: None for name in VARIANT_NAMES}
    counters = {name: defaultdict(int) for name in VARIANT_NAMES}
    integrity = []
    class_names = records[0]['class_names']
    if any(record.get('class_names') != class_names for record in records):
        raise ValueError(f'{dataset} has inconsistent class order across records.')
    primary_values = {str(record['primary_variant']) for record in records}
    if len(primary_values) != 1:
        raise ValueError(f'{dataset} mixed primary variants: {primary_values}.')
    primary = next(iter(primary_values))

    mechanism = defaultdict(list)
    class_mechanism = {
        idx: defaultdict(list) for idx in range(len(class_names))}
    for record in records:
        for variant in VARIANT_NAMES:
            stats = record['variants'][variant]
            matrices[variant] = add_matrix(
                matrices[variant], stats['confusion']['matrix'])
            for key in (
                    'changed_pixels', 'improved_pixels', 'harmed_pixels',
                    'wrong_to_wrong_pixels', 'help_minus_harm'):
                counters[variant][key] += int(stats.get(key, 0))
        for view in record.get('views', []):
            integrity.append(view.get('baseline_reconstruction_max_abs'))
            gates = view.get('presence_gate', [])
            seeds = view.get('seed_counts', [])
            seed_entropy = view.get('seed_entropy', [])
            affinities = view.get('visual_affinity', [])
            updated = set(view.get('e2e_updated_classes', []))
            for class_idx in range(len(class_names)):
                gate = gates[class_idx] if class_idx < len(gates) else None
                seed = seeds[class_idx] if class_idx < len(seeds) else None
                ent = seed_entropy[class_idx] if class_idx < len(seed_entropy) else None
                affinity = affinities[class_idx] if class_idx < len(affinities) else []
                class_mechanism[class_idx]['presence_gate'].append(gate)
                class_mechanism[class_idx]['seed_pixels'].append(seed)
                class_mechanism[class_idx]['seed_entropy'].append(ent)
                class_mechanism[class_idx]['visual_affinity_abs'].append(
                    mean([abs(value) for value in affinity]))
                class_mechanism[class_idx]['e2e_updated'].append(
                    1.0 if class_idx in updated else 0.0)
                for weight_name in ('anchor', 'visual', 'full', 'e2e'):
                    row = class_weight_row(view, weight_name, class_idx)
                    if row:
                        class_mechanism[class_idx][
                            f'{weight_name}_effective_count'].append(
                                row.get('effective_prompt_count'))
                        class_mechanism[class_idx][
                            f'{weight_name}_max_weight'].append(
                                row.get('max_weight'))
                gt_rows = view.get('seed_gt_diagnostics', [])
                if class_idx < len(gt_rows):
                    gt_row = gt_rows[class_idx]
                    class_mechanism[class_idx]['seed_purity'].append(
                        gt_row.get('seed_purity'))
                    class_mechanism[class_idx]['valid_seed_pixels'].append(
                        gt_row.get('valid_seed_pixels'))
                    class_mechanism[class_idx]['correct_seed_pixels'].append(
                        gt_row.get('correct_seed_pixels'))
                candidate_rows = view.get('candidate_stats', [])
                if class_idx < len(candidate_rows):
                    candidates = candidate_rows[class_idx]
                    presences = [row.get('presence') for row in candidates]
                    final_means = [row.get('final_mean') for row in candidates]
                    kept = [row.get('kept_candidate_count') for row in candidates]
                    finite_presence = [float(value) for value in presences
                                       if finite(value)]
                    finite_final = [float(value) for value in final_means
                                    if finite(value)]
                    finite_kept = [float(value) for value in kept
                                   if finite(value)]
                    class_mechanism[class_idx]['prompt_presence_std'].append(
                        std(finite_presence)
                        if finite_presence else None)
                    class_mechanism[class_idx]['prompt_final_mean_range'].append(
                        max(finite_final) - min(finite_final)
                        if finite_final else None)
                    class_mechanism[class_idx]['prompt_kept_candidate_range'].append(
                        max(finite_kept) - min(finite_kept)
                        if finite_kept else None)
            for block in view.get('head_change', []):
                variant = str(block.get('variant', 'unknown'))
                for row in block.get('classes', []):
                    class_idx = int(row.get('class_index', -1))
                    if class_idx not in class_mechanism:
                        continue
                    for key in (
                            'semantic_mean_delta', 'semantic_abs_delta',
                            'instance_mean_delta', 'instance_abs_delta',
                            'presence_delta', 'final_mean_delta',
                            'semantic_instance_mask_iou'):
                        class_mechanism[class_idx][
                            f'{variant}_{key}'].append(row.get(key))
            for variant, rows in view.get(
                    'language_fusion_stats', {}).items():
                for row in rows:
                    class_idx = int(row.get('class_index', -1))
                    if class_idx not in class_mechanism:
                        continue
                    for key in (
                            'residual_to_anchor_norm', 'raw_candidate_count',
                            'kept_candidate_count'):
                        class_mechanism[class_idx][
                            f'{variant}_{key}'].append(row.get(key))
            mechanism['anchor_entropy_mean'].append(
                view.get('anchor_entropy_mean'))
            mechanism['remoteclip_global_norm'].append(
                view.get('remoteclip_global_norm'))
            for role, snapshot in view.get('cuda_memory', {}).items():
                if not isinstance(snapshot, dict):
                    continue
                for key in (
                        'allocated_mb', 'reserved_mb',
                        'peak_allocated_mb', 'peak_reserved_mb'):
                    mechanism[f'cuda_{role}_{key}'].append(
                        snapshot.get(key))
            for trajectory_name in (
                    'entropy_trajectory', 'full_trajectory',
                    'no_anchor_trajectory', 'no_presence_trajectory'):
                trajectory = view.get(trajectory_name, [])
                if trajectory:
                    mechanism[f'{trajectory_name}_loss_start'].append(
                        trajectory[0].get('loss'))
                    mechanism[f'{trajectory_name}_loss_end'].append(
                        trajectory[-1].get('loss'))
                    mechanism[f'{trajectory_name}_grad_norm'].append(
                        mean([step.get('grad_norm') for step in trajectory]))
            for class_trace in view.get('e2e_trajectory', []):
                steps = class_trace.get('steps', [])
                if steps:
                    mechanism['e2e_loss_start'].append(steps[0].get('loss'))
                    mechanism['e2e_loss_end'].append(steps[-1].get('loss'))
                    mechanism['e2e_grad_norm'].append(
                        mean([step.get('grad_norm') for step in steps]))
                    for stage in (
                            'before_forward', 'after_forward',
                            'after_backward'):
                        mechanism[
                            f'e2e_cuda_{stage}_peak_allocated_mb'].append(
                                max([
                                    float(step.get('cuda_memory', {})
                                          .get(stage, {})
                                          .get('peak_allocated_mb'))
                                    for step in steps
                                    if finite(
                                        step.get('cuda_memory', {})
                                        .get(stage, {})
                                        .get('peak_allocated_mb'))
                                ] or [None]))

    metrics = {name: confusion_metrics(matrix) for name, matrix in matrices.items()}
    baseline = metrics['baseline']
    valid_pixels = int(sum(sum(row) for row in matrices['baseline']))
    variant_rows = []
    for name in VARIANT_NAMES:
        current = metrics[name]
        count = counters[name]
        changed = count['changed_pixels']
        variant_rows.append(dict(
            dataset=dataset,
            variant=name,
            images=len(records),
            miou=current['miou'],
            delta_miou=current['miou'] - baseline['miou'],
            aacc=current['aacc'],
            delta_aacc=current['aacc'] - baseline['aacc'],
            changed_pixels=changed,
            changed_ratio=safe_div(changed, valid_pixels),
            improved_pixels=count['improved_pixels'],
            harmed_pixels=count['harmed_pixels'],
            help_minus_harm=count['help_minus_harm'],
            change_precision=safe_div(count['improved_pixels'], changed),
        ))

    class_rows = []
    for class_idx, class_name in enumerate(class_names):
        for name in VARIANT_NAMES:
            current_iou = metrics[name]['iou'][class_idx]
            baseline_iou = baseline['iou'][class_idx]
            class_rows.append(dict(
                dataset=dataset,
                class_index=class_idx,
                class_name=class_name,
                variant=name,
                iou=(float(current_iou * 100.0)
                     if finite(current_iou) else None),
                delta_iou=(float((current_iou - baseline_iou) * 100.0)
                           if finite(current_iou)
                           and finite(baseline_iou) else None),
                gt_pixels=int(baseline['gt'][class_idx]),
                pred_pixels=int(metrics[name]['pred'][class_idx]),
            ))
    class_diagnostic_rows = []
    for class_idx, class_name in enumerate(class_names):
        row = dict(
            dataset=dataset,
            class_index=class_idx,
            class_name=class_name,
            primary_delta_iou=(
                class_rows[
                    class_idx * len(VARIANT_NAMES)
                    + VARIANT_NAMES.index(primary)]['delta_iou']),
        )
        for key, values in class_mechanism[class_idx].items():
            row[key] = mean(values)
        correct = sum(
            value for value in class_mechanism[class_idx].get(
                'correct_seed_pixels', []) if finite(value))
        total = sum(
            value for value in class_mechanism[class_idx].get(
                'valid_seed_pixels', []) if finite(value))
        row['seed_purity_weighted'] = safe_div(correct, total)
        class_diagnostic_rows.append(row)

    mechanism_row = dict(
        dataset=dataset,
        views=sum(len(record.get('views', [])) for record in records),
        baseline_reconstruction_max_abs=max(
            [float(value) for value in integrity if finite(value)] or [0.0]),
    )
    for key, values in mechanism.items():
        mechanism_row[key] = mean(values)
    primary_row = next(row for row in variant_rows if row['variant'] == primary)
    return dict(
        dataset=dataset,
        primary=primary,
        variant_rows=variant_rows,
        class_rows=class_rows,
        class_diagnostic_rows=class_diagnostic_rows,
        mechanism_row=mechanism_row,
        primary_delta=primary_row['delta_miou'],
    )


def build_attribution(variant_rows):
    lookup = {
        (row['dataset'], row['variant']): row['delta_miou']
        for row in variant_rows
    }
    datasets = sorted({row['dataset'] for row in variant_rows})
    comparisons = [
        ('prompt_pool_best', 'pool_max', 'baseline'),
        ('static_anchor_reground', 'anchor_regrounded', 'baseline'),
        ('anchor_mass_vs_uniform', 'anchor_regrounded', 'uniform_regrounded'),
        ('visual_vs_uniform_reground', 'visual_regrounded', 'uniform_regrounded'),
        ('entropy_plus_visual', 'full_regrounded_surrogate', 'visual_regrounded'),
        ('reground_vs_output', 'full_regrounded_surrogate', 'full_output'),
        ('e2e_vs_surrogate', 'full_regrounded_e2e', 'full_regrounded_surrogate'),
        ('anchor_effect_output', 'full_output', 'full_no_anchor_output'),
        ('presence_gate_effect_output', 'full_output', 'full_no_presence_gate_output'),
    ]
    rows = []
    for label, left, right in comparisons:
        deltas = [lookup[(dataset, left)] - lookup[(dataset, right)]
                  for dataset in datasets]
        rows.append(dict(
            component=label,
            left=left,
            right=right,
            mean_delta_miou=mean(deltas),
            positive_datasets=sum(value > 0 for value in deltas),
            zero_datasets=sum(value == 0 for value in deltas),
            negative_datasets=sum(value < 0 for value in deltas),
            per_dataset={dataset: value for dataset, value in zip(datasets, deltas)},
        ))
    return rows


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit('No role_prompt_tta.rank*.jsonl inputs found.')
    grouped, duplicates = load_records(paths)
    expected = [name.lower() for name in args.expected_datasets]
    missing = sorted(set(expected) - set(grouped))
    unexpected = sorted(set(grouped) - set(expected))
    summaries = [summarize_dataset(name, grouped[name])
                 for name in expected if name in grouped]
    variant_rows = [row for item in summaries for row in item['variant_rows']]
    class_rows = [row for item in summaries for row in item['class_rows']]
    class_diagnostics = [
        row for item in summaries for row in item['class_diagnostic_rows']]
    mechanism_rows = [item['mechanism_row'] for item in summaries]
    attribution = build_attribution(variant_rows) if summaries else []
    primary_positive = sum(item['primary_delta'] > 0 for item in summaries)
    gate_passed = (
        not missing
        and not unexpected
        and primary_positive >= int(args.positive_dataset_gate))
    variant_rank = []
    for variant in VARIANT_NAMES:
        rows = [row for row in variant_rows if row['variant'] == variant]
        variant_rank.append(dict(
            variant=variant,
            mean_delta_miou=mean([row['delta_miou'] for row in rows]),
            positive_datasets=sum(row['delta_miou'] > 0 for row in rows),
            negative_datasets=sum(row['delta_miou'] < 0 for row in rows),
        ))
    variant_rank.sort(
        key=lambda row: row['mean_delta_miou']
        if row['mean_delta_miou'] is not None else -float('inf'),
        reverse=True)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'dataset_variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'per_class.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'class_diagnostics.csv'), class_diagnostics)
    write_csv(os.path.join(args.out_dir, 'mechanism.csv'), mechanism_rows)
    attribution_csv = []
    for row in attribution:
        flat = dict(row)
        flat['per_dataset'] = json.dumps(
            flat['per_dataset'], ensure_ascii=False, sort_keys=True)
        attribution_csv.append(flat)
    write_csv(os.path.join(args.out_dir, 'component_attribution.csv'), attribution_csv)
    payload = dict(
        schema_version=SCHEMA_VERSION,
        source_files=paths,
        duplicate_records=duplicates,
        expected_datasets=expected,
        missing_datasets=missing,
        unexpected_datasets=unexpected,
        primary_positive_datasets=primary_positive,
        positive_dataset_gate=int(args.positive_dataset_gate),
        gate_passed=gate_passed,
        variant_rank=variant_rank,
        component_attribution=attribution,
        datasets=summaries,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'decision_report.md'), 'w', encoding='utf-8') as handle:
        handle.write('# Role Prompt TTA decision report\n\n')
        handle.write(
            f'- Dataset gate: **{primary_positive}/{len(summaries)}** positive; '
            f'required {args.positive_dataset_gate}; passed={gate_passed}.\n')
        if missing:
            handle.write(f'- Missing datasets: {", ".join(missing)}.\n')
        handle.write('\n## Variant ranking\n\n')
        for row in variant_rank:
            handle.write(
                f'- `{row["variant"]}`: mean ΔmIoU '
                f'{row["mean_delta_miou"]:.4f}, positive '
                f'{row["positive_datasets"]}/{len(summaries)}.\n')
        handle.write('\n## Component attribution\n\n')
        for row in attribution:
            handle.write(
                f'- `{row["component"]}`: mean paired ΔmIoU '
                f'{row["mean_delta_miou"]:.4f}; positive '
                f'{row["positive_datasets"]}/{len(summaries)}.\n')
        handle.write(
            '\nGT seed purity, per-class mechanism statistics, weight '
            'concentration, gradient norms and exact head changes are in the '
            'CSV/JSON files. They are diagnostics only and never enter TTA.\n')
    print(json.dumps({
        'out_dir': os.path.abspath(args.out_dir),
        'datasets': [item['dataset'] for item in summaries],
        'missing': missing,
        'primary_positive': primary_positive,
        'gate_passed': gate_passed,
        'best_variant': variant_rank[0] if variant_rank else None,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
