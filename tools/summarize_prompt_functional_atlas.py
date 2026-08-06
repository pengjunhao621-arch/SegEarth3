#!/usr/bin/env python3
"""Summarize the SAM3 prompt functional atlas without tuning on labels."""

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

from prompt_functional_atlas_definitions import (
    PROMPT_SLOTS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)))


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else None


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
    total = sum(sum(row) for row in matrix)
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


def pearson(rows, left, right):
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows if finite(row.get(left)) and finite(row.get(right))
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return safe_div(numerator, x_scale * y_scale)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-dataset', default='udd5')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


def load_records(paths, expected_dataset):
    records = []
    seen = set()
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get('schema_version', -1)) != SCHEMA_VERSION:
                    raise ValueError(f'Unexpected schema at {path}:{line_number}.')
                if record.get('settings', {}).get('protocol') \
                        != 'prompt_functional_atlas_v2':
                    raise ValueError(f'Not an atlas-v2 record: {path}:{line_number}.')
                if str(record.get('dataset_name', '')).lower() != expected_dataset:
                    raise ValueError(
                        f'Expected {expected_dataset}, got '
                        f'{record.get("dataset_name")!r}.')
                if tuple(record.get('variants', {})) != VARIANT_NAMES:
                    raise ValueError(
                        f'Variant contract mismatch at {path}:{line_number}.')
                key = str(record.get('img_path'))
                if key in seen:
                    raise ValueError(f'Duplicate atlas image: {key}.')
                seen.add(key)
                records.append(record)
    if not records:
        raise ValueError('No prompt functional atlas records were found.')
    return records


def summarize(records, tolerance):
    class_names = records[0]['class_names']
    if any(record['class_names'] != class_names for record in records):
        raise ValueError('Class order differs across atlas records.')
    matrices = {name: None for name in VARIANT_NAMES}
    counters = {name: defaultdict(int) for name in VARIANT_NAMES}
    prompt_values = {
        (class_idx, prompt_idx): defaultdict(list)
        for class_idx in range(len(class_names))
        for prompt_idx in range(len(PROMPT_SLOTS))
    }
    head_values = defaultdict(lambda: defaultdict(list))
    e2e_values = defaultdict(lambda: defaultdict(list))
    control_values = defaultdict(list)
    selected_counts = defaultdict(int)

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
            control_values['baseline_reconstruction_max_abs'].append(
                view.get('baseline_reconstruction_max_abs'))
            for key, value in view.get(
                    'literal_reground_parity_max_abs', {}).items():
                control_values[f'literal_reground_{key}_max_abs'].append(value)
            candidate_stats = view.get('candidate_stats', [])
            selected = view.get('presence_selected_indices', [])
            for class_idx, class_rows in enumerate(candidate_stats):
                if class_idx < len(selected):
                    selected_counts[(class_idx, int(selected[class_idx]))] += 1
                for prompt_idx, row in enumerate(class_rows):
                    target = prompt_values[(class_idx, prompt_idx)]
                    for key in (
                            'token_count', 'word_count', 'character_count',
                            'presence', 'raw_candidate_count',
                            'kept_candidate_count', 'semantic_mean',
                            'instance_mean', 'final_mean'):
                        target[key].append(row.get(key))
                    target['prompt'].append(row.get('prompt'))
            for block in view.get('head_change', []):
                variant = str(block.get('variant'))
                for row in block.get('classes', []):
                    class_idx = int(row['class_index'])
                    target = head_values[(variant, class_idx)]
                    for key, value in row.items():
                        if key not in ('class_index', 'class_name'):
                            target[key].append(value)
            for trace in view.get('e2e_trajectory', []):
                class_idx = int(trace['class_index'])
                steps = trace.get('steps', [])
                if steps:
                    target = e2e_values[class_idx]
                    target['loss'].append(steps[-1].get('loss'))
                    target['grad_norm'].extend(
                        step.get('grad_norm') for step in steps)
                    target['post_update_loss'].append(
                        steps[-1].get('post_update_loss'))
                    target['post_update_weight_l1'].append(
                        steps[-1].get('post_update_weight_l1'))

    metrics = {
        name: confusion_metrics(matrix) for name, matrix in matrices.items()
    }
    baseline = metrics['baseline']
    valid_pixels = int(sum(sum(row) for row in matrices['baseline']))
    variant_rows = []
    class_rows = []
    for variant in VARIANT_NAMES:
        metric = metrics[variant]
        changed = counters[variant]['changed_pixels']
        variant_rows.append(dict(
            variant=variant,
            images=len(records),
            miou=metric['miou'],
            delta_miou=metric['miou'] - baseline['miou'],
            aacc=metric['aacc'],
            delta_aacc=metric['aacc'] - baseline['aacc'],
            changed_ratio=safe_div(changed, valid_pixels),
            change_precision=safe_div(
                counters[variant]['improved_pixels'], changed),
            improved_pixels=counters[variant]['improved_pixels'],
            harmed_pixels=counters[variant]['harmed_pixels'],
            help_minus_harm=counters[variant]['help_minus_harm'],
        ))
        for class_idx, class_name in enumerate(class_names):
            iou = metric['iou'][class_idx]
            baseline_iou = baseline['iou'][class_idx]
            class_rows.append(dict(
                variant=variant,
                class_index=class_idx,
                class_name=class_name,
                iou=iou * 100.0 if finite(iou) else None,
                delta_iou=(iou - baseline_iou) * 100.0
                if finite(iou) and finite(baseline_iou) else None,
                gt_pixels=int(baseline['gt'][class_idx]),
                pred_pixels=int(metric['pred'][class_idx]),
            ))

    class_lookup = {
        (row['variant'], row['class_index']): row for row in class_rows
    }
    prompt_rows = []
    for class_idx, class_name in enumerate(class_names):
        for prompt_idx, (variant, role) in enumerate(PROMPT_SLOTS):
            values = prompt_values[(class_idx, prompt_idx)]
            row = dict(
                class_index=class_idx,
                class_name=class_name,
                prompt_index=prompt_idx,
                prompt_role=role,
                prompt=(values['prompt'][0] if values['prompt'] else None),
                selected_by_presence_images=selected_counts[
                    (class_idx, prompt_idx)],
                selected_by_presence_ratio=safe_div(
                    selected_counts[(class_idx, prompt_idx)], len(records)),
                slot_class_iou=class_lookup[(variant, class_idx)]['iou'],
                slot_class_delta_iou=class_lookup[
                    (variant, class_idx)]['delta_iou'],
            )
            for key, numbers in values.items():
                if key != 'prompt':
                    row[key] = mean(numbers)
            prompt_rows.append(row)

    head_rows = []
    for (variant, class_idx), values in sorted(head_values.items()):
        row = dict(
            variant=variant,
            class_index=class_idx,
            class_name=class_names[class_idx],
        )
        for key, numbers in values.items():
            row[key] = mean(numbers)
        head_rows.append(row)

    e2e_rows = []
    for class_idx, values in sorted(e2e_values.items()):
        pre = mean(values['loss'])
        post = mean(values['post_update_loss'])
        e2e_rows.append(dict(
            class_index=class_idx,
            class_name=class_names[class_idx],
            updates=len(values['loss']),
            loss_before=pre,
            loss_after=post,
            loss_delta=(post - pre) if finite(pre) and finite(post) else None,
            grad_norm=mean(values['grad_norm']),
            weight_l1=mean(values['post_update_weight_l1']),
        ))

    controls = {
        key: max([float(value) for value in values if finite(value)] or [0.0])
        for key, values in control_values.items()
    }
    required_controls = {
        'baseline_reconstruction_max_abs',
        'literal_reground_final_max_abs',
        'literal_reground_semantic_max_abs',
        'literal_reground_semantic_raw_max_abs',
        'literal_reground_instance_max_abs',
        'literal_reground_presence_max_abs',
    }
    missing_controls = sorted(required_controls - set(controls))
    if missing_controls:
        raise ValueError(
            f'Missing required identity controls: {missing_controls}.')
    controls['literal_identity_passed'] = all(
        controls[key] <= tolerance for key in required_controls)
    correlations = [
        dict(left=left, right='slot_class_delta_iou',
             pearson=pearson(prompt_rows, left, 'slot_class_delta_iou'))
        for left in (
            'token_count', 'word_count', 'presence',
            'kept_candidate_count', 'semantic_mean', 'instance_mean')
    ]
    correlations_per_class = []
    for class_idx, class_name in enumerate(class_names):
        class_prompt_rows = [
            row for row in prompt_rows if row['class_index'] == class_idx
        ]
        for left in (
                'token_count', 'word_count', 'presence',
                'kept_candidate_count', 'semantic_mean', 'instance_mean'):
            correlations_per_class.append(dict(
                class_index=class_idx,
                class_name=class_name,
                left=left,
                right='slot_class_delta_iou',
                pearson=pearson(
                    class_prompt_rows, left, 'slot_class_delta_iou'),
            ))
    variant_lookup = {row['variant']: row for row in variant_rows}
    prompt_role_rows = []
    literal_class = {
        row['class_index']: row
        for row in class_rows if row['variant'] == 'prompt_0_literal'
    }
    for variant, role in PROMPT_SLOTS:
        rows = [row for row in class_rows if row['variant'] == variant]
        class_deltas_to_literal = [
            row['iou'] - literal_class[row['class_index']]['iou']
            for row in rows
            if finite(row['iou'])
            and finite(literal_class[row['class_index']]['iou'])
        ]
        prompt_role_rows.append(dict(
            variant=variant,
            prompt_role=role,
            miou=variant_lookup[variant]['miou'],
            delta_miou_to_baseline=variant_lookup[variant]['delta_miou'],
            mean_class_delta_iou_to_literal=mean(class_deltas_to_literal),
            classes_better_than_literal=sum(
                value > 0.0 for value in class_deltas_to_literal),
            classes_worse_than_literal=sum(
                value < 0.0 for value in class_deltas_to_literal),
        ))
    comparisons = []
    for name, left, right in (
            ('background_literal_control',
             'full_bg_literal_output', 'full_output'),
            ('presence_selection', 'presence_selected_output', 'baseline'),
            ('presence_weighting', 'presence_weighted_output', 'baseline'),
            ('e2e_weight_update', 'e2e_output', 'full_output'),
            ('e2e_with_literal_background',
             'e2e_bg_literal_output', 'full_bg_literal_output'),
            ('literal_reground_identity', 'literal_regrounded', 'baseline'),
            ('residual_010_effect',
             'anchor_residual_010_regrounded', 'literal_regrounded'),
            ('residual_025_effect',
             'anchor_residual_025_regrounded', 'literal_regrounded'),
            ('sequence_reground_vs_output',
             'full_sequence_regrounded', 'full_output')):
        comparisons.append(dict(
            comparison=name,
            left=left,
            right=right,
            delta_miou=(
                variant_lookup[left]['miou'] - variant_lookup[right]['miou']),
        ))
    best_prompt_role = max(prompt_role_rows, key=lambda row: row['miou'])
    output_candidates = [
        'pool_mean', 'pool_max', 'anchor_output', 'visual_output',
        'entropy_output', 'full_output', 'full_bg_literal_output',
        'presence_selected_output', 'presence_weighted_output',
        'e2e_output', 'e2e_bg_literal_output',
    ]
    best_output = max(
        (variant_lookup[name] for name in output_candidates),
        key=lambda row: row['miou'])
    residual_candidates = [
        'anchor_residual_010_regrounded',
        'anchor_residual_025_regrounded',
    ]
    best_residual = max(
        (variant_lookup[name] for name in residual_candidates),
        key=lambda row: row['miou'])
    e2e_loss_deltas = [
        row['loss_delta'] for row in e2e_rows if finite(row['loss_delta'])
    ]
    directional_evidence = dict(
        controls_passed=controls['literal_identity_passed'],
        best_prompt_role=dict(
            variant=best_prompt_role['variant'],
            role=best_prompt_role['prompt_role'],
            miou=best_prompt_role['miou'],
            delta_miou_to_baseline=best_prompt_role[
                'delta_miou_to_baseline'],
        ),
        best_independent_output=dict(
            variant=best_output['variant'],
            miou=best_output['miou'],
            delta_miou_to_baseline=best_output['delta_miou'],
        ),
        presence=dict(
            selected_delta_miou=variant_lookup[
                'presence_selected_output']['delta_miou'],
            weighted_delta_miou=variant_lookup[
                'presence_weighted_output']['delta_miou'],
        ),
        e2e=dict(
            output_delta_miou_to_full=(
                variant_lookup['e2e_output']['miou']
                - variant_lookup['full_output']['miou']),
            mean_post_minus_pre_objective=mean(e2e_loss_deltas),
        ),
        reground=dict(
            literal_identity_delta_miou=variant_lookup[
                'literal_regrounded']['delta_miou'],
            best_bounded_residual=best_residual['variant'],
            best_bounded_residual_delta_miou_to_literal=(
                best_residual['miou']
                - variant_lookup['literal_regrounded']['miou']),
            full_sequence_delta_miou_to_output=(
                variant_lookup['full_sequence_regrounded']['miou']
                - variant_lookup['full_output']['miou']),
        ),
    )
    return dict(
        class_names=class_names,
        variant_rows=variant_rows,
        class_rows=class_rows,
        prompt_rows=prompt_rows,
        head_rows=head_rows,
        e2e_rows=e2e_rows,
        controls=controls,
        correlations=correlations,
        correlations_per_class=correlations_per_class,
        prompt_role_rows=prompt_role_rows,
        comparisons=comparisons,
        directional_evidence=directional_evidence,
    )


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit('No prompt_atlas.rank*.jsonl inputs found.')
    expected_dataset = str(args.expected_dataset).lower()
    if expected_dataset == 'isaid':
        raise SystemExit('iSAID is excluded from project evaluation.')
    records = load_records(paths, expected_dataset)
    summary = summarize(records, args.integrity_tolerance)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'dataset_variants.csv'),
              summary['variant_rows'])
    write_csv(os.path.join(args.out_dir, 'per_class.csv'),
              summary['class_rows'])
    write_csv(os.path.join(args.out_dir, 'prompt_function.csv'),
              summary['prompt_rows'])
    write_csv(os.path.join(args.out_dir, 'head_path.csv'),
              summary['head_rows'])
    write_csv(os.path.join(args.out_dir, 'e2e_update.csv'),
              summary['e2e_rows'])
    write_csv(os.path.join(args.out_dir, 'comparisons.csv'),
              summary['comparisons'])
    write_csv(os.path.join(args.out_dir, 'correlations.csv'),
              summary['correlations'])
    write_csv(os.path.join(args.out_dir, 'correlations_per_class.csv'),
              summary['correlations_per_class'])
    write_csv(os.path.join(args.out_dir, 'prompt_role_summary.csv'),
              summary['prompt_role_rows'])
    payload = dict(
        schema_version=SCHEMA_VERSION,
        dataset=expected_dataset,
        source_files=paths,
        records=len(records),
        **summary,
    )
    with open(os.path.join(args.out_dir, 'summary.json'),
              'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'decision_report.md'),
              'w', encoding='utf-8') as handle:
        handle.write('# SAM3 prompt functional atlas\n\n')
        handle.write(
            f'- Dataset: `{expected_dataset}`; images: {len(records)}.\n')
        handle.write(
            '- Literal cached/re-ground identity passed: '
            f'`{summary["controls"]["literal_identity_passed"]}`.\n')
        handle.write('\n## Main comparisons\n\n')
        for row in summary['comparisons']:
            handle.write(
                f'- `{row["comparison"]}`: ΔmIoU '
                f'{row["delta_miou"]:.4f} '
                f'(`{row["left"]}` minus `{row["right"]}`).\n')
        handle.write('\n## Prompt-function correlations\n\n')
        for row in summary['correlations']:
            value = row['pearson']
            rendered = f'{value:.4f}' if finite(value) else 'N/A'
            handle.write(
                f'- `{row["left"]}` vs slot-class ΔIoU: {rendered}.\n')
        evidence = summary['directional_evidence']
        handle.write('\n## Direction-first evidence\n\n')
        handle.write(
            '- Best controlled prompt role: '
            f'`{evidence["best_prompt_role"]["role"]}` '
            f'(ΔmIoU vs baseline '
            f'{evidence["best_prompt_role"]["delta_miou_to_baseline"]:.4f}).\n')
        handle.write(
            '- Best independent-output path: '
            f'`{evidence["best_independent_output"]["variant"]}` '
            f'(ΔmIoU vs baseline '
            f'{evidence["best_independent_output"]["delta_miou_to_baseline"]:.4f}).\n')
        handle.write(
            '- E2E post-minus-pre objective: '
            f'{evidence["e2e"]["mean_post_minus_pre_objective"]!r}; '
            'compare its sign with the segmentation delta before retaining '
            'the objective.\n')
        handle.write(
            '- Best bounded re-ground residual ΔmIoU vs literal re-ground: '
            f'{evidence["reground"]["best_bounded_residual_delta_miou_to_literal"]:.4f}; '
            'full-sequence re-ground ΔmIoU vs output fusion: '
            f'{evidence["reground"]["full_sequence_delta_miou_to_output"]:.4f}.\n')
        handle.write(
            '\nThese are diagnostic associations, not prompt-selection '
            'rules. Ground-truth metrics never enter inference.\n')
    print(json.dumps(dict(
        out_dir=os.path.abspath(args.out_dir),
        dataset=expected_dataset,
        records=len(records),
        literal_identity_passed=(
            summary['controls']['literal_identity_passed']),
        comparisons=summary['comparisons'],
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
