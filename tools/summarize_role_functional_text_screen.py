#!/usr/bin/env python3
"""Summarize role-specific SAM3 text, combination, and residual diagnostics."""

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

from role_functional_text_definitions import (
    DEFAULT_SETTING,
    PROTOCOL,
    RESIDUAL_SETTINGS,
    ROLE_FIELDS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    combo_variant_name,
    reference_variant,
    sensitivity_variant_name,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def mean(values):
    values = [float(value) for value in values
              if isinstance(value, (int, float))
              and not isinstance(value, bool) and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def add_matrix(target, source):
    source = [[int(value) for value in row] for row in source]
    if target is None:
        return source
    return [[left + right for left, right in zip(left_row, right_row)]
            for left_row, right_row in zip(target, source)]


def confusion_metrics(matrix):
    count = len(matrix)
    tp = [float(matrix[index][index]) for index in range(count)]
    gt = [float(sum(row)) for row in matrix]
    pred = [float(sum(matrix[row][col] for row in range(count)))
            for col in range(count)]
    union = [gt[index] + pred[index] - tp[index]
             for index in range(count)]
    iou = [safe_div(tp[index], union[index]) for index in range(count)]
    valid = [value for value in iou if value is not None]
    return dict(
        iou=iou,
        gt=gt,
        pred=pred,
        miou=100.0 * mean(valid),
        aacc=100.0 * safe_div(sum(tp), sum(gt)),
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
    if 'isaid' in expected:
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
                    raise ValueError(
                        f'Unexpected schema at {path}:{line_number}.')
                if record.get('settings', {}).get('protocol') != PROTOCOL:
                    raise ValueError(
                        f'Wrong protocol at {path}:{line_number}.')
                dataset = str(record.get('dataset_name', '')).lower()
                if dataset not in expected:
                    raise ValueError(f'Unexpected dataset {dataset!r}.')
                if tuple(record.get('variants', {})) != VARIANT_NAMES:
                    raise ValueError(
                        f'Variant contract mismatch at {path}:{line_number}.')
                key = (dataset, str(record.get('img_path')))
                rank = int(record.get('rank', 0))
                if key in seen:
                    if rank in seen[key]:
                        raise ValueError(
                            f'Duplicate image record on rank {rank}: {key}.')
                    prior = next(iter(seen[key].values()))
                    for field in ('class_names', 'valid_pixels', 'variants'):
                        if prior.get(field) != record.get(field):
                            raise ValueError(
                                f'Inconsistent cross-rank padding duplicate '
                                f'for {key}: {field}.')
                    seen[key][rank] = record
                    duplicate_counts[dataset] += 1
                    continue
                seen[key] = {rank: record}
                records.append(record)
    if not records:
        raise ValueError('No role-functional text records were found.')
    observed = {str(record['dataset_name']).lower() for record in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing expected datasets: {missing}.')
    return records, missing, dict(duplicate_counts)


def binary_auc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else (0.5 if pos == neg else 0.0)
    return wins / (len(positive) * len(negative))


def summarize_dataset(dataset, records, tolerance):
    class_names = records[0]['class_names']
    if any(record['class_names'] != class_names for record in records):
        raise ValueError(f'Class order differs within {dataset}.')
    matrices = {name: None for name in VARIANT_NAMES}
    counters = {name: defaultdict(int) for name in VARIANT_NAMES}
    candidate_values = defaultdict(lambda: defaultdict(list))
    integrity = defaultdict(list)

    for record in records:
        for name in VARIANT_NAMES:
            stats = record['variants'][name]
            matrices[name] = add_matrix(
                matrices[name], stats['confusion']['matrix'])
            for key in ('changed_pixels', 'improved_pixels', 'harmed_pixels',
                        'wrong_to_wrong_pixels', 'help_minus_harm'):
                counters[name][key] += int(stats.get(key, 0))
        for view in record.get('views', []):
            for key in ('baseline_reconstruction_max_abs',
                        'no_update_identity_max_abs',
                        'native_recomposition_max_abs',
                        'diagnostic_cpu_bytes'):
                integrity[key].append(view.get(key))
            for row in view.get('candidate_rows', []):
                key = (int(row['class_index']), str(row['role']),
                       int(row['slot']))
                for field, value in row.items():
                    candidate_values[key][field].append(value)

    metrics = {name: confusion_metrics(matrix)
               for name, matrix in matrices.items()}
    if matrices['combo_p0_s0_i0'] != matrices['baseline']:
        raise ValueError(
            f'{dataset}: full-image no-update predictions differ from the '
            'official baseline.')
    valid_pixels = int(sum(sum(row) for row in matrices['baseline']))
    variant_rows = []
    class_rows = []
    for name in VARIANT_NAMES:
        reference = reference_variant(name)
        metric = metrics[name]
        ref_metric = metrics[reference]
        official = metrics['baseline']
        changed = counters[name]['changed_pixels']
        variant_rows.append(dict(
            dataset=dataset,
            variant=name,
            reference_variant=reference,
            images=len(records),
            miou=metric['miou'],
            delta_miou_to_reference=metric['miou'] - ref_metric['miou'],
            delta_miou_to_official=metric['miou'] - official['miou'],
            aacc=metric['aacc'],
            changed_ratio_to_reference=safe_div(changed, valid_pixels),
            improved_pixels=counters[name]['improved_pixels'],
            harmed_pixels=counters[name]['harmed_pixels'],
            help_minus_harm=counters[name]['help_minus_harm'],
        ))
        for class_index, class_name in enumerate(class_names):
            value = metric['iou'][class_index]
            ref_value = ref_metric['iou'][class_index]
            official_value = official['iou'][class_index]
            class_rows.append(dict(
                dataset=dataset,
                variant=name,
                reference_variant=reference,
                class_index=class_index,
                class_name=class_name,
                iou=None if value is None else 100.0 * value,
                delta_iou_to_reference=(
                    None if value is None or ref_value is None
                    else 100.0 * (value - ref_value)),
                delta_iou_to_official=(
                    None if value is None or official_value is None
                    else 100.0 * (value - official_value)),
                gt_pixels=int(official['gt'][class_index]),
                pred_pixels=int(metric['pred'][class_index]),
            ))

    candidate_rows = []
    presence_groups = defaultdict(lambda: dict(labels=[], anchor=[], candidate=[]))
    for (class_index, role, slot), fields in sorted(candidate_values.items()):
        row = dict(
            dataset=dataset,
            class_index=class_index,
            class_name=class_names[class_index],
            role=role,
            slot=slot,
        )
        for field in (
                'candidate_prompt', 'token_count', 'presence_delta',
                'semantic_abs_delta_mean', 'instance_abs_delta_mean',
                'anchor_kept_count',
                'candidate_kept_count_at_anchor_presence',
                'raw_candidate_count', 'raw_object_score_mean'):
            values = fields.get(field, [])
            row[field] = (
                values[0] if field == 'candidate_prompt' and values
                else mean(values))
        candidate_rows.append(row)
        if role == 'presence':
            group = presence_groups[(class_index, slot)]
            group['labels'].extend(bool(value)
                                   for value in fields.get('gt_present', []))
            group['anchor'].extend(float(value)
                                   for value in fields.get('anchor_presence', []))
            group['candidate'].extend(float(value) for value in
                                      fields.get('candidate_presence', []))

    presence_rows = []
    for (class_index, slot), values in sorted(presence_groups.items()):
        labels = values['labels']
        anchor = values['anchor']
        candidate = values['candidate']
        count = min(len(labels), len(anchor), len(candidate))
        labels, anchor, candidate = labels[:count], anchor[:count], candidate[:count]
        label_float = [1.0 if value else 0.0 for value in labels]
        anchor_brier = mean([(score - label) ** 2
                             for score, label in zip(anchor, label_float)])
        candidate_brier = mean([(score - label) ** 2
                                for score, label in zip(candidate, label_float)])
        anchor_auc = binary_auc(labels, anchor)
        candidate_auc = binary_auc(labels, candidate)
        presence_rows.append(dict(
            dataset=dataset,
            class_index=class_index,
            class_name=class_names[class_index],
            slot=slot,
            samples=count,
            positive_samples=sum(labels),
            anchor_auc=anchor_auc,
            candidate_auc=candidate_auc,
            delta_auc=(
                None if anchor_auc is None or candidate_auc is None
                else candidate_auc - anchor_auc),
            anchor_brier=anchor_brier,
            candidate_brier=candidate_brier,
            delta_brier=(
                None if anchor_brier is None or candidate_brier is None
                else candidate_brier - anchor_brier),
            anchor_accuracy=mean([
                float((score >= 0.5) == label)
                for score, label in zip(anchor, labels)]),
            candidate_accuracy=mean([
                float((score >= 0.5) == label)
                for score, label in zip(candidate, labels)]),
        ))

    maxima = {
        key: max([float(value) for value in values
                  if isinstance(value, (int, float))] or [0.0])
        for key, values in integrity.items()
    }
    for key in ('baseline_reconstruction_max_abs',
                'no_update_identity_max_abs',
                'native_recomposition_max_abs'):
        if maxima.get(key, 0.0) > tolerance:
            raise ValueError(
                f'{dataset}: {key}={maxima[key]} exceeds {tolerance}.')
    controls = dict(
        dataset=dataset,
        images=len(records),
        **maxima,
        peak_diagnostic_cpu_mb=(
            max(integrity.get('diagnostic_cpu_bytes') or [0])
            / (1024.0 * 1024.0)),
    )
    return (variant_rows, class_rows, candidate_rows, presence_rows,
            controls)


def build_sensitivity_rows(variant_rows):
    lookup = {(row['dataset'], row['variant']): row for row in variant_rows}
    output = []
    for dataset in sorted({row['dataset'] for row in variant_rows}):
        for role, _, count in ROLE_FIELDS:
            for slot in range(1, count + 1):
                deltas = []
                for setting, alpha, clip in RESIDUAL_SETTINGS:
                    if setting == DEFAULT_SETTING:
                        indices = dict(presence=0, semantic=0, instance=0)
                        indices[role] = slot
                        variant = combo_variant_name(
                            indices['presence'], indices['semantic'],
                            indices['instance'])
                    else:
                        variant = sensitivity_variant_name(role, slot, setting)
                    row = lookup[(dataset, variant)]
                    delta = float(row['delta_miou_to_official'])
                    deltas.append(delta)
                    output.append(dict(
                        dataset=dataset,
                        role=role,
                        slot=slot,
                        setting=setting,
                        alpha=alpha,
                        clip=clip,
                        variant=variant,
                        miou=row['miou'],
                        delta_miou=delta,
                    ))
                status = (
                    'stable_positive' if min(deltas) > 0.0
                    else 'unsupported_in_grid' if max(deltas) <= 0.0
                    else 'partial_or_sensitive')
                for row in output[-len(RESIDUAL_SETTINGS):]:
                    row.update(dict(
                        status=status,
                        positive_setting_count=sum(value > 0.0
                                                   for value in deltas),
                        best_delta_miou=max(deltas),
                        worst_delta_miou=min(deltas),
                        delta_spread=max(deltas) - min(deltas),
                    ))
    return output


def build_interaction_rows(variant_rows):
    lookup = {(row['dataset'], row['variant']): row for row in variant_rows}
    output = []
    for dataset in sorted({row['dataset'] for row in variant_rows}):
        for p in range(3):
            for s in range(4):
                for i in range(3):
                    if sum(value > 0 for value in (p, s, i)) < 2:
                        continue
                    name = combo_variant_name(p, s, i)
                    delta = lookup[(dataset, name)]['delta_miou_to_official']
                    additive = 0.0
                    if p:
                        additive += lookup[(
                            dataset, combo_variant_name(p, 0, 0)
                        )]['delta_miou_to_official']
                    if s:
                        additive += lookup[(
                            dataset, combo_variant_name(0, s, 0)
                        )]['delta_miou_to_official']
                    if i:
                        additive += lookup[(
                            dataset, combo_variant_name(0, 0, i)
                        )]['delta_miou_to_official']
                    output.append(dict(
                        dataset=dataset,
                        variant=name,
                        presence_slot=p,
                        semantic_slot=s,
                        instance_slot=i,
                        delta_miou=delta,
                        additive_single_head_delta=additive,
                        interaction_delta=delta - additive,
                        cancels_positive_singles=(
                            additive > 0.0 and delta < additive),
                    ))
    return output


def write_markdown(path, records, missing, duplicate_counts, variant_rows,
                   sensitivity_rows, controls):
    datasets = sorted({row['dataset'] for row in variant_rows})
    lines = [
        '# Role-functional text screen summary', '',
        f'- Unique images: {len(records)}',
        f'- Missing datasets: {missing or "none"}',
        f'- DDP padding duplicates removed: {duplicate_counts or "none"}',
        '', '## Dataset results', '',
        '| Dataset | Official mIoU | Best fixed combination | Delta | '
        'Best shared-text control | Delta |',
        '|---|---:|---|---:|---|---:|',
    ]
    for dataset in datasets:
        rows = [row for row in variant_rows if row['dataset'] == dataset]
        official = next(row for row in rows if row['variant'] == 'baseline')
        combinations = [row for row in rows
                        if row['variant'].startswith('combo_')]
        shared = [row for row in rows
                  if row['variant'].startswith('shared_')]
        best_combo = max(combinations,
                         key=lambda row: row['delta_miou_to_official'])
        best_shared = max(shared,
                          key=lambda row: row['delta_miou_to_official'])
        lines.append(
            f"| {dataset} | {official['miou']:.3f} | "
            f"{best_combo['variant']} | "
            f"{best_combo['delta_miou_to_official']:+.3f} | "
            f"{best_shared['variant']} | "
            f"{best_shared['delta_miou_to_official']:+.3f} |")
    lines.extend(['', '## Residual stability', '',
                  '| Dataset | Role/slot | Status | Positive settings | '
                  'Best delta | Worst delta | Spread |',
                  '|---|---|---|---:|---:|---:|---:|'])
    groups = {}
    for row in sensitivity_rows:
        groups[(row['dataset'], row['role'], row['slot'])] = row
    for (dataset, role, slot), row in sorted(groups.items()):
        lines.append(
            f"| {dataset} | {role}/{slot} | {row['status']} | "
            f"{row['positive_setting_count']}/{len(RESIDUAL_SETTINGS)} | "
            f"{row['best_delta_miou']:+.3f} | "
            f"{row['worst_delta_miou']:+.3f} | "
            f"{row['delta_spread']:.3f} |")
    lines.extend(['', '## Integrity and memory', '',
                  '| Dataset | Cached/native max error | No-update max error | '
                  'Recomposition max error | Peak diagnostic CPU MB |',
                  '|---|---:|---:|---:|---:|'])
    for row in controls:
        lines.append(
            f"| {row['dataset']} | "
            f"{row.get('baseline_reconstruction_max_abs', 0.0):.3g} | "
            f"{row.get('no_update_identity_max_abs', 0.0):.3g} | "
            f"{row.get('native_recomposition_max_abs', 0.0):.3g} | "
            f"{row.get('peak_diagnostic_cpu_mb', 0.0):.1f} |")
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    records, missing, duplicate_counts = load_records(
        paths, args.expected_datasets, args.allow_incomplete)
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)

    variant_rows, class_rows = [], []
    candidate_rows, presence_rows, controls = [], [], []
    for dataset in sorted(grouped):
        result = summarize_dataset(
            dataset, grouped[dataset], args.integrity_tolerance)
        variant_rows.extend(result[0])
        class_rows.extend(result[1])
        candidate_rows.extend(result[2])
        presence_rows.extend(result[3])
        controls.append(result[4])
    sensitivity_rows = build_sensitivity_rows(variant_rows)
    interaction_rows = build_interaction_rows(variant_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'dataset_variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'class_variants.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'candidate_responses.csv'), candidate_rows)
    write_csv(os.path.join(args.out_dir, 'presence_scores.csv'), presence_rows)
    write_csv(os.path.join(args.out_dir, 'residual_sensitivity.csv'),
              sensitivity_rows)
    write_csv(os.path.join(args.out_dir, 'combination_interactions.csv'),
              interaction_rows)
    write_csv(os.path.join(args.out_dir, 'integrity_and_memory.csv'), controls)
    payload = dict(
        protocol=PROTOCOL,
        records=len(records),
        missing_datasets=missing,
        duplicate_counts=duplicate_counts,
        dataset_variants=variant_rows,
        class_variants=class_rows,
        candidate_responses=candidate_rows,
        presence_scores=presence_rows,
        residual_sensitivity=sensitivity_rows,
        combination_interactions=interaction_rows,
        controls=controls,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(
        os.path.join(args.out_dir, 'summary.md'), records, missing,
        duplicate_counts, variant_rows, sensitivity_rows, controls)


if __name__ == '__main__':
    main()
