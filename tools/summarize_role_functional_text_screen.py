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
    COMPLETION_PROTOCOL,
    COMPLETION_SCHEMA_VERSION,
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
                        'completion_noop_max_abs',
                        'native_recomposition_max_abs',
                        'diagnostic_cpu_bytes'):
                integrity[key].append(view.get(key))
            for row in view.get('candidate_rows', []):
                key = (int(row['class_index']), str(row['role']),
                       int(row['slot']))
                for field, value in row.items():
                    candidate_values[key][field].append(value)

    # ``combo_p0_s0_i0`` is the defined no-update control, not an independent
    # method output.  In legacy sliding-window records, class aliases were
    # re-anchored as ``baseline + variant - crop_anchor`` before argmax.  The
    # algebraic zero can leave tiny floating-point cancellation at exact class
    # ties even when every crop passes the numerical identity checks below.
    # Canonicalize this control to the protected baseline for aggregation; the
    # cached/native, per-crop no-update and recomposition tolerances remain the
    # guards against an actual inference-path mismatch.
    matrices['combo_p0_s0_i0'] = [
        list(row) for row in matrices['baseline']]
    counters['combo_p0_s0_i0'] = defaultdict(
        int, counters['baseline'])
    metrics = {name: confusion_metrics(matrix)
               for name, matrix in matrices.items()}
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
                'completion_noop_max_abs',
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


def build_completion_rows(records, variant_rows):
    """Aggregate native versus anchor-admission for requested combinations."""
    grouped = defaultdict(list)
    for record in records:
        completion = record.get('role_text_completion')
        if completion is not None:
            grouped[str(record['dataset_name']).lower()].append(completion)
    if not grouped:
        return [], []

    official = {
        row['dataset']: row for row in variant_rows
        if row['variant'] == 'baseline'
    }
    rows, class_rows = [], []
    for dataset, payloads in sorted(grouped.items()):
        first = payloads[0]
        if (int(first.get('schema_version', -1)) != COMPLETION_SCHEMA_VERSION
                or first.get('protocol') != COMPLETION_PROTOCOL):
            raise ValueError(f'{dataset}: completion protocol/schema mismatch.')
        selected = tuple(first['selected_combination'])
        targets = {tuple(value) for value in first.get(
            'target_combinations', [])}
        names = tuple(first.get('variants', {}))
        native_matrices = {name: None for name in names}
        admission_matrices = {name: None for name in names}
        counters = {name: defaultdict(int) for name in names}
        slots = {}
        for payload in payloads:
            if (tuple(payload.get('selected_combination', ())) != selected
                    or {tuple(value) for value in payload.get(
                        'target_combinations', [])} != targets
                    or tuple(payload.get('variants', {})) != names):
                raise ValueError(
                    f'{dataset}: completion contract changed within a run.')
            for name, values in payload['variants'].items():
                slots[name] = tuple(values['slots'])
                native_matrices[name] = add_matrix(
                    native_matrices[name],
                    values['native_confusion']['matrix'])
                admission_matrices[name] = add_matrix(
                    admission_matrices[name],
                    values['anchor_admission_confusion']['matrix'])
                for field in ('changed_pixels', 'improved_pixels',
                              'harmed_pixels', 'help_minus_harm'):
                    counters[name][field] += int(values.get(field, 0))

        for name in names:
            native = confusion_metrics(native_matrices[name])
            admission = confusion_metrics(admission_matrices[name])
            p_slot, s_slot, i_slot = slots[name]
            base_miou = float(official[dataset]['miou'])
            rows.append(dict(
                dataset=dataset,
                variant=name,
                presence_slot=p_slot,
                semantic_slot=s_slot,
                instance_slot=i_slot,
                is_current_selected=(slots[name] == selected),
                is_target_completion=(slots[name] in targets),
                is_all_nonzero=all(value > 0 for value in slots[name]),
                images=len(payloads),
                official_miou=base_miou,
                native_miou=native['miou'],
                native_delta_to_official=native['miou'] - base_miou,
                anchor_admission_miou=admission['miou'],
                anchor_admission_delta_to_official=(
                    admission['miou'] - base_miou),
                anchor_admission_delta_to_native=(
                    admission['miou'] - native['miou']),
                changed_pixels=counters[name]['changed_pixels'],
                improved_pixels=counters[name]['improved_pixels'],
                harmed_pixels=counters[name]['harmed_pixels'],
                help_minus_harm=counters[name]['help_minus_harm'],
            ))
            for class_index, class_name in enumerate(
                    next(record['class_names'] for record in records
                         if str(record['dataset_name']).lower() == dataset)):
                native_iou = native['iou'][class_index]
                admission_iou = admission['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    presence_slot=p_slot,
                    semantic_slot=s_slot,
                    instance_slot=i_slot,
                    is_current_selected=(slots[name] == selected),
                    class_index=class_index,
                    class_name=class_name,
                    native_iou=(None if native_iou is None
                                else 100.0 * native_iou),
                    anchor_admission_iou=(
                        None if admission_iou is None
                        else 100.0 * admission_iou),
                    anchor_admission_delta_iou=(
                        None if native_iou is None or admission_iou is None
                        else 100.0 * (admission_iou - native_iou)),
                ))
    return rows, class_rows


def write_completion_markdown(path, rows):
    datasets = sorted({row['dataset'] for row in rows})
    lines = [
        '# Role-text completion and anchor-admission screen', '',
        '## Current selected combination', '',
        '| Dataset | Combination | Native mIoU | Anchor-admission mIoU | '
        'Admission delta |',
        '|---|---|---:|---:|---:|',
    ]
    for dataset in datasets:
        row = next(item for item in rows
                   if item['dataset'] == dataset
                   and item['is_current_selected'])
        slots = (row['presence_slot'], row['semantic_slot'],
                 row['instance_slot'])
        lines.append(
            f"| {dataset} | P{slots[0]}+S{slots[1]}+I{slots[2]} | "
            f"{row['native_miou']:.3f} | "
            f"{row['anchor_admission_miou']:.3f} | "
            f"{row['anchor_admission_delta_to_native']:+.3f} |")
    lines.extend([
        '', '## Best all-nonzero completion candidate', '',
        '| Dataset | Best native | Delta to official | Best with anchor '
        'admission | Delta to official | Target vs current best |',
        '|---|---|---:|---|---:|---:|',
    ])
    for dataset in datasets:
        candidates = [row for row in rows
                      if row['dataset'] == dataset
                      and row['is_all_nonzero']]
        if not candidates:
            continue
        native = max(candidates, key=lambda row: row['native_miou'])
        admission = max(
            candidates, key=lambda row: row['anchor_admission_miou'])
        selected = next(row for row in rows
                        if row['dataset'] == dataset
                        and row['is_current_selected'])
        targets = [row for row in candidates
                   if row['is_target_completion']]
        target_best = max(
            [value for row in targets for value in (
                row['native_miou'], row['anchor_admission_miou'])],
            default=None)

        def label(row):
            return (f"P{row['presence_slot']}+S{row['semantic_slot']}+"
                    f"I{row['instance_slot']}")

        lines.append(
            f"| {dataset} | {label(native)} "
            f"({native['native_miou']:.3f}) | "
            f"{native['native_delta_to_official']:+.3f} | "
            f"{label(admission)} "
            f"({admission['anchor_admission_miou']:.3f}) | "
            f"{admission['anchor_admission_delta_to_official']:+.3f} | "
            + ("n/a |" if target_best is None else
               f"{target_best - selected['native_miou']:+.3f} |"))
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _factorial_components(value_at, presence_slot, semantic_slot,
                          instance_slot):
    f000 = float(value_at(0, 0, 0))
    fp = float(value_at(presence_slot, 0, 0))
    fs = float(value_at(0, semantic_slot, 0))
    fi = float(value_at(0, 0, instance_slot))
    fps = float(value_at(presence_slot, semantic_slot, 0))
    fpi = float(value_at(presence_slot, 0, instance_slot))
    fsi = float(value_at(0, semantic_slot, instance_slot))
    fpsi = float(value_at(
        presence_slot, semantic_slot, instance_slot))
    main_p = fp - f000
    main_s = fs - f000
    main_i = fi - f000
    gamma_ps = fps - fp - fs + f000
    gamma_pi = fpi - fp - fi + f000
    gamma_si = fsi - fs - fi + f000
    gamma_psi = (
        fpsi - fps - fpi - fsi + fp + fs + fi - f000)
    delta = fpsi - f000
    total_interaction = delta - main_p - main_s - main_i
    reconstruction = (
        main_p + main_s + main_i + gamma_ps + gamma_pi + gamma_si
        + gamma_psi)
    return dict(
        baseline=f000,
        full=fpsi,
        delta=delta,
        main_p=main_p,
        main_s=main_s,
        main_i=main_i,
        gamma_ps=gamma_ps,
        gamma_pi=gamma_pi,
        gamma_si=gamma_si,
        gamma_psi=gamma_psi,
        total_interaction=total_interaction,
        reconstruction_error=delta - reconstruction,
    )


def _add_factorial_fields(row, prefix, components):
    row.update({
        f'baseline_{prefix}': components['baseline'],
        f'full_{prefix}': components['full'],
        f'delta_{prefix}': components['delta'],
        f'main_p_{prefix}': components['main_p'],
        f'main_s_{prefix}': components['main_s'],
        f'main_i_{prefix}': components['main_i'],
        f'gamma_ps_{prefix}': components['gamma_ps'],
        f'gamma_pi_{prefix}': components['gamma_pi'],
        f'gamma_si_{prefix}': components['gamma_si'],
        f'gamma_psi_{prefix}': components['gamma_psi'],
        f'total_interaction_{prefix}': components['total_interaction'],
        f'reconstruction_error_{prefix}': components[
            'reconstruction_error'],
    })


def _interaction_summary_row(dataset, interaction, values):
    values = [float(value) for value in values]
    return dict(
        dataset=dataset,
        interaction=interaction,
        combinations=len(values),
        mean_gamma=mean(values),
        mean_abs_gamma=mean([abs(value) for value in values]),
        positive_count=sum(value > 0.0 for value in values),
        negative_count=sum(value < 0.0 for value in values),
        zero_count=sum(value == 0.0 for value in values),
        max_gamma=max(values),
        min_gamma=min(values),
    )


def build_factorial_diagnosis(variant_rows, class_rows):
    variant_lookup = {
        (row['dataset'], row['variant']): row for row in variant_rows}
    class_lookup = {
        (row['dataset'], int(row['class_index']), row['variant']): row
        for row in class_rows
    }
    datasets = sorted({row['dataset'] for row in variant_rows})
    factorial_rows = []
    class_factorial_rows = []
    for dataset in datasets:
        def dataset_value(field):
            return lambda p, s, i: variant_lookup[(
                dataset, combo_variant_name(p, s, i))][field]

        for presence_slot in range(1, 3):
            for semantic_slot in range(1, 4):
                for instance_slot in range(1, 3):
                    row = dict(
                        dataset=dataset,
                        presence_slot=presence_slot,
                        semantic_slot=semantic_slot,
                        instance_slot=instance_slot,
                        variant=combo_variant_name(
                            presence_slot, semantic_slot, instance_slot),
                    )
                    for field, prefix in (
                            ('miou', 'miou'),
                            ('aacc', 'aacc'),
                            ('improved_pixels', 'improved_pixels'),
                            ('harmed_pixels', 'harmed_pixels'),
                            ('help_minus_harm', 'net_pixels')):
                        components = _factorial_components(
                            dataset_value(field), presence_slot,
                            semantic_slot, instance_slot)
                        _add_factorial_fields(row, prefix, components)
                    factorial_rows.append(row)

        class_indices = sorted({
            int(row['class_index']) for row in class_rows
            if row['dataset'] == dataset})
        for class_index in class_indices:
            baseline = class_lookup[(
                dataset, class_index, 'baseline')]

            def class_value(p, s, i):
                value = class_lookup[(
                    dataset, class_index,
                    combo_variant_name(p, s, i))]['iou']
                if value is None:
                    raise ValueError(
                        f'{dataset}/{class_index}: undefined class IoU.')
                return value

            for presence_slot in range(1, 3):
                for semantic_slot in range(1, 4):
                    for instance_slot in range(1, 3):
                        components = _factorial_components(
                            class_value, presence_slot, semantic_slot,
                            instance_slot)
                        row = dict(
                            dataset=dataset,
                            class_index=class_index,
                            class_name=baseline['class_name'],
                            presence_slot=presence_slot,
                            semantic_slot=semantic_slot,
                            instance_slot=instance_slot,
                            variant=combo_variant_name(
                                presence_slot, semantic_slot, instance_slot),
                        )
                        _add_factorial_fields(row, 'iou', components)
                        class_factorial_rows.append(row)

    # Pairwise terms are independent of the third active slot. Collapse the
    # repeated values so summary counts reflect unique factorial contrasts.
    summary_rows = []
    macro = defaultdict(list)
    for dataset in datasets:
        rows = [row for row in factorial_rows
                if row['dataset'] == dataset]
        unique = {
            'ps': {(row['presence_slot'], row['semantic_slot']):
                   row['gamma_ps_miou'] for row in rows},
            'pi': {(row['presence_slot'], row['instance_slot']):
                   row['gamma_pi_miou'] for row in rows},
            'si': {(row['semantic_slot'], row['instance_slot']):
                   row['gamma_si_miou'] for row in rows},
            'psi': {(row['presence_slot'], row['semantic_slot'],
                     row['instance_slot']): row['gamma_psi_miou']
                    for row in rows},
            'total': {(row['presence_slot'], row['semantic_slot'],
                       row['instance_slot']):
                      row['total_interaction_miou'] for row in rows},
        }
        for interaction, mapping in unique.items():
            values = list(mapping.values())
            summary_rows.append(_interaction_summary_row(
                dataset, interaction, values))
            macro[interaction].extend(values)
    for interaction, values in sorted(macro.items()):
        summary_rows.append(_interaction_summary_row(
            'macro', interaction, values))
    return factorial_rows, class_factorial_rows, summary_rows


def write_interaction_markdown(path, factorial_rows, summary_rows):
    datasets = sorted({row['dataset'] for row in factorial_rows})
    lines = [
        '# Native-fusion-aware role interaction diagnosis', '',
        '- Source: the existing 36 fixed P/S/I combinations; no new SAM3 '
        'forward.',
        '- Terms are counterfactual performance interactions, not feature-level '
        'causal decompositions.',
        '- Existing records support aggregate mIoU/class-IoU and corrected/'
        'harmed-pixel counts. They do not contain per-pixel S/I winner maps or '
        'per-query admission identities.',
        '', '## Best all-role combination per dataset', '',
        '| Dataset | Variant | Delta mIoU | Main-effect sum | Total interaction '
        '| Gamma PS | Gamma PI | Gamma SI | Gamma PSI |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in datasets:
        rows = [row for row in factorial_rows
                if row['dataset'] == dataset]
        best = max(rows, key=lambda row: row['delta_miou'])
        main_sum = (best['main_p_miou'] + best['main_s_miou']
                    + best['main_i_miou'])
        lines.append(
            f"| {dataset} | {best['variant']} | "
            f"{best['delta_miou']:+.3f} | {main_sum:+.3f} | "
            f"{best['total_interaction_miou']:+.3f} | "
            f"{best['gamma_ps_miou']:+.3f} | "
            f"{best['gamma_pi_miou']:+.3f} | "
            f"{best['gamma_si_miou']:+.3f} | "
            f"{best['gamma_psi_miou']:+.3f} |")
    lines.extend([
        '', '## Mean absolute interaction by dataset', '',
        '| Dataset | PS | PI | SI | PSI | Total non-additivity |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    lookup = {(row['dataset'], row['interaction']): row
              for row in summary_rows}
    for dataset in datasets + ['macro']:
        values = [lookup[(dataset, name)]['mean_abs_gamma']
                  for name in ('ps', 'pi', 'si', 'psi', 'total')]
        lines.append(
            f'| {dataset} | ' + ' | '.join(
                f'{value:.3f}' for value in values) + ' |')
    lines.extend([
        '', '## Maximum and minimum all-role interaction', '',
        '| Dataset | Maximum-interaction variant | Interaction | '
        'Minimum-interaction variant | '
        'Interaction |',
        '|---|---|---:|---|---:|',
    ])
    for dataset in datasets:
        rows = [row for row in factorial_rows
                if row['dataset'] == dataset]
        high = max(rows, key=lambda row: row['total_interaction_miou'])
        low = min(rows, key=lambda row: row['total_interaction_miou'])
        lines.append(
            f"| {dataset} | {high['variant']} | "
            f"{high['total_interaction_miou']:+.3f} | "
            f"{low['variant']} | {low['total_interaction_miou']:+.3f} |")
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


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
    factorial_rows, class_factorial_rows, interaction_summary_rows = (
        build_factorial_diagnosis(variant_rows, class_rows))
    completion_rows, completion_class_rows = build_completion_rows(
        records, variant_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'dataset_variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'class_variants.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'candidate_responses.csv'), candidate_rows)
    write_csv(os.path.join(args.out_dir, 'presence_scores.csv'), presence_rows)
    write_csv(os.path.join(args.out_dir, 'residual_sensitivity.csv'),
              sensitivity_rows)
    write_csv(os.path.join(args.out_dir, 'combination_interactions.csv'),
              interaction_rows)
    write_csv(os.path.join(args.out_dir, 'factorial_interactions.csv'),
              factorial_rows)
    write_csv(os.path.join(args.out_dir, 'factorial_class_interactions.csv'),
              class_factorial_rows)
    write_csv(os.path.join(args.out_dir, 'role_interaction_summary.csv'),
              interaction_summary_rows)
    write_csv(os.path.join(args.out_dir, 'integrity_and_memory.csv'), controls)
    write_csv(os.path.join(args.out_dir, 'completion_variants.csv'),
              completion_rows)
    write_csv(os.path.join(args.out_dir, 'completion_class_variants.csv'),
              completion_class_rows)
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
        factorial_interactions=factorial_rows,
        factorial_class_interactions=class_factorial_rows,
        role_interaction_summary=interaction_summary_rows,
        controls=controls,
        completion_variants=completion_rows,
        completion_class_variants=completion_class_rows,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(
        os.path.join(args.out_dir, 'summary.md'), records, missing,
        duplicate_counts, variant_rows, sensitivity_rows, controls)
    write_interaction_markdown(
        os.path.join(args.out_dir, 'interaction_diagnosis.md'),
        factorial_rows, interaction_summary_rows)
    if completion_rows:
        write_completion_markdown(
            os.path.join(args.out_dir, 'completion_summary.md'),
            completion_rows)


if __name__ == '__main__':
    main()
