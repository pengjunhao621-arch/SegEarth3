#!/usr/bin/env python3
"""Summarize PI role-compatibility replays from role-functional JSONL logs."""

import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    PI_PROTOCOL,
    PI_SCHEMA_VERSION,
    PI_VARIANT_NAMES,
)
from tools.summarize_role_functional_text_screen import (
    DATASETS,
    add_matrix,
    confusion_metrics,
    expand_inputs,
    load_records,
    safe_div,
    write_csv,
)


COUNT_FIELDS = (
    'class_flip_pixels',
    'improved_pixels',
    'harmed_pixels',
    'wrong_to_wrong_pixels',
    'any_head_switch_pixels',
    'class_flip_with_head_switch',
    'class_flip_without_head_switch',
    'improved_with_head_switch',
    'improved_without_head_switch',
    'harmed_with_head_switch',
    'harmed_without_head_switch',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+',
                        default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def _weighted_mean(total, weight):
    return total / weight if weight else None


def summarize_dataset(dataset, records):
    class_names = records[0]['class_names']
    matrices = {name: None for name in PI_VARIANT_NAMES}
    variant_counts = {
        name: defaultdict(int) for name in PI_VARIANT_NAMES}
    mechanism = defaultdict(int)
    margin_sums = defaultdict(float)
    margin_weights = defaultdict(int)
    class_counts = defaultdict(lambda: defaultdict(int))
    query_counts = defaultdict(lambda: defaultdict(int))
    query_margins = defaultdict(lambda: dict(p0=None, p1=None))
    query_prompts = {}
    slots = set()

    for record in records:
        pi = record.get('pi_role_compatibility')
        if not isinstance(pi, dict):
            raise ValueError(
                f'{dataset}: record lacks PI role-compatibility output.')
        if (int(pi.get('schema_version', -1)) != PI_SCHEMA_VERSION
                or pi.get('protocol') != PI_PROTOCOL):
            raise ValueError(f'{dataset}: PI protocol/schema mismatch.')
        if tuple(pi.get('variants', {})) != PI_VARIANT_NAMES:
            raise ValueError(f'{dataset}: PI variant contract mismatch.')
        slots.add((int(pi['presence_slot']), int(pi['instance_slot'])))

        for name in PI_VARIANT_NAMES:
            stats = pi['variants'][name]
            matrices[name] = add_matrix(
                matrices[name], stats['confusion']['matrix'])
            for field in ('changed_pixels', 'improved_pixels',
                          'harmed_pixels', 'help_minus_harm'):
                variant_counts[name][field] += int(stats.get(field, 0))

        values = pi['mechanism']
        for field in COUNT_FIELDS:
            mechanism[field] += int(values.get(field, 0))
        for field in ('mean_gt_margin_delta',
                      'corrected_gt_margin_delta',
                      'harmed_gt_margin_delta'):
            value = values.get(field)
            if value is not None:
                if field == 'mean_gt_margin_delta':
                    weight = int(record['valid_pixels'])
                elif field == 'corrected_gt_margin_delta':
                    weight = int(values.get('improved_pixels', 0))
                else:
                    weight = int(values.get('harmed_pixels', 0))
                margin_sums[field] += float(value) * weight
                margin_weights[field] += weight
        for row in values.get('class_rows', []):
            key = int(row['class_index'])
            for field, value in row.items():
                if field.endswith('_pixels'):
                    class_counts[key][field] += int(value)

        for view in record.get('views', []):
            view_pi = view.get('pi_diagnosis')
            if not isinstance(view_pi, dict):
                continue
            for row in view_pi.get('query_rows', []):
                key = int(row['class_index'])
                query_prompts.setdefault(key, (
                    row.get('presence_prompt'), row.get('instance_prompt')))
                for field in (
                        'candidate_query_count', 'admitted_p0', 'admitted_p1',
                        'admission_added', 'admission_removed',
                        'admission_unchanged', 'anchor_admitted_p0'):
                    query_counts[key][field] += int(row.get(field, 0))
                for source, target in (
                        ('min_abs_margin_p0', 'p0'),
                        ('min_abs_margin_p1', 'p1')):
                    value = row.get(source)
                    if value is not None:
                        current = query_margins[key][target]
                        query_margins[key][target] = (
                            float(value) if current is None
                            else min(current, float(value)))

    if len(slots) != 1:
        raise ValueError(f'{dataset}: inconsistent PI slots: {slots}.')
    presence_slot, instance_slot = next(iter(slots))
    metrics = {
        name: confusion_metrics(matrix) for name, matrix in matrices.items()
    }
    base = metrics['pi_native_p0_i0']
    variant_rows = []
    class_rows = []
    for name in PI_VARIANT_NAMES:
        metric = metrics[name]
        variant_rows.append(dict(
            dataset=dataset,
            presence_slot=presence_slot,
            instance_slot=instance_slot,
            variant=name,
            images=len(records),
            miou=metric['miou'],
            delta_miou=metric['miou'] - base['miou'],
            aacc=metric['aacc'],
            improved_pixels=variant_counts[name]['improved_pixels'],
            harmed_pixels=variant_counts[name]['harmed_pixels'],
            help_minus_harm=variant_counts[name]['help_minus_harm'],
        ))
        for class_index, class_name in enumerate(class_names):
            iou = metric['iou'][class_index]
            base_iou = base['iou'][class_index]
            class_rows.append(dict(
                dataset=dataset,
                presence_slot=presence_slot,
                instance_slot=instance_slot,
                variant=name,
                class_index=class_index,
                class_name=class_name,
                iou=None if iou is None else 100.0 * iou,
                delta_iou=(None if iou is None or base_iou is None
                           else 100.0 * (iou - base_iou)),
                gt_pixels=int(base['gt'][class_index]),
            ))

    def score(name):
        return metrics[name]['miou']

    n00 = score('pi_native_p0_i0')
    n10 = score('pi_native_p1_i0')
    n01 = score('pi_native_p0_i1')
    n11 = score('pi_native_p1_i1')
    b00 = score('pi_branch_once_p0_i0')
    b10 = score('pi_branch_once_p1_i0')
    b01 = score('pi_branch_once_p0_i1')
    b11 = score('pi_branch_once_p1_i1')
    contrast = dict(
        dataset=dataset,
        presence_slot=presence_slot,
        instance_slot=instance_slot,
        baseline_miou=n00,
        p_only_delta=n10 - n00,
        i_only_delta=n01 - n00,
        pi_full_delta=n11 - n00,
        native_gamma_pi=n11 - n10 - n01 + n00,
        branch_gamma_pi=b11 - b10 - b01 + b00,
        branch_once_anchor_delta=b00 - n00,
        branch_once_full_delta=b11 - n11,
        allocation_by_text_interaction=b11 - b00 - n11 + n00,
        admission_site_effect=(
            n11 - score('pi_freeze_admission_p1_i1')),
        amplitude_site_effect=(
            n11 - score('pi_freeze_amplitude_p1_i1')),
        head_winner_switch_effect=(
            n11 - score('pi_hold_i_only_winner_p1_i1')),
    )

    valid_pixels = sum(int(record['valid_pixels']) for record in records)
    mechanism_row = dict(
        dataset=dataset,
        presence_slot=presence_slot,
        instance_slot=instance_slot,
        images=len(records),
        valid_pixels=valid_pixels,
        **mechanism,
        head_switch_flip_fraction=safe_div(
            mechanism['class_flip_with_head_switch'],
            mechanism['class_flip_pixels']),
    )
    for field in margin_sums:
        mechanism_row[field] = _weighted_mean(
            margin_sums[field], margin_weights[field])

    mechanism_class_rows = []
    query_rows = []
    for class_index, class_name in enumerate(class_names):
        counts = class_counts[class_index]
        mechanism_class_rows.append(dict(
            dataset=dataset,
            class_index=class_index,
            class_name=class_name,
            **counts,
            added_support_precision=safe_div(
                counts['added_support_gt_pixels'],
                counts['added_support_pixels']),
            removed_support_precision=safe_div(
                counts['removed_support_gt_pixels'],
                counts['removed_support_pixels']),
        ))
        query_rows.append(dict(
            dataset=dataset,
            class_index=class_index,
            class_name=class_name,
            presence_prompt=query_prompts.get(class_index, (None, None))[0],
            instance_prompt=query_prompts.get(class_index, (None, None))[1],
            **query_counts[class_index],
            minimum_abs_margin_p0=query_margins[class_index]['p0'],
            minimum_abs_margin_p1=query_margins[class_index]['p1'],
        ))
    return (variant_rows, class_rows, contrast, mechanism_row,
            mechanism_class_rows, query_rows)


def write_markdown(path, records, missing, contrasts, mechanisms):
    lookup = {row['dataset']: row for row in mechanisms}
    lines = [
        '# Native-fusion-aware PI role compatibility', '',
        f'- Unique images: {len(records)}',
        f'- Missing datasets: {missing or "none"}',
        '- Every replay reuses the same SAM3 grounding outputs; the returned '
        'model prediction remains the official baseline.', '',
        '- Admission/amplitude/head-winner columns are conditional '
        'counterfactual effects (`full - frozen`), not an additive '
        'decomposition.', '',
        '## PI performance and intervention sites', '',
        '| Dataset | P/I slots | P only | I only | PI full | Gamma PI | '
        'Admission | Amplitude | Head winner | Branch once on PI |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in contrasts:
        lines.append(
            f"| {row['dataset']} | P{row['presence_slot']}/"
            f"I{row['instance_slot']} | {row['p_only_delta']:+.3f} | "
            f"{row['i_only_delta']:+.3f} | {row['pi_full_delta']:+.3f} | "
            f"{row['native_gamma_pi']:+.3f} | "
            f"{row['admission_site_effect']:+.3f} | "
            f"{row['amplitude_site_effect']:+.3f} | "
            f"{row['head_winner_switch_effect']:+.3f} | "
            f"{row['branch_once_full_delta']:+.3f} |")
    lines.extend([
        '', '## Pixel mechanism', '',
        '| Dataset | Class flips | Any head switch | Flip with head switch | '
        'Flip without head switch | Improved | Harmed |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in contrasts:
        item = lookup[row['dataset']]
        lines.append(
            f"| {row['dataset']} | {item['class_flip_pixels']} | "
            f"{item['any_head_switch_pixels']} | "
            f"{item['class_flip_with_head_switch']} | "
            f"{item['class_flip_without_head_switch']} | "
            f"{item['improved_pixels']} | {item['harmed_pixels']} |")
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    records, missing, duplicate_counts = load_records(
        expand_inputs(args.inputs), args.expected_datasets,
        args.allow_incomplete)
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)

    variants, classes, contrasts, mechanisms = [], [], [], []
    mechanism_classes, queries = [], []
    for dataset in sorted(grouped):
        result = summarize_dataset(dataset, grouped[dataset])
        variants.extend(result[0])
        classes.extend(result[1])
        contrasts.append(result[2])
        mechanisms.append(result[3])
        mechanism_classes.extend(result[4])
        queries.extend(result[5])

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'pi_variants.csv'), variants)
    write_csv(os.path.join(args.out_dir, 'pi_class_variants.csv'), classes)
    write_csv(os.path.join(args.out_dir, 'pi_contrasts.csv'), contrasts)
    write_csv(os.path.join(args.out_dir, 'pi_mechanism_events.csv'),
              mechanisms)
    write_csv(os.path.join(args.out_dir, 'pi_mechanism_classes.csv'),
              mechanism_classes)
    write_csv(os.path.join(args.out_dir, 'pi_query_admission.csv'), queries)
    payload = dict(
        protocol=PI_PROTOCOL,
        records=len(records),
        missing_datasets=missing,
        duplicate_counts=duplicate_counts,
        variants=variants,
        class_variants=classes,
        contrasts=contrasts,
        mechanism_events=mechanisms,
        mechanism_classes=mechanism_classes,
        query_admission=queries,
    )
    with open(os.path.join(args.out_dir, 'pi_summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(
        os.path.join(args.out_dir, 'pi_summary.md'), records, missing,
        contrasts, mechanisms)


if __name__ == '__main__':
    main()
