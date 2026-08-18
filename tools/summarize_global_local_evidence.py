#!/usr/bin/env python3
"""Summarize frozen Global/Local complementary-evidence screening."""

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
    GLOBAL_LOCAL_EVIDENCE_PROTOCOL as GLV_PROTOCOL,
    GLOBAL_LOCAL_EVIDENCE_SCHEMA_VERSION as GLV_SCHEMA_VERSION,
    GLOBAL_LOCAL_EVIDENCE_VARIANT_NAMES as GLV_VARIANT_NAMES,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


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
                payload = record.get('global_local_evidence')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1))
                        != GLV_SCHEMA_VERSION
                        or payload.get('protocol') != GLV_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != GLV_VARIANT_NAMES):
                    raise ValueError(
                        f'{path}:{line_number}: global/local contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['global_local_evidence'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No global_local_evidence_v1 records found.')
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def family_of(name):
    if name.startswith('glv_anchor_'):
        return 'anchor'
    if name.startswith('glv_text_'):
        return 'text'
    return 'reference'


def reference_of(name):
    if name == 'glv_official' or name.startswith('glv_anchor_'):
        return 'glv_official'
    return 'glv_best_text'


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)

    variant_rows, class_rows = [], []
    complement_rows, complement_class_rows, image_rows = [], [], []
    metrics_by_dataset = {}
    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in GLV_VARIANT_NAMES}
        counters = {name: defaultdict(int) for name in GLV_VARIANT_NAMES}
        first = values[0]['global_local_evidence']
        source = first['global_source']
        endpoint = first['reference_endpoint']
        selection = tuple(first['selected_slots'])
        admission = first['selected_admission']
        comp_matrices = {
            family: None for family in ('anchor', 'text')}
        comp_counts = {
            family: defaultdict(int) for family in ('anchor', 'text')}
        comp_score_sums = defaultdict(float)
        class_counts = {
            family: defaultdict(lambda: defaultdict(int))
            for family in ('anchor', 'text')}

        for record in values:
            payload = record['global_local_evidence']
            if (payload['global_source'] != source
                    or payload['reference_endpoint'] != endpoint
                    or tuple(payload['selected_slots']) != selection
                    or payload['selected_admission'] != admission):
                raise ValueError(f'{dataset}: experiment metadata drifted.')
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in (
                        'changed_pixels', 'improved_pixels', 'harmed_pixels',
                        'help_minus_harm', 'background_to_foreground_pixels',
                        'foreground_to_background_pixels',
                        'foreground_to_foreground_pixels'):
                    counters[name][field] += int(stats.get(field, 0))
            for family, stats in payload['complementarity'].items():
                comp_matrices[family] = add_matrix(
                    comp_matrices[family],
                    stats['oracle_confusion']['matrix'])
                for field in (
                        'valid_pixels', 'disagreement_pixels',
                        'local_only_correct_pixels',
                        'global_only_correct_pixels', 'both_correct_pixels',
                        'both_wrong_pixels', 'score_entries'):
                    comp_counts[family][field] += int(stats[field])
                comp_score_sums[family] += (
                    float(stats['mean_abs_score_difference'])
                    * int(stats['score_entries']))
                for row in stats['class_rows']:
                    index = int(row['class_index'])
                    for field in (
                            'target_pixels', 'local_only_correct_pixels',
                            'global_only_correct_pixels',
                            'both_correct_pixels', 'both_wrong_pixels'):
                        class_counts[family][index][field] += int(row[field])

                local_metric = confusion_metrics(
                    payload['variants'][stats['local_variant']]
                    ['confusion']['matrix'])
                global_metric = confusion_metrics(
                    payload['variants'][stats['global_variant']]
                    ['confusion']['matrix'])
                oracle_metric = confusion_metrics(
                    stats['oracle_confusion']['matrix'])
                image_rows.append(dict(
                    dataset=dataset,
                    img_path=record['img_path'],
                    family=family,
                    global_source=source,
                    local_miou=local_metric['miou'],
                    global_miou=global_metric['miou'],
                    oracle_miou=oracle_metric['miou'],
                    oracle_headroom=(
                        oracle_metric['miou']
                        - max(local_metric['miou'], global_metric['miou'])),
                    local_only_correct_pixels=int(
                        stats['local_only_correct_pixels']),
                    global_only_correct_pixels=int(
                        stats['global_only_correct_pixels']),
                    disagreement_pixels=int(stats['disagreement_pixels']),
                ))

        metrics = {
            name: confusion_metrics(matrix) for name, matrix in matrices.items()
        }
        metrics_by_dataset[dataset] = metrics
        for name in GLV_VARIANT_NAMES:
            metric = metrics[name]
            reference_name = reference_of(name)
            reference = metrics[reference_name]
            variant_rows.append(dict(
                dataset=dataset,
                variant=name,
                family=family_of(name),
                images=len(values),
                global_source=source,
                reference_endpoint=endpoint,
                selected_slots=list(selection),
                selected_admission=admission,
                miou=metric['miou'],
                aacc=metric['aacc'],
                delta_to_reference=metric['miou'] - reference['miou'],
                **counters[name],
            ))
            for class_index, class_name in enumerate(
                    values[0]['class_names']):
                value = metric['iou'][class_index]
                ref_value = reference['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset,
                    variant=name,
                    family=family_of(name),
                    class_index=class_index,
                    class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_reference=(
                        None if value is None or ref_value is None
                        else 100.0 * (value - ref_value)),
                ))

        for family in ('anchor', 'text'):
            local = metrics[f'glv_{family}_local']
            global_value = metrics[f'glv_{family}_global']
            oracle = confusion_metrics(comp_matrices[family])
            reference = metrics[
                'glv_official' if family == 'anchor'
                else 'glv_best_text']
            count = comp_counts[family]
            best_endpoint = max(local['miou'], global_value['miou'])
            complement_rows.append(dict(
                dataset=dataset,
                family=family,
                images=len(values),
                global_source=source,
                reference_endpoint=endpoint,
                reference_miou=reference['miou'],
                local_miou=local['miou'],
                global_miou=global_value['miou'],
                global_minus_local=global_value['miou'] - local['miou'],
                oracle_miou=oracle['miou'],
                oracle_headroom_over_best_endpoint=(
                    oracle['miou'] - best_endpoint),
                disagreement_ratio=(
                    count['disagreement_pixels'] / count['valid_pixels']),
                local_only_correct_ratio=(
                    count['local_only_correct_pixels']
                    / count['valid_pixels']),
                global_only_correct_ratio=(
                    count['global_only_correct_pixels']
                    / count['valid_pixels']),
                both_wrong_ratio=(
                    count['both_wrong_pixels'] / count['valid_pixels']),
                mean_abs_score_difference=(
                    comp_score_sums[family] / count['score_entries']),
            ))
            for class_index, class_name in enumerate(values[0]['class_names']):
                row = class_counts[family][class_index]
                target = row['target_pixels']
                complement_class_rows.append(dict(
                    dataset=dataset,
                    family=family,
                    class_index=class_index,
                    class_name=class_name,
                    target_pixels=target,
                    local_only_correct_pixels=(
                        row['local_only_correct_pixels']),
                    global_only_correct_pixels=(
                        row['global_only_correct_pixels']),
                    local_only_correct_ratio=(
                        row['local_only_correct_pixels'] / target
                        if target else None),
                    global_only_correct_ratio=(
                        row['global_only_correct_pixels'] / target
                        if target else None),
                ))

    fixed_rows = []
    for family in ('anchor', 'text'):
        candidates = [
            name for name in GLV_VARIANT_NAMES
            if name.startswith(f'glv_{family}_mix_')
            or name == f'glv_{family}_max']
        for name in candidates:
            reference_deltas, endpoint_deltas = [], []
            for dataset, metrics in metrics_by_dataset.items():
                metric = metrics[name]['miou']
                reference = metrics[
                    'glv_official' if family == 'anchor'
                    else 'glv_best_text']['miou']
                endpoint = max(
                    metrics[f'glv_{family}_local']['miou'],
                    metrics[f'glv_{family}_global']['miou'])
                reference_deltas.append(metric - reference)
                endpoint_deltas.append(metric - endpoint)
            fixed_rows.append(dict(
                family=family,
                variant=name,
                datasets=len(reference_deltas),
                positive_vs_reference=sum(x > 0 for x in reference_deltas),
                macro_delta_vs_reference=(
                    sum(reference_deltas) / len(reference_deltas)),
                beats_best_endpoint=sum(x > 0 for x in endpoint_deltas),
                macro_delta_vs_best_endpoint=(
                    sum(endpoint_deltas) / len(endpoint_deltas)),
                min_delta_vs_reference=min(reference_deltas),
                max_delta_vs_reference=max(reference_deltas),
            ))
    interaction_rows = []
    operators = [
        name[len('glv_anchor_'):]
        for name in GLV_VARIANT_NAMES
        if name.startswith('glv_anchor_mix_') or name == 'glv_anchor_max']
    for operator in operators:
        values = []
        for dataset, metrics in sorted(metrics_by_dataset.items()):
            anchor_gain = (
                metrics[f'glv_anchor_{operator}']['miou']
                - metrics['glv_official']['miou'])
            text_gain = (
                metrics[f'glv_text_{operator}']['miou']
                - metrics['glv_best_text']['miou'])
            interaction = text_gain - anchor_gain
            values.append(interaction)
            interaction_rows.append(dict(
                dataset=dataset,
                operator=operator,
                anchor_visual_gain=anchor_gain,
                text_visual_gain=text_gain,
                role_text_visual_interaction=interaction,
            ))
        interaction_rows.append(dict(
            dataset='macro',
            operator=operator,
            anchor_visual_gain=None,
            text_visual_gain=None,
            role_text_visual_interaction=sum(values) / len(values),
            positive_interaction_datasets=sum(value > 0 for value in values),
        ))
    return (variant_rows, class_rows, complement_rows,
            complement_class_rows, image_rows, fixed_rows, interaction_rows)


def write_report(path, complement_rows, fixed_rows, interaction_rows,
                 missing, duplicates):
    lines = [
        '# Global–Local Evidence v1',
        '',
        'This is a complementarity screen. Local and Global each retain full '
        'SAM3 P/S/I grounding; fixed score mixtures are controls, not the '
        'proposed learned coherent-memory adapter.',
        '',
        '## Endpoint complementarity',
        '',
        '| Dataset | Family | Global source | Reference | Local | Global | '
        'Oracle | Oracle headroom | L-only correct | G-only correct |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in complement_rows:
        lines.append(
            f'| {row["dataset"]} | {row["family"]} | '
            f'{row["global_source"]} | {row["reference_miou"]:.3f} | '
            f'{row["local_miou"]:.3f} | {row["global_miou"]:.3f} | '
            f'{row["oracle_miou"]:.3f} | '
            f'{row["oracle_headroom_over_best_endpoint"]:+.3f} | '
            f'{100.0 * row["local_only_correct_ratio"]:.2f}% | '
            f'{100.0 * row["global_only_correct_ratio"]:.2f}% |')
    lines.extend([
        '',
        '## Fixed Global/Local fusion controls',
        '',
        '| Family | Variant | Positive vs current | Macro vs current | '
        'Beats both endpoints | Macro vs best endpoint | Range vs current |',
        '|---|---|---:|---:|---:|---:|---:|',
    ])
    for row in sorted(
            fixed_rows,
            key=lambda item: (
                item['family'], -item['macro_delta_vs_reference'])):
        lines.append(
            f'| {row["family"]} | {row["variant"]} | '
            f'{row["positive_vs_reference"]}/{row["datasets"]} | '
            f'{row["macro_delta_vs_reference"]:+.3f} | '
            f'{row["beats_best_endpoint"]}/{row["datasets"]} | '
            f'{row["macro_delta_vs_best_endpoint"]:+.3f} | '
            f'[{row["min_delta_vs_reference"]:+.3f}, '
            f'{row["max_delta_vs_reference"]:+.3f}] |')
    lines.extend([
        '',
        '## Role-Text × visual interaction',
        '',
        '| Operator | Positive interaction | Macro interaction |',
        '|---|---:|---:|',
    ])
    for row in interaction_rows:
        if row['dataset'] != 'macro':
            continue
        lines.append(
            f'| {row["operator"]} | '
            f'{row["positive_interaction_datasets"]}/6 | '
            f'{row["role_text_visual_interaction"]:+.3f} |')
    lines.extend([
        '',
        f'- Missing datasets: {missing or "none"}',
        f'- Deduplicated DDP records: {duplicates or "none"}',
        '- Oracle rows use GT only to test whether complementary evidence '
        'exists; they are not deployable selection results.',
        '- A reusable fusion signal requires one fixed variant across datasets. '
        'Per-dataset endpoint or weight selection is diagnostic only.',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    expected = tuple(str(value).lower() for value in args.expected_datasets)
    records, missing, duplicates = load_records(
        args.inputs, expected, args.allow_incomplete)
    outputs = summarize(records)
    (variant_rows, class_rows, complement_rows,
     complement_class_rows, image_rows, fixed_rows,
     interaction_rows) = outputs
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'global_local_variants.csv'),
              variant_rows)
    write_csv(os.path.join(args.out_dir, 'global_local_classes.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'global_local_complementarity.csv'),
              complement_rows)
    write_csv(os.path.join(args.out_dir, 'global_local_complementarity_classes.csv'),
              complement_class_rows)
    write_csv(os.path.join(args.out_dir, 'global_local_images.csv'), image_rows)
    write_csv(os.path.join(args.out_dir, 'global_local_fixed_mix.csv'),
              fixed_rows)
    write_csv(os.path.join(args.out_dir, 'global_local_interactions.csv'),
              interaction_rows)
    report = os.path.join(args.out_dir, 'global_local_evidence.md')
    write_report(report, complement_rows, fixed_rows, interaction_rows,
                 missing, duplicates)
    print(report)


if __name__ == '__main__':
    main()
