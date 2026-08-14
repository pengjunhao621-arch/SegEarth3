#!/usr/bin/env python3
"""Summarize boundary-local refinement and anchor-clamped replay."""

import argparse
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
    BOUNDARY_REPLAY_PROTOCOL,
    BOUNDARY_REPLAY_SCHEMA_VERSION,
    BOUNDARY_REPLAY_VARIANT_NAMES,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')
COUNTERS = (
    'changed_pixels', 'improved_pixels', 'harmed_pixels',
    'wrong_to_wrong_pixels', 'help_minus_harm',
    'changed_gt_boundary_pixels', 'improved_gt_boundary_pixels',
    'harmed_gt_boundary_pixels',
)


def mean(values):
    values = [float(value) for value in values if value is not None
              and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


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
                payload = record.get('boundary_replay')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1))
                        != BOUNDARY_REPLAY_SCHEMA_VERSION
                        or payload.get('protocol') != BOUNDARY_REPLAY_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != BOUNDARY_REPLAY_VARIANT_NAMES):
                    raise ValueError(
                        f'{path}:{line_number}: boundary/replay contract '
                        'mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    prior = seen[key]
                    if prior['boundary_replay'] != payload:
                        raise ValueError(
                            f'Inconsistent DDP duplicate for {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No boundary_replay_v1 records found.')
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    return records, missing, dict(duplicates)


def variant_metadata(name):
    if name in ('br_official', 'br_best_full', 'br_replay_p',
                'br_replay_s', 'br_replay_i', 'br_replay_all'):
        family = 'reference' if name in (
            'br_official', 'br_best_full') else 'replay'
        role = name[len('br_replay_'):] if family == 'replay' else None
        return dict(family=family, base=None, guide=None,
                    strength=None, replay_role=role)
    body = name[len('br_'):]
    base, guide, strength = body.split('_')
    return dict(family='boundary', base=base, guide=guide,
                strength=strength, replay_role=None)


def summarize_variants(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    rows, class_rows = [], []
    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in BOUNDARY_REPLAY_VARIANT_NAMES}
        counters = {
            name: defaultdict(int) for name in BOUNDARY_REPLAY_VARIANT_NAMES}
        selected = tuple(values[0]['boundary_replay'][
            'selected_combination'])
        admission = values[0]['boundary_replay']['selected_admission']
        for record in values:
            payload = record['boundary_replay']
            if (tuple(payload['selected_combination']) != selected
                    or payload['selected_admission'] != admission):
                raise ValueError(f'{dataset}: selection drifted.')
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in COUNTERS:
                    counters[name][field] += int(stats.get(field, 0))
        metrics = {name: confusion_metrics(matrix)
                   for name, matrix in matrices.items()}
        official = metrics['br_official']
        best_full = metrics['br_best_full']
        for name in BOUNDARY_REPLAY_VARIANT_NAMES:
            metric = metrics[name]
            metadata = variant_metadata(name)
            reference_name = (
                'br_official' if metadata.get('base') == 'official'
                else 'br_best_full')
            if metadata['family'] != 'boundary':
                reference_name = (
                    name if metadata['family'] == 'reference'
                    else 'br_best_full')
            reference = metrics[reference_name]
            rows.append(dict(
                dataset=dataset, variant=name, images=len(values),
                selected_presence_slot=selected[0],
                selected_semantic_slot=selected[1],
                selected_instance_slot=selected[2],
                selected_admission=admission,
                miou=metric['miou'], aacc=metric['aacc'],
                delta_to_official=metric['miou'] - official['miou'],
                delta_to_best_full=metric['miou'] - best_full['miou'],
                delta_to_reference=metric['miou'] - reference['miou'],
                reference_variant=reference_name,
                **metadata, **counters[name],
            ))
            for class_index, class_name in enumerate(
                    values[0]['class_names']):
                value = metric['iou'][class_index]
                ref_value = reference['iou'][class_index]
                official_value = official['iou'][class_index]
                best_value = best_full['iou'][class_index]
                class_rows.append(dict(
                    dataset=dataset, variant=name,
                    class_index=class_index, class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_reference=(
                        None if value is None or ref_value is None
                        else 100.0 * (value - ref_value)),
                    delta_iou_to_official=(
                        None if value is None or official_value is None
                        else 100.0 * (value - official_value)),
                    delta_iou_to_best_full=(
                        None if value is None or best_value is None
                        else 100.0 * (value - best_value)),
                    **metadata,
                ))
    return rows, class_rows


def expand_query_paths(row):
    paths = []
    if row['source'] == 'anchor' and row.get('admitted_official'):
        paths.append('official_anchor')
    if row['source'] == 'anchor' and row.get('admitted_best'):
        paths.append('best_anchor')
    if row['source'] == 'candidate' and row.get('admitted_best'):
        paths.append('best_candidate')
    return paths


def summarize_queries(records):
    raw_rows = []
    for record in records:
        dataset = str(record['dataset_name']).lower()
        for view in record.get('views', []):
            payload = view.get('boundary_replay_diagnosis')
            if payload is None:
                continue
            for source in payload.get('query_rows', []):
                row = dict(source)
                row.update(
                    dataset=dataset, img_path=record.get('img_path'),
                    view_id=view.get('view_id'))
                raw_rows.append(row)

    expanded = []
    for row in raw_rows:
        for path in expand_query_paths(row):
            expanded.append(dict(row, path=path))
    groups = defaultdict(list)
    for row in expanded:
        groups[(row['dataset'], row['path'], -1)].append(row)
        groups[(row['dataset'], row['path'], int(row['class_index']))].append(
            row)
    summary = []
    for (dataset, path, class_index), values in sorted(groups.items()):
        class_name = 'all' if class_index < 0 else values[0]['class_name']
        improved = sum(float(row['grid_iou_delta']) > 0 for row in values)
        harmed = sum(float(row['grid_iou_delta']) < 0 for row in values)
        crossed_up = sum(
            float(row['grid_before_iou']) < 0.5
            <= float(row['grid_after_iou']) for row in values)
        crossed_down = sum(
            float(row['grid_after_iou']) < 0.5
            <= float(row['grid_before_iou']) for row in values)
        changed = sum(int(row['grid_changed_pixels']) for row in values)
        outside = sum(
            int(row['grid_changed_outside_boundary_zone']) for row in values)
        summary.append(dict(
            dataset=dataset, path=path, class_index=class_index,
            class_name=class_name, queries=len(values),
            mean_before_iou=mean(row['grid_before_iou'] for row in values),
            mean_after_iou=mean(row['grid_after_iou'] for row in values),
            mean_iou_delta=mean(row['grid_iou_delta'] for row in values),
            mean_before_boundary_f1=mean(
                row['grid_before_boundary_f1'] for row in values),
            mean_after_boundary_f1=mean(
                row['grid_after_boundary_f1'] for row in values),
            mean_boundary_f1_delta=mean(
                row['grid_boundary_f1_delta'] for row in values),
            improved_queries=improved, harmed_queries=harmed,
            unchanged_iou_queries=len(values) - improved - harmed,
            crossed_iou_050_up=crossed_up,
            crossed_iou_050_down=crossed_down,
            changed_grid_pixels=changed,
            changed_outside_boundary_zone=outside,
            outside_boundary_change_ratio=safe_div(outside, changed),
            net_correct_support_pixels=sum(
                int(row['grid_added_correct_pixels'])
                + int(row['grid_removed_false_pixels'])
                - int(row['grid_added_false_pixels'])
                - int(row['grid_removed_correct_pixels'])
                for row in values),
        ))
    return raw_rows, summary


def summarize_replay_integrity(records):
    rows = []
    for record in records:
        dataset = str(record['dataset_name']).lower()
        for view in record.get('views', []):
            payload = view.get('boundary_replay_diagnosis')
            if payload is None:
                continue
            for row in payload.get('class_rows', []):
                rows.append(dict(
                    row, dataset=dataset,
                    img_path=record.get('img_path'),
                    view_id=view.get('view_id')))
    summary = []
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset']].append(row)
    for dataset, values in sorted(groups.items()):
        summary.append(dict(
            dataset=dataset, class_views=len(values),
            anchor_trace_max_abs=max(
                float(row['anchor_trace_max_abs']) for row in values),
            anchor_replay_max_abs=max(
                float(row['anchor_replay_max_abs']) for row in values),
            replay_p_abs_mean=mean(
                row['replay_p_abs_mean'] for row in values),
            replay_s_abs_mean=mean(
                row['replay_s_abs_mean'] for row in values),
            replay_i_abs_mean=mean(
                row['replay_i_abs_mean'] for row in values),
            replay_all_abs_mean=mean(
                row['replay_all_abs_mean'] for row in values),
        ))
    return rows, summary


def build_mechanism_chain(variant_rows, query_summary):
    rows = []
    datasets = sorted({row['dataset'] for row in variant_rows})
    for dataset in datasets:
        for base, paths in (
                ('official', ('official_anchor',)),
                ('best', ('best_anchor', 'best_candidate'))):
            final = find(
                variant_rows, dataset,
                f'br_{base}_block23_l050')
            query_groups = [
                row for row in query_summary
                if row['dataset'] == dataset and row['class_index'] == -1
                and row['path'] in paths]
            total = sum(int(row['queries']) for row in query_groups)

            def weighted(field):
                return (sum(float(row[field]) * int(row['queries'])
                            for row in query_groups) / total
                        if total else None)

            changed = sum(
                int(row['changed_grid_pixels']) for row in query_groups)
            outside = sum(
                int(row['changed_outside_boundary_zone'])
                for row in query_groups)
            rows.append(dict(
                dataset=dataset, base=base,
                final_variant=final['variant'],
                final_miou=final['miou'],
                final_miou_delta=final['delta_to_reference'],
                final_improved_pixels=final['improved_pixels'],
                final_harmed_pixels=final['harmed_pixels'],
                final_help_minus_harm=final['help_minus_harm'],
                query_paths='+'.join(paths), queries=total,
                mean_query_iou_delta=weighted('mean_iou_delta'),
                mean_query_boundary_f1_delta=weighted(
                    'mean_boundary_f1_delta'),
                improved_queries=sum(
                    int(row['improved_queries']) for row in query_groups),
                harmed_queries=sum(
                    int(row['harmed_queries']) for row in query_groups),
                crossed_iou_050_up=sum(
                    int(row['crossed_iou_050_up']) for row in query_groups),
                crossed_iou_050_down=sum(
                    int(row['crossed_iou_050_down']) for row in query_groups),
                outside_boundary_change_ratio=safe_div(outside, changed),
            ))
    return rows


def find(rows, dataset, variant):
    return next(row for row in rows
                if row['dataset'] == dataset and row['variant'] == variant)


def write_markdown(
        path, variant_rows, query_summary, mechanism_chain, integrity,
        duplicates):
    datasets = sorted({row['dataset'] for row in variant_rows})
    lines = [
        '# Boundary-local refinement and anchor-clamped replay', '',
        f'- DDP padding duplicates removed: {duplicates or "none"}',
        '- Primary metric: final whole-image mIoU. Query IoU/boundary F1 are '
        'mechanism diagnostics on the block23 grid.',
        '- Boundary refinement never changes class identity, object score, '
        'Presence or query admission.', '',
        '## Final mIoU: boundary-local refinement', '',
        '| Dataset | Official | Best-full | B23 .25 Δ(O/B) | B23 .50 Δ(O/B) '
        '| B23 1.00 Δ(O/B) | RGB .50 Δ(O/B) | Uniform .50 Δ(O/B) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in datasets:
        official = find(variant_rows, dataset, 'br_official')
        best = find(variant_rows, dataset, 'br_best_full')

        def pair(guide, strength):
            left = find(
                variant_rows, dataset,
                f'br_official_{guide}_{strength}')['delta_to_reference']
            right = find(
                variant_rows, dataset,
                f'br_best_{guide}_{strength}')['delta_to_reference']
            return f'{left:+.3f}/{right:+.3f}'

        lines.append(
            f"| {dataset} | {official['miou']:.3f} | {best['miou']:.3f} | "
            f"{pair('block23', 'l025')} | {pair('block23', 'l050')} | "
            f"{pair('block23', 'l100')} | {pair('rgb', 'l050')} | "
            f"{pair('uniform', 'l050')} |")

    lines.extend([
        '', '## Final mIoU: layer-wise anchor-clamped replay', '',
        '| Dataset | Best-full | Replay P | Δ | Replay S | Δ | Replay I | Δ '
        '| Replay all | Δ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for dataset in datasets:
        best = find(variant_rows, dataset, 'br_best_full')
        values = [find(variant_rows, dataset, f'br_replay_{role}')
                  for role in ('p', 's', 'i', 'all')]
        lines.append(
            f"| {dataset} | {best['miou']:.3f} | "
            + ' | '.join(
                f"{row['miou']:.3f} | {row['delta_to_best_full']:+.3f}"
                for row in values) + ' |')

    lines.extend([
        '', '## Block23 query mechanism at strength 0.50', '',
        '| Dataset/path | Queries | mean ΔIoU | mean ΔBoundary-F1 | '
        'improved/harmed | IoU .50 up/down | outside-boundary change |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in query_summary:
        if row['class_index'] != -1:
            continue
        lines.append(
            f"| {row['dataset']}/{row['path']} | {row['queries']} | "
            f"{row['mean_iou_delta']:+.5f} | "
            f"{row['mean_boundary_f1_delta']:+.5f} | "
            f"{row['improved_queries']}/{row['harmed_queries']} | "
            f"{row['crossed_iou_050_up']}/{row['crossed_iou_050_down']} | "
            f"{row['outside_boundary_change_ratio']:.4f} |")

    lines.extend([
        '', '## Mechanism chain at block23 strength 0.50', '',
        '| Dataset/base | final ΔmIoU | query ΔIoU | query ΔBoundary-F1 | '
        'query improved/harmed | final help-harm pixels |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in mechanism_chain:
        query_iou = row['mean_query_iou_delta']
        query_boundary = row['mean_query_boundary_f1_delta']
        lines.append(
            f"| {row['dataset']}/{row['base']} | "
            f"{row['final_miou_delta']:+.3f} | "
            f"{('n/a' if query_iou is None else f'{query_iou:+.5f}')} | "
            f"{('n/a' if query_boundary is None else f'{query_boundary:+.5f}')} | "
            f"{row['improved_queries']}/{row['harmed_queries']} | "
            f"{row['final_help_minus_harm']} |")

    lines.extend([
        '', '## Replay integrity', '',
        '| Dataset | Anchor trace max abs | Anchor-text replay max abs |',
        '|---|---:|---:|',
    ])
    for row in integrity:
        lines.append(
            f"| {row['dataset']} | {row['anchor_trace_max_abs']:.8g} | "
            f"{row['anchor_replay_max_abs']:.8g} |")
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--expected-datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--integrity-tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    expected = tuple(str(value).lower() for value in args.expected_datasets)
    if 'isaid' in expected:
        raise ValueError('iSAID is excluded from project evaluation.')
    records, missing, duplicates = load_records(
        args.inputs, expected, args.allow_incomplete)
    variant_rows, class_rows = summarize_variants(records)
    query_rows, query_summary = summarize_queries(records)
    replay_rows, replay_integrity = summarize_replay_integrity(records)
    mechanism_chain = build_mechanism_chain(variant_rows, query_summary)
    for row in replay_integrity:
        if max(row['anchor_trace_max_abs'], row['anchor_replay_max_abs']) > (
                args.integrity_tolerance):
            raise ValueError(
                f"{row['dataset']}: replay integrity exceeded tolerance.")
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'variants.csv'), variant_rows)
    write_csv(os.path.join(args.out_dir, 'class_variants.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'boundary_queries.csv'), query_rows)
    write_csv(os.path.join(args.out_dir, 'boundary_query_summary.csv'),
              query_summary)
    write_csv(os.path.join(args.out_dir, 'mechanism_chain.csv'),
              mechanism_chain)
    write_csv(os.path.join(args.out_dir, 'replay_class_views.csv'), replay_rows)
    write_csv(os.path.join(args.out_dir, 'replay_integrity.csv'),
              replay_integrity)
    payload = dict(
        protocol=BOUNDARY_REPLAY_PROTOCOL,
        records=len(records), missing_datasets=missing,
        duplicate_counts=duplicates,
        variants=variant_rows, class_variants=class_rows,
        boundary_query_summary=query_summary,
        mechanism_chain=mechanism_chain,
        replay_integrity=replay_integrity,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(
        os.path.join(args.out_dir, 'summary.md'), variant_rows,
        query_summary, mechanism_chain, replay_integrity, duplicates)


if __name__ == '__main__':
    main()
