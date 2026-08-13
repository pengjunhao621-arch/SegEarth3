#!/usr/bin/env python3
"""Summarize PE semantic interventions and instance-query evidence."""

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
    PE_FEATURE_NAMES,
    PE_PROTOCOL,
    PE_SCHEMA_VERSION,
    PE_VARIANT_NAMES,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def mean(values):
    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def pearson(left, right):
    pairs = [(float(x), float(y)) for x, y in zip(left, right)
             if x is not None and y is not None
             and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs)
        * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def binary_auc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def load_records(patterns, expected):
    paths = sorted(set(path for pattern in patterns
                       for path in (glob.glob(pattern) or [pattern])
                       if os.path.isfile(path)))
    records, seen, duplicates = [], {}, defaultdict(int)
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                dataset = str(record.get('dataset_name', '')).lower()
                payload = record.get('pe_role_evidence')
                if dataset not in expected or payload is None:
                    continue
                if (int(payload.get('schema_version', -1)) != PE_SCHEMA_VERSION
                        or payload.get('protocol') != PE_PROTOCOL
                        or tuple(payload.get('variants', {}))
                        != PE_VARIANT_NAMES):
                    raise ValueError(f'{dataset}: PE record contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                rank = int(record.get('rank', 0))
                if key in seen:
                    prior = seen[key]
                    if prior['pe_role_evidence'] != payload:
                        raise ValueError(
                            f'Inconsistent DDP duplicate for {key}.')
                    duplicates[dataset] += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No PE role-evidence records found.')
    return records, dict(duplicates)


def summarize_semantic(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record['dataset_name']).lower()].append(record)
    rows, class_rows = [], []
    for dataset, values in sorted(grouped.items()):
        matrices = {name: None for name in PE_VARIANT_NAMES}
        counters = {name: defaultdict(int) for name in PE_VARIANT_NAMES}
        selected = tuple(values[0]['pe_role_evidence'][
            'selected_combination'])
        admission = next(
            view['pe_diagnosis']['selected_admission']
            for view in values[0]['views']
            if view.get('pe_diagnosis') is not None)
        for record in values:
            payload = record['pe_role_evidence']
            if tuple(payload['selected_combination']) != selected:
                raise ValueError(f'{dataset}: selected composition drifted.')
            for name, stats in payload['variants'].items():
                matrices[name] = add_matrix(
                    matrices[name], stats['confusion']['matrix'])
                for field in (
                        'changed_pixels', 'improved_pixels', 'harmed_pixels',
                        'help_minus_harm', 'changed_boundary_pixels',
                        'improved_boundary_pixels', 'harmed_boundary_pixels'):
                    counters[name][field] += int(stats.get(field, 0))
        metrics = {name: confusion_metrics(value)
                   for name, value in matrices.items()}
        reference_name = f'pe_sem_unrefined_{admission}'
        reference = metrics[reference_name]
        for name in PE_VARIANT_NAMES:
            metric = metrics[name]
            source = name[len('pe_sem_'):].rsplit('_', 1)[0]
            mode = name.rsplit('_', 1)[1]
            # Restore the two-word admission label.
            if name.endswith('_anchor_admission'):
                source = name[len('pe_sem_'):-len('_anchor_admission')]
                mode = 'anchor_admission'
            row = dict(
                dataset=dataset, variant=name, source=source,
                admission=mode, selected_presence_slot=selected[0],
                selected_semantic_slot=selected[1],
                selected_instance_slot=selected[2],
                selected_admission=admission,
                is_selected_admission=(mode == admission),
                images=len(values), miou=metric['miou'],
                delta_to_selected_unrefined=(
                    metric['miou'] - reference['miou']),
                aacc=metric['aacc'],
                **counters[name],
            )
            rows.append(row)
            for index, class_name in enumerate(values[0]['class_names']):
                value = metric['iou'][index]
                ref = reference['iou'][index]
                class_rows.append(dict(
                    dataset=dataset, variant=name, source=source,
                    admission=mode, class_index=index,
                    class_name=class_name,
                    iou=None if value is None else 100.0 * value,
                    delta_iou_to_selected_unrefined=(
                        None if value is None or ref is None
                        else 100.0 * (value - ref)),
                ))
    return rows, class_rows


def summarize_queries(records):
    query_rows = []
    for record in records:
        dataset = str(record['dataset_name']).lower()
        for view in record.get('views', []):
            payload = view.get('pe_diagnosis')
            if payload is None:
                continue
            for source in payload.get('query_rows', []):
                row = dict(source)
                row.update(dataset=dataset, img_path=record.get('img_path'),
                           view_id=view.get('view_id'))
                query_rows.append(row)

    summary = []
    groups = defaultdict(list)
    for row in query_rows:
        groups[(row['dataset'], int(row['class_index']))].append(row)
        groups[(row['dataset'], -1)].append(row)
    metrics = ('inside_coherence', 'ring_separation', 'boundary_agreement')
    for (dataset, class_index), values in sorted(groups.items()):
        class_name = ('all' if class_index < 0 else values[0]['class_name'])
        labels = [float(row['gt_iou']) >= 0.5 for row in values]
        for field in ('object_score', 'admission_score'):
            scores = [float(row[field]) for row in values]
            summary.append(dict(
                dataset=dataset, class_index=class_index,
                class_name=class_name, feature='native', metric=field,
                queries=len(values), useful_queries=sum(labels),
                mean_score=mean(scores),
                pearson_with_gt_iou=pearson(
                    scores, [row['gt_iou'] for row in values]),
                pearson_with_gt_precision=pearson(
                    scores, [row['gt_precision'] for row in values]),
                auc_useful_iou_ge_050=binary_auc(labels, scores),
            ))
        for feature in PE_FEATURE_NAMES:
            for metric in metrics:
                field = f'{feature}_{metric}'
                scores = [float(row[field]) for row in values]
                summary.append(dict(
                    dataset=dataset, class_index=class_index,
                    class_name=class_name, feature=feature, metric=metric,
                    queries=len(values), useful_queries=sum(labels),
                    mean_score=mean(scores),
                    pearson_with_gt_iou=pearson(
                        scores, [row['gt_iou'] for row in values]),
                    pearson_with_gt_precision=pearson(
                        scores, [row['gt_precision'] for row in values]),
                    auc_useful_iou_ge_050=binary_auc(labels, scores),
                ))
    return query_rows, summary


def write_markdown(path, semantic_rows, query_summary, duplicates):
    datasets = sorted({row['dataset'] for row in semantic_rows})
    lines = [
        '# PE role-evidence screen', '',
        f'- DDP padding duplicates removed: {duplicates or "none"}',
        '- Semantic variants change prediction; instance PE evidence is '
        'diagnostic only.',
        '', '## Semantic spatial guidance', '',
        '| Dataset | Selected path | Unrefined | Best PE block | Delta | '
        'Final FPN delta | RGB delta |',
        '|---|---|---:|---|---:|---:|---:|',
    ]
    for dataset in datasets:
        rows = [row for row in semantic_rows
                if row['dataset'] == dataset and row['is_selected_admission']]
        reference = next(row for row in rows if row['source'] == 'unrefined')
        pe = [row for row in rows if row['source'].startswith('block')]
        best = max(pe, key=lambda row: row['miou'])
        fpn = next(row for row in rows if row['source'] == 'fpn_final')
        rgb = next(row for row in rows if row['source'] == 'rgb')
        slots = (reference['selected_presence_slot'],
                 reference['selected_semantic_slot'],
                 reference['selected_instance_slot'])
        lines.append(
            f"| {dataset} | P{slots[0]}+S{slots[1]}+I{slots[2]} "
            f"({reference['selected_admission']}) | {reference['miou']:.3f} | "
            f"{best['source']} ({best['miou']:.3f}) | "
            f"{best['delta_to_selected_unrefined']:+.3f} | "
            f"{fpn['delta_to_selected_unrefined']:+.3f} | "
            f"{rgb['delta_to_selected_unrefined']:+.3f} |")
    lines.extend([
        '', '## Instance query-region evidence', '',
        '| Dataset | Best PE feature/metric | PE correlation | PE AUC | '
        'Native object-score correlation | Native AUC | Queries |',
        '|---|---|---:|---:|---:|---:|---:|',
    ])
    for dataset in datasets:
        rows = [row for row in query_summary
                if row['dataset'] == dataset and row['class_index'] == -1]
        pe_rows = [row for row in rows if row['feature'] != 'native']
        best = max(pe_rows, key=lambda row: (
            -float('inf') if row['pearson_with_gt_iou'] is None
            else row['pearson_with_gt_iou']))
        native = next(row for row in rows
                      if row['metric'] == 'object_score')
        corr = best['pearson_with_gt_iou']
        auc = best['auc_useful_iou_ge_050']
        native_corr = native['pearson_with_gt_iou']
        native_auc = native['auc_useful_iou_ge_050']
        lines.append(
            f"| {dataset} | {best['feature']}/{best['metric']} | "
            f"{('n/a' if corr is None else f'{corr:.3f}')} | "
            f"{('n/a' if auc is None else f'{auc:.3f}')} | "
            f"{('n/a' if native_corr is None else f'{native_corr:.3f}')} | "
            f"{('n/a' if native_auc is None else f'{native_auc:.3f}')} | "
            f"{best['queries']} |")
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
    records, duplicates = load_records(args.inputs, expected)
    observed = {str(row['dataset_name']).lower() for row in records}
    missing = [name for name in expected if name not in observed]
    if missing and not args.allow_incomplete:
        raise ValueError(f'Missing datasets: {missing}.')
    semantic_rows, class_rows = summarize_semantic(records)
    query_rows, query_summary = summarize_queries(records)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'semantic_variants.csv'), semantic_rows)
    write_csv(os.path.join(args.out_dir, 'semantic_class_variants.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'instance_queries.csv'), query_rows)
    write_csv(os.path.join(args.out_dir, 'instance_query_summary.csv'),
              query_summary)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(dict(
            protocol=PE_PROTOCOL, records=len(records),
            missing_datasets=missing, duplicate_counts=duplicates,
            semantic_variants=semantic_rows,
            semantic_class_variants=class_rows,
            instance_query_summary=query_summary,
        ), handle, indent=2, ensure_ascii=False)
    write_markdown(os.path.join(args.out_dir, 'summary.md'),
                   semantic_rows, query_summary, duplicates)


if __name__ == '__main__':
    main()
