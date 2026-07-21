#!/usr/bin/env python3
"""Aggregate candidate-region diagnostics and issue an explicit go/no-go verdict."""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--decision-source', default='hybrid')
    parser.add_argument(
        '--selection-datasets', default='udd5,vdd,vaihingen')
    parser.add_argument('--min-iou25-recall', type=float, default=0.50)
    parser.add_argument('--min-high-purity-rate', type=float, default=0.60)
    parser.add_argument('--min-top2-recovery-rate', type=float, default=0.20)
    parser.add_argument('--min-switch-regions', type=int, default=50)
    parser.add_argument('--min-switch-classes', type=int, default=2)
    return parser.parse_args()


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


def iter_records(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def add_numeric(target, source):
    for key, value in source.items():
        if (
                key in ('source', 'class_index', 'class_name', 'is_background')
                or not isinstance(value, (int, float))):
            continue
        target[key] += value


def metrics(row):
    result = dict(row)
    result.update(
        covered_valid_ratio=safe_div(
            row.get('covered_valid_pixels'), row.get('valid_pixels')),
        overlap_ratio=safe_div(
            row.get('overlap_valid_pixels'),
            row.get('covered_valid_pixels')),
        mean_purity=safe_div(
            row.get('purity_sum'), row.get('evaluated_regions')),
        area_weighted_purity=safe_div(
            row.get('purity_pixel_sum'), row.get('region_pixels')),
        high_purity_rate=safe_div(
            row.get('high_purity_regions'), row.get('evaluated_regions')),
        baseline_consistency=safe_div(
            row.get('baseline_consistency_sum'),
            row.get('evaluated_regions')),
        keep_rate=safe_div(
            row.get('action_keep_regions'), row.get('evaluated_regions')),
        switch_rate=safe_div(
            row.get('action_switch_regions'), row.get('evaluated_regions')),
        abstain_rate=safe_div(
            row.get('action_abstain_regions'), row.get('evaluated_regions')),
        top2_recovery_rate=safe_div(
            row.get('action_switch_regions'),
            row.get('high_purity_top1_wrong_regions')),
        top2_label_coverage=safe_div(
            row.get('top2_label_covered_regions'),
            row.get('evaluated_regions')),
        gt_component_recall_iou25=safe_div(
            row.get('gt_recalled_iou025'), row.get('gt_components')),
        gt_component_recall_iou50=safe_div(
            row.get('gt_recalled_iou05'), row.get('gt_components')),
        gt_component_recall_coverage50=safe_div(
            row.get('gt_recalled_coverage05'), row.get('gt_components')),
        gt_component_recall_coverage80=safe_div(
            row.get('gt_recalled_coverage08'), row.get('gt_components')),
        mean_best_iou=safe_div(
            row.get('best_iou_sum'), row.get('gt_components')),
        mean_best_coverage=safe_div(
            row.get('best_coverage_sum'), row.get('gt_components')),
        fragmented_gt_rate=safe_div(
            row.get('fragmented_gt_components'), row.get('gt_components')),
        mean_fragment_links=safe_div(
            row.get('fragment_links'), row.get('gt_components')),
        fg_gt_component_recall_iou25=safe_div(
            row.get('fg_gt_recalled_iou025'),
            row.get('fg_gt_components')),
        fg_gt_component_recall_iou50=safe_div(
            row.get('fg_gt_recalled_iou05'),
            row.get('fg_gt_components')),
        fg_gt_component_recall_coverage50=safe_div(
            row.get('fg_gt_recalled_coverage05'),
            row.get('fg_gt_components')),
    )
    return result


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return 'NA' if value is None else f'{100.0 * value:.1f}%'


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')
    source_buckets = defaultdict(lambda: defaultdict(float))
    class_buckets = defaultdict(lambda: defaultdict(float))
    image_counts = defaultdict(int)
    for record in iter_records(paths):
        stats = record.get('candidate_region_quality_stats')
        if not stats:
            continue
        dataset = str(
            stats.get('dataset_name') or record.get('dataset_name')
            or 'unknown').lower()
        image_counts[dataset] += 1
        for row in stats.get('source_rows') or []:
            source = str(row['source'])
            add_numeric(source_buckets[(dataset, source)], row)
            add_numeric(source_buckets[('all', source)], row)
        for row in stats.get('class_rows') or []:
            source = str(row['source'])
            class_idx = int(row['class_index'])
            class_name = str(row['class_name'])
            is_background = bool(row.get('is_background', False))
            add_numeric(
                class_buckets[(
                    dataset, source, class_idx, class_name, is_background)],
                row)
            add_numeric(
                class_buckets[(
                    'all', source, class_idx, class_name, is_background)],
                row)

    source_rows = []
    for (dataset, source), bucket in sorted(source_buckets.items()):
        row = metrics(bucket)
        row = dict(
            dataset=dataset,
            source=source,
            images=(sum(image_counts.values())
                    if dataset == 'all' else image_counts[dataset]),
            **row,
        )
        source_rows.append(row)
    class_rows = []
    switch_class_counts = defaultdict(int)
    for key, bucket in sorted(class_buckets.items()):
        dataset, source, class_idx, class_name, is_background = key
        row = dict(
            dataset=dataset,
            source=source,
            class_index=class_idx,
            class_name=class_name,
            is_background=is_background,
            **bucket,
        )
        row['switch_rate'] = safe_div(
            bucket.get('action_switch_regions'), bucket.get('regions'))
        class_rows.append(row)
        if (
                not is_background
                and bucket.get('action_switch_regions', 0) > 0):
            switch_class_counts[(dataset, source)] += 1

    selected = {
        name.strip().lower()
        for name in args.selection_datasets.split(',') if name.strip()
    }
    thresholds = dict(
        min_iou25_recall=args.min_iou25_recall,
        min_high_purity_rate=args.min_high_purity_rate,
        min_top2_recovery_rate=args.min_top2_recovery_rate,
        min_switch_regions=args.min_switch_regions,
        min_switch_classes=args.min_switch_classes,
    )
    decisions = []
    for row in source_rows:
        if row['source'] != args.decision_source:
            continue
        dataset = row['dataset']
        if dataset != 'all' and dataset not in selected:
            continue
        gates = dict(
            proposal_recall=(row.get('fg_gt_component_recall_iou25') or 0.0)
            >= args.min_iou25_recall,
            region_purity=(row.get('high_purity_rate') or 0.0)
            >= args.min_high_purity_rate,
            top2_recoverability=(row.get('top2_recovery_rate') or 0.0)
            >= args.min_top2_recovery_rate,
            switch_sample_count=row.get('action_switch_regions', 0)
            >= args.min_switch_regions,
            switch_class_diversity=switch_class_counts[(dataset, row['source'])]
            >= args.min_switch_classes,
        )
        decisions.append(dict(
            dataset=dataset,
            source=row['source'],
            passed=all(gates.values()),
            passed_gates=sum(gates.values()),
            gates=gates,
            metrics={
                'fg_gt_component_recall_iou25': row.get(
                    'fg_gt_component_recall_iou25'),
                'high_purity_rate': row.get('high_purity_rate'),
                'top2_recovery_rate': row.get('top2_recovery_rate'),
                'switch_regions': row.get('action_switch_regions', 0),
                'switch_classes': switch_class_counts[(
                    dataset, row['source'])],
            },
        ))
    dataset_decisions = [
        row for row in decisions if row['dataset'] != 'all']
    all_decision = next(
        (row for row in decisions if row['dataset'] == 'all'), None)
    observed_datasets = {row['dataset'] for row in dataset_decisions}
    missing_datasets = sorted(selected - observed_datasets)
    if missing_datasets:
        verdict = 'INCOMPLETE'
    elif dataset_decisions and all(row['passed'] for row in dataset_decisions):
        verdict = 'GO'
    elif all_decision and all_decision['passed']:
        verdict = 'CONDITIONAL_GO'
    else:
        verdict = 'NO_GO'

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'source_quality.csv'), source_rows)
    write_csv(os.path.join(args.out_dir, 'class_actions.csv'), class_rows)
    result = dict(
        verdict=verdict,
        decision_source=args.decision_source,
        selection_datasets=sorted(selected),
        thresholds=thresholds,
        input_files=paths,
        missing_datasets=missing_datasets,
        decisions=decisions,
    )
    with open(os.path.join(args.out_dir, 'feasibility.json'), 'w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    report = [
        '# Candidate Region Quality Feasibility', '',
        f'**Verdict: {verdict}**', '',
        ('GO requires the same hybrid proposal definition to pass all five '
         'gates on every selected dataset. CONDITIONAL_GO means the pooled '
         'data pass but at least one dataset does not; NO_GO means even the '
         'pooled evidence is insufficient. INCOMPLETE means at least one '
         'required dataset output is missing.'), '',
        '| Dataset | IoU@0.25 recall | High-purity | Top2 recovery | Switch regions | Switch classes | Pass |',
        '|---|---:|---:|---:|---:|---:|:---:|',
    ]
    for decision in decisions:
        values = decision['metrics']
        report.append(
            f"| {decision['dataset']} | "
            f"{fmt(values['fg_gt_component_recall_iou25'])} | "
            f"{fmt(values['high_purity_rate'])} | "
            f"{fmt(values['top2_recovery_rate'])} | "
            f"{int(values['switch_regions'])} | "
            f"{int(values['switch_classes'])} | "
            f"{'yes' if decision['passed'] else 'no'} |")
    report.extend([
        '',
        'Interpretation:', '',
        '- Low proposal recall: improve/replace region construction before training.',
        '- Low purity: grouped masks are mixing physical objects; tighten grouping or split with semantic boundaries.',
        '- Low top2 recovery: a top1/top2 verifier is the wrong action space; expand candidate classes or change the task.',
        '- Too few switch samples/classes: supervision will be imbalanced; do not train a verifier yet.',
        '- Excess fragmentation/overlap is reported in `source_quality.csv` and should be treated as an efficiency/routing risk.',
    ])
    with open(os.path.join(args.out_dir, 'report.md'), 'w') as handle:
        handle.write('\n'.join(report) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
