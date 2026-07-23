#!/usr/bin/env python3
"""Summarize query-provenance diagnostics with predeclared controls and gates."""

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict


TOPOLOGY_METRICS = (
    'dominant_source_coverage',
    'effective_source_count',
    'switch_boundary_density',
    'consensus_fraction',
)
QUALITY_DIRECTIONS = {
    'dominant_source_coverage': 1.0,
    'effective_source_count': -1.0,
    'switch_boundary_density': -1.0,
    'consensus_fraction': 1.0,
}
ACTION_DIRECTIONS = {
    'dominant_source_coverage': -1.0,
    'effective_source_count': 1.0,
    'switch_boundary_density': 1.0,
    'consensus_fraction': -1.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--expected-datasets',
        default='udd5,vdd,vaihingen,potsdam,openearthmap,loveda,isaid',
    )
    parser.add_argument('--min-foreground-regions', type=int, default=50)
    parser.add_argument('--max-reconstruction-mae', type=float, default=0.05)
    parser.add_argument('--min-action-delta', type=float, default=0.02)
    parser.add_argument('--min-improvable-rate', type=float, default=0.05)
    parser.add_argument(
        '--min-mean-positive-oracle-gain', type=float, default=0.01)
    parser.add_argument('--min-conditional-lift', type=float, default=0.02)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if os.path.isfile(pattern) else []))
    return sorted(set(paths))


def iter_records(paths):
    for path in paths:
        with open(path) as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record['_source_file'] = path
                record['_source_line'] = line_number
                yield record


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def safe_mean(values):
    values = [float(value) for value in values if finite_number(value)]
    return sum(values) / len(values) if values else None


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def bin_index(value, edges):
    if not finite_number(value):
        return -1
    for index, edge in enumerate(edges):
        if float(value) < edge:
            return index
    return len(edges)


def control_cell(row):
    area = max(1.0, float(row.get('area_pixels', 1)))
    return (
        int(row.get('class_index', -1)),
        bin_index(
            row.get('final_score_mean'),
            (0.05, 0.10, 0.20, 0.40, 0.60, 0.80),
        ),
        int(math.floor(math.log(area, 2))),
        bin_index(
            row.get('instance_head_fraction'),
            (0.10, 0.50, 0.90),
        ),
    )


def pearson(pairs):
    if len(pairs) < 3:
        return None
    xs = [float(pair[0]) for pair in pairs]
    ys = [float(pair[1]) for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def fisher_interval(correlation, count):
    if correlation is None or count <= 3:
        return None, None
    clipped = min(0.999999, max(-0.999999, float(correlation)))
    z_value = math.atanh(clipped)
    half_width = 1.96 / math.sqrt(count - 3)
    return math.tanh(z_value - half_width), math.tanh(
        z_value + half_width)


def residual_correlation(rows, feature, target):
    cells = defaultdict(list)
    for row in rows:
        if finite_number(row.get(feature)) and finite_number(row.get(target)):
            cells[control_cell(row)].append(row)
    pairs = []
    used_cells = 0
    for cell_rows in cells.values():
        if len(cell_rows) < 3:
            continue
        mean_feature = safe_mean(row[feature] for row in cell_rows)
        mean_target = safe_mean(row[target] for row in cell_rows)
        used_cells += 1
        pairs.extend(
            (
                float(row[feature]) - mean_feature,
                float(row[target]) - mean_target,
            )
            for row in cell_rows
        )
    correlation = pearson(pairs)
    low, high = fisher_interval(correlation, len(pairs))
    return dict(
        correlation=correlation,
        ci95_low=low,
        ci95_high=high,
        samples=len(pairs),
        control_cells=used_cells,
    )


def write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted(set().union(
        *(row.keys() for row in rows))) if rows else []
    with open(path, 'w', newline='') as handle:
        if not fieldnames:
            handle.write('')
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_records(records):
    images = []
    regions = []
    for record in records:
        stats = record.get('query_topology_stats')
        if not isinstance(stats, dict):
            continue
        dataset = str(
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown'
        ).lower()
        image = dict(
            dataset=dataset,
            img_path=record.get('img_path'),
            rank=record.get('rank'),
            schema_version=stats.get('schema_version'),
            diagnostic_height=(
                stats.get('diagnostic_shape') or [None, None])[0],
            diagnostic_width=(
                stats.get('diagnostic_shape') or [None, None])[1],
            prediction_sha1=stats.get('prediction_sha1'),
            raw_record_count=stats.get('raw_record_count'),
            expected_raw_record_count=stats.get(
                'expected_raw_record_count'),
            raw_record_complete=stats.get('raw_record_complete'),
            reported_candidate_count=stats.get(
                'reported_candidate_count'),
            reported_kept_count=stats.get('reported_kept_count'),
            selected_kept_count=stats.get('selected_kept_count'),
            kept_selection_complete=stats.get(
                'kept_selection_complete'),
            reconstruction_mae=stats.get('reconstruction_mae'),
            reconstruction_max_error=stats.get(
                'reconstruction_max_error'),
            region_count=stats.get('region_count'),
            dropped_region_count=stats.get(
                'dropped_region_count'),
            npz_path=(stats.get('artifact_paths') or {}).get('npz'),
            sources_path=(
                stats.get('artifact_paths') or {}).get('sources'),
            source_file=record.get('_source_file'),
            source_line=record.get('_source_line'),
        )
        images.append(image)
        for region in stats.get('regions') or []:
            row = dict(region)
            row.update(
                dataset=dataset,
                img_path=record.get('img_path'),
                rank=record.get('rank'),
                prediction_sha1=stats.get('prediction_sha1'),
                reconstruction_mae=stats.get('reconstruction_mae'),
                kept_selection_complete=stats.get(
                    'kept_selection_complete'),
                source_file=record.get('_source_file'),
                source_line=record.get('_source_line'),
            )
            regions.append(row)
    return images, regions


def build_conditional_rows(regions):
    output = []
    datasets = sorted(set(row['dataset'] for row in regions))
    targets = ('gt_purity', 'baseline_local_iou', 'best_action_delta')
    for dataset in datasets:
        foreground = [
            row for row in regions
            if row['dataset'] == dataset
            and not bool(row.get('is_background'))
        ]
        for target in targets:
            for metric in TOPOLOGY_METRICS:
                true_stats = residual_correlation(
                    foreground, metric, target)
                control_metric = f'control_roll_{metric}'
                control_stats = residual_correlation(
                    foreground, control_metric, target)
                direction = (
                    ACTION_DIRECTIONS[metric]
                    if target == 'best_action_delta'
                    else QUALITY_DIRECTIONS[metric]
                )
                true_corr = true_stats['correlation']
                control_corr = control_stats['correlation']
                directional_lift = None
                if true_corr is not None:
                    directional_lift = (
                        direction * true_corr
                        - abs(control_corr or 0.0)
                    )
                output.append(dict(
                    dataset=dataset,
                    target=target,
                    metric=metric,
                    expected_direction=int(direction),
                    control_metric=control_metric,
                    true_correlation=true_corr,
                    true_ci95_low=true_stats['ci95_low'],
                    true_ci95_high=true_stats['ci95_high'],
                    control_correlation=control_corr,
                    control_ci95_low=control_stats['ci95_low'],
                    control_ci95_high=control_stats['ci95_high'],
                    samples=true_stats['samples'],
                    control_cells=true_stats['control_cells'],
                    directional_lift=directional_lift,
                ))
    return output


def build_formation_rows(regions):
    groups = defaultdict(list)
    for row in regions:
        if not bool(row.get('is_background')):
            groups[(row['dataset'], row.get('formation'))].append(row)
    output = []
    for (dataset, formation), rows in sorted(groups.items()):
        output.append(dict(
            dataset=dataset,
            formation=formation,
            regions=len(rows),
            pixels=sum(int(row.get('area_pixels', 0)) for row in rows),
            mean_gt_purity=safe_mean(
                row.get('gt_purity') for row in rows),
            mean_baseline_local_iou=safe_mean(
                row.get('baseline_local_iou') for row in rows),
            mean_best_action_delta=safe_mean(
                row.get('best_action_delta') for row in rows),
            improvable_rate=safe_div(
                sum(
                    1 for row in rows
                    if finite_number(row.get('best_action_delta'))
                    and float(row['best_action_delta']) >= 0.02
                ),
                len(rows),
            ),
            mean_dominant_source_coverage=safe_mean(
                row.get('dominant_source_coverage') for row in rows),
            mean_effective_source_count=safe_mean(
                row.get('effective_source_count') for row in rows),
            mean_switch_boundary_density=safe_mean(
                row.get('switch_boundary_density') for row in rows),
            mean_consensus_fraction=safe_mean(
                row.get('consensus_fraction') for row in rows),
        ))
    return output


def build_action_rows(regions, min_delta):
    groups = defaultdict(list)
    for row in regions:
        if not bool(row.get('is_background')):
            groups[row['dataset']].append(row)
    output = []
    for dataset, rows in sorted(groups.items()):
        deltas = [
            float(row['best_action_delta'])
            for row in rows
            if finite_number(row.get('best_action_delta'))
        ]
        winners = Counter(
            row.get('best_action') for row in rows
            if row.get('best_action'))
        output.append(dict(
            dataset=dataset,
            foreground_regions=len(rows),
            oracle_improvable_regions=sum(
                delta >= min_delta for delta in deltas),
            oracle_improvable_rate=safe_div(
                sum(delta >= min_delta for delta in deltas),
                len(deltas),
            ),
            mean_best_action_delta=safe_mean(deltas),
            mean_positive_oracle_gain=safe_mean(
                max(0.0, delta) for delta in deltas),
            median_best_action_delta=(
                sorted(deltas)[len(deltas) // 2] if deltas else None),
            dominant_query_wins=winners['dominant_query'],
            query_union_wins=winners['query_union'],
            query_consensus_wins=winners['query_consensus'],
            semantic_completion_wins=winners['semantic_completion'],
        ))
    return output


def build_gate_rows(
        images,
        regions,
        conditional_rows,
        action_rows,
        expected_datasets,
        args,
):
    images_by_dataset = defaultdict(list)
    regions_by_dataset = defaultdict(list)
    for row in images:
        images_by_dataset[row['dataset']].append(row)
    for row in regions:
        if not bool(row.get('is_background')):
            regions_by_dataset[row['dataset']].append(row)
    action_by_dataset = {
        row['dataset']: row for row in action_rows
    }
    signal_by_dataset = defaultdict(list)
    for row in conditional_rows:
        if finite_number(row.get('directional_lift')):
            signal_by_dataset[row['dataset']].append(
                float(row['directional_lift']))

    output = []
    for dataset in expected_datasets:
        dataset_images = images_by_dataset.get(dataset, [])
        dataset_regions = regions_by_dataset.get(dataset, [])
        action = action_by_dataset.get(dataset, {})
        max_lift = (
            max(signal_by_dataset[dataset])
            if signal_by_dataset.get(dataset) else None
        )
        present = bool(dataset_images)
        selection_complete = (
            present
            and all(
                bool(row.get('kept_selection_complete'))
                and bool(row.get('raw_record_complete'))
                    for row in dataset_images)
        )
        reconstruction_mae = safe_mean(
            row.get('reconstruction_mae') for row in dataset_images)
        reconstruction_ok = (
            reconstruction_mae is not None
            and reconstruction_mae <= args.max_reconstruction_mae
        )
        sample_ok = len(dataset_regions) >= args.min_foreground_regions
        oracle_rate = action.get('oracle_improvable_rate')
        oracle_gain = action.get('mean_positive_oracle_gain')
        oracle_ok = (
            finite_number(oracle_rate)
            and float(oracle_rate) >= args.min_improvable_rate
            and finite_number(oracle_gain)
            and float(oracle_gain)
            >= args.min_mean_positive_oracle_gain
        )
        signal_ok = (
            finite_number(max_lift)
            and float(max_lift) >= args.min_conditional_lift
        )
        passed = bool(
            present
            and selection_complete
            and reconstruction_ok
            and sample_ok
            and oracle_ok
            and signal_ok
        )
        failures = []
        if not present:
            failures.append('missing_dataset')
        if present and not selection_complete:
            failures.append('query_record_or_kept_selection_incomplete')
        if not reconstruction_ok:
            failures.append('instance_reconstruction_failed')
        if not sample_ok:
            failures.append('insufficient_foreground_regions')
        if not oracle_ok:
            failures.append('insufficient_spatial_action_oracle')
        if not signal_ok:
            failures.append('topology_not_incremental_over_roll_control')
        output.append(dict(
            dataset=dataset,
            images=len(dataset_images),
            foreground_regions=len(dataset_regions),
            selection_complete=selection_complete,
            reconstruction_mae=reconstruction_mae,
            max_conditional_directional_lift=max_lift,
            oracle_improvable_rate=oracle_rate,
            mean_positive_oracle_gain=oracle_gain,
            passed=passed,
            failures=';'.join(failures),
        ))
    return output


def render_report(path, decision, gates, action_rows, conditional_rows):
    lines = [
        '# Query-topology feasibility report',
        '',
        f"Decision: **{decision['decision']}**",
        '',
        (
            'This is a prediction-preserving diagnostic. Ground truth is used '
            'only for offline region quality and action-oracle measurement.'
        ),
        '',
        '## Dataset gates',
        '',
        '| Dataset | FG regions | Reconstruction MAE | Oracle rate | '
        'Positive oracle gain | Conditional lift | Pass |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in gates:
        lines.append(
            '| {dataset} | {foreground_regions} | {recon} | {oracle} | '
            '{gain} | {lift} | {passed} |'.format(
                dataset=row['dataset'],
                foreground_regions=row['foreground_regions'],
                recon=_format(row.get('reconstruction_mae')),
                oracle=_format(row.get('oracle_improvable_rate')),
                gain=_format(row.get('mean_positive_oracle_gain')),
                lift=_format(row.get(
                    'max_conditional_directional_lift')),
                passed='yes' if row['passed'] else 'no',
            )
        )
    lines.extend([
        '',
        '## Interpretation contract',
        '',
        '- A positive oracle says a fixed, GT-free candidate action sometimes '
        'contains better spatial support; it is not an implementable selector.',
        '- A conditional topology signal must remain after class, confidence, '
        'area, and semantic/instance-mix control and exceed the deterministic '
        'spatial-roll null.',
        '- Any missing dataset, incomplete kept-query capture, or failed '
        'instance reconstruction makes the evidence incomplete.',
        '- Only an all-dataset pass justifies designing a prediction-changing '
        'topology-aware mechanism. It does not establish that such a mechanism '
        'will improve mIoU.',
        '',
        '## Failed gates',
        '',
    ])
    failed = [row for row in gates if not row['passed']]
    if failed:
        for row in failed:
            lines.append(f"- {row['dataset']}: {row['failures']}")
    else:
        lines.append('- None.')
    lines.extend([
        '',
        'Detailed evidence is in `region_topology.csv`, '
        '`conditional_signal.csv`, `formation_summary.csv`, '
        '`action_oracle.csv`, and `image_summary.csv`.',
        '',
    ])
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))


def _format(value):
    return 'n/a' if not finite_number(value) else f'{float(value):.4f}'


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit('No query-topology JSONL files matched --inputs.')
    records = list(iter_records(paths))
    images, regions = flatten_records(records)
    if not images:
        raise SystemExit(
            'No records contained query_topology_stats dictionaries.')
    expected = [
        item.strip().lower()
        for item in args.expected_datasets.split(',')
        if item.strip()
    ]
    conditional_rows = build_conditional_rows(regions)
    formation_rows = build_formation_rows(regions)
    action_rows = build_action_rows(regions, args.min_action_delta)
    gate_rows = build_gate_rows(
        images,
        regions,
        conditional_rows,
        action_rows,
        expected,
        args,
    )
    missing = [
        row['dataset'] for row in gate_rows if row['images'] == 0
    ]
    if missing:
        verdict = 'INCOMPLETE'
    elif all(row['passed'] for row in gate_rows):
        verdict = 'GO_TO_PREDICTION_METHOD_DESIGN'
    else:
        verdict = 'NO_GO_FOR_QUERY_TOPOLOGY_METHOD'
    decision = dict(
        decision=verdict,
        expected_datasets=expected,
        input_files=paths,
        image_records=len(images),
        region_records=len(regions),
        thresholds=dict(
            min_foreground_regions=args.min_foreground_regions,
            max_reconstruction_mae=args.max_reconstruction_mae,
            min_action_delta=args.min_action_delta,
            min_improvable_rate=args.min_improvable_rate,
            min_mean_positive_oracle_gain=(
                args.min_mean_positive_oracle_gain),
            min_conditional_lift=args.min_conditional_lift,
        ),
        dataset_gates=gate_rows,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'image_summary.csv'), images)
    write_csv(os.path.join(args.out_dir, 'region_topology.csv'), regions)
    write_csv(
        os.path.join(args.out_dir, 'conditional_signal.csv'),
        conditional_rows,
    )
    write_csv(
        os.path.join(args.out_dir, 'formation_summary.csv'),
        formation_rows,
    )
    write_csv(
        os.path.join(args.out_dir, 'action_oracle.csv'),
        action_rows,
    )
    write_csv(
        os.path.join(args.out_dir, 'dataset_gate.csv'),
        gate_rows,
    )
    with open(os.path.join(args.out_dir, 'decision.json'), 'w') as handle:
        json.dump(decision, handle, indent=2, allow_nan=False)
    render_report(
        os.path.join(args.out_dir, 'report.md'),
        decision,
        gate_rows,
        action_rows,
        conditional_rows,
    )
    print(json.dumps(
        dict(
            decision=verdict,
            images=len(images),
            regions=len(regions),
            out_dir=args.out_dir,
        ),
        allow_nan=False,
    ))


if __name__ == '__main__':
    main()
