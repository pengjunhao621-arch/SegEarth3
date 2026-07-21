import argparse
import math
import os
from collections import defaultdict

from summarize_candidate_residual_trajectory import (
    add_mechanism,
    add_metric,
    expand_inputs,
    iter_records,
    mechanism_values,
    metric_values,
    new_mechanism_bucket,
    new_metric_bucket,
    safe_div,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate a pre-registered low-effective-rank image gate on '
            'candidate residual trajectory JSONL files.'))
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--gate-layer', type=int, default=0)
    parser.add_argument(
        '--gate-descriptor', default='effective_rank')
    parser.add_argument('--gate-quantile', type=float, default=0.20)
    return parser.parse_args()


def record_identity(record, record_index):
    return str(
        record.get('img_path')
        or f"{record.get('rank', 0)}:{record_index}")


def record_dataset(record, stats):
    return str(
        stats.get('dataset_name')
        or record.get('dataset_name')
        or 'unknown')


def get_descriptor(stats, layer, descriptor):
    for row in stats.get('feature_stats') or []:
        if int(row.get('layer_index', -1)) != int(layer):
            continue
        variant_name = str(row.get('variant_name') or '')
        if not variant_name.endswith('_raw'):
            continue
        value = row.get(descriptor)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def build_gate_membership(paths, layer, descriptor, quantile):
    values = defaultdict(list)
    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get(
            'candidate_residual_trajectory_stats') or {}
        if not stats:
            continue
        dataset = record_dataset(record, stats)
        image_id = record_identity(record, record_index)
        value = get_descriptor(stats, layer, descriptor)
        if value is not None:
            values[dataset].append((value, image_id))

    selected = {}
    threshold_rows = []
    quantile = min(max(float(quantile), 0.0), 1.0)
    for dataset, items in sorted(values.items()):
        ordered = sorted(items, key=lambda item: (item[0], item[1]))
        count = len(ordered)
        selected_count = (
            0 if count == 0
            else max(1, int(math.ceil(quantile * count))))
        selected_items = ordered[:selected_count]
        selected[dataset] = {
            image_id for _, image_id in selected_items
        }
        threshold_rows.append(dict(
            dataset_name=dataset,
            gate_layer=int(layer),
            gate_descriptor=str(descriptor),
            requested_quantile=quantile,
            total_images=count,
            selected_images=selected_count,
            selected_ratio=safe_div(selected_count, count),
            maximum_selected_value=(
                selected_items[-1][0] if selected_items else None),
            minimum_remaining_value=(
                ordered[selected_count][0]
                if selected_count < count else None),
            dataset_minimum=ordered[0][0] if ordered else None,
            dataset_median=(
                ordered[count // 2][0] if ordered else None),
            dataset_maximum=ordered[-1][0] if ordered else None,
        ))
    return selected, threshold_rows


def make_dataset_rows(summary):
    rows = []
    for key, bucket in sorted(summary.items()):
        (
            gate_group, dataset, layer, variant, control, trial,
            score, rank, regime, task,
        ) = key
        rows.append(dict(
            gate_group=gate_group,
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            control_name=control,
            trial=trial,
            score_name=score,
            candidate_rank=rank,
            regime_name=regime,
            task_name=task,
            **metric_values(bucket),
        ))
    return rows


def make_pair_rows(summary):
    rows = []
    for key, bucket in sorted(summary.items()):
        (
            gate_group, dataset, layer, variant, control, trial,
            score, rank, regime, base_idx, base_name,
            candidate_idx, candidate_name, task,
        ) = key
        rows.append(dict(
            gate_group=gate_group,
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            control_name=control,
            trial=trial,
            score_name=score,
            candidate_rank=rank,
            regime_name=regime,
            task_name=task,
            base_class_index=base_idx,
            base_class_name=base_name,
            candidate_class_index=candidate_idx,
            candidate_class_name=candidate_name,
            **metric_values(bucket),
        ))
    return rows


def make_mechanism_rows(summary):
    rows = []
    for key, bucket in sorted(summary.items()):
        (
            gate_group, dataset, layer, variant,
            control, trial, rank,
        ) = key
        rows.append(dict(
            gate_group=gate_group,
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            control_name=control,
            trial=trial,
            candidate_rank=rank,
            **mechanism_values(bucket),
        ))
    return rows


def make_null_rows(dataset_rows):
    controls = defaultdict(list)
    for row in dataset_rows:
        if row['control_name'] == 'observed':
            continue
        key = (
            row['gate_group'],
            row['dataset_name'],
            row['layer_index'],
            row['variant_name'],
            row['score_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['task_name'],
        )
        controls[key].append(row)

    result = []
    for observed in dataset_rows:
        if observed['control_name'] != 'observed':
            continue
        key = (
            observed['gate_group'],
            observed['dataset_name'],
            observed['layer_index'],
            observed['variant_name'],
            observed['score_name'],
            observed['candidate_rank'],
            observed['regime_name'],
            observed['task_name'],
        )
        comparisons = controls.get(key, [])
        for control_name in sorted({
                row['control_name'] for row in comparisons}):
            selected = [
                row for row in comparisons
                if row['control_name'] == control_name
            ]
            auc_values = [
                row['matched_auroc'] for row in selected
                if row['matched_auroc'] is not None
            ]
            ap_values = [
                row['matched_auprc_lift'] for row in selected
                if row['matched_auprc_lift'] is not None
            ]
            null_auc_max = max(auc_values) if auc_values else None
            null_ap_max = max(ap_values) if ap_values else None
            result.append(dict(
                gate_group=observed['gate_group'],
                dataset_name=observed['dataset_name'],
                layer_index=observed['layer_index'],
                variant_name=observed['variant_name'],
                score_name=observed['score_name'],
                candidate_rank=observed['candidate_rank'],
                regime_name=observed['regime_name'],
                task_name=observed['task_name'],
                null_control_name=control_name,
                null_trials=len(selected),
                observed_auroc=observed['matched_auroc'],
                null_auroc_mean=safe_div(
                    sum(auc_values), len(auc_values)),
                null_auroc_max=null_auc_max,
                auroc_minus_best_null=(
                    None
                    if observed['matched_auroc'] is None
                    or null_auc_max is None
                    else observed['matched_auroc']
                    - null_auc_max),
                observed_auprc_lift=(
                    observed['matched_auprc_lift']),
                null_auprc_lift_mean=safe_div(
                    sum(ap_values), len(ap_values)),
                null_auprc_lift_max=null_ap_max,
                auprc_lift_minus_best_null=(
                    None
                    if observed['matched_auprc_lift'] is None
                    or null_ap_max is None
                    else observed['matched_auprc_lift']
                    - null_ap_max),
            ))
    return result


def make_decision_rows(dataset_rows, null_rows):
    groups = defaultdict(list)
    for row in dataset_rows:
        if row['control_name'] != 'observed':
            continue
        key = (
            row['gate_group'],
            row['layer_index'],
            row['variant_name'],
            row['score_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['task_name'],
        )
        groups[key].append(row)

    null_lookup = defaultdict(list)
    for row in null_rows:
        key = (
            row['gate_group'],
            row['dataset_name'],
            row['layer_index'],
            row['variant_name'],
            row['score_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['task_name'],
        )
        null_lookup[key].append(row)

    result = []
    for key, rows in sorted(groups.items()):
        (
            gate_group, layer, variant, score,
            rank, regime, task,
        ) = key
        auc_values = [
            row['matched_auroc'] for row in rows
            if row['matched_auroc'] is not None
        ]
        ap_values = [
            row['matched_auprc_lift'] for row in rows
            if row['matched_auprc_lift'] is not None
        ]
        auc_gaps = []
        ap_gaps = []
        beats_all = 0
        for row in rows:
            comparisons = null_lookup.get((
                gate_group,
                row['dataset_name'],
                row['layer_index'],
                row['variant_name'],
                row['score_name'],
                row['candidate_rank'],
                row['regime_name'],
                row['task_name'],
            ), [])
            valid_auc = [
                item['auroc_minus_best_null']
                for item in comparisons
                if item['auroc_minus_best_null'] is not None
            ]
            valid_ap = [
                item['auprc_lift_minus_best_null']
                for item in comparisons
                if item['auprc_lift_minus_best_null'] is not None
            ]
            if valid_auc:
                auc_gaps.append(min(valid_auc))
            if valid_ap:
                ap_gaps.append(min(valid_ap))
            if (
                    valid_auc and valid_ap
                    and min(valid_auc) > 0.0
                    and min(valid_ap) > 0.0):
                beats_all += 1
        result.append(dict(
            gate_group=gate_group,
            layer_index=layer,
            variant_name=variant,
            score_name=score,
            candidate_rank=rank,
            regime_name=regime,
            task_name=task,
            datasets=len(rows),
            mean_matched_auroc=safe_div(
                sum(auc_values), len(auc_values)),
            datasets_auroc_ge_0p60=sum(
                value >= 0.60 for value in auc_values),
            mean_matched_auprc_lift=safe_div(
                sum(ap_values), len(ap_values)),
            datasets_positive_auprc_lift=sum(
                value > 0.0 for value in ap_values),
            mean_auroc_minus_best_null=safe_div(
                sum(auc_gaps), len(auc_gaps)),
            positive_auroc_null_gap_datasets=sum(
                value > 0.0 for value in auc_gaps),
            mean_auprc_lift_minus_best_null=safe_div(
                sum(ap_gaps), len(ap_gaps)),
            positive_auprc_null_gap_datasets=sum(
                value > 0.0 for value in ap_gaps),
            datasets_beating_all_nulls=beats_all,
        ))
    result.sort(
        key=lambda row: (
            row['gate_group'] == 'low_quantile',
            row['datasets_beating_all_nulls'],
            row['positive_auprc_null_gap_datasets'],
            row['positive_auroc_null_gap_datasets'],
            row['mean_auprc_lift_minus_best_null']
            if row['mean_auprc_lift_minus_best_null'] is not None
            else -1e9,
            row['mean_auroc_minus_best_null']
            if row['mean_auroc_minus_best_null'] is not None
            else -1e9,
        ),
        reverse=True,
    )
    for index, row in enumerate(result):
        row['decision_rank'] = index + 1
    return result


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')

    selected, threshold_rows = build_gate_membership(
        paths,
        args.gate_layer,
        args.gate_descriptor,
        args.gate_quantile,
    )
    dataset_summary = defaultdict(new_metric_bucket)
    pair_summary = defaultdict(new_metric_bucket)
    mechanism_summary = defaultdict(new_mechanism_bucket)
    seen_images = set()

    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get(
            'candidate_residual_trajectory_stats') or {}
        if not stats:
            continue
        dataset = record_dataset(record, stats)
        image_id = record_identity(record, record_index)
        if dataset not in selected:
            continue
        gate_group = (
            'low_quantile'
            if image_id in selected[dataset]
            else 'remaining'
        )
        seen_images.add((dataset, image_id))

        for row in stats.get('mechanism_stats') or []:
            key = (
                gate_group,
                dataset,
                row.get('layer_index'),
                row.get('variant_name'),
                row.get('control_name'),
                row.get('trial'),
                row.get('candidate_rank'),
            )
            add_mechanism(
                mechanism_summary[key], image_id, row)

        for row in stats.get('matched_stats') or []:
            row_scope = str(row.get('row_scope') or 'both')
            common = (
                gate_group,
                dataset,
                row.get('layer_index'),
                row.get('variant_name'),
                row.get('control_name'),
                row.get('trial'),
                row.get('score_name'),
                row.get('candidate_rank'),
                row.get('regime_name'),
            )
            pair = (
                row.get('base_class_index'),
                row.get('base_class_name'),
                row.get('candidate_class_index'),
                row.get('candidate_class_name'),
            )
            for task_name, metric in (
                    row.get('task_metrics') or {}).items():
                if row_scope in ('dataset', 'both'):
                    add_metric(
                        dataset_summary[common + (task_name,)],
                        image_id,
                        metric,
                    )
                if row_scope in ('pair', 'both'):
                    add_metric(
                        pair_summary[
                            common + pair + (task_name,)],
                        image_id,
                        metric,
                    )

    dataset_rows = make_dataset_rows(dataset_summary)
    pair_rows = make_pair_rows(pair_summary)
    mechanism_rows = make_mechanism_rows(mechanism_summary)
    null_rows = make_null_rows(dataset_rows)
    decision_rows = make_decision_rows(
        dataset_rows, null_rows)

    write_csv(
        os.path.join(args.out_dir, 'gate_thresholds.csv'),
        threshold_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'gated_trajectory_dataset_summary.csv'),
        dataset_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'gated_trajectory_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'gated_trajectory_mechanism_summary.csv'),
        mechanism_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'gated_trajectory_null_summary.csv'),
        null_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'gated_trajectory_decision_summary.csv'),
        decision_rows)

    print(f'Read {len(paths)} JSONL files.')
    print(f'Found gated trajectory stats for {len(seen_images)} images.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
