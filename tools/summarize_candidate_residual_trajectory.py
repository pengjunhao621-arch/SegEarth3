import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize candidate residual trajectories, recovery mechanisms, '
            'and matched null controls.'))
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(set(paths))


def iter_records(paths):
    for path in paths:
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return numerator / denominator


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as file:
            file.write('')
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_metric_bucket():
    return dict(
        images=set(),
        strata=0,
        positive_pixels=0,
        negative_pixels=0,
        comparison_weight=0,
        auroc_weighted=0.0,
        ap_weight=0,
        auprc_weighted=0.0,
        auprc_lift_weighted=0.0,
    )


def add_metric(bucket, image_id, metric):
    positive = int(metric.get('positive_pixels') or 0)
    negative = int(metric.get('negative_pixels') or 0)
    comparison_weight = int(metric.get('comparison_weight') or 0)
    bucket['images'].add(image_id)
    bucket['strata'] += 1
    bucket['positive_pixels'] += positive
    bucket['negative_pixels'] += negative
    bucket['comparison_weight'] += comparison_weight
    if metric.get('auroc') is not None:
        bucket['auroc_weighted'] += (
            float(metric['auroc']) * comparison_weight)
    if metric.get('auprc') is not None:
        bucket['auprc_weighted'] += (
            float(metric['auprc']) * positive)
        bucket['auprc_lift_weighted'] += (
            float(metric.get('auprc_lift') or 0.0) * positive)
        bucket['ap_weight'] += positive


def metric_values(bucket):
    return dict(
        images=len(bucket['images']),
        matched_strata=int(bucket['strata']),
        positive_pixels=int(bucket['positive_pixels']),
        negative_pixels=int(bucket['negative_pixels']),
        matched_auroc=safe_div(
            bucket['auroc_weighted'],
            bucket['comparison_weight'],
        ),
        matched_auprc=safe_div(
            bucket['auprc_weighted'],
            bucket['ap_weight'],
        ),
        matched_auprc_lift=safe_div(
            bucket['auprc_lift_weighted'],
            bucket['ap_weight'],
        ),
    )


def new_mechanism_bucket():
    return dict(images=set(), sums=defaultdict(int))


MECHANISM_FIELDS = (
    'eligible_pixels',
    'help_pixels',
    'harm_pixels',
    'help_positive_candidate_gain_pixels',
    'help_positive_margin_gain_pixels',
    'help_candidate_dominant_pixels',
    'help_competitor_only_pixels',
    'help_crossing_pixels',
    'harm_positive_margin_gain_pixels',
)


def add_mechanism(bucket, image_id, row):
    bucket['images'].add(image_id)
    for name in MECHANISM_FIELDS:
        bucket['sums'][name] += int(row.get(name) or 0)


def mechanism_values(bucket):
    values = dict(
        images=len(bucket['images']),
        **{
            name: int(bucket['sums'][name])
            for name in MECHANISM_FIELDS
        },
    )
    help_pixels = values['help_pixels']
    harm_pixels = values['harm_pixels']
    values.update(
        help_positive_candidate_gain_ratio=safe_div(
            values['help_positive_candidate_gain_pixels'],
            help_pixels,
        ),
        help_positive_margin_gain_ratio=safe_div(
            values['help_positive_margin_gain_pixels'],
            help_pixels,
        ),
        help_candidate_dominant_ratio=safe_div(
            values['help_candidate_dominant_pixels'],
            help_pixels,
        ),
        help_competitor_only_ratio=safe_div(
            values['help_competitor_only_pixels'],
            help_pixels,
        ),
        help_crossing_ratio=safe_div(
            values['help_crossing_pixels'],
            help_pixels,
        ),
        harm_positive_margin_gain_ratio=safe_div(
            values['harm_positive_margin_gain_pixels'],
            harm_pixels,
        ),
    )
    return values


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')

    dataset_metrics = defaultdict(new_metric_bucket)
    pair_metrics = defaultdict(new_metric_bucket)
    margin_metrics = defaultdict(new_metric_bucket)
    mechanism_summary = defaultdict(new_mechanism_bucket)
    inventory = {}
    image_feature_rows = []
    seen_images = set()

    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get(
            'candidate_residual_trajectory_stats') or {}
        if not stats:
            continue
        dataset = str(
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown')
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{record_index}")
        seen_images.add((dataset, image_id))

        for feature_index, row in enumerate(
                stats.get('feature_stats') or []):
            image_feature_rows.append(dict(
                dataset_name=dataset,
                image_id=image_id,
                feature_index=feature_index,
                layer_index=row.get('layer_index'),
                variant_name=row.get('variant_name'),
                spatial_pixels=row.get('spatial_pixels'),
                feature_channels=row.get('feature_channels'),
                mean_norm_ratio=row.get('mean_norm_ratio'),
                common_energy_ratio=row.get('common_energy_ratio'),
                mean_token_cosine=row.get('mean_token_cosine'),
                effective_rank=row.get('effective_rank'),
                top1_variance_ratio=row.get(
                    'top1_variance_ratio'),
                top4_variance_ratio=row.get(
                    'top4_variance_ratio'),
                top8_variance_ratio=row.get(
                    'top8_variance_ratio'),
                seed_ratio=row.get('seed_ratio'),
                seed_class_count=row.get('seed_class_count'),
                seed_prototype_dispersion=row.get(
                    'seed_prototype_dispersion'),
            ))

        for row in stats.get('source_inventory') or []:
            key = (
                dataset,
                row.get('layer_index'),
                row.get('variant_name'),
                row.get('control_name'),
                row.get('trial'),
            )
            if key not in inventory:
                inventory[key] = dict(
                    dataset_name=dataset,
                    layer_index=row.get('layer_index'),
                    variant_name=row.get('variant_name'),
                    control_name=row.get('control_name'),
                    trial=row.get('trial'),
                    strengths='|'.join(
                        str(value)
                        for value in row.get('strengths') or []),
                    source_names='|'.join(
                        str(value)
                        for value in row.get('source_names') or []),
                    images=0,
                )
            inventory[key]['images'] += 1

        for row in stats.get('mechanism_stats') or []:
            key = (
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
                dataset,
                row.get('layer_index'),
                row.get('variant_name'),
                row.get('control_name'),
                row.get('trial'),
                row.get('score_name'),
                row.get('candidate_rank'),
                row.get('regime_name'),
            )
            margin = (
                row.get('margin_name'),
                row.get('margin_lower'),
                row.get('margin_upper'),
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
                        dataset_metrics[common + (task_name,)],
                        image_id,
                        metric,
                    )
                    add_metric(
                        margin_metrics[
                            common + margin + (task_name,)],
                        image_id,
                        metric,
                    )
                if row_scope in ('pair', 'both'):
                    add_metric(
                        pair_metrics[
                            common + pair + (task_name,)],
                        image_id,
                        metric,
                    )

    dataset_rows = []
    dataset_index = {}
    for key, bucket in sorted(dataset_metrics.items()):
        (
            dataset, layer, variant, control, trial, score,
            rank, regime, task,
        ) = key
        row = dict(
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
        )
        dataset_rows.append(row)
        dataset_index[key] = row

    pair_rows = []
    for key, bucket in sorted(pair_metrics.items()):
        (
            dataset, layer, variant, control, trial, score,
            rank, regime, base_idx, base_name,
            candidate_idx, candidate_name, task,
        ) = key
        pair_rows.append(dict(
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

    margin_rows = []
    for key, bucket in sorted(margin_metrics.items()):
        (
            dataset, layer, variant, control, trial, score,
            rank, regime, margin_name, margin_lower,
            margin_upper, task,
        ) = key
        margin_rows.append(dict(
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            control_name=control,
            trial=trial,
            score_name=score,
            candidate_rank=rank,
            regime_name=regime,
            margin_name=margin_name,
            margin_lower=margin_lower,
            margin_upper=margin_upper,
            task_name=task,
            **metric_values(bucket),
        ))

    mechanism_rows = []
    mechanism_index = {}
    for key, bucket in sorted(mechanism_summary.items()):
        dataset, layer, variant, control, trial, rank = key
        row = dict(
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            control_name=control,
            trial=trial,
            candidate_rank=rank,
            **mechanism_values(bucket),
        )
        mechanism_rows.append(row)
        mechanism_index[key] = row

    control_lookup = defaultdict(list)
    for row in dataset_rows:
        if row['control_name'] == 'observed':
            continue
        key = (
            row['dataset_name'],
            row['layer_index'],
            row['variant_name'],
            row['score_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['task_name'],
        )
        control_lookup[key].append(row)

    null_rows = []
    for observed in dataset_rows:
        if observed['control_name'] != 'observed':
            continue
        key = (
            observed['dataset_name'],
            observed['layer_index'],
            observed['variant_name'],
            observed['score_name'],
            observed['candidate_rank'],
            observed['regime_name'],
            observed['task_name'],
        )
        controls = control_lookup.get(key, [])
        for control_name in sorted({
                row['control_name'] for row in controls}):
            selected = [
                row for row in controls
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
            null_auc_mean = safe_div(
                sum(auc_values), len(auc_values))
            null_ap_mean = safe_div(
                sum(ap_values), len(ap_values))
            null_rows.append(dict(
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
                null_auroc_mean=null_auc_mean,
                null_auroc_max=(
                    max(auc_values) if auc_values else None),
                auroc_minus_null_mean=(
                    None
                    if observed['matched_auroc'] is None
                    or null_auc_mean is None
                    else observed['matched_auroc']
                    - null_auc_mean),
                auroc_minus_best_null=(
                    None
                    if observed['matched_auroc'] is None
                    or not auc_values
                    else observed['matched_auroc']
                    - max(auc_values)),
                observed_auprc_lift=(
                    observed['matched_auprc_lift']),
                null_auprc_lift_mean=null_ap_mean,
                null_auprc_lift_max=(
                    max(ap_values) if ap_values else None),
                auprc_lift_minus_null_mean=(
                    None
                    if observed['matched_auprc_lift'] is None
                    or null_ap_mean is None
                    else observed['matched_auprc_lift']
                    - null_ap_mean),
                auprc_lift_minus_best_null=(
                    None
                    if observed['matched_auprc_lift'] is None
                    or not ap_values
                    else observed['matched_auprc_lift']
                    - max(ap_values)),
            ))

    decision_groups = defaultdict(list)
    for observed in dataset_rows:
        if observed['control_name'] != 'observed':
            continue
        key = (
            observed['layer_index'],
            observed['variant_name'],
            observed['score_name'],
            observed['candidate_rank'],
            observed['regime_name'],
            observed['task_name'],
        )
        decision_groups[key].append(observed)

    null_by_observed = defaultdict(list)
    for row in null_rows:
        key = (
            row['dataset_name'],
            row['layer_index'],
            row['variant_name'],
            row['score_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['task_name'],
        )
        null_by_observed[key].append(row)

    decision_rows = []
    for key, rows in sorted(decision_groups.items()):
        layer, variant, score, rank, regime, task = key
        auc_values = [
            row['matched_auroc'] for row in rows
            if row['matched_auroc'] is not None
        ]
        ap_values = [
            row['matched_auprc_lift'] for row in rows
            if row['matched_auprc_lift'] is not None
        ]
        auc_null_gaps = []
        ap_null_gaps = []
        beats_all = 0
        for row in rows:
            comparison_key = (
                row['dataset_name'],
                row['layer_index'],
                row['variant_name'],
                row['score_name'],
                row['candidate_rank'],
                row['regime_name'],
                row['task_name'],
            )
            comparisons = null_by_observed.get(
                comparison_key, [])
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
                auc_null_gaps.append(min(valid_auc))
            if valid_ap:
                ap_null_gaps.append(min(valid_ap))
            if (
                    valid_auc and valid_ap
                    and min(valid_auc) > 0.0
                    and min(valid_ap) > 0.0):
                beats_all += 1
        decision_rows.append(dict(
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
                sum(auc_null_gaps), len(auc_null_gaps)),
            positive_auroc_null_gap_datasets=sum(
                value > 0.0 for value in auc_null_gaps),
            mean_auprc_lift_minus_best_null=safe_div(
                sum(ap_null_gaps), len(ap_null_gaps)),
            positive_auprc_null_gap_datasets=sum(
                value > 0.0 for value in ap_null_gaps),
            datasets_beating_all_nulls=beats_all,
        ))

    decision_rows.sort(
        key=lambda row: (
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
    for index, row in enumerate(decision_rows):
        row['decision_rank'] = index + 1

    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_dataset_summary.csv'),
        dataset_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_margin_summary.csv'),
        margin_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_mechanism_summary.csv'),
        mechanism_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_null_summary.csv'),
        null_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_decision_summary.csv'),
        decision_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_source_inventory.csv'),
        list(inventory.values()))
    write_csv(
        os.path.join(
            args.out_dir, 'trajectory_image_feature_summary.csv'),
        image_feature_rows)

    print(f'Read {len(paths)} JSONL files.')
    print(f'Found trajectory stats for {len(seen_images)} images.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
