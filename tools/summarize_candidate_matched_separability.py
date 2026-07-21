import argparse
import csv
import glob
import hashlib
import json
import math
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize pair/rank/margin-matched candidate separability, '
            'matched null controls, and image-fold threshold diagnostics.'))
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--min-positive', type=int, default=64)
    parser.add_argument('--min-negative', type=int, default=64)
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


def image_fold(image_id, folds):
    digest = hashlib.sha1(image_id.encode('utf-8')).hexdigest()
    return int(digest[:12], 16) % folds


def new_metric_bucket():
    return dict(
        images=set(),
        strata=0,
        valid_metric_strata=0,
        positive_pixels=0,
        negative_pixels=0,
        comparison_weight=0,
        auroc_weighted=0.0,
        auprc_weighted=0.0,
        auprc_lift_weighted=0.0,
        prior_weighted=0.0,
        ap_weight=0,
        positive_clip_low_pixels=0,
        positive_clip_high_pixels=0,
        negative_clip_low_pixels=0,
        negative_clip_high_pixels=0,
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
        bucket['valid_metric_strata'] += 1
        bucket['auroc_weighted'] += (
            float(metric['auroc']) * comparison_weight)
    if metric.get('auprc') is not None:
        bucket['auprc_weighted'] += float(metric['auprc']) * positive
        bucket['auprc_lift_weighted'] += (
            float(metric.get('auprc_lift') or 0.0) * positive)
        bucket['prior_weighted'] += (
            float(metric.get('prior') or 0.0) * positive)
        bucket['ap_weight'] += positive
    for name in (
            'positive_clip_low_pixels',
            'positive_clip_high_pixels',
            'negative_clip_low_pixels',
            'negative_clip_high_pixels'):
        bucket[name] += int(metric.get(name) or 0)


def metric_row(bucket):
    positive = bucket['positive_pixels']
    negative = bucket['negative_pixels']
    total = positive + negative
    clipped = (
        bucket['positive_clip_low_pixels']
        + bucket['positive_clip_high_pixels']
        + bucket['negative_clip_low_pixels']
        + bucket['negative_clip_high_pixels']
    )
    return dict(
        images=len(bucket['images']),
        matched_strata=int(bucket['strata']),
        valid_metric_strata=int(bucket['valid_metric_strata']),
        valid_metric_strata_ratio=safe_div(
            bucket['valid_metric_strata'], bucket['strata']),
        positive_pixels=int(positive),
        negative_pixels=int(negative),
        global_positive_prior=safe_div(positive, total),
        matched_auroc=safe_div(
            bucket['auroc_weighted'], bucket['comparison_weight']),
        matched_auprc=safe_div(
            bucket['auprc_weighted'], bucket['ap_weight']),
        matched_prior=safe_div(
            bucket['prior_weighted'], bucket['ap_weight']),
        matched_auprc_lift=safe_div(
            bucket['auprc_lift_weighted'], bucket['ap_weight']),
        auprc_positive_pixels=int(bucket['ap_weight']),
        auprc_positive_coverage=safe_div(
            bucket['ap_weight'], positive),
        comparison_weight=int(bucket['comparison_weight']),
        score_clip_ratio=safe_div(clipped, total),
    )


def threshold_bucket():
    return dict(
        pair_pixels=0,
        selected_pixels=0,
        help_pixels=0,
        harm_pixels=0,
        neutral_pixels=0,
    )


def add_threshold(bucket, pair_pixels, row):
    bucket['pair_pixels'] += int(pair_pixels or 0)
    for name in (
            'selected_pixels', 'help_pixels', 'harm_pixels',
            'neutral_pixels'):
        bucket[name] += int(row.get(name) or 0)


def choose_threshold(train_by_threshold):
    candidates = []
    for threshold, bucket in train_by_threshold.items():
        net = bucket['help_pixels'] - bucket['harm_pixels']
        precision = safe_div(
            bucket['help_pixels'], bucket['selected_pixels']) or 0.0
        candidates.append((
            net,
            precision,
            -bucket['selected_pixels'],
            threshold,
        ))
    candidates.append((0, 1.0, 0, math.inf))
    return max(candidates)[-1]


def threshold_result_row(bucket):
    net = bucket['help_pixels'] - bucket['harm_pixels']
    return dict(
        pair_pixels=int(bucket['pair_pixels']),
        selected_pixels=int(bucket['selected_pixels']),
        selected_ratio=safe_div(
            bucket['selected_pixels'], bucket['pair_pixels']),
        help_pixels=int(bucket['help_pixels']),
        harm_pixels=int(bucket['harm_pixels']),
        neutral_pixels=int(bucket['neutral_pixels']),
        changed_precision=safe_div(
            bucket['help_pixels'], bucket['selected_pixels']),
        net_improved_pixels=int(net),
        candidate_stratum_accuracy_delta=safe_div(
            net, bucket['pair_pixels']),
    )


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')
    folds = max(2, int(args.folds))

    pair_metrics = defaultdict(new_metric_bucket)
    dataset_metrics = defaultdict(new_metric_bucket)
    threshold_pair = defaultdict(threshold_bucket)
    threshold_global = defaultdict(threshold_bucket)
    dataset_valid_by_fold = defaultdict(int)
    seen_images = set()

    for record in iter_records(paths):
        stats = (
            record.get('candidate_internal_verifier_stats')
            or record.get('position_bias_candidate_stats')
            or {}
        )
        matched_rows = stats.get('matched_stats') or []
        if not matched_rows:
            continue
        dataset = (
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown'
        )
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{len(seen_images)}")
        fold = image_fold(image_id, folds)
        image_key = (dataset, image_id)
        if image_key not in seen_images:
            seen_images.add(image_key)
            dataset_valid_by_fold[(dataset, fold)] += int(
                stats.get('valid_pixels') or 0)

        for row in matched_rows:
            common = (
                dataset,
                row.get('family_name'),
                row.get('candidate_rank'),
                row.get('regime_name'),
                row.get('margin_name'),
            )
            dataset_common_keys = [common]
            if row.get('margin_name') != 'matched_all_margins':
                dataset_common_keys.append(
                    common[:-1] + ('matched_all_margins',))
            pair = (
                row.get('base_class_index'),
                row.get('base_class_name'),
                row.get('candidate_class_index'),
                row.get('candidate_class_name'),
            )
            control = (
                row.get('control_name'),
                row.get('trial'),
            )
            for task_name, metric in (
                    row.get('task_metrics') or {}).items():
                pair_key = common + pair + control + (task_name,)
                add_metric(pair_metrics[pair_key], image_id, metric)
                for dataset_common in dataset_common_keys:
                    dataset_key = (
                        dataset_common + control + (task_name,))
                    add_metric(
                        dataset_metrics[dataset_key], image_id, metric)

            if row.get('control_name') != 'observed':
                continue
            for threshold_row in row.get('threshold_stats') or []:
                threshold = float(threshold_row['threshold'])
                pair_key = common + pair + (fold, threshold)
                add_threshold(
                    threshold_pair[pair_key],
                    row.get('pair_pixels'),
                    threshold_row,
                )
                for dataset_common in dataset_common_keys:
                    global_key = dataset_common + (fold, threshold)
                    add_threshold(
                        threshold_global[global_key],
                        row.get('pair_pixels'),
                        threshold_row,
                    )

    pair_rows = []
    for key, bucket in sorted(pair_metrics.items()):
        (
            dataset, family, rank, regime, margin_name,
            base_idx, base_name, candidate_idx, candidate_name,
            control, trial, task_name,
        ) = key
        row = dict(
            dataset_name=dataset,
            family_name=family,
            candidate_rank=rank,
            regime_name=regime,
            margin_name=margin_name,
            base_class_index=base_idx,
            base_class_name=base_name,
            candidate_class_index=candidate_idx,
            candidate_class_name=candidate_name,
            control_name=control,
            trial=trial,
            task_name=task_name,
        )
        row.update(metric_row(bucket))
        row['meets_min_counts'] = (
            row['positive_pixels'] >= args.min_positive
            and row['negative_pixels'] >= args.min_negative
        )
        pair_rows.append(row)

    dataset_rows = []
    dataset_index = {}
    for key, bucket in sorted(dataset_metrics.items()):
        (
            dataset, family, rank, regime, margin_name,
            control, trial, task_name,
        ) = key
        row = dict(
            dataset_name=dataset,
            family_name=family,
            candidate_rank=rank,
            regime_name=regime,
            margin_name=margin_name,
            control_name=control,
            trial=trial,
            task_name=task_name,
        )
        row.update(metric_row(bucket))
        row['meets_min_counts'] = (
            row['positive_pixels'] >= args.min_positive
            and row['negative_pixels'] >= args.min_negative
        )
        dataset_rows.append(row)
        dataset_index[key] = row

    null_rows = []
    observed_keys = [
        key for key in dataset_index
        if key[5] == 'observed'
    ]
    for key in sorted(observed_keys):
        (
            dataset, family, rank, regime, margin_name,
            _, _, task_name,
        ) = key
        observed = dataset_index[key]
        controls = [
            row for other_key, row in dataset_index.items()
            if (
                other_key[:5] == key[:5]
                and other_key[7] == task_name
                and other_key[5] != 'observed'
                and row.get('matched_auroc') is not None
            )
        ]
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
            ap_lifts = [
                row['matched_auprc_lift'] for row in selected
                if row['matched_auprc_lift'] is not None
            ]
            null_auc_mean = (
                sum(auc_values) / len(auc_values)
                if auc_values else None)
            null_auc_max = max(auc_values) if auc_values else None
            null_ap_lift_mean = (
                sum(ap_lifts) / len(ap_lifts)
                if ap_lifts else None)
            null_ap_lift_max = max(ap_lifts) if ap_lifts else None
            null_rows.append(dict(
                dataset_name=dataset,
                family_name=family,
                candidate_rank=rank,
                regime_name=regime,
                margin_name=margin_name,
                task_name=task_name,
                control_name=control_name,
                trials=len(selected),
                observed_auroc=observed.get('matched_auroc'),
                null_auroc_mean=null_auc_mean,
                null_auroc_max=null_auc_max,
                auroc_minus_null_mean=(
                    None if observed.get('matched_auroc') is None
                    or null_auc_mean is None
                    else observed['matched_auroc'] - null_auc_mean),
                auroc_minus_best_null=(
                    None if observed.get('matched_auroc') is None
                    or null_auc_max is None
                    else observed['matched_auroc'] - null_auc_max),
                observed_auprc_lift=observed.get(
                    'matched_auprc_lift'),
                null_auprc_lift_mean=null_ap_lift_mean,
                null_auprc_lift_max=null_ap_lift_max,
                auprc_lift_minus_best_null=(
                    None if observed.get('matched_auprc_lift') is None
                    or null_ap_lift_max is None
                    else observed['matched_auprc_lift']
                    - null_ap_lift_max),
            ))

    def build_holdout_rows(source, scope):
        grouped = defaultdict(dict)
        for key, bucket in source.items():
            group_key = key[:-2]
            fold, threshold = key[-2:]
            grouped[group_key][(fold, threshold)] = bucket
        rows = []
        for group_key, entries in sorted(grouped.items()):
            thresholds = sorted({
                threshold for _, threshold in entries})
            for test_fold in range(folds):
                train_by_threshold = {}
                for threshold in thresholds:
                    aggregate = threshold_bucket()
                    for fold in range(folds):
                        if fold == test_fold:
                            continue
                        bucket = entries.get((fold, threshold))
                        if bucket:
                            add_threshold(
                                aggregate, bucket['pair_pixels'], bucket)
                    train_by_threshold[threshold] = aggregate
                selected_threshold = choose_threshold(train_by_threshold)
                test_bucket = (
                    threshold_bucket()
                    if math.isinf(selected_threshold)
                    else entries.get(
                        (test_fold, selected_threshold),
                        threshold_bucket())
                )
                row = dict(
                    scope=scope,
                    test_fold=test_fold,
                    selected_threshold=(
                        'inf' if math.isinf(selected_threshold)
                        else selected_threshold),
                    threshold_selection='GT_train_fold_oracle',
                )
                if scope == 'pair':
                    (
                        dataset, family, rank, regime, margin_name,
                        base_idx, base_name, candidate_idx, candidate_name,
                    ) = group_key
                    row.update(
                        dataset_name=dataset,
                        family_name=family,
                        candidate_rank=rank,
                        regime_name=regime,
                        margin_name=margin_name,
                        base_class_index=base_idx,
                        base_class_name=base_name,
                        candidate_class_index=candidate_idx,
                        candidate_class_name=candidate_name,
                    )
                else:
                    dataset, family, rank, regime, margin_name = group_key
                    row.update(
                        dataset_name=dataset,
                        family_name=family,
                        candidate_rank=rank,
                        regime_name=regime,
                        margin_name=margin_name,
                    )
                row.update(threshold_result_row(test_bucket))
                valid_pixels = dataset_valid_by_fold.get(
                    (row['dataset_name'], test_fold), 0)
                row['dataset_valid_pixels'] = int(valid_pixels)
                row['dataset_pixel_accuracy_delta'] = safe_div(
                    row['net_improved_pixels'], valid_pixels)
                rows.append(row)
        return rows

    holdout_pair_rows = build_holdout_rows(threshold_pair, 'pair')
    holdout_global_rows = build_holdout_rows(threshold_global, 'global')

    holdout_summary_buckets = defaultdict(threshold_bucket)
    for row in holdout_pair_rows + holdout_global_rows:
        key = (
            row['scope'],
            row['dataset_name'],
            row['family_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['margin_name'],
        )
        bucket = holdout_summary_buckets[key]
        bucket['pair_pixels'] += int(row.get('pair_pixels') or 0)
        for name in (
                'selected_pixels', 'help_pixels', 'harm_pixels',
                'neutral_pixels'):
            bucket[name] += int(row.get(name) or 0)

    holdout_summary_rows = []
    for key, bucket in sorted(holdout_summary_buckets.items()):
        scope, dataset, family, rank, regime, margin_name = key
        row = dict(
            scope=scope,
            dataset_name=dataset,
            family_name=family,
            candidate_rank=rank,
            regime_name=regime,
            margin_name=margin_name,
            threshold_selection='GT_train_fold_oracle',
            folds=folds,
        )
        row.update(threshold_result_row(bucket))
        dataset_valid_pixels = sum(
            dataset_valid_by_fold.get((dataset, fold), 0)
            for fold in range(folds)
        )
        row['dataset_valid_pixels'] = int(dataset_valid_pixels)
        row['dataset_pixel_accuracy_delta'] = safe_div(
            row['net_improved_pixels'], dataset_valid_pixels)
        holdout_summary_rows.append(row)

    decision_rows = []
    null_lookup = defaultdict(list)
    for row in null_rows:
        key = (
            row['dataset_name'],
            row['family_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['margin_name'],
            row['task_name'],
        )
        null_lookup[key].append(row)
    holdout_lookup = {
        (
            row['scope'],
            row['dataset_name'],
            row['family_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['margin_name'],
        ): row
        for row in holdout_summary_rows
    }
    for key in sorted(observed_keys):
        (
            dataset, family, rank, regime, margin_name,
            _, _, task_name,
        ) = key
        observed = dataset_index[key]
        comparison_key = (
            dataset, family, rank, regime, margin_name, task_name)
        comparisons = null_lookup.get(comparison_key, [])
        beats_all_nulls = bool(comparisons) and all(
            row.get('auroc_minus_best_null') is not None
            and row['auroc_minus_best_null'] > 0
            for row in comparisons
        )
        global_holdout = holdout_lookup.get((
            'global', dataset, family, rank, regime, margin_name))
        pair_holdout = holdout_lookup.get((
            'pair', dataset, family, rank, regime, margin_name))
        decision_rows.append(dict(
            dataset_name=dataset,
            family_name=family,
            candidate_rank=rank,
            regime_name=regime,
            margin_name=margin_name,
            task_name=task_name,
            positive_pixels=observed['positive_pixels'],
            negative_pixels=observed['negative_pixels'],
            matched_auroc=observed['matched_auroc'],
            matched_auprc_lift=observed['matched_auprc_lift'],
            beats_all_matched_nulls=beats_all_nulls,
            meets_auc_0p60=(
                observed['matched_auroc'] is not None
                and observed['matched_auroc'] >= 0.60),
            meets_auprc_lift_0p02=(
                observed['matched_auprc_lift'] is not None
                and observed['matched_auprc_lift'] >= 0.02),
            global_holdout_dataset_delta=(
                None if global_holdout is None
                else global_holdout['dataset_pixel_accuracy_delta']),
            pair_holdout_dataset_delta=(
                None if pair_holdout is None
                else pair_holdout['dataset_pixel_accuracy_delta']),
            passes_diagnostic_bar=(
                task_name == 'switch_utility'
                and observed['matched_auroc'] is not None
                and observed['matched_auroc'] >= 0.60
                and observed['matched_auprc_lift'] is not None
                and observed['matched_auprc_lift'] >= 0.02
                and beats_all_nulls
                and global_holdout is not None
                and (
                    global_holdout['dataset_pixel_accuracy_delta']
                    is not None
                    and global_holdout[
                        'dataset_pixel_accuracy_delta'] > 0
                )
            ),
        ))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(
            args.out_dir, 'candidate_matched_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'candidate_matched_dataset_summary.csv'),
        dataset_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'candidate_matched_null_summary.csv'),
        null_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'candidate_matched_holdout_pair.csv'),
        holdout_pair_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'candidate_matched_holdout_summary.csv'),
        holdout_summary_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'candidate_matched_decision_summary.csv'),
        decision_rows)

    print(f'Read {len(paths)} files.')
    print(f'Found matched stats for {len(seen_images)} images.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
