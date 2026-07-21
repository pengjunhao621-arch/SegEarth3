import argparse
import bisect
import csv
import math
import os
from collections import defaultdict


DEFAULT_VARIANTS = (
    'global_a0p25,global_a0p5,global_a0p75,global_a1,'
    'robust,balanced'
)
DEFAULT_DESCRIPTORS = (
    'common_energy_ratio,mean_norm_ratio,mean_token_cosine,'
    'effective_rank,top1_variance_ratio,top4_variance_ratio,'
    'top8_variance_ratio,seed_ratio,seed_class_count,'
    'seed_prototype_dispersion'
)
DEFAULT_TASKS = 'switch_utility,baseline_risk'
_DATASET_INDEX_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate simple no-GT scene-common selectors with strict '
            'leave-one-dataset-out validation.'))
    parser.add_argument('--input', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--layers', default='0,2')
    parser.add_argument('--variants', default=DEFAULT_VARIANTS)
    parser.add_argument('--candidate-ranks', default='2,3')
    parser.add_argument('--tasks', default=DEFAULT_TASKS)
    parser.add_argument('--descriptors', default=DEFAULT_DESCRIPTORS)
    parser.add_argument(
        '--gate-modes',
        default='absolute,within_dataset_quantile',
        help=(
            'absolute learns a numeric threshold on source datasets; '
            'within_dataset_quantile transfers only an unlabeled image '
            'percentile.'))
    parser.add_argument(
        '--quantiles',
        default='0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90')
    parser.add_argument('--min-gate-ratio', type=float, default=0.05)
    parser.add_argument('--max-gate-ratio', type=float, default=0.90)
    return parser.parse_args()


def parse_names(value):
    return [
        item.strip()
        for item in str(value or '').split(',')
        if item.strip()
    ]


def parse_ints(value):
    return sorted({
        int(item.strip())
        for item in str(value or '').split(',')
        if item.strip()
    })


def parse_floats(value):
    return sorted({
        float(item.strip())
        for item in str(value or '').split(',')
        if item.strip()
    })


def to_float(value):
    if value in (None, ''):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_int(value):
    numeric = to_float(value)
    return 0 if numeric is None else int(numeric)


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return numerator / denominator


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def quantile(values, probability):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = min(max(probability, 0.0), 1.0) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


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


def load_records(path, layers, variants, ranks, tasks, descriptors):
    structures = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: dict(descriptors={}, tasks={}))))
    with open(path, 'r', newline='') as file:
        for row in csv.DictReader(file):
            layer = to_int(row.get('layer_index'))
            variant = str(row.get('variant_name') or '')
            rank = to_int(row.get('candidate_rank'))
            task = str(row.get('task_name') or '')
            if (
                    layer not in layers
                    or variant not in variants
                    or rank not in ranks
                    or task not in tasks
                    or row.get('regime_name') != 'argmax_competition'):
                continue
            dataset = str(row.get('dataset_name') or 'unknown')
            image_id = str(row.get('image_id') or '')
            record = structures[(layer, variant, rank)][dataset][image_id]
            for descriptor in descriptors:
                value = to_float(row.get(descriptor))
                if value is not None:
                    record['descriptors'][descriptor] = value
            record['tasks'][task] = dict(
                variant_auroc=to_float(row.get('variant_auroc')),
                raw_auroc=to_float(row.get('raw_auroc')),
                variant_ap=to_float(row.get('variant_auprc_lift')),
                raw_ap=to_float(row.get('raw_auprc_lift')),
                positive_pixels=to_int(row.get('positive_pixels')),
                negative_pixels=to_int(row.get('negative_pixels')),
            )
    return structures


def dataset_index(records, descriptor, tasks):
    cache_key = (id(records), descriptor, tuple(tasks))
    cached = _DATASET_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ordered = sorted([
        (
            record['descriptors'][descriptor],
            record,
        )
        for record in records.values()
        if descriptor in record['descriptors']
    ], key=lambda item: item[0])
    values = [item[0] for item in ordered]
    result = dict(
        values=values,
        images=len(records),
        eligible_images=len(values),
        tasks={},
    )
    for task in tasks:
        raw_auc_sum = 0.0
        variant_auc_sum = 0.0
        oracle_auc_sum = 0.0
        auc_weight = 0
        raw_ap_sum = 0.0
        variant_ap_sum = 0.0
        oracle_ap_sum = 0.0
        ap_weight = 0
        prefix_auc_delta = [0.0]
        prefix_ap_delta = [0.0]
        prefix_joint_positive = [0]

        for record in records.values():
            metric = record['tasks'].get(task)
            if metric is None:
                continue
            variant_auc = metric.get('variant_auroc')
            raw_auc = metric.get('raw_auroc')
            variant_ap = metric.get('variant_ap')
            raw_ap = metric.get('raw_ap')
            current_auc_weight = (
                int(metric.get('positive_pixels') or 0)
                * int(metric.get('negative_pixels') or 0)
            )
            current_ap_weight = int(
                metric.get('positive_pixels') or 0)
            if (
                    variant_auc is not None
                    and raw_auc is not None
                    and current_auc_weight > 0):
                raw_auc_sum += raw_auc * current_auc_weight
                variant_auc_sum += variant_auc * current_auc_weight
                oracle_auc_sum += (
                    max(variant_auc, raw_auc) * current_auc_weight)
                auc_weight += current_auc_weight
            if (
                    variant_ap is not None
                    and raw_ap is not None
                    and current_ap_weight > 0):
                raw_ap_sum += raw_ap * current_ap_weight
                variant_ap_sum += variant_ap * current_ap_weight
                oracle_ap_sum += (
                    max(variant_ap, raw_ap) * current_ap_weight)
                ap_weight += current_ap_weight

        for _, record in ordered:
            metric = record['tasks'].get(task)
            auc_delta = 0.0
            ap_delta = 0.0
            joint_positive = 0
            if metric is not None:
                variant_auc = metric.get('variant_auroc')
                raw_auc = metric.get('raw_auroc')
                variant_ap = metric.get('variant_ap')
                raw_ap = metric.get('raw_ap')
                current_auc_weight = (
                    int(metric.get('positive_pixels') or 0)
                    * int(metric.get('negative_pixels') or 0)
                )
                current_ap_weight = int(
                    metric.get('positive_pixels') or 0)
                if (
                        variant_auc is not None
                        and raw_auc is not None
                        and current_auc_weight > 0):
                    auc_delta = (
                        variant_auc - raw_auc) * current_auc_weight
                if (
                        variant_ap is not None
                        and raw_ap is not None
                        and current_ap_weight > 0):
                    ap_delta = (
                        variant_ap - raw_ap) * current_ap_weight
                joint_positive = int(
                    variant_auc is not None
                    and raw_auc is not None
                    and variant_ap is not None
                    and raw_ap is not None
                    and variant_auc > raw_auc
                    and variant_ap >= raw_ap)
            prefix_auc_delta.append(
                prefix_auc_delta[-1] + auc_delta)
            prefix_ap_delta.append(
                prefix_ap_delta[-1] + ap_delta)
            prefix_joint_positive.append(
                prefix_joint_positive[-1] + joint_positive)

        result['tasks'][task] = dict(
            raw_auc_sum=raw_auc_sum,
            variant_auc_sum=variant_auc_sum,
            oracle_auc_sum=oracle_auc_sum,
            auc_weight=auc_weight,
            raw_ap_sum=raw_ap_sum,
            variant_ap_sum=variant_ap_sum,
            oracle_ap_sum=oracle_ap_sum,
            ap_weight=ap_weight,
            prefix_auc_delta=prefix_auc_delta,
            prefix_ap_delta=prefix_ap_delta,
            prefix_joint_positive=prefix_joint_positive,
        )
    _DATASET_INDEX_CACHE[cache_key] = result
    return result


def evaluate_dataset(records, descriptor, direction, gate_mode, parameter,
                     tasks):
    index = dataset_index(records, descriptor, tasks)
    values = index['values']
    threshold = (
        quantile(values, parameter)
        if gate_mode == 'within_dataset_quantile'
        else float(parameter)
    )
    if threshold is None:
        selected_start = 0
        selected_end = 0
    elif direction == 'high':
        selected_start = bisect.bisect_left(values, threshold)
        selected_end = len(values)
    else:
        selected_start = 0
        selected_end = bisect.bisect_right(values, threshold)
    selected_images = selected_end - selected_start

    result = dict(
        images=index['images'],
        eligible_images=index['eligible_images'],
        selected_images=selected_images,
        gate_ratio=safe_div(
            selected_images, index['eligible_images']),
        applied_threshold=threshold,
        task_metrics={},
    )
    for task in tasks:
        task_index = index['tasks'][task]

        def selected_prefix(prefix):
            return (
                prefix[selected_end] - prefix[selected_start])

        auc_weight = task_index['auc_weight']
        ap_weight = task_index['ap_weight']
        selected_auc_delta = selected_prefix(
            task_index['prefix_auc_delta'])
        selected_ap_delta = selected_prefix(
            task_index['prefix_ap_delta'])
        positive_selected = selected_prefix(
            task_index['prefix_joint_positive'])
        raw_auc = safe_div(
            task_index['raw_auc_sum'], auc_weight)
        variant_auc = safe_div(
            task_index['variant_auc_sum'], auc_weight)
        gated_auc = safe_div(
            task_index['raw_auc_sum'] + selected_auc_delta,
            auc_weight,
        )
        oracle_auc = safe_div(
            task_index['oracle_auc_sum'], auc_weight)
        raw_ap = safe_div(
            task_index['raw_ap_sum'], ap_weight)
        variant_ap = safe_div(
            task_index['variant_ap_sum'], ap_weight)
        gated_ap = safe_div(
            task_index['raw_ap_sum'] + selected_ap_delta,
            ap_weight,
        )
        oracle_ap = safe_div(
            task_index['oracle_ap_sum'], ap_weight)
        result['task_metrics'][task] = dict(
            raw_auroc=raw_auc,
            always_variant_auroc=variant_auc,
            gated_auroc=gated_auc,
            oracle_auroc=oracle_auc,
            gated_minus_raw_auroc=(
                None
                if gated_auc is None or raw_auc is None
                else gated_auc - raw_auc),
            always_variant_minus_raw_auroc=(
                None
                if variant_auc is None or raw_auc is None
                else variant_auc - raw_auc),
            oracle_minus_raw_auroc=(
                None
                if oracle_auc is None or raw_auc is None
                else oracle_auc - raw_auc),
            raw_auprc_lift=raw_ap,
            always_variant_auprc_lift=variant_ap,
            gated_auprc_lift=gated_ap,
            oracle_auprc_lift=oracle_ap,
            gated_minus_raw_auprc_lift=(
                None
                if gated_ap is None or raw_ap is None
                else gated_ap - raw_ap),
            always_variant_minus_raw_auprc_lift=(
                None
                if variant_ap is None or raw_ap is None
                else variant_ap - raw_ap),
            oracle_minus_raw_auprc_lift=(
                None
                if oracle_ap is None or raw_ap is None
                else oracle_ap - raw_ap),
            selected_joint_positive_ratio=safe_div(
                positive_selected, selected_images),
        )
    return result


def flatten_dataset_result(result, prefix=''):
    row = dict(
        **{
            f'{prefix}images': result.get('images'),
            f'{prefix}eligible_images': result.get('eligible_images'),
            f'{prefix}selected_images': result.get('selected_images'),
            f'{prefix}gate_ratio': result.get('gate_ratio'),
            f'{prefix}applied_threshold': result.get('applied_threshold'),
        }
    )
    for task, metrics in result.get('task_metrics', {}).items():
        for name, value in metrics.items():
            row[f'{prefix}{task}_{name}'] = value
    return row


def aggregate_training(dataset_results, tasks):
    endpoint_values = []
    positive_dataset_task_pairs = 0
    positive_endpoints = 0
    task_means = {}
    for task in tasks:
        auc_values = [
            result['task_metrics'][task]['gated_minus_raw_auroc']
            for result in dataset_results
            if task in result['task_metrics']
        ]
        ap_values = [
            result['task_metrics'][task][
                'gated_minus_raw_auprc_lift']
            for result in dataset_results
            if task in result['task_metrics']
        ]
        mean_auc = mean(auc_values)
        mean_ap = mean(ap_values)
        task_means[task] = dict(
            mean_auc=mean_auc,
            mean_ap=mean_ap,
            positive_auc_datasets=sum(
                value is not None and value > 0
                for value in auc_values),
            positive_ap_datasets=sum(
                value is not None and value > 0
                for value in ap_values),
        )
        endpoint_values.extend([
            value for value in (mean_auc, mean_ap)
            if value is not None
        ])
        for auc_value, ap_value in zip(auc_values, ap_values):
            positive_dataset_task_pairs += int(
                auc_value is not None
                and ap_value is not None
                and auc_value > 0
                and ap_value >= 0)
            positive_endpoints += int(
                auc_value is not None and auc_value > 0)
            positive_endpoints += int(
                ap_value is not None and ap_value > 0)

    gate_ratios = [
        result.get('gate_ratio')
        for result in dataset_results
        if result.get('gate_ratio') is not None
    ]
    return dict(
        positive_dataset_task_pairs=positive_dataset_task_pairs,
        positive_endpoints=positive_endpoints,
        worst_mean_endpoint=(
            min(endpoint_values) if endpoint_values else -float('inf')),
        mean_endpoint=mean(endpoint_values),
        mean_gate_ratio=mean(gate_ratios),
        task_means=task_means,
    )


def selection_key(summary, min_gate_ratio, max_gate_ratio):
    gate_ratio = summary.get('mean_gate_ratio')
    valid_gate = (
        gate_ratio is not None
        and min_gate_ratio <= gate_ratio <= max_gate_ratio
    )
    return (
        int(valid_gate),
        int(summary.get('positive_dataset_task_pairs') or 0),
        int(summary.get('positive_endpoints') or 0),
        float(summary.get('worst_mean_endpoint') or -float('inf')),
        float(summary.get('mean_endpoint') or -float('inf')),
    )


def threshold_parameters(train_records, descriptor, gate_mode, quantiles):
    if gate_mode == 'within_dataset_quantile':
        return list(quantiles)
    values = []
    for records in train_records:
        values.extend([
            record['descriptors'].get(descriptor)
            for record in records.values()
            if record['descriptors'].get(descriptor) is not None
        ])
    return sorted({
        quantile(values, probability)
        for probability in quantiles
        if quantile(values, probability) is not None
    })


def fit_rule(train_by_dataset, descriptor, gate_mode, quantiles, tasks,
             min_gate_ratio, max_gate_ratio):
    train_records = list(train_by_dataset.values())
    candidates = []
    for direction in ('high', 'low'):
        for parameter in threshold_parameters(
                train_records, descriptor, gate_mode, quantiles):
            results = [
                evaluate_dataset(
                    records,
                    descriptor,
                    direction,
                    gate_mode,
                    parameter,
                    tasks,
                )
                for records in train_records
            ]
            summary = aggregate_training(results, tasks)
            candidates.append(dict(
                direction=direction,
                parameter=parameter,
                summary=summary,
            ))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: selection_key(
            item['summary'], min_gate_ratio, max_gate_ratio),
    )


def fixed_rule_rows(structures, descriptors, gate_modes, quantiles, tasks,
                    min_gate_ratio, max_gate_ratio):
    rows = []
    datasets = sorted({
        dataset
        for by_dataset in structures.values()
        for dataset in by_dataset
    })
    for structure, by_dataset in sorted(structures.items()):
        layer, variant, rank = structure
        if len(by_dataset) < 2:
            continue
        for descriptor in descriptors:
            for gate_mode in gate_modes:
                for heldout in datasets:
                    if heldout not in by_dataset:
                        continue
                    train = {
                        dataset: records
                        for dataset, records in by_dataset.items()
                        if dataset != heldout
                    }
                    fitted = fit_rule(
                        train,
                        descriptor,
                        gate_mode,
                        quantiles,
                        tasks,
                        min_gate_ratio,
                        max_gate_ratio,
                    )
                    if fitted is None:
                        continue
                    heldout_result = evaluate_dataset(
                        by_dataset[heldout],
                        descriptor,
                        fitted['direction'],
                        gate_mode,
                        fitted['parameter'],
                        tasks,
                    )
                    row = dict(
                        heldout_dataset=heldout,
                        layer_index=layer,
                        variant_name=variant,
                        candidate_rank=rank,
                        descriptor_name=descriptor,
                        gate_mode=gate_mode,
                        direction=fitted['direction'],
                        learned_parameter=fitted['parameter'],
                        train_positive_dataset_task_pairs=(
                            fitted['summary'][
                                'positive_dataset_task_pairs']),
                        train_positive_endpoints=(
                            fitted['summary']['positive_endpoints']),
                        train_worst_mean_endpoint=(
                            fitted['summary']['worst_mean_endpoint']),
                        train_mean_endpoint=(
                            fitted['summary']['mean_endpoint']),
                        train_mean_gate_ratio=(
                            fitted['summary']['mean_gate_ratio']),
                    )
                    for task, summary in (
                            fitted['summary']['task_means'].items()):
                        row.update({
                            f'train_{task}_mean_auc_gain':
                                summary['mean_auc'],
                            f'train_{task}_mean_ap_gain':
                                summary['mean_ap'],
                            f'train_{task}_positive_auc_datasets':
                                summary['positive_auc_datasets'],
                            f'train_{task}_positive_ap_datasets':
                                summary['positive_ap_datasets'],
                        })
                    row.update(flatten_dataset_result(
                        heldout_result, prefix='heldout_'))
                    rows.append(row)
    return rows


def summarize_fixed_rules(rows, tasks):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row['layer_index'],
            row['variant_name'],
            row['candidate_rank'],
            row['descriptor_name'],
            row['gate_mode'],
        )
        groups[key].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        layer, variant, rank, descriptor, gate_mode = key
        row = dict(
            layer_index=layer,
            variant_name=variant,
            candidate_rank=rank,
            descriptor_name=descriptor,
            gate_mode=gate_mode,
            folds=len(group),
            mean_gate_ratio=mean([
                item.get('heldout_gate_ratio') for item in group]),
            selected_high_folds=sum(
                item.get('direction') == 'high' for item in group),
            selected_low_folds=sum(
                item.get('direction') == 'low' for item in group),
        )
        all_mean_endpoints = []
        minimum_positive_folds = len(group)
        for task in tasks:
            auc_values = [
                item.get(
                    f'heldout_{task}_gated_minus_raw_auroc')
                for item in group
            ]
            ap_values = [
                item.get(
                    f'heldout_{task}_gated_minus_raw_auprc_lift')
                for item in group
            ]
            mean_auc = mean(auc_values)
            mean_ap = mean(ap_values)
            positive_auc = sum(
                value is not None and value > 0 for value in auc_values)
            positive_ap = sum(
                value is not None and value > 0 for value in ap_values)
            minimum_positive_folds = min(
                minimum_positive_folds, positive_auc, positive_ap)
            all_mean_endpoints.extend([
                value for value in (mean_auc, mean_ap)
                if value is not None
            ])
            row.update({
                f'{task}_mean_heldout_auc_gain': mean_auc,
                f'{task}_positive_auc_folds': positive_auc,
                f'{task}_mean_heldout_ap_gain': mean_ap,
                f'{task}_positive_ap_folds': positive_ap,
                f'{task}_mean_always_variant_auc_gain': mean([
                    item.get(
                        f'heldout_{task}_'
                        'always_variant_minus_raw_auroc')
                    for item in group
                ]),
                f'{task}_mean_always_variant_ap_gain': mean([
                    item.get(
                        f'heldout_{task}_'
                        'always_variant_minus_raw_auprc_lift')
                    for item in group
                ]),
                f'{task}_mean_oracle_auc_gain': mean([
                    item.get(
                        f'heldout_{task}_oracle_minus_raw_auroc')
                    for item in group
                ]),
                f'{task}_mean_oracle_ap_gain': mean([
                    item.get(
                        f'heldout_{task}_oracle_minus_raw_auprc_lift')
                    for item in group
                ]),
            })
        row['worst_mean_heldout_endpoint'] = (
            min(all_mean_endpoints)
            if all_mean_endpoints else None)
        row['mean_heldout_endpoint'] = mean(all_mean_endpoints)
        row['minimum_positive_folds'] = minimum_positive_folds
        summary_rows.append(row)
    return summary_rows


def select_rule_per_fold(fold_rows, tasks, min_gate_ratio, max_gate_ratio):
    by_fold = defaultdict(list)
    for row in fold_rows:
        by_fold[row['heldout_dataset']].append(row)

    selected = []
    for heldout, rows in sorted(by_fold.items()):
        def key(row):
            gate_ratio = row.get('train_mean_gate_ratio')
            valid_gate = (
                gate_ratio is not None
                and min_gate_ratio <= gate_ratio <= max_gate_ratio
            )
            return (
                int(valid_gate),
                int(row.get(
                    'train_positive_dataset_task_pairs') or 0),
                int(row.get('train_positive_endpoints') or 0),
                float(row.get(
                    'train_worst_mean_endpoint') or -float('inf')),
                float(row.get(
                    'train_mean_endpoint') or -float('inf')),
            )

        best = max(rows, key=key)
        selected.append(best.copy())

    summary = dict(
        folds=len(selected),
        unique_structures=len({
            (
                row['layer_index'],
                row['variant_name'],
                row['candidate_rank'],
                row['descriptor_name'],
                row['gate_mode'],
            )
            for row in selected
        }),
        mean_gate_ratio=mean([
            row.get('heldout_gate_ratio') for row in selected]),
    )
    endpoints = []
    for task in tasks:
        auc_values = [
            row.get(f'heldout_{task}_gated_minus_raw_auroc')
            for row in selected
        ]
        ap_values = [
            row.get(
                f'heldout_{task}_gated_minus_raw_auprc_lift')
            for row in selected
        ]
        mean_auc = mean(auc_values)
        mean_ap = mean(ap_values)
        endpoints.extend([
            value for value in (mean_auc, mean_ap)
            if value is not None
        ])
        summary.update({
            f'{task}_mean_heldout_auc_gain': mean_auc,
            f'{task}_positive_auc_folds': sum(
                value is not None and value > 0 for value in auc_values),
            f'{task}_mean_heldout_ap_gain': mean_ap,
            f'{task}_positive_ap_folds': sum(
                value is not None and value > 0 for value in ap_values),
        })
    summary['worst_mean_heldout_endpoint'] = (
        min(endpoints) if endpoints else None)
    summary['mean_heldout_endpoint'] = mean(endpoints)
    return selected, [summary]


def rank_decisions(summary_rows, tasks):
    rows = []
    for row in summary_rows:
        folds = int(row.get('folds') or 0)
        endpoint_positive = []
        for task in tasks:
            endpoint_positive.extend([
                int(row.get(f'{task}_positive_auc_folds') or 0),
                int(row.get(f'{task}_positive_ap_folds') or 0),
            ])
        decision = row.copy()
        decision['passes_strict_4_of_6'] = bool(
            folds >= 6
            and endpoint_positive
            and min(endpoint_positive) >= 4
            and (row.get('worst_mean_heldout_endpoint') or -1.0) > 0
            and 0.05 <= (row.get('mean_gate_ratio') or 0.0) <= 0.90
        )
        rows.append(decision)
    rows.sort(
        key=lambda row: (
            int(row['passes_strict_4_of_6']),
            int(row.get('minimum_positive_folds') or 0),
            float(row.get(
                'worst_mean_heldout_endpoint') or -float('inf')),
            float(row.get(
                'mean_heldout_endpoint') or -float('inf')),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows):
        row['decision_rank'] = index + 1
    return rows


def main():
    args = parse_args()
    layers = set(parse_ints(args.layers))
    variants = set(parse_names(args.variants))
    ranks = set(parse_ints(args.candidate_ranks))
    tasks = parse_names(args.tasks)
    descriptors = parse_names(args.descriptors)
    gate_modes = parse_names(args.gate_modes)
    quantiles = parse_floats(args.quantiles)

    structures = load_records(
        args.input,
        layers,
        variants,
        ranks,
        set(tasks),
        descriptors,
    )
    if not structures:
        raise RuntimeError('No scene-common image records matched.')

    fold_rows = fixed_rule_rows(
        structures,
        descriptors,
        gate_modes,
        quantiles,
        tasks,
        float(args.min_gate_ratio),
        float(args.max_gate_ratio),
    )
    summary_rows = summarize_fixed_rules(fold_rows, tasks)
    selected_rows, selected_summary = select_rule_per_fold(
        fold_rows,
        tasks,
        float(args.min_gate_ratio),
        float(args.max_gate_ratio),
    )
    decision_rows = rank_decisions(summary_rows, tasks)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(args.out_dir, 'loco_fold_summary.csv'),
        fold_rows)
    write_csv(
        os.path.join(args.out_dir, 'loco_fixed_rule_summary.csv'),
        summary_rows)
    write_csv(
        os.path.join(args.out_dir, 'loco_selected_fold.csv'),
        selected_rows)
    write_csv(
        os.path.join(args.out_dir, 'loco_selected_summary.csv'),
        selected_summary)
    write_csv(
        os.path.join(args.out_dir, 'loco_decision_summary.csv'),
        decision_rows)

    strict_count = sum(
        bool(row.get('passes_strict_4_of_6'))
        for row in decision_rows)
    print(f'Loaded {len(structures)} layer/variant/rank structures.')
    print(f'Evaluated {len(fold_rows)} leave-one-dataset-out folds.')
    print(f'Strict passing fixed rules: {strict_count}')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
