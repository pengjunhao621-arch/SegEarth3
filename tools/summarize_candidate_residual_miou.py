import argparse
import math
import os
from collections import defaultdict

from summarize_candidate_residual_trajectory import (
    expand_inputs,
    iter_records,
    safe_div,
    write_csv,
)


RULE_FIELDS = (
    'layer_index',
    'variant_name',
    'score_name',
    'candidate_rank',
    'regime_name',
    'unit_name',
    'region_reducer',
    'threshold_type',
    'threshold_value',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Aggregate exact candidate-residual counterfactual confusion '
            'statistics and evaluate transferable LODO rules.'))
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--gate-layer', type=int, default=0)
    parser.add_argument(
        '--gate-descriptor', default='effective_rank')
    parser.add_argument('--gate-quantile', type=float, default=0.20)
    parser.add_argument(
        '--gate-groups',
        default='low_quantile,all,remaining')
    parser.add_argument(
        '--lodo-gate-group', default='low_quantile')
    parser.add_argument(
        '--lodo-selection-policies',
        default='mean_best,nonnegative_best')
    return parser.parse_args()


def parse_names(value):
    return [
        item.strip().lower()
        for item in str(value or '').split(',')
        if item.strip()
    ]


def record_identity(record, record_index):
    return str(
        record.get('img_path')
        or f"{record.get('rank', 0)}:{record_index}")


def record_dataset(record, stats):
    return str(
        stats.get('dataset_name')
        or record.get('dataset_name')
        or 'unknown')


def zero_matrix(size):
    return [[0 for _ in range(size)] for _ in range(size)]


def add_matrix(target, source):
    for row_idx, row in enumerate(source):
        for col_idx, value in enumerate(row):
            target[row_idx][col_idx] += int(value)


def copy_matrix(matrix):
    return [list(row) for row in matrix]


def sparse_delta(row, class_count):
    result = zero_matrix(class_count)
    sparse = row.get('confusion_delta_sparse')
    if sparse is not None:
        for gt_idx, pred_idx, value in sparse:
            result[int(gt_idx)][int(pred_idx)] += int(value)
        return result
    dense = row.get('confusion_delta')
    if dense is not None:
        add_matrix(result, dense)
    return result


def confusion_metrics(matrix):
    class_count = len(matrix)
    total = sum(sum(row) for row in matrix)
    intersections = [matrix[idx][idx] for idx in range(class_count)]
    gt_area = [sum(matrix[idx]) for idx in range(class_count)]
    pred_area = [
        sum(matrix[row_idx][col_idx] for row_idx in range(class_count))
        for col_idx in range(class_count)
    ]
    unions = [
        gt_area[idx] + pred_area[idx] - intersections[idx]
        for idx in range(class_count)
    ]
    ious = [
        safe_div(intersections[idx], unions[idx])
        if unions[idx] > 0 else None
        for idx in range(class_count)
    ]
    accuracies = [
        safe_div(intersections[idx], gt_area[idx])
        if gt_area[idx] > 0 else None
        for idx in range(class_count)
    ]
    valid_ious = [value for value in ious if value is not None]
    valid_acc = [value for value in accuracies if value is not None]
    return dict(
        total_pixels=total,
        pixel_accuracy=safe_div(sum(intersections), total),
        mean_accuracy=(
            sum(valid_acc) / len(valid_acc) if valid_acc else None),
        miou=(sum(valid_ious) / len(valid_ious) if valid_ious else None),
        intersections=intersections,
        unions=unions,
        ious=ious,
    )


def descriptor_value(stats, layer, descriptor):
    values = []
    for row in stats.get('feature_stats') or []:
        if int(row.get('layer_index', -1)) != int(layer):
            continue
        variant_name = str(row.get('variant_name') or '')
        if not variant_name.endswith('_raw'):
            continue
        value = row.get(descriptor)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def build_gate_membership(records, layer, descriptor, quantile):
    values = defaultdict(list)
    for record in records:
        value = descriptor_value(
            record['stats'], layer, descriptor)
        record['gate_descriptor_value'] = value
        if value is not None:
            values[record['dataset']].append(
                (value, record['image_id']))

    quantile = min(max(float(quantile), 0.0), 1.0)
    selected = {}
    threshold_rows = []
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


def gate_applies(group, dataset, image_id, selected):
    is_selected = image_id in selected.get(dataset, set())
    if group == 'low_quantile':
        return is_selected
    if group == 'remaining':
        return not is_selected
    if group == 'all':
        return True
    raise ValueError(f'Unknown gate group: {group}')


def rule_key(row):
    return tuple(row.get(field) for field in RULE_FIELDS)


def rule_dict(key):
    return dict(zip(RULE_FIELDS, key))


def new_bucket(class_count):
    return dict(
        delta=zero_matrix(class_count),
        row_images=set(),
        activated_images=set(),
        eligible_pixels=0,
        selected_diag_pixels=0,
        potential_changed_pixels=0,
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        neutral_pixels=0,
        proposed_regions=0,
        retained_regions=0,
        raw_mask_available_images=0,
    )


def add_counterfactual(bucket, image_id, row, class_count):
    bucket['row_images'].add(image_id)
    changed = int(row.get('changed_pixels') or 0)
    if changed > 0:
        bucket['activated_images'].add(image_id)
    bucket['eligible_pixels'] += int(row.get('eligible_pixels') or 0)
    bucket['selected_diag_pixels'] += int(
        row.get('selected_diag_pixels') or 0)
    bucket['potential_changed_pixels'] += int(
        row.get('potential_changed_pixels') or 0)
    bucket['changed_pixels'] += changed
    bucket['improved_pixels'] += int(
        row.get('improved_pixels') or 0)
    bucket['harmed_pixels'] += int(row.get('harmed_pixels') or 0)
    bucket['neutral_pixels'] += int(row.get('neutral_pixels') or 0)
    bucket['proposed_regions'] += int(
        row.get('proposed_regions') or 0)
    bucket['retained_regions'] += int(
        row.get('retained_regions') or 0)
    bucket['raw_mask_available_images'] += int(
        bool(row.get('raw_mask_available', False)))
    add_matrix(
        bucket['delta'],
        sparse_delta(row, class_count),
    )


def build_curve_rows(records, selected, gate_groups):
    dataset_info = {}
    for record in records:
        dataset = record['dataset']
        class_count = len(record['baseline_confusion'])
        if dataset not in dataset_info:
            dataset_info[dataset] = dict(
                class_count=class_count,
                class_names=list(record['class_names']),
                baseline=zero_matrix(class_count),
                images=set(),
            )
        info = dataset_info[dataset]
        if info['class_count'] != class_count:
            raise ValueError(
                f'Class-count mismatch inside dataset {dataset}.')
        add_matrix(info['baseline'], record['baseline_confusion'])
        info['images'].add(record['image_id'])

    buckets = {}
    for record in records:
        dataset = record['dataset']
        class_count = dataset_info[dataset]['class_count']
        for group in gate_groups:
            if not gate_applies(
                    group, dataset, record['image_id'], selected):
                continue
            for row in (
                    record['stats'].get('counterfactual_stats') or []):
                key = (group, dataset, rule_key(row))
                if key not in buckets:
                    buckets[key] = new_bucket(class_count)
                add_counterfactual(
                    buckets[key],
                    record['image_id'],
                    row,
                    class_count,
                )

    curve_rows = []
    class_rows = []
    for (group, dataset, key), bucket in sorted(
            buckets.items(), key=lambda item: str(item[0])):
        info = dataset_info[dataset]
        baseline = info['baseline']
        counterfactual = copy_matrix(baseline)
        add_matrix(counterfactual, bucket['delta'])
        base_metrics = confusion_metrics(baseline)
        new_metrics = confusion_metrics(counterfactual)
        base_miou = base_metrics['miou']
        new_miou = new_metrics['miou']
        changed = bucket['changed_pixels']
        improved = bucket['improved_pixels']
        harmed = bucket['harmed_pixels']
        row = dict(
            gate_group=group,
            dataset_name=dataset,
            **rule_dict(key),
            dataset_images=len(info['images']),
            row_images=len(bucket['row_images']),
            activated_images=len(bucket['activated_images']),
            baseline_miou=base_miou,
            counterfactual_miou=new_miou,
            miou_delta=(
                None if base_miou is None or new_miou is None
                else new_miou - base_miou),
            baseline_pixel_accuracy=base_metrics['pixel_accuracy'],
            counterfactual_pixel_accuracy=(
                new_metrics['pixel_accuracy']),
            pixel_accuracy_delta=(
                None
                if base_metrics['pixel_accuracy'] is None
                or new_metrics['pixel_accuracy'] is None
                else new_metrics['pixel_accuracy']
                - base_metrics['pixel_accuracy']),
            eligible_pixels=bucket['eligible_pixels'],
            selected_diag_pixels=bucket['selected_diag_pixels'],
            potential_changed_pixels=(
                bucket['potential_changed_pixels']),
            changed_pixels=changed,
            changed_ratio=safe_div(
                changed, base_metrics['total_pixels']),
            improved_pixels=improved,
            harmed_pixels=harmed,
            neutral_pixels=bucket['neutral_pixels'],
            net_correct_pixels=improved - harmed,
            changed_correct_ratio=safe_div(improved, changed),
            utility_precision=safe_div(
                improved, improved + harmed),
            proposed_regions=bucket['proposed_regions'],
            retained_regions=bucket['retained_regions'],
            raw_mask_available_images=(
                bucket['raw_mask_available_images']),
        )
        curve_rows.append(row)
        for class_idx, class_name in enumerate(info['class_names']):
            base_iou = base_metrics['ious'][class_idx]
            new_iou = new_metrics['ious'][class_idx]
            class_rows.append(dict(
                gate_group=group,
                dataset_name=dataset,
                **rule_dict(key),
                class_index=class_idx,
                class_name=class_name,
                baseline_intersection=(
                    base_metrics['intersections'][class_idx]),
                baseline_union=base_metrics['unions'][class_idx],
                counterfactual_intersection=(
                    new_metrics['intersections'][class_idx]),
                counterfactual_union=(
                    new_metrics['unions'][class_idx]),
                baseline_iou=base_iou,
                counterfactual_iou=new_iou,
                iou_delta=(
                    None
                    if base_iou is None or new_iou is None
                    else new_iou - base_iou),
            ))
    return curve_rows, class_rows


def best_row(rows):
    eligible = [
        row for row in rows
        if row.get('miou_delta') is not None
        and int(row.get('changed_pixels') or 0) > 0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row['miou_delta']),
            float(row.get('utility_precision') or -1.0),
            -float(row.get('changed_ratio') or 0.0),
        ),
    )


def build_oracle_rows(curve_rows):
    groups = defaultdict(list)
    for row in curve_rows:
        groups[(row['gate_group'], row['dataset_name'])].append(row)
    result = []
    for (gate_group, dataset), rows in sorted(groups.items()):
        selected = best_row(rows)
        if selected is None:
            continue
        result.append(dict(
            oracle_scope='all_rules',
            **selected,
        ))
        for unit_name in ('pixel', 'component', 'raw_mask'):
            unit_selected = best_row([
                row for row in rows
                if row['unit_name'] == unit_name
            ])
            if unit_selected is not None:
                result.append(dict(
                    oracle_scope=unit_name,
                    **unit_selected,
                ))
    return result


def build_lodo_rows(curve_rows, gate_group, policies):
    by_dataset_rule = {}
    datasets = sorted({
        row['dataset_name'] for row in curve_rows
        if row['gate_group'] == gate_group
    })
    for row in curve_rows:
        if row['gate_group'] != gate_group:
            continue
        by_dataset_rule[
            (row['dataset_name'], rule_key(row))
        ] = row

    scopes = {
        'all_rules': None,
        'pixel': 'pixel',
        'component': 'component',
        'raw_mask': 'raw_mask',
    }
    result = []
    for heldout in datasets:
        train_datasets = [
            dataset for dataset in datasets if dataset != heldout]
        for scope_name, unit_filter in scopes.items():
            candidate_rules = set.intersection(*[
                {
                    key
                    for dataset, key in by_dataset_rule
                    if dataset == train_dataset
                    and (
                        unit_filter is None
                        or rule_dict(key)['unit_name'] == unit_filter)
                }
                for train_dataset in train_datasets
            ]) if train_datasets else set()
            for policy in policies:
                candidates = []
                for key in candidate_rules:
                    train_rows = [
                        by_dataset_rule[(dataset, key)]
                        for dataset in train_datasets
                    ]
                    deltas = [
                        row.get('miou_delta') for row in train_rows]
                    if any(value is None for value in deltas):
                        continue
                    if policy == 'nonnegative_best' and min(deltas) < 0:
                        continue
                    mean_delta = sum(deltas) / len(deltas)
                    min_delta = min(deltas)
                    mean_precision = sum(
                        float(row.get('utility_precision') or 0.0)
                        for row in train_rows
                    ) / len(train_rows)
                    candidates.append((
                        mean_delta,
                        min_delta,
                        mean_precision,
                        key,
                    ))
                if not candidates:
                    result.append(dict(
                        gate_group=gate_group,
                        heldout_dataset=heldout,
                        training_datasets='|'.join(train_datasets),
                        selection_scope=scope_name,
                        selection_policy=policy,
                        selected_rule_found=False,
                    ))
                    continue
                candidates.sort(reverse=True, key=lambda item: item[:3])
                mean_delta, min_delta, mean_precision, key = candidates[0]
                heldout_row = by_dataset_rule.get((heldout, key))
                result.append(dict(
                    gate_group=gate_group,
                    heldout_dataset=heldout,
                    training_datasets='|'.join(train_datasets),
                    selection_scope=scope_name,
                    selection_policy=policy,
                    selected_rule_found=True,
                    training_mean_miou_delta=mean_delta,
                    training_min_miou_delta=min_delta,
                    training_mean_utility_precision=mean_precision,
                    **rule_dict(key),
                    heldout_baseline_miou=(
                        None if heldout_row is None
                        else heldout_row.get('baseline_miou')),
                    heldout_counterfactual_miou=(
                        None if heldout_row is None
                        else heldout_row.get(
                            'counterfactual_miou')),
                    heldout_miou_delta=(
                        None if heldout_row is None
                        else heldout_row.get('miou_delta')),
                    heldout_changed_ratio=(
                        None if heldout_row is None
                        else heldout_row.get('changed_ratio')),
                    heldout_utility_precision=(
                        None if heldout_row is None
                        else heldout_row.get('utility_precision')),
                    heldout_improved_pixels=(
                        None if heldout_row is None
                        else heldout_row.get('improved_pixels')),
                    heldout_harmed_pixels=(
                        None if heldout_row is None
                        else heldout_row.get('harmed_pixels')),
                ))
    return result


def build_lodo_decision_rows(lodo_rows):
    groups = defaultdict(list)
    for row in lodo_rows:
        groups[(
            row.get('gate_group'),
            row.get('selection_scope'),
            row.get('selection_policy'),
        )].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        valid = [
            row for row in rows
            if row.get('selected_rule_found')
            and row.get('heldout_miou_delta') is not None
        ]
        deltas = [
            float(row['heldout_miou_delta']) for row in valid]
        precisions = [
            float(row['heldout_utility_precision'])
            for row in valid
            if row.get('heldout_utility_precision') is not None
        ]
        changed_ratios = [
            float(row['heldout_changed_ratio'])
            for row in valid
            if row.get('heldout_changed_ratio') is not None
        ]
        result.append(dict(
            gate_group=key[0],
            selection_scope=key[1],
            selection_policy=key[2],
            heldout_datasets=len(rows),
            rules_found=len(valid),
            positive_heldout_datasets=sum(
                value > 0.0 for value in deltas),
            nonnegative_heldout_datasets=sum(
                value >= 0.0 for value in deltas),
            all_heldout_nonnegative=bool(
                deltas
                and len(valid) == len(rows)
                and min(deltas) >= 0.0),
            mean_heldout_miou_delta=(
                sum(deltas) / len(deltas) if deltas else None),
            minimum_heldout_miou_delta=(
                min(deltas) if deltas else None),
            maximum_heldout_miou_delta=(
                max(deltas) if deltas else None),
            mean_heldout_utility_precision=(
                sum(precisions) / len(precisions)
                if precisions else None),
            mean_heldout_changed_ratio=(
                sum(changed_ratios) / len(changed_ratios)
                if changed_ratios else None),
        ))
    return result


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')

    records = []
    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get('candidate_residual_miou_stats') or {}
        baseline_confusion = stats.get('baseline_confusion')
        if not stats or not baseline_confusion:
            continue
        class_names = list(stats.get('class_names') or [])
        if len(class_names) != len(baseline_confusion):
            class_names = [
                f'class_{idx}'
                for idx in range(len(baseline_confusion))
            ]
        records.append(dict(
            dataset=record_dataset(record, stats),
            image_id=record_identity(record, record_index),
            baseline_confusion=baseline_confusion,
            class_names=class_names,
            stats=stats,
        ))
    if not records:
        raise ValueError(
            'No candidate_residual_miou_stats records were found.')

    selected, threshold_rows = build_gate_membership(
        records,
        args.gate_layer,
        args.gate_descriptor,
        args.gate_quantile,
    )
    gate_groups = parse_names(args.gate_groups)
    curve_rows, class_rows = build_curve_rows(
        records, selected, gate_groups)
    oracle_rows = build_oracle_rows(curve_rows)
    lodo_rows = build_lodo_rows(
        curve_rows,
        args.lodo_gate_group,
        parse_names(args.lodo_selection_policies),
    )
    lodo_decision_rows = build_lodo_decision_rows(lodo_rows)

    write_csv(
        os.path.join(args.out_dir, 'gate_thresholds.csv'),
        threshold_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'counterfactual_curve.csv'),
        curve_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'counterfactual_class_curve.csv'),
        class_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'counterfactual_oracle.csv'),
        oracle_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'counterfactual_lodo.csv'),
        lodo_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'counterfactual_lodo_decision.csv'),
        lodo_decision_rows)

    print(f'Read {len(paths)} JSONL files.')
    print(f'Found counterfactual stats for {len(records)} images.')
    print(
        'Datasets: '
        + ', '.join(sorted({record['dataset'] for record in records})))
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
