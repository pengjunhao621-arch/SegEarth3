import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


COUNT_FEATURES = (
    'feature_pixels',
    'predicted_pixels',
    'raw_top1_pixels',
    'semantic_support_pixels',
    'instance_support_pixels',
    'final_support_pixels',
    'spatial_support_pixels',
    'semantic_instance_union_pixels',
    'semantic_instance_intersection_pixels',
    'semantic_only_pixels',
    'instance_only_pixels',
    'low_presence_spatial_pixels',
    'low_margin_top1_pixels',
    'threshold_reject_pixels',
    'head_agree_top1_pixels',
    'local_core_pixels',
    'top_competitor_pixels',
    'catch_all_competitor_pixels',
)

SUM_FEATURES = (
    'local_consistency_sum',
    'effective_presence_support_sum',
    'final_score_sum',
    'semantic_score_sum',
    'instance_score_sum',
    'no_presence_score_sum',
    'prompt_gap_sum',
)

LODO_FEATURES = (
    'predicted_ratio',
    'semantic_support_ratio',
    'instance_support_ratio',
    'final_support_ratio',
    'spatial_support_ratio',
    'semantic_instance_iou',
    'semantic_only_ratio',
    'instance_only_ratio',
    'low_presence_spatial_ratio',
    'low_margin_top1_ratio',
    'threshold_reject_ratio',
    'head_agree_top1_ratio',
    'local_consistency',
    'local_core_ratio',
    'effective_presence_mean',
    'semantic_expansion_over_final',
    'instance_expansion_over_final',
    'no_presence_expansion_over_final',
    'prompt_gap_mean',
    'prompt_winner_entropy',
    'catch_all_competitor_ratio',
    'presence_under_score',
    'catch_all_pressure_score',
    'competition_score',
    'weak_dual_score',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize state-action atlas JSONL files.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--lodo-min-gt-pixels', type=int, default=64)
    parser.add_argument(
        '--lodo-min-positive-iou-delta',
        type=float,
        default=0.01,
    )
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


def safe_div(numerator, denominator, default=None):
    if denominator in (None, 0):
        return default
    return float(numerator) / float(denominator)


def add_matrix(target, source):
    if source is None:
        return
    if not target:
        target.extend([[0 for _ in row] for row in source])
    for row_idx, row in enumerate(source):
        for col_idx, value in enumerate(row):
            target[row_idx][col_idx] += int(value)


def confusion_metrics(matrix):
    if not matrix:
        return dict(
            total_pixels=0,
            pixel_accuracy=None,
            miou=None,
            ious=[],
        )
    class_count = len(matrix)
    intersections = [matrix[idx][idx] for idx in range(class_count)]
    gt_pixels = [sum(matrix[idx]) for idx in range(class_count)]
    pred_pixels = [
        sum(matrix[row_idx][col_idx] for row_idx in range(class_count))
        for col_idx in range(class_count)
    ]
    unions = [
        gt_pixels[idx] + pred_pixels[idx] - intersections[idx]
        for idx in range(class_count)
    ]
    ious = [
        safe_div(intersections[idx], unions[idx])
        if unions[idx] > 0 else None
        for idx in range(class_count)
    ]
    valid_ious = [value for value in ious if value is not None]
    total = sum(gt_pixels)
    return dict(
        total_pixels=total,
        pixel_accuracy=safe_div(sum(intersections), total),
        miou=(
            sum(valid_ious) / len(valid_ious)
            if valid_ious else None),
        ious=ious,
    )


def class_iou(row):
    return safe_div(
        row.get('intersection_pixels', 0),
        row.get('union_pixels', 0),
    )


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def derive_feature_ratios(row):
    result = dict(row)
    feature_pixels = result.get('feature_pixels', 0)
    predicted_pixels = result.get('predicted_pixels', 0)
    raw_top1_pixels = result.get('raw_top1_pixels', 0)
    spatial_pixels = result.get('spatial_support_pixels', 0)
    union_pixels = result.get(
        'semantic_instance_union_pixels', 0)
    final_pixels = result.get('final_support_pixels', 0)
    result.update(
        predicted_ratio=safe_div(
            predicted_pixels, feature_pixels, 0.0),
        semantic_support_ratio=safe_div(
            result.get('semantic_support_pixels', 0),
            feature_pixels,
            0.0,
        ),
        instance_support_ratio=safe_div(
            result.get('instance_support_pixels', 0),
            feature_pixels,
            0.0,
        ),
        final_support_ratio=safe_div(
            final_pixels, feature_pixels, 0.0),
        spatial_support_ratio=safe_div(
            spatial_pixels, feature_pixels, 0.0),
        semantic_instance_iou=safe_div(
            result.get(
                'semantic_instance_intersection_pixels', 0),
            union_pixels,
            0.0,
        ),
        semantic_only_ratio=safe_div(
            result.get('semantic_only_pixels', 0),
            feature_pixels,
            0.0,
        ),
        instance_only_ratio=safe_div(
            result.get('instance_only_pixels', 0),
            feature_pixels,
            0.0,
        ),
        low_presence_spatial_ratio=safe_div(
            result.get('low_presence_spatial_pixels', 0),
            spatial_pixels,
            0.0,
        ),
        low_margin_top1_ratio=safe_div(
            result.get('low_margin_top1_pixels', 0),
            raw_top1_pixels,
            0.0,
        ),
        threshold_reject_ratio=safe_div(
            result.get('threshold_reject_pixels', 0),
            raw_top1_pixels,
            0.0,
        ),
        head_agree_top1_ratio=safe_div(
            result.get('head_agree_top1_pixels', 0),
            raw_top1_pixels,
            0.0,
        ),
        local_consistency=safe_div(
            result.get('local_consistency_sum', 0.0),
            predicted_pixels,
            0.0,
        ),
        local_core_ratio=safe_div(
            result.get('local_core_pixels', 0),
            predicted_pixels,
            0.0,
        ),
        effective_presence_mean=safe_div(
            result.get(
                'effective_presence_support_sum', 0.0),
            spatial_pixels,
            0.0,
        ),
        final_score_mean=safe_div(
            result.get('final_score_sum', 0.0),
            feature_pixels,
            0.0,
        ),
        semantic_score_mean=safe_div(
            result.get('semantic_score_sum', 0.0),
            feature_pixels,
            0.0,
        ),
        instance_score_mean=safe_div(
            result.get('instance_score_sum', 0.0),
            feature_pixels,
            0.0,
        ),
        no_presence_score_mean=safe_div(
            result.get('no_presence_score_sum', 0.0),
            feature_pixels,
            0.0,
        ),
        semantic_expansion_over_final=safe_div(
            result.get('semantic_support_pixels', 0),
            max(final_pixels, 1),
            0.0,
        ),
        instance_expansion_over_final=safe_div(
            result.get('instance_support_pixels', 0),
            max(final_pixels, 1),
            0.0,
        ),
        no_presence_expansion_over_final=safe_div(
            spatial_pixels,
            max(final_pixels, 1),
            0.0,
        ),
        prompt_gap_mean=safe_div(
            result.get('prompt_gap_sum', 0.0),
            feature_pixels,
            0.0,
        ),
        catch_all_competitor_ratio=safe_div(
            result.get('catch_all_competitor_pixels', 0),
            raw_top1_pixels,
            0.0,
        ),
    )
    return result


def summarize(records):
    dataset_base = defaultdict(list)
    dataset_oracle = defaultdict(list)
    action_groups = defaultdict(lambda: dict(
        confusion=[],
        images=0,
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
        net_correct_pixels=0,
    ))
    class_groups = defaultdict(lambda: defaultdict(float))
    feature_groups = defaultdict(lambda: defaultdict(float))
    feature_meta = {}
    pair_groups = defaultdict(lambda: defaultdict(float))
    image_rows = []
    image_class_rows = []
    image_oracle_confusions = defaultdict(list)
    dataset_actions = defaultdict(set)

    for record in records:
        stats = record.get('state_action_atlas_stats')
        if not stats:
            continue
        dataset = str(
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown')
        img_path = record.get('img_path')
        base_confusion = stats.get('baseline_confusion')
        oracle_confusion = stats.get('pixel_oracle_confusion')
        dataset_base[dataset].append(base_confusion)
        dataset_oracle[dataset].append(oracle_confusion)
        base_metrics = confusion_metrics(base_confusion)

        action_by_name = {
            row['action_name']: row
            for row in stats.get('action_rows') or []
        }
        best_image_action = None
        best_image_miou = -math.inf
        for action_name, action in action_by_name.items():
            dataset_actions[dataset].add(action_name)
            key = (dataset, action_name)
            group = action_groups[key]
            group['images'] += 1
            add_matrix(group['confusion'], action.get('confusion'))
            for field in (
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels',
                    'wrong_to_wrong_pixels',
                    'net_correct_pixels'):
                group[field] += int(action.get(field, 0))
            metrics = confusion_metrics(action.get('confusion'))
            delta = (
                metrics['miou'] - base_metrics['miou']
                if (
                    metrics['miou'] is not None
                    and base_metrics['miou'] is not None)
                else None
            )
            image_rows.append(dict(
                dataset_name=dataset,
                img_path=img_path,
                action_name=action_name,
                baseline_miou=base_metrics['miou'],
                action_miou=metrics['miou'],
                miou_delta=delta,
                baseline_pixel_accuracy=(
                    base_metrics['pixel_accuracy']),
                action_pixel_accuracy=metrics['pixel_accuracy'],
                changed_pixels=action.get('changed_pixels', 0),
                improved_pixels=action.get('improved_pixels', 0),
                harmed_pixels=action.get('harmed_pixels', 0),
                net_correct_pixels=action.get(
                    'net_correct_pixels', 0),
            ))
            if (
                    metrics['miou'] is not None
                    and metrics['miou'] > best_image_miou):
                best_image_miou = metrics['miou']
                best_image_action = action_name
        if best_image_action is not None:
            add_matrix(
                image_oracle_confusions[dataset],
                action_by_name[best_image_action]['confusion'],
            )

        class_actions = defaultdict(dict)
        for row in stats.get('class_action_rows') or []:
            action_name = str(row['action_name'])
            class_idx = int(row['class_index'])
            class_actions[class_idx][action_name] = row
            key = (dataset, action_name, class_idx)
            bucket = class_groups[key]
            for field in (
                    'gt_pixels', 'pred_pixels',
                    'intersection_pixels', 'union_pixels',
                    'changed_pixels', 'improved_pixels',
                    'harmed_pixels'):
                bucket[field] += int(row.get(field, 0))
            bucket['class_name'] = row.get('class_name')

        for feature in stats.get('class_feature_rows') or []:
            class_idx = int(feature['class_index'])
            key = (dataset, class_idx)
            bucket = feature_groups[key]
            for field in COUNT_FEATURES + SUM_FEATURES:
                bucket[field] += float(feature.get(field, 0))
            bucket['images'] += 1
            bucket['prompt_winner_entropy_sum'] += float(
                feature.get('prompt_winner_entropy', 0.0))
            bucket['presence_under_score_sum'] += float(
                feature.get('presence_under_score', 0.0))
            bucket['catch_all_pressure_score_sum'] += float(
                feature.get('catch_all_pressure_score', 0.0))
            bucket['competition_score_sum'] += float(
                feature.get('competition_score', 0.0))
            bucket['weak_dual_score_sum'] += float(
                feature.get('weak_dual_score', 0.0))
            feature_meta[key] = dict(
                class_name=feature.get('class_name'),
                query_count=feature.get('query_count'),
                is_catch_all=feature.get('is_catch_all'),
            )

            image_class = derive_feature_ratios(dict(
                dataset_name=dataset,
                img_path=img_path,
                **feature,
            ))
            base_class = class_actions.get(
                class_idx, {}).get('baseline')
            base_iou = class_iou(base_class) if base_class else None
            image_class['baseline_iou'] = base_iou
            best_action = 'baseline'
            best_delta = 0.0
            for action_name, action_row in (
                    class_actions.get(class_idx, {}).items()):
                action_iou = class_iou(action_row)
                delta = (
                    action_iou - base_iou
                    if (
                        action_iou is not None
                        and base_iou is not None)
                    else None
                )
                image_class[f'iou_delta__{action_name}'] = delta
                if delta is not None and delta > best_delta:
                    best_delta = delta
                    best_action = action_name
            image_class['best_action'] = best_action
            image_class['best_iou_delta'] = best_delta
            if base_class is not None:
                image_class['gt_pixels'] = base_class.get(
                    'gt_pixels', 0)
                image_class['baseline_union_pixels'] = (
                    base_class.get('union_pixels', 0))
            image_class_rows.append(image_class)

        for row in stats.get('pair_action_rows') or []:
            key = (
                dataset,
                str(row['action_name']),
                int(row['gt_class_index']),
                int(row['base_pred_class_index']),
            )
            bucket = pair_groups[key]
            for field in (
                    'pixels', 'changed_pixels',
                    'corrected_pixels',
                    'changed_to_other_wrong_pixels'):
                bucket[field] += int(row.get(field, 0))
            bucket['gt_class_name'] = row.get(
                'gt_class_name')
            bucket['base_pred_class_name'] = row.get(
                'base_pred_class_name')

    overall_rows = []
    for (dataset, action_name), group in sorted(
            action_groups.items()):
        base_matrix = []
        for matrix in dataset_base[dataset]:
            add_matrix(base_matrix, matrix)
        base = confusion_metrics(base_matrix)
        action = confusion_metrics(group['confusion'])
        overall_rows.append(dict(
            dataset_name=dataset,
            action_name=action_name,
            images=group['images'],
            valid_pixels=base['total_pixels'],
            baseline_miou=base['miou'],
            action_miou=action['miou'],
            miou_delta=(
                action['miou'] - base['miou']),
            baseline_pixel_accuracy=base['pixel_accuracy'],
            action_pixel_accuracy=action['pixel_accuracy'],
            pixel_accuracy_delta=(
                action['pixel_accuracy']
                - base['pixel_accuracy']),
            changed_pixels=group['changed_pixels'],
            improved_pixels=group['improved_pixels'],
            harmed_pixels=group['harmed_pixels'],
            wrong_to_wrong_pixels=(
                group['wrong_to_wrong_pixels']),
            net_correct_pixels=group['net_correct_pixels'],
            changed_ratio=safe_div(
                group['changed_pixels'],
                base['total_pixels']),
            utility_precision=safe_div(
                group['improved_pixels'],
                group['improved_pixels']
                + group['harmed_pixels']),
        ))

    class_rows = []
    for (dataset, action_name, class_idx), bucket in sorted(
            class_groups.items()):
        baseline_bucket = class_groups.get(
            (dataset, 'baseline', class_idx), {})
        baseline_iou = class_iou(baseline_bucket)
        action_iou = class_iou(bucket)
        class_rows.append(dict(
            dataset_name=dataset,
            action_name=action_name,
            class_index=class_idx,
            class_name=bucket.get('class_name'),
            baseline_iou=baseline_iou,
            action_iou=action_iou,
            iou_delta=(
                action_iou - baseline_iou
                if (
                    action_iou is not None
                    and baseline_iou is not None)
                else None),
            **{
                field: int(bucket.get(field, 0))
                for field in (
                    'gt_pixels', 'pred_pixels',
                    'intersection_pixels', 'union_pixels',
                    'changed_pixels', 'improved_pixels',
                    'harmed_pixels')
            },
            utility_precision=safe_div(
                bucket.get('improved_pixels', 0),
                bucket.get('improved_pixels', 0)
                + bucket.get('harmed_pixels', 0)),
        ))

    state_rows = []
    for (dataset, class_idx), bucket in sorted(
            feature_groups.items()):
        images = max(1, int(bucket.get('images', 0)))
        row = dict(
            dataset_name=dataset,
            class_index=class_idx,
            **feature_meta.get((dataset, class_idx), {}),
        )
        for field in COUNT_FEATURES:
            row[field] = int(bucket.get(field, 0))
        for field in SUM_FEATURES:
            row[field] = float(bucket.get(field, 0.0))
        row['prompt_winner_entropy'] = (
            bucket.get('prompt_winner_entropy_sum', 0.0)
            / images)
        for field in (
                'presence_under_score',
                'catch_all_pressure_score',
                'competition_score',
                'weak_dual_score'):
            row[field] = (
                bucket.get(f'{field}_sum', 0.0) / images)
        row = derive_feature_ratios(row)
        best_action = 'baseline'
        best_delta = 0.0
        for action_name in dataset_actions[dataset]:
            action_bucket = class_groups.get(
                (dataset, action_name, class_idx))
            baseline_bucket = class_groups.get(
                (dataset, 'baseline', class_idx))
            if not action_bucket or not baseline_bucket:
                continue
            action_iou = class_iou(action_bucket)
            baseline_iou = class_iou(baseline_bucket)
            if action_iou is None or baseline_iou is None:
                continue
            delta = action_iou - baseline_iou
            row[f'iou_delta__{action_name}'] = delta
            if delta > best_delta:
                best_delta = delta
                best_action = action_name
        row['best_action'] = best_action
        row['best_iou_delta'] = best_delta
        state_rows.append(row)

    pair_rows = []
    for key, bucket in sorted(pair_groups.items()):
        dataset, action_name, gt_idx, pred_idx = key
        pair_rows.append(dict(
            dataset_name=dataset,
            action_name=action_name,
            gt_class_index=gt_idx,
            gt_class_name=bucket.get('gt_class_name'),
            base_pred_class_index=pred_idx,
            base_pred_class_name=bucket.get(
                'base_pred_class_name'),
            pixels=int(bucket.get('pixels', 0)),
            changed_pixels=int(
                bucket.get('changed_pixels', 0)),
            corrected_pixels=int(
                bucket.get('corrected_pixels', 0)),
            changed_to_other_wrong_pixels=int(
                bucket.get(
                    'changed_to_other_wrong_pixels', 0)),
            changed_ratio=safe_div(
                bucket.get('changed_pixels', 0),
                bucket.get('pixels', 0)),
            corrected_ratio=safe_div(
                bucket.get('corrected_pixels', 0),
                bucket.get('pixels', 0)),
            correction_precision=safe_div(
                bucket.get('corrected_pixels', 0),
                bucket.get('changed_pixels', 0)),
        ))

    oracle_rows = []
    for dataset in sorted(dataset_base):
        base_matrix = []
        pixel_oracle_matrix = []
        for matrix in dataset_base[dataset]:
            add_matrix(base_matrix, matrix)
        for matrix in dataset_oracle[dataset]:
            add_matrix(pixel_oracle_matrix, matrix)
        base = confusion_metrics(base_matrix)
        pixel_oracle = confusion_metrics(pixel_oracle_matrix)
        image_oracle = confusion_metrics(
            image_oracle_confusions[dataset])
        dataset_action_rows = [
            row for row in overall_rows
            if row['dataset_name'] == dataset
        ]
        best_single = max(
            dataset_action_rows,
            key=lambda row: row['action_miou'],
        )
        per_class_best = []
        for class_idx in range(len(base['ious'])):
            candidates = []
            for action_name in dataset_actions[dataset]:
                bucket = class_groups.get(
                    (dataset, action_name, class_idx))
                if bucket:
                    value = class_iou(bucket)
                    if value is not None:
                        candidates.append((value, action_name))
            if candidates:
                per_class_best.append(max(candidates)[0])
        oracle_rows.append(dict(
            dataset_name=dataset,
            baseline_miou=base['miou'],
            best_single_action=best_single['action_name'],
            best_single_action_miou=best_single['action_miou'],
            best_single_action_delta=best_single['miou_delta'],
            image_oracle_miou=image_oracle['miou'],
            image_oracle_delta=(
                image_oracle['miou'] - base['miou']),
            class_oracle_miou=(
                sum(per_class_best) / len(per_class_best)
                if per_class_best else None),
            class_oracle_delta=(
                (
                    sum(per_class_best) / len(per_class_best)
                    - base['miou']
                ) if per_class_best else None),
            pixel_oracle_miou=pixel_oracle['miou'],
            pixel_oracle_delta=(
                pixel_oracle['miou'] - base['miou']),
        ))

    return dict(
        overall=overall_rows,
        classes=class_rows,
        states=state_rows,
        pairs=pair_rows,
        images=image_rows,
        image_classes=image_class_rows,
        oracles=oracle_rows,
    )


def lodo_nearest_centroid(
        rows, min_gt_pixels, min_positive_delta):
    datasets = sorted({
        row['dataset_name'] for row in rows})
    output = []
    action_fields = sorted({
        field
        for row in rows
        for field in row
        if field.startswith('iou_delta__')
    })
    actions = [
        field.split('__', 1)[1]
        for field in action_fields
    ]
    usable = [
        row for row in rows
        if int(row.get('gt_pixels', 0) or 0) >= min_gt_pixels
        and int(
            row.get('baseline_union_pixels', 0) or 0) > 0
    ]
    for held_out in datasets:
        train = [
            row for row in usable
            if row['dataset_name'] != held_out]
        test = [
            row for row in usable
            if row['dataset_name'] == held_out]
        if not train or not test:
            continue

        x_train = [
            [float(row.get(field, 0.0) or 0.0)
             for field in LODO_FEATURES]
            for row in train
        ]
        feature_count = len(LODO_FEATURES)
        means = [
            sum(vector[idx] for vector in x_train)
            / len(x_train)
            for idx in range(feature_count)
        ]
        stds = []
        for idx in range(feature_count):
            variance = sum(
                (vector[idx] - means[idx]) ** 2
                for vector in x_train
            ) / len(x_train)
            stds.append(
                math.sqrt(variance)
                if variance >= 1e-16 else 1.0)
        x_train = [
            [
                (vector[idx] - means[idx]) / stds[idx]
                for idx in range(feature_count)
            ]
            for vector in x_train
        ]

        labels = []
        for row in train:
            best_action = 'baseline'
            best_delta = float(min_positive_delta)
            for action in actions:
                value = row.get(f'iou_delta__{action}')
                if value is not None and float(value) > best_delta:
                    best_action = action
                    best_delta = float(value)
            labels.append(best_action)
        centroid_sums = defaultdict(
            lambda: [0.0] * feature_count)
        centroid_counts = defaultdict(int)
        for vector, label in zip(x_train, labels):
            centroid_counts[label] += 1
            for idx, value in enumerate(vector):
                centroid_sums[label][idx] += value
        centroids = {
            label: [
                value / centroid_counts[label]
                for value in values
            ]
            for label, values in centroid_sums.items()
        }

        selected_deltas = []
        positive_choices = 0
        useful_choices = 0
        correct_labels = 0
        for row in test:
            vector = [
                float(row.get(field, 0.0) or 0.0)
                for field in LODO_FEATURES
            ]
            vector = [
                (vector[idx] - means[idx]) / stds[idx]
                for idx in range(feature_count)
            ]
            predicted = min(
                centroids,
                key=lambda label: sum(
                    (
                        vector[idx]
                        - centroids[label][idx]
                    ) ** 2
                    for idx in range(feature_count)
                ),
            )
            selected = float(
                row.get(f'iou_delta__{predicted}', 0.0)
                or 0.0)
            selected_deltas.append(selected)
            if predicted != 'baseline':
                positive_choices += 1
                if selected > 0:
                    useful_choices += 1

            oracle_label = 'baseline'
            oracle_delta = float(min_positive_delta)
            for action in actions:
                value = row.get(f'iou_delta__{action}')
                if value is not None and float(value) > oracle_delta:
                    oracle_label = action
                    oracle_delta = float(value)
            if predicted == oracle_label:
                correct_labels += 1

        output.append(dict(
            held_out_dataset=held_out,
            train_rows=len(train),
            test_rows=len(test),
            available_labels=','.join(sorted(centroids)),
            mean_selected_class_iou_delta=(
                sum(selected_deltas) / len(selected_deltas)),
            positive_selected_ratio=safe_div(
                positive_choices, len(test)),
            positive_selection_precision=safe_div(
                useful_choices, positive_choices),
            best_action_label_accuracy=safe_div(
                correct_labels, len(test)),
            note=(
                'Diagnostic class-level transfer only; '
                'not a realizable routed mIoU.'),
        ))
    return output


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(
            f'No JSONL inputs matched: {args.inputs}')
    tables = summarize(iter_records(paths))
    lodo_rows = lodo_nearest_centroid(
        tables['image_classes'],
        args.lodo_min_gt_pixels,
        args.lodo_min_positive_iou_delta,
    )
    outputs = {
        'state_action_overall.csv': tables['overall'],
        'state_action_class_summary.csv': tables['classes'],
        'state_action_state_summary.csv': tables['states'],
        'state_action_pair_summary.csv': tables['pairs'],
        'state_action_image_summary.csv': tables['images'],
        'state_action_image_class.csv': tables['image_classes'],
        'state_action_oracle_summary.csv': tables['oracles'],
        'state_action_lodo_diagnostic.csv': lodo_rows,
    }
    for name, rows in outputs.items():
        write_csv(os.path.join(args.out_dir, name), rows)
    print(f'Read {len(paths)} JSONL files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
