import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SAM3 region-contrastive readout diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--selection-datasets',
        default='udd5,vdd,vaihingen',
        help='Comma-separated development datasets for shared-action ranking.')
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
    intersection = [matrix[idx][idx] for idx in range(class_count)]
    gt_pixels = [sum(matrix[idx]) for idx in range(class_count)]
    pred_pixels = [
        sum(matrix[row][idx] for row in range(class_count))
        for idx in range(class_count)
    ]
    union = [
        gt_pixels[idx] + pred_pixels[idx] - intersection[idx]
        for idx in range(class_count)
    ]
    ious = [
        safe_div(intersection[idx], union[idx])
        if union[idx] > 0 else None
        for idx in range(class_count)
    ]
    valid = [value for value in ious if value is not None]
    total = sum(gt_pixels)
    return dict(
        total_pixels=total,
        pixel_accuracy=safe_div(sum(intersection), total),
        miou=sum(valid) / len(valid) if valid else None,
        ious=ious,
    )


def write_csv(path, rows, default_fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(default_fields)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, selection_datasets):
    baseline = defaultdict(list)
    pixel_oracle = defaultdict(list)
    exact_baseline = defaultdict(list)
    exact_threshold_disabled = defaultdict(list)
    exact_action_groups = defaultdict(lambda: dict(
        confusion=[],
        images=0,
    ))
    route_groups = defaultdict(lambda: defaultdict(float))
    class_names_by_dataset = {}
    images = defaultdict(int)
    region_counts = defaultdict(int)
    action_groups = defaultdict(lambda: dict(
        confusion=[],
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
        net_correct_pixels=0,
    ))
    source_groups = defaultdict(lambda: defaultdict(float))
    class_groups = defaultdict(lambda: defaultdict(float))
    pair_groups = defaultdict(lambda: defaultdict(float))
    region_groups = defaultdict(lambda: defaultdict(float))

    for record in records:
        stats = record.get('region_contrastive_readout_stats')
        if not stats:
            continue
        dataset = str(
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown')
        if stats.get('class_names'):
            class_names_by_dataset[dataset] = list(
                stats['class_names'])
        images[dataset] += 1
        region_counts[dataset] += int(stats.get('region_count', 0))
        add_matrix(baseline[dataset], stats.get('baseline_confusion'))
        add_matrix(
            pixel_oracle[dataset],
            stats.get('pixel_oracle_confusion'),
        )
        add_matrix(
            exact_baseline[dataset],
            stats.get('exact_baseline_confusion'),
        )
        add_matrix(
            exact_threshold_disabled[dataset],
            stats.get('exact_threshold_disabled_confusion'),
        )
        exact_action = str(
            stats.get('exact_action_name') or 'unknown')
        exact_confusion = stats.get('exact_applied_confusion')
        if exact_confusion is not None:
            exact_bucket = exact_action_groups[
                (dataset, exact_action)]
            add_matrix(exact_bucket['confusion'], exact_confusion)
            exact_bucket['images'] += 1
        for row in stats.get('route_rows') or []:
            key = (
                dataset,
                exact_action,
                str(row['route_name']),
                str(row['source_name']),
                float(row['blend']),
            )
            bucket = route_groups[key]
            for field in (
                    'eligible_pixels',
                    'accepted_pixels',
                    'gt_foreground_pixels',
                    'gt_background_pixels',
                    'baseline_wrong_pixels',
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels',
                    'wrong_to_wrong_pixels'):
                bucket[field] += int(row.get(field, 0))

        for row in stats.get('action_rows') or []:
            action = str(row['action_name'])
            bucket = action_groups[(dataset, action)]
            add_matrix(bucket['confusion'], row.get('confusion'))
            for field in (
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels',
                    'wrong_to_wrong_pixels',
                    'net_correct_pixels'):
                bucket[field] += int(row.get(field, 0))

        for row in stats.get('source_rows') or []:
            source = str(row['source_name'])
            bucket = source_groups[(dataset, source)]
            for field in (
                    'covered_pixels',
                    'baseline_wrong_covered_pixels',
                    'wrong_top1_recovers_gt_pixels',
                    'wrong_topk_contains_gt_pixels',
                    'wrong_gt_beats_base_pixels'):
                bucket[field] += int(row.get(field, 0))
            bucket['wrong_gt_vs_base_margin_sum'] += float(
                row.get('wrong_gt_vs_base_margin_sum', 0.0))

        for row in stats.get('class_rows') or []:
            key = (
                dataset,
                str(row['action_name']),
                int(row['class_index']),
            )
            bucket = class_groups[key]
            bucket['class_name'] = row.get('class_name')
            for field in (
                    'gt_pixels',
                    'pred_pixels',
                    'baseline_pred_pixels',
                    'intersection_pixels',
                    'union_pixels',
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels'):
                bucket[field] += int(row.get(field, 0))

        for row in stats.get('pair_rows') or []:
            key = (
                dataset,
                str(row['action_name']),
                int(row['gt_class_index']),
                int(row['base_pred_class_index']),
            )
            bucket = pair_groups[key]
            bucket['gt_class_name'] = row.get('gt_class_name')
            bucket['base_pred_class_name'] = row.get(
                'base_pred_class_name')
            for field in (
                    'pixels',
                    'changed_pixels',
                    'corrected_pixels'):
                bucket[field] += int(row.get(field, 0))

        for row in stats.get('region_rows') or []:
            class_idx = row.get('class_index')
            key = (
                dataset,
                str(row['source_name']),
                '__all__' if class_idx is None else int(class_idx),
            )
            bucket = region_groups[key]
            bucket['class_name'] = row.get('class_name')
            for field in (
                    'regions',
                    'valid_regions',
                    'correct_regions',
                    'high_purity_regions',
                    'high_purity_correct_regions',
                    'origin_correct_regions',
                    'area_pixels'):
                bucket[field] += int(row.get(field, 0) or 0)
            for field in ('purity_sum', 'margin_sum'):
                bucket[field] += float(row.get(field, 0.0) or 0.0)

    overall_rows = []
    action_metrics = defaultdict(dict)
    for (dataset, action), bucket in sorted(action_groups.items()):
        base_metrics = confusion_metrics(baseline[dataset])
        metrics = confusion_metrics(bucket['confusion'])
        action_metrics[dataset][action] = metrics
        overall_rows.append(dict(
            dataset_name=dataset,
            action_name=action,
            images=images[dataset],
            regions=region_counts[dataset],
            valid_pixels=base_metrics['total_pixels'],
            baseline_miou=base_metrics['miou'],
            action_miou=metrics['miou'],
            miou_delta=(
                metrics['miou'] - base_metrics['miou']
                if metrics['miou'] is not None
                and base_metrics['miou'] is not None
                else None),
            baseline_pixel_accuracy=(
                base_metrics['pixel_accuracy']),
            action_pixel_accuracy=metrics['pixel_accuracy'],
            pixel_accuracy_delta=(
                metrics['pixel_accuracy']
                - base_metrics['pixel_accuracy']
                if metrics['pixel_accuracy'] is not None
                and base_metrics['pixel_accuracy'] is not None
                else None),
            changed_pixels=bucket['changed_pixels'],
            improved_pixels=bucket['improved_pixels'],
            harmed_pixels=bucket['harmed_pixels'],
            wrong_to_wrong_pixels=(
                bucket['wrong_to_wrong_pixels']),
            net_correct_pixels=bucket['net_correct_pixels'],
            utility_precision=safe_div(
                bucket['improved_pixels'],
                bucket['improved_pixels']
                + bucket['harmed_pixels']),
        ))

    source_rows = []
    for (dataset, source), bucket in sorted(source_groups.items()):
        base_metrics = confusion_metrics(baseline[dataset])
        wrong_pixels = (
            base_metrics['total_pixels']
            - sum(
                baseline[dataset][idx][idx]
                for idx in range(len(baseline[dataset]))
            )
        )
        covered_wrong = bucket['baseline_wrong_covered_pixels']
        source_rows.append(dict(
            dataset_name=dataset,
            source_name=source,
            covered_pixels=bucket['covered_pixels'],
            coverage_ratio=safe_div(
                bucket['covered_pixels'],
                base_metrics['total_pixels']),
            baseline_wrong_covered_pixels=covered_wrong,
            baseline_wrong_coverage=safe_div(
                covered_wrong, wrong_pixels),
            wrong_top1_recovers_gt_pixels=(
                bucket['wrong_top1_recovers_gt_pixels']),
            wrong_top1_recovers_gt_ratio=safe_div(
                bucket['wrong_top1_recovers_gt_pixels'],
                covered_wrong),
            wrong_topk_contains_gt_pixels=(
                bucket['wrong_topk_contains_gt_pixels']),
            wrong_topk_contains_gt_ratio=safe_div(
                bucket['wrong_topk_contains_gt_pixels'],
                covered_wrong),
            wrong_gt_beats_base_pixels=(
                bucket['wrong_gt_beats_base_pixels']),
            wrong_gt_beats_base_ratio=safe_div(
                bucket['wrong_gt_beats_base_pixels'],
                covered_wrong),
            mean_wrong_gt_vs_base_margin=safe_div(
                bucket['wrong_gt_vs_base_margin_sum'],
                covered_wrong),
        ))

    class_rows = []
    for (dataset, action, class_idx), bucket in sorted(
            class_groups.items()):
        iou = safe_div(
            bucket['intersection_pixels'],
            bucket['union_pixels'],
        )
        base_bucket = class_groups.get(
            (dataset, 'baseline', class_idx), {})
        base_iou = safe_div(
            base_bucket.get('intersection_pixels', 0),
            base_bucket.get('union_pixels', 0),
        )
        class_rows.append(dict(
            dataset_name=dataset,
            action_name=action,
            class_index=class_idx,
            class_name=bucket.get('class_name'),
            gt_pixels=bucket['gt_pixels'],
            pred_pixels=bucket['pred_pixels'],
            baseline_pred_pixels=bucket['baseline_pred_pixels'],
            baseline_iou=base_iou,
            action_iou=iou,
            iou_delta=(
                iou - base_iou
                if iou is not None and base_iou is not None
                else None),
            changed_pixels=bucket['changed_pixels'],
            improved_pixels=bucket['improved_pixels'],
            harmed_pixels=bucket['harmed_pixels'],
        ))

    pair_rows = []
    for (
            dataset, action, gt_idx, pred_idx), bucket in sorted(
                pair_groups.items()):
        pair_rows.append(dict(
            dataset_name=dataset,
            action_name=action,
            gt_class_index=gt_idx,
            gt_class_name=bucket.get('gt_class_name'),
            base_pred_class_index=pred_idx,
            base_pred_class_name=bucket.get(
                'base_pred_class_name'),
            pixels=bucket['pixels'],
            changed_pixels=bucket['changed_pixels'],
            corrected_pixels=bucket['corrected_pixels'],
            correction_ratio=safe_div(
                bucket['corrected_pixels'],
                bucket['pixels']),
            correction_precision=safe_div(
                bucket['corrected_pixels'],
                bucket['changed_pixels']),
        ))

    region_rows = []
    for (dataset, source, class_idx), bucket in sorted(
            region_groups.items(), key=lambda item: str(item[0])):
        valid_regions = bucket['valid_regions']
        region_rows.append(dict(
            dataset_name=dataset,
            source_name=source,
            class_index=class_idx,
            class_name=bucket.get('class_name'),
            valid_regions=valid_regions,
            correct_regions=bucket['correct_regions'],
            region_accuracy=safe_div(
                bucket['correct_regions'], valid_regions),
            high_purity_regions=bucket['high_purity_regions'],
            high_purity_correct_regions=(
                bucket['high_purity_correct_regions']),
            high_purity_accuracy=safe_div(
                bucket['high_purity_correct_regions'],
                bucket['high_purity_regions']),
            origin_correct_regions=bucket['origin_correct_regions'],
            origin_accuracy=safe_div(
                bucket['origin_correct_regions'], valid_regions),
            mean_region_purity=safe_div(
                bucket['purity_sum'], valid_regions),
            mean_region_margin=safe_div(
                bucket['margin_sum'], valid_regions),
            area_pixels=bucket['area_pixels'],
        ))

    oracle_rows = []
    for dataset in sorted(baseline):
        base = confusion_metrics(baseline[dataset])
        oracle = confusion_metrics(pixel_oracle[dataset])
        actions = action_metrics.get(dataset, {})
        candidates = [
            (name, metrics)
            for name, metrics in actions.items()
            if name != 'baseline' and metrics['miou'] is not None
        ]
        best_name = 'baseline'
        best_metrics = base
        if candidates:
            candidate_name, candidate_metrics = max(
                candidates, key=lambda item: item[1]['miou'])
            if (
                    base['miou'] is None
                    or candidate_metrics['miou'] > base['miou']):
                best_name = candidate_name
                best_metrics = candidate_metrics
        source_by_name = {
            row['source_name']: row
            for row in source_rows
            if row['dataset_name'] == dataset
        }
        true_row = source_by_name.get('true_zcontrast_top', {})
        control_names = (
            'rect_zcontrast_top',
            'shift_zcontrast_top',
            'classperm_zcontrast_top',
        )
        best_control_top1 = max(
            [
                source_by_name.get(name, {}).get(
                    'wrong_top1_recovers_gt_ratio')
                for name in control_names
                if source_by_name.get(name, {}).get(
                    'wrong_top1_recovers_gt_ratio') is not None
            ] or [0.0]
        )
        true_top1 = true_row.get(
            'wrong_top1_recovers_gt_ratio')
        oracle_rows.append(dict(
            dataset_name=dataset,
            baseline_miou=base['miou'],
            best_fixed_action=best_name,
            best_fixed_miou=best_metrics['miou'],
            best_fixed_delta=(
                best_metrics['miou'] - base['miou']
                if best_metrics['miou'] is not None
                and base['miou'] is not None
                else None),
            pixel_oracle_miou=oracle['miou'],
            pixel_oracle_delta=(
                oracle['miou'] - base['miou']
                if oracle['miou'] is not None
                and base['miou'] is not None
                else None),
            true_region_wrong_top1_recovery=true_top1,
            best_control_wrong_top1_recovery=best_control_top1,
            true_region_recovery_lift_over_control=(
                true_top1 - best_control_top1
                if true_top1 is not None else None),
            true_region_beats_all_controls=(
                true_top1 is not None
                and true_top1 > best_control_top1),
        ))

    selection_datasets = {
        name.strip().lower()
        for name in selection_datasets.split(',')
        if name.strip()
    }
    transfer_groups = defaultdict(list)
    for row in overall_rows:
        if row['action_name'] == 'baseline':
            continue
        if row['miou_delta'] is not None:
            transfer_groups[row['action_name']].append(row)
    transfer_rows = []
    for action_name, rows in sorted(transfer_groups.items()):
        development = [
            row['miou_delta']
            for row in rows
            if row['dataset_name'].lower() in selection_datasets
        ]
        held_out = [
            row['miou_delta']
            for row in rows
            if row['dataset_name'].lower() not in selection_datasets
        ]
        all_deltas = [row['miou_delta'] for row in rows]
        transfer_rows.append(dict(
            action_name=action_name,
            development_datasets=','.join(sorted(selection_datasets)),
            development_count=len(development),
            development_mean_delta=(
                sum(development) / len(development)
                if development else None),
            development_min_delta=(
                min(development) if development else None),
            development_positive_count=sum(
                value > 0 for value in development),
            held_out_count=len(held_out),
            held_out_mean_delta=(
                sum(held_out) / len(held_out)
                if held_out else None),
            held_out_min_delta=(
                min(held_out) if held_out else None),
            held_out_positive_count=sum(
                value > 0 for value in held_out),
            all_count=len(all_deltas),
            all_mean_delta=sum(all_deltas) / len(all_deltas),
            all_min_delta=min(all_deltas),
            all_positive_count=sum(
                value > 0 for value in all_deltas),
        ))

    exact_rows = []
    exact_class_rows = []
    for dataset in sorted(exact_baseline):
        base = confusion_metrics(exact_baseline[dataset])
        disabled = confusion_metrics(
            exact_threshold_disabled[dataset])
        for action_name, metrics, action_images in (
                ('baseline', base, images[dataset]),
                ('threshold_disabled', disabled, images[dataset])):
            exact_rows.append(dict(
                dataset_name=dataset,
                action_name=action_name,
                images=action_images,
                valid_pixels=metrics['total_pixels'],
                baseline_miou=base['miou'],
                action_miou=metrics['miou'],
                miou_delta=(
                    metrics['miou'] - base['miou']
                    if metrics['miou'] is not None
                    and base['miou'] is not None else None),
                baseline_pixel_accuracy=base['pixel_accuracy'],
                action_pixel_accuracy=metrics['pixel_accuracy'],
                pixel_accuracy_delta=(
                    metrics['pixel_accuracy']
                    - base['pixel_accuracy']
                    if metrics['pixel_accuracy'] is not None
                    and base['pixel_accuracy'] is not None else None),
            ))

        for (action_dataset, action_name), bucket in sorted(
                exact_action_groups.items()):
            if action_dataset != dataset:
                continue
            if action_name == 'threshold_disabled':
                continue
            metrics = confusion_metrics(bucket['confusion'])
            exact_rows.append(dict(
                dataset_name=dataset,
                action_name=action_name,
                images=bucket['images'],
                valid_pixels=metrics['total_pixels'],
                baseline_miou=base['miou'],
                action_miou=metrics['miou'],
                miou_delta=(
                    metrics['miou'] - base['miou']
                    if metrics['miou'] is not None
                    and base['miou'] is not None else None),
                baseline_pixel_accuracy=base['pixel_accuracy'],
                action_pixel_accuracy=metrics['pixel_accuracy'],
                pixel_accuracy_delta=(
                    metrics['pixel_accuracy']
                    - base['pixel_accuracy']
                    if metrics['pixel_accuracy'] is not None
                    and base['pixel_accuracy'] is not None else None),
            ))

        matrices = {
            'baseline': exact_baseline[dataset],
            'threshold_disabled': exact_threshold_disabled[dataset],
        }
        for (action_dataset, action_name), bucket in (
                exact_action_groups.items()):
            if action_dataset == dataset:
                matrices[action_name] = bucket['confusion']
        for action_name, matrix in sorted(matrices.items()):
            metrics = confusion_metrics(matrix)
            for class_idx, action_iou in enumerate(metrics['ious']):
                baseline_iou = (
                    base['ious'][class_idx]
                    if class_idx < len(base['ious']) else None)
                exact_class_rows.append(dict(
                    dataset_name=dataset,
                    action_name=action_name,
                    class_index=class_idx,
                    class_name=(
                        class_names_by_dataset.get(dataset, [])[class_idx]
                        if class_idx < len(
                            class_names_by_dataset.get(dataset, []))
                        else None),
                    baseline_iou=baseline_iou,
                    action_iou=action_iou,
                    iou_delta=(
                        action_iou - baseline_iou
                        if action_iou is not None
                        and baseline_iou is not None else None),
                    gt_pixels=sum(matrix[class_idx]),
                    pred_pixels=sum(
                        matrix[row_idx][class_idx]
                        for row_idx in range(len(matrix))),
                ))

    route_rows = []
    for (
            dataset,
            action_name,
            route_name,
            source_name,
            blend), bucket in sorted(route_groups.items()):
        accepted = bucket['accepted_pixels']
        improved = bucket['improved_pixels']
        harmed = bucket['harmed_pixels']
        route_rows.append(dict(
            dataset_name=dataset,
            action_name=action_name,
            route_name=route_name,
            source_name=source_name,
            blend=blend,
            eligible_pixels=int(bucket['eligible_pixels']),
            accepted_pixels=int(accepted),
            accepted_ratio=safe_div(
                accepted, bucket['eligible_pixels']),
            gt_foreground_pixels=int(bucket['gt_foreground_pixels']),
            gt_background_pixels=int(bucket['gt_background_pixels']),
            gt_foreground_ratio=safe_div(
                bucket['gt_foreground_pixels'],
                bucket['eligible_pixels']),
            baseline_wrong_pixels=int(bucket['baseline_wrong_pixels']),
            changed_pixels=int(bucket['changed_pixels']),
            improved_pixels=int(improved),
            harmed_pixels=int(harmed),
            wrong_to_wrong_pixels=int(
                bucket['wrong_to_wrong_pixels']),
            utility_precision=safe_div(
                improved, improved + harmed),
            net_correct_pixels=int(improved - harmed),
        ))

    return {
        'overall': overall_rows,
        'sources': source_rows,
        'classes': class_rows,
        'pairs': pair_rows,
        'regions': region_rows,
        'oracle': oracle_rows,
        'transfer': transfer_rows,
        'exact': exact_rows,
        'exact_classes': exact_class_rows,
        'routes': route_rows,
    }


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')
    tables = summarize(
        iter_records(paths),
        args.selection_datasets,
    )
    outputs = {
        'region_readout_overall.csv': (
            tables['overall'],
            [
                'dataset_name', 'action_name', 'images', 'regions',
                'valid_pixels', 'baseline_miou', 'action_miou',
                'miou_delta', 'baseline_pixel_accuracy',
                'action_pixel_accuracy', 'pixel_accuracy_delta',
                'changed_pixels', 'improved_pixels', 'harmed_pixels',
                'wrong_to_wrong_pixels', 'net_correct_pixels',
                'utility_precision',
            ],
        ),
        'region_readout_source_summary.csv': (
            tables['sources'],
            [
                'dataset_name', 'source_name', 'covered_pixels',
                'coverage_ratio', 'baseline_wrong_covered_pixels',
                'baseline_wrong_coverage',
                'wrong_top1_recovers_gt_pixels',
                'wrong_top1_recovers_gt_ratio',
                'wrong_topk_contains_gt_pixels',
                'wrong_topk_contains_gt_ratio',
                'wrong_gt_beats_base_pixels',
                'wrong_gt_beats_base_ratio',
                'mean_wrong_gt_vs_base_margin',
            ],
        ),
        'region_readout_class_summary.csv': (
            tables['classes'],
            [
                'dataset_name', 'action_name', 'class_index',
                'class_name', 'gt_pixels', 'pred_pixels',
                'baseline_pred_pixels', 'baseline_iou', 'action_iou',
                'iou_delta', 'changed_pixels', 'improved_pixels',
                'harmed_pixels',
            ],
        ),
        'region_readout_pair_summary.csv': (
            tables['pairs'],
            [
                'dataset_name', 'action_name', 'gt_class_index',
                'gt_class_name', 'base_pred_class_index',
                'base_pred_class_name', 'pixels', 'changed_pixels',
                'corrected_pixels', 'correction_ratio',
                'correction_precision',
            ],
        ),
        'region_readout_region_summary.csv': (
            tables['regions'],
            [
                'dataset_name', 'source_name', 'class_index',
                'class_name', 'valid_regions', 'correct_regions',
                'region_accuracy', 'high_purity_regions',
                'high_purity_correct_regions', 'high_purity_accuracy',
                'origin_correct_regions', 'origin_accuracy',
                'mean_region_purity', 'mean_region_margin',
                'area_pixels',
            ],
        ),
        'region_readout_oracle_summary.csv': (
            tables['oracle'],
            [
                'dataset_name', 'baseline_miou', 'best_fixed_action',
                'best_fixed_miou', 'best_fixed_delta',
                'pixel_oracle_miou', 'pixel_oracle_delta',
                'true_region_wrong_top1_recovery',
                'best_control_wrong_top1_recovery',
                'true_region_recovery_lift_over_control',
                'true_region_beats_all_controls',
            ],
        ),
        'region_readout_transfer_summary.csv': (
            tables['transfer'],
            [
                'action_name', 'development_datasets',
                'development_count', 'development_mean_delta',
                'development_min_delta',
                'development_positive_count', 'held_out_count',
                'held_out_mean_delta', 'held_out_min_delta',
                'held_out_positive_count', 'all_count',
                'all_mean_delta', 'all_min_delta',
                'all_positive_count',
            ],
        ),
        'region_readout_exact_summary.csv': (
            tables['exact'],
            [
                'dataset_name', 'action_name', 'images',
                'valid_pixels', 'baseline_miou', 'action_miou',
                'miou_delta', 'baseline_pixel_accuracy',
                'action_pixel_accuracy', 'pixel_accuracy_delta',
            ],
        ),
        'region_readout_exact_class_summary.csv': (
            tables['exact_classes'],
            [
                'dataset_name', 'action_name', 'class_index',
                'class_name', 'baseline_iou', 'action_iou', 'iou_delta',
                'gt_pixels', 'pred_pixels',
            ],
        ),
        'region_readout_route_summary.csv': (
            tables['routes'],
            [
                'dataset_name', 'action_name', 'route_name',
                'source_name', 'blend', 'eligible_pixels',
                'accepted_pixels', 'accepted_ratio',
                'gt_foreground_pixels', 'gt_background_pixels',
                'gt_foreground_ratio', 'baseline_wrong_pixels',
                'changed_pixels', 'improved_pixels', 'harmed_pixels',
                'wrong_to_wrong_pixels', 'utility_precision',
                'net_correct_pixels',
            ],
        ),
    }
    for filename, (rows, fields) in outputs.items():
        write_csv(
            os.path.join(args.out_dir, filename),
            rows,
            fields,
        )
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
