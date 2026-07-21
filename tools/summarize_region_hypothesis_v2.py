import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize prediction-preserving SAM3 region-hypothesis '
            'diagnostics.'))
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--selection-datasets',
        default='udd5,vdd,vaihingen',
        help='Datasets used for shared-action and leave-one-dataset-out tests.')
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.isfile(pattern):
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
            gt_pixels=[],
            pred_pixels=[],
            intersections=[],
            unions=[],
        )
    class_count = len(matrix)
    intersections = [matrix[idx][idx] for idx in range(class_count)]
    gt_pixels = [sum(matrix[idx]) for idx in range(class_count)]
    pred_pixels = [
        sum(matrix[row][idx] for row in range(class_count))
        for idx in range(class_count)
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
        gt_pixels=gt_pixels,
        pred_pixels=pred_pixels,
        intersections=intersections,
        unions=unions,
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


def sum_fields(bucket, row, fields):
    for field in fields:
        bucket[field] += row.get(field, 0) or 0


def collect(records):
    action_groups = defaultdict(lambda: dict(
        confusion=[],
        images=0,
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
    ))
    formula_groups = defaultdict(lambda: defaultdict(float))
    selector_groups = defaultdict(lambda: defaultdict(float))
    quality_groups = defaultdict(lambda: defaultdict(float))
    class_names = {}
    dataset_images = defaultdict(int)
    dataset_regions = defaultdict(int)
    overlap_sum = defaultdict(float)
    overlap_count = defaultdict(int)

    for record in records:
        stats = record.get('region_hypothesis_v2_stats')
        if not stats:
            continue
        dataset = str(
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown').lower()
        if stats.get('class_names'):
            class_names[dataset] = list(stats['class_names'])
        dataset_images[dataset] += 1
        dataset_regions[dataset] += int(stats.get('region_count', 0))
        overlap = stats.get('overlap_pixel_ratio')
        if overlap is not None:
            overlap_sum[dataset] += float(overlap)
            overlap_count[dataset] += 1

        for row in stats.get('action_rows') or []:
            key = (dataset, str(row['action_name']))
            bucket = action_groups[key]
            add_matrix(bucket['confusion'], row.get('confusion'))
            bucket['images'] += 1
            sum_fields(
                bucket,
                row,
                (
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels',
                    'wrong_to_wrong_pixels',
                ),
            )

        for row in stats.get('formula_rows') or []:
            key = (
                dataset,
                str(row['formula_name']),
                str(row['gt_role']),
                str(row['area_bin']),
                str(row['shape_bin']),
            )
            bucket = formula_groups[key]
            sum_fields(
                bucket,
                row,
                (
                    'regions',
                    'correct_regions',
                    'high_purity_regions',
                    'high_purity_correct_regions',
                    'origin_correct_regions',
                    'margin_sum',
                    'entropy_sum',
                    'purity_sum',
                ),
            )

        for row in stats.get('selector_rows') or []:
            key = (
                dataset,
                str(row['selector_name']),
                str(row['threshold']),
            )
            bucket = selector_groups[key]
            sum_fields(
                bucket,
                row,
                (
                    'regions',
                    'correct_regions',
                    'formula_oracle_match_regions',
                    'useful_regions',
                    'high_purity_regions',
                ),
            )
            for name, count in (
                    row.get('selected_formula_counts') or {}).items():
                bucket[f'formula_count__{name}'] += int(count)

        for row in stats.get('quality_bin_rows') or []:
            key = (
                dataset,
                str(row['selector_name']),
                float(row['quality_lower']),
                float(row['quality_upper']),
            )
            bucket = quality_groups[key]
            for field, value in row.items():
                if field in (
                        'selector_name',
                        'quality_lower',
                        'quality_upper'):
                    continue
                bucket[field] += value or 0

    return dict(
        action_groups=action_groups,
        formula_groups=formula_groups,
        selector_groups=selector_groups,
        quality_groups=quality_groups,
        class_names=class_names,
        dataset_images=dataset_images,
        dataset_regions=dataset_regions,
        overlap_sum=overlap_sum,
        overlap_count=overlap_count,
    )


def build_action_rows(collected):
    rows = []
    class_rows = []
    metrics_by_key = {}
    action_groups = collected['action_groups']
    class_names = collected['class_names']
    for (dataset, action), bucket in sorted(action_groups.items()):
        metrics = confusion_metrics(bucket['confusion'])
        metrics_by_key[(dataset, action)] = metrics
        baseline = confusion_metrics(
            action_groups[(dataset, 'baseline')]['confusion'])
        changed = int(bucket['changed_pixels'])
        improved = int(bucket['improved_pixels'])
        harmed = int(bucket['harmed_pixels'])
        rows.append(dict(
            dataset=dataset,
            action_name=action,
            action_family=action.split('__', 1)[0],
            images=int(bucket['images']),
            regions=int(collected['dataset_regions'][dataset]),
            mean_overlap_pixel_ratio=safe_div(
                collected['overlap_sum'][dataset],
                collected['overlap_count'][dataset]),
            total_pixels=metrics['total_pixels'],
            pixel_accuracy=metrics['pixel_accuracy'],
            miou=metrics['miou'],
            miou_points=(
                metrics['miou'] * 100.0
                if metrics['miou'] is not None else None),
            baseline_miou_points=(
                baseline['miou'] * 100.0
                if baseline['miou'] is not None else None),
            delta_miou_points=(
                (metrics['miou'] - baseline['miou']) * 100.0
                if metrics['miou'] is not None
                and baseline['miou'] is not None else None),
            changed_pixels=changed,
            changed_ratio=safe_div(changed, metrics['total_pixels']),
            improved_pixels=improved,
            harmed_pixels=harmed,
            wrong_to_wrong_pixels=int(
                bucket['wrong_to_wrong_pixels']),
            net_correct_pixels=improved - harmed,
            correction_precision=safe_div(
                improved, improved + harmed),
            harmful_change_rate=safe_div(harmed, changed),
        ))
        names = class_names.get(dataset, [])
        for class_idx, iou in enumerate(metrics['ious']):
            base_iou = (
                baseline['ious'][class_idx]
                if class_idx < len(baseline['ious']) else None)
            class_rows.append(dict(
                dataset=dataset,
                action_name=action,
                class_index=class_idx,
                class_name=(
                    names[class_idx]
                    if class_idx < len(names) else str(class_idx)),
                gt_pixels=metrics['gt_pixels'][class_idx],
                pred_pixels=metrics['pred_pixels'][class_idx],
                intersection_pixels=metrics['intersections'][class_idx],
                union_pixels=metrics['unions'][class_idx],
                iou=iou,
                iou_points=iou * 100.0 if iou is not None else None,
                baseline_iou_points=(
                    base_iou * 100.0
                    if base_iou is not None else None),
                delta_iou_points=(
                    (iou - base_iou) * 100.0
                    if iou is not None and base_iou is not None
                    else None),
            ))
    return rows, class_rows, metrics_by_key


def build_formula_rows(groups):
    rows = []
    for key, bucket in sorted(groups.items()):
        dataset, formula, role, area_bin, shape_bin = key
        regions = int(bucket['regions'])
        high = int(bucket['high_purity_regions'])
        rows.append(dict(
            dataset=dataset,
            formula_name=formula,
            gt_role=role,
            area_bin=area_bin,
            shape_bin=shape_bin,
            regions=regions,
            region_accuracy=safe_div(
                bucket['correct_regions'], regions),
            high_purity_regions=high,
            high_purity_accuracy=safe_div(
                bucket['high_purity_correct_regions'], high),
            origin_prompt_accuracy=safe_div(
                bucket['origin_correct_regions'], regions),
            mean_margin=safe_div(bucket['margin_sum'], regions),
            mean_entropy=safe_div(bucket['entropy_sum'], regions),
            mean_gt_purity=safe_div(bucket['purity_sum'], regions),
        ))
    return rows


def build_selector_rows(groups):
    rows = []
    for key, bucket in sorted(groups.items()):
        dataset, selector, threshold = key
        regions = int(bucket['regions'])
        row = dict(
            dataset=dataset,
            selector_name=selector,
            threshold=threshold,
            regions=regions,
            region_accuracy=safe_div(
                bucket['correct_regions'], regions),
            formula_oracle_match_rate=safe_div(
                bucket['formula_oracle_match_regions'], regions),
            useful_region_rate=safe_div(
                bucket['useful_regions'], regions),
            high_purity_region_rate=safe_div(
                bucket['high_purity_regions'], regions),
        )
        formula_counts = {
            field.replace('formula_count__', '', 1): int(value)
            for field, value in bucket.items()
            if field.startswith('formula_count__')
        }
        row['selected_formula_counts'] = json.dumps(
            formula_counts, sort_keys=True)
        rows.append(row)
    return rows


def build_quality_rows(groups):
    rows = []
    for key, bucket in sorted(groups.items()):
        dataset, selector, lower, upper = key
        regions = int(bucket['regions'])
        row = dict(
            dataset=dataset,
            selector_name=selector,
            quality_lower=lower,
            quality_upper=upper,
            regions=regions,
            region_accuracy=safe_div(
                bucket['correct_regions'], regions),
            useful_region_rate=safe_div(
                bucket['useful_regions'], regions),
        )
        for field, value in bucket.items():
            if field.endswith('_sum'):
                row[field[:-4] + '_mean'] = safe_div(value, regions)
        rows.append(row)
    return rows


def build_oracle_rows(action_rows):
    by_dataset = defaultdict(dict)
    for row in action_rows:
        by_dataset[row['dataset']][row['action_name']] = row
    rows = []
    projection_rows = []
    for dataset, actions in sorted(by_dataset.items()):
        baseline = actions.get('baseline', {})
        selectors = [
            row for name, row in actions.items()
            if name.startswith('selector__')
        ]
        fixed = [
            row for name, row in actions.items()
            if name.startswith('formula__')
        ]
        projections = [
            row for name, row in actions.items()
            if name.startswith('projection__')
        ]
        best_selector = max(
            selectors,
            key=lambda row: row['delta_miou_points'],
            default=None,
        )
        best_fixed = max(
            fixed,
            key=lambda row: row['delta_miou_points'],
            default=None,
        )
        best_projection = max(
            projections,
            key=lambda row: row['delta_miou_points'],
            default=None,
        )
        rows.append(dict(
            dataset=dataset,
            baseline_miou_points=baseline.get('miou_points'),
            formula_oracle_delta=actions.get(
                'oracle__formula', {}).get('delta_miou_points'),
            region_oracle_delta=actions.get(
                'oracle__region', {}).get('delta_miou_points'),
            projection_oracle_delta=actions.get(
                'oracle__projection', {}).get('delta_miou_points'),
            best_fixed_formula=(
                best_fixed['action_name'] if best_fixed else None),
            best_fixed_formula_delta=(
                best_fixed['delta_miou_points']
                if best_fixed else None),
            best_projection=(
                best_projection['action_name']
                if best_projection else None),
            best_projection_delta=(
                best_projection['delta_miou_points']
                if best_projection else None),
            best_no_gt_selector=(
                best_selector['action_name']
                if best_selector else None),
            best_no_gt_selector_delta=(
                best_selector['delta_miou_points']
                if best_selector else None),
            best_no_gt_correction_precision=(
                best_selector['correction_precision']
                if best_selector else None),
            best_no_gt_harmful_change_rate=(
                best_selector['harmful_change_rate']
                if best_selector else None),
        ))
        projection_rows.extend(projections)
    return rows, projection_rows


def build_shared_and_lodo(action_rows, selection_datasets):
    selected = {
        name.strip().lower()
        for name in selection_datasets.split(',')
        if name.strip()
    }
    deltas = defaultdict(dict)
    for row in action_rows:
        if (
                row['dataset'] in selected
                and row['action_name'].startswith('selector__')):
            deltas[row['action_name']][row['dataset']] = (
                row['delta_miou_points'])
    eligible = {
        action: values
        for action, values in deltas.items()
        if selected.issubset(values)
    }
    shared_rows = []
    for action, values in sorted(eligible.items()):
        dataset_values = [values[name] for name in sorted(selected)]
        shared_rows.append(dict(
            action_name=action,
            datasets=','.join(sorted(selected)),
            mean_delta_miou_points=(
                sum(dataset_values) / len(dataset_values)),
            minimum_delta_miou_points=min(dataset_values),
            positive_dataset_count=sum(
                value > 0 for value in dataset_values),
            nonnegative_dataset_count=sum(
                value >= 0 for value in dataset_values),
            per_dataset_delta=json.dumps(
                values, sort_keys=True),
        ))
    shared_rows.sort(
        key=lambda row: (
            row['minimum_delta_miou_points'],
            row['mean_delta_miou_points'],
        ),
        reverse=True,
    )

    lodo_rows = []
    for heldout in sorted(selected):
        train = sorted(selected - {heldout})
        candidates = []
        for action, values in eligible.items():
            if not all(name in values for name in train + [heldout]):
                continue
            train_values = [values[name] for name in train]
            candidates.append((
                sum(train_values) / len(train_values),
                min(train_values),
                action,
            ))
        if not candidates:
            continue
        _, _, chosen = max(candidates)
        lodo_rows.append(dict(
            heldout_dataset=heldout,
            training_datasets=','.join(train),
            selected_action=chosen,
            training_mean_delta_miou_points=safe_div(
                sum(eligible[chosen][name] for name in train),
                len(train)),
            training_min_delta_miou_points=min(
                eligible[chosen][name] for name in train),
            heldout_delta_miou_points=eligible[chosen][heldout],
        ))
    return shared_rows, lodo_rows


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit('No JSONL inputs matched.')
    collected = collect(iter_records(paths))
    action_rows, class_rows, _ = build_action_rows(collected)
    formula_rows = build_formula_rows(collected['formula_groups'])
    selector_rows = build_selector_rows(collected['selector_groups'])
    quality_rows = build_quality_rows(collected['quality_groups'])
    oracle_rows, projection_rows = build_oracle_rows(action_rows)
    shared_rows, lodo_rows = build_shared_and_lodo(
        action_rows, args.selection_datasets)

    outputs = {
        'rh2_exact_summary.csv': (
            action_rows,
            ('dataset', 'action_name', 'miou_points',
             'delta_miou_points')),
        'rh2_exact_class_summary.csv': (
            class_rows,
            ('dataset', 'action_name', 'class_index', 'class_name',
             'iou_points', 'delta_iou_points')),
        'rh2_formula_summary.csv': (
            formula_rows,
            ('dataset', 'formula_name', 'gt_role', 'area_bin',
             'shape_bin', 'regions', 'region_accuracy')),
        'rh2_selector_region_summary.csv': (
            selector_rows,
            ('dataset', 'selector_name', 'threshold', 'regions',
             'region_accuracy', 'useful_region_rate')),
        'rh2_quality_bin_summary.csv': (
            quality_rows,
            ('dataset', 'selector_name', 'quality_lower',
             'quality_upper', 'regions', 'region_accuracy',
             'useful_region_rate')),
        'rh2_oracle_summary.csv': (
            oracle_rows,
            ('dataset', 'baseline_miou_points',
             'formula_oracle_delta', 'region_oracle_delta',
             'projection_oracle_delta', 'best_no_gt_selector_delta')),
        'rh2_projection_summary.csv': (
            projection_rows,
            ('dataset', 'action_name', 'miou_points',
             'delta_miou_points', 'correction_precision',
             'harmful_change_rate')),
        'rh2_shared_selector_summary.csv': (
            shared_rows,
            ('action_name', 'datasets', 'mean_delta_miou_points',
             'minimum_delta_miou_points', 'positive_dataset_count')),
        'rh2_lodo_summary.csv': (
            lodo_rows,
            ('heldout_dataset', 'training_datasets',
             'selected_action', 'training_mean_delta_miou_points',
             'heldout_delta_miou_points')),
    }
    for filename, (rows, fields) in outputs.items():
        write_csv(os.path.join(args.out_dir, filename), rows, fields)
    print(f'Loaded {len(paths)} files.')
    print(f'Wrote {len(outputs)} summaries to {args.out_dir}.')


if __name__ == '__main__':
    main()
