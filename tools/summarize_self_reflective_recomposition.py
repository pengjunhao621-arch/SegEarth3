import argparse
import csv
import glob
import json
import os
from collections import defaultdict


SETTING_FIELDS = (
    'dataset_name',
    'self_reflection_sources',
    'self_reflection_max_side',
    'self_reflection_temperature',
    'self_reflection_rank_weight',
    'self_reflection_local_kernel',
    'self_reflection_min_source_sharpness',
    'self_reflection_source_topk',
    'self_reflection_candidate_topk',
    'self_reflection_min_candidate_sources',
    'self_reflection_min_core_sources',
    'self_reflection_min_aux_sources',
    'self_reflection_min_candidate_persistence',
    'self_reflection_max_base_persistence',
    'self_reflection_min_consensus_advantage',
    'self_reflection_min_region_consistency',
    'self_reflection_blend',
    'self_reflection_allow_background_switch',
    'self_reflection_recover_reject',
    'self_reflection_reject_min_persistence',
    'self_reflection_reject_min_sources',
    'self_reflection_reject_min_local_consistency',
    'self_reflection_reject_score_margin',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SR3 prediction-changing JSONL results.')
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
        miou=(sum(valid_ious) / len(valid_ious) if valid_ious else None),
        ious=ious,
    )


def settings_key(record):
    return tuple(record.get(field) for field in SETTING_FIELDS)


def settings_dict(key):
    return dict(zip(SETTING_FIELDS, key))


def summarize(records):
    groups = defaultdict(lambda: dict(
        images=0,
        base_confusion=[],
        method_confusion=[],
        overall=defaultdict(float),
        routes=defaultdict(lambda: defaultdict(float)),
        classes=defaultdict(lambda: defaultdict(float)),
        class_names={},
        pairs=defaultdict(lambda: defaultdict(float)),
        pair_names={},
        sources=defaultdict(lambda: defaultdict(float)),
    ))

    for record in records:
        stats = record.get('self_reflective_recomposition_stats')
        if not stats:
            continue
        key = settings_key(record)
        group = groups[key]
        group['images'] += 1
        add_matrix(group['base_confusion'], stats.get('base_confusion'))
        add_matrix(group['method_confusion'], stats.get('method_confusion'))

        for field, value in (stats.get('overall') or {}).items():
            if isinstance(value, (int, float)):
                group['overall'][field] += value

        for row in stats.get('route_stats') or []:
            route = str(row.get('route_name'))
            for field, value in row.items():
                if field != 'route_name' and isinstance(value, (int, float)):
                    group['routes'][route][field] += value

        for row in stats.get('class_stats') or []:
            class_idx = int(row['class_index'])
            group['class_names'][class_idx] = row.get('class_name')
            for field, value in row.items():
                if field not in ('class_index', 'class_name'):
                    if isinstance(value, (int, float)):
                        group['classes'][class_idx][field] += value

        for row in stats.get('pair_stats') or []:
            pair = (
                int(row['gt_class_index']),
                int(row['base_pred_class_index']),
            )
            group['pair_names'][pair] = (
                row.get('gt_class_name'),
                row.get('base_pred_class_name'),
            )
            for field, value in row.items():
                if not field.endswith('_index') and not field.endswith('_name'):
                    if isinstance(value, (int, float)):
                        group['pairs'][pair][field] += value

        for row in stats.get('source_rows') or []:
            source = str(row.get('source_name'))
            bucket = group['sources'][source]
            bucket['images'] += 1
            for field in (
                    'available_classes',
                    'mean_weight',
                    'mean_sharpness',
                    'mean_local_consistency'):
                value = row.get(field)
                if isinstance(value, (int, float)):
                    bucket[field] += value

    overall_rows = []
    route_rows = []
    class_rows = []
    pair_rows = []
    source_rows = []
    for key, group in sorted(groups.items()):
        settings = settings_dict(key)
        base = confusion_metrics(group['base_confusion'])
        method = confusion_metrics(group['method_confusion'])
        overall = dict(group['overall'])
        row = dict(settings)
        row.update(
            images=group['images'],
            valid_pixels=base['total_pixels'],
            baseline_miou=base['miou'],
            method_miou=method['miou'],
            miou_delta=method['miou'] - base['miou'],
            baseline_pixel_accuracy=base['pixel_accuracy'],
            method_pixel_accuracy=method['pixel_accuracy'],
            pixel_accuracy_delta=(
                method['pixel_accuracy'] - base['pixel_accuracy']),
            **overall,
        )
        row['mean_source_count'] = safe_div(
            row.pop('source_count', 0), group['images'])
        row['apply_ratio'] = safe_div(
            row.get('apply_pixels', 0), row['valid_pixels'])
        row['changed_ratio'] = safe_div(
            row.get('changed_pixels', 0), row['valid_pixels'])
        row['utility_precision'] = safe_div(
            row.get('improved_pixels', 0),
            row.get('improved_pixels', 0) + row.get('harmed_pixels', 0),
        )
        overall_rows.append(row)

        for route_name, bucket in group['routes'].items():
            route = dict(settings)
            route.update(route_name=route_name, **dict(bucket))
            route['apply_ratio'] = safe_div(
                route.get('apply_pixels', 0), base['total_pixels'])
            route['utility_precision'] = safe_div(
                route.get('improved_pixels', 0),
                route.get('improved_pixels', 0)
                + route.get('harmed_pixels', 0),
            )
            route_rows.append(route)

        for class_idx, bucket in group['classes'].items():
            class_row = dict(settings)
            class_row.update(
                class_index=class_idx,
                class_name=group['class_names'].get(class_idx),
                **dict(bucket),
            )
            class_row['baseline_iou'] = base['ious'][class_idx]
            class_row['method_iou'] = method['ious'][class_idx]
            class_row['iou_delta'] = (
                method['ious'][class_idx] - base['ious'][class_idx])
            class_row['utility_precision'] = safe_div(
                class_row.get('improved_pixels', 0),
                class_row.get('improved_pixels', 0)
                + class_row.get('harmed_pixels', 0),
            )
            class_rows.append(class_row)

        for pair, bucket in group['pairs'].items():
            gt_name, pred_name = group['pair_names'][pair]
            pair_row = dict(settings)
            pair_row.update(
                gt_class_index=pair[0],
                gt_class_name=gt_name,
                base_pred_class_index=pair[1],
                base_pred_class_name=pred_name,
                **dict(bucket),
            )
            pair_row['changed_ratio'] = safe_div(
                pair_row.get('changed_pixels', 0),
                pair_row.get('pixels', 0),
            )
            pair_row['corrected_ratio'] = safe_div(
                pair_row.get('corrected_pixels', 0),
                pair_row.get('pixels', 0),
            )
            pair_rows.append(pair_row)

        for source_name, bucket in group['sources'].items():
            source_row = dict(settings)
            source_row.update(
                source_name=source_name,
                images=int(bucket.get('images', 0)),
            )
            images = max(1, source_row['images'])
            for field in (
                    'available_classes',
                    'mean_weight',
                    'mean_sharpness',
                    'mean_local_consistency'):
                source_row[field] = bucket.get(field, 0.0) / images
            source_rows.append(source_row)

    pair_rows.sort(key=lambda row: (
        row.get('dataset_name') or '',
        -(row.get('pixels') or 0),
    ))
    return overall_rows, route_rows, class_rows, pair_rows, source_rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for field in SETTING_FIELDS:
        if any(field in row for row in rows):
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(
            f'No JSONL inputs matched: {args.inputs}')
    rows = summarize(iter_records(paths))
    names = (
        'sr3_overall_summary.csv',
        'sr3_route_summary.csv',
        'sr3_class_summary.csv',
        'sr3_pair_summary.csv',
        'sr3_source_summary.csv',
    )
    for name, table in zip(names, rows):
        write_csv(os.path.join(args.out_dir, name), table)
    print(f'Read {len(paths)} JSONL files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
