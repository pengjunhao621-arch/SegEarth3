import argparse
import csv
import glob
import json
import os
from collections import defaultdict


SETTING_FIELDS = (
    'dataset_name',
    'ccer_sources',
    'ccer_seed_rule',
    'ccer_min_seed_pixels',
    'ccer_min_separation',
    'ccer_full_reliability_separation',
    'ccer_temperature',
    'ccer_strength',
    'ccer_mode',
    'ccer_apply_mode',
    'ccer_candidate_topk',
    'ccer_min_calibrated_margin',
    'ccer_min_reliable_sources',
    'ccer_max_base_margin',
    'ccer_preserve_reject',
    'ccer_min_local_consistency',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize class-conditional evidence recomposition JSONL '
            'diagnostics.'))
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


def setting_key(record):
    return tuple(record.get(field) for field in SETTING_FIELDS)


def setting_dict(key):
    return dict(zip(SETTING_FIELDS, key))


def add_matrix(target, source):
    if source is None:
        return
    if not target:
        target.extend([
            [0 for _ in row]
            for row in source
        ])
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
        miou=(
            sum(valid_ious) / len(valid_ious)
            if valid_ious else None),
        intersections=intersections,
        unions=unions,
        ious=ious,
    )


def add_weighted(bucket, key, value, weight):
    if value is None or weight <= 0:
        return
    bucket[f'__sum__{key}'] += float(value) * weight
    bucket[f'__den__{key}'] += weight


def summarize(records):
    groups = defaultdict(lambda: dict(
        images=0,
        base_confusion=[],
        method_confusion=[],
        overall=defaultdict(float),
        classes=defaultdict(lambda: defaultdict(float)),
        class_names={},
        pairs=defaultdict(lambda: defaultdict(float)),
        pair_names={},
        calibration=defaultdict(lambda: defaultdict(float)),
        calibration_names={},
    ))

    for record in records:
        stats = record.get(
            'class_conditional_recomposition_stats')
        if not stats:
            continue
        key = setting_key(record)
        group = groups[key]
        group['images'] += 1
        add_matrix(group['base_confusion'], stats.get('base_confusion'))
        add_matrix(
            group['method_confusion'], stats.get('method_confusion'))

        overall = stats.get('overall') or {}
        for field, value in overall.items():
            if value is None:
                continue
            if field.startswith('mean_'):
                add_weighted(
                    group['overall'],
                    field,
                    value,
                    int(overall.get('apply_pixels') or 0),
                )
            elif isinstance(value, (int, float)):
                group['overall'][field] += value

        for row in stats.get('class_stats') or []:
            class_idx = row.get('class_index')
            if class_idx is None:
                continue
            bucket = group['classes'][int(class_idx)]
            group['class_names'][int(class_idx)] = row.get('class_name')
            for field, value in row.items():
                if field in ('class_index', 'class_name') or value is None:
                    continue
                if isinstance(value, (int, float)):
                    bucket[field] += value

        for row in stats.get('pair_stats') or []:
            gt_idx = row.get('gt_class_index')
            pred_idx = row.get('base_pred_class_index')
            if gt_idx is None or pred_idx is None:
                continue
            pair_key = (int(gt_idx), int(pred_idx))
            bucket = group['pairs'][pair_key]
            group['pair_names'][pair_key] = (
                row.get('gt_class_name'),
                row.get('base_pred_class_name'),
            )
            for field, value in row.items():
                if field.endswith('_index') or field.endswith('_name'):
                    continue
                if value is not None and isinstance(value, (int, float)):
                    bucket[field] += value

        for row in stats.get('calibration_rows') or []:
            class_idx = row.get('class_index')
            source_name = row.get('source_name')
            if class_idx is None or source_name is None:
                continue
            calibration_key = (int(class_idx), str(source_name))
            bucket = group['calibration'][calibration_key]
            group['calibration_names'][calibration_key] = row.get(
                'class_name')
            weight = max(1, int(row.get('seed_pixels') or 0))
            bucket['images'] += 1
            bucket['seed_pixels'] += int(row.get('seed_pixels') or 0)
            bucket['reliable_images'] += int(
                float(row.get('reliability') or 0) > 0)
            for field in (
                    'positive_center',
                    'negative_center',
                    'center_gap',
                    'standardized_separation',
                    'reliability'):
                add_weighted(bucket, field, row.get(field), weight)

    overall_rows = []
    class_rows = []
    pair_rows = []
    calibration_rows = []
    for key, group in sorted(groups.items()):
        settings = setting_dict(key)
        base_metrics = confusion_metrics(group['base_confusion'])
        method_metrics = confusion_metrics(group['method_confusion'])
        overall = group['overall']
        row = dict(settings)
        row.update(
            images=group['images'],
            valid_pixels=base_metrics['total_pixels'],
            baseline_miou=base_metrics['miou'],
            method_miou=method_metrics['miou'],
            miou_delta=(
                method_metrics['miou'] - base_metrics['miou']),
            baseline_pixel_accuracy=base_metrics['pixel_accuracy'],
            method_pixel_accuracy=method_metrics['pixel_accuracy'],
            pixel_accuracy_delta=(
                method_metrics['pixel_accuracy']
                - base_metrics['pixel_accuracy']),
        )
        for field, value in overall.items():
            if not field.startswith('__'):
                row[field] = value
        for field, value in overall.items():
            if field.startswith('__sum__'):
                mean_field = field[len('__sum__'):]
                row[mean_field] = safe_div(
                    value, overall.get(f'__den__{mean_field}', 0))
        row['seed_ratio'] = safe_div(
            row.get('seed_pixels', 0), row['valid_pixels'])
        row['proposal_change_ratio'] = safe_div(
            row.get('proposal_change_pixels', 0), row['valid_pixels'])
        row['apply_ratio'] = safe_div(
            row.get('apply_pixels', 0), row['valid_pixels'])
        row['changed_ratio'] = safe_div(
            row.get('changed_pixels', 0), row['valid_pixels'])
        row['utility_precision'] = safe_div(
            row.get('improved_pixels', 0),
            row.get('improved_pixels', 0)
            + row.get('harmed_pixels', 0),
        )
        overall_rows.append(row)

        for class_idx, bucket in sorted(group['classes'].items()):
            class_row = dict(settings)
            class_row.update(
                class_index=class_idx,
                class_name=group['class_names'].get(class_idx),
                **{
                    field: value
                    for field, value in bucket.items()
                },
            )
            base_iou = base_metrics['ious'][class_idx]
            method_iou = method_metrics['ious'][class_idx]
            class_row.update(
                baseline_iou=base_iou,
                method_iou=method_iou,
                iou_delta=(
                    method_iou - base_iou
                    if base_iou is not None
                    and method_iou is not None else None),
                baseline_recall=safe_div(
                    bucket.get('base_correct_pixels', 0),
                    bucket.get('pixels', 0)),
                method_recall=safe_div(
                    bucket.get('method_correct_pixels', 0),
                    bucket.get('pixels', 0)),
            )
            class_rows.append(class_row)

        for pair_key, bucket in group['pairs'].items():
            pair_row = dict(settings)
            gt_name, pred_name = group['pair_names'][pair_key]
            pair_row.update(
                gt_class_index=pair_key[0],
                gt_class_name=gt_name,
                base_pred_class_index=pair_key[1],
                base_pred_class_name=pred_name,
                **{
                    field: value
                    for field, value in bucket.items()
                },
            )
            pair_row['changed_ratio'] = safe_div(
                pair_row.get('changed_pixels', 0),
                pair_row.get('pixels', 0))
            pair_row['corrected_ratio'] = safe_div(
                pair_row.get('corrected_pixels', 0),
                pair_row.get('pixels', 0))
            pair_rows.append(pair_row)

        for calibration_key, bucket in group['calibration'].items():
            calibration_row = dict(settings)
            calibration_row.update(
                class_index=calibration_key[0],
                class_name=group['calibration_names'].get(
                    calibration_key),
                source_name=calibration_key[1],
                images=int(bucket.get('images', 0)),
                reliable_images=int(bucket.get('reliable_images', 0)),
                seed_pixels=int(bucket.get('seed_pixels', 0)),
            )
            calibration_row['reliable_image_ratio'] = safe_div(
                calibration_row['reliable_images'],
                calibration_row['images'])
            for field, value in bucket.items():
                if not field.startswith('__sum__'):
                    continue
                mean_field = field[len('__sum__'):]
                calibration_row[f'mean_{mean_field}'] = safe_div(
                    value, bucket.get(f'__den__{mean_field}', 0))
            calibration_rows.append(calibration_row)

    pair_rows.sort(
        key=lambda row: (
            row.get('dataset_name') or '',
            -(row.get('pixels') or 0),
        ))
    calibration_rows.sort(
        key=lambda row: (
            row.get('dataset_name') or '',
            row.get('class_index') or 0,
            row.get('source_name') or '',
        ))
    return (
        overall_rows,
        class_rows,
        pair_rows,
        calibration_rows,
    )


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for field in SETTING_FIELDS:
        if any(field in row for row in rows):
            fieldnames.append(field)
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(
            f'No JSONL inputs matched: {args.inputs}')
    records = list(iter_records(paths))
    (
        overall_rows,
        class_rows,
        pair_rows,
        calibration_rows,
    ) = summarize(records)
    write_csv(
        os.path.join(args.out_dir, 'ccer_overall_summary.csv'),
        overall_rows)
    write_csv(
        os.path.join(args.out_dir, 'ccer_class_summary.csv'),
        class_rows)
    write_csv(
        os.path.join(args.out_dir, 'ccer_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(args.out_dir, 'ccer_calibration_summary.csv'),
        calibration_rows)
    print(f'Read {len(records)} records from {len(paths)} files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
