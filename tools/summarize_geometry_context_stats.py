import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize GeoER geometry/context diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/geoer_geom/vdd/geom_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for geometry_pair_summary.csv, '
             'geometry_class_summary.csv, and geometry_role_summary.csv.')
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
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_div(num, den):
    return None if den == 0 else num / den


def first(values):
    for value in values:
        if value not in (None, ''):
            return value
    return None


def class_tokens(name):
    tokens = set()
    raw = (name or '').lower().replace('/', ',').replace('_', ',')
    for item in raw.split(','):
        item = item.strip()
        if item:
            tokens.add(item)
    if name:
        tokens.add(name.strip().lower())
    return tokens


def infer_role(name):
    tokens = class_tokens(name)
    if tokens & {'background', 'other', 'clutter'}:
        return 'reject_background'
    if tokens & {'vehicle', 'car'}:
        return 'object_vehicle'
    if tokens & {'building', 'house'}:
        return 'object_building'
    if tokens & {'roof'}:
        return 'built_horizontal_surface'
    if tokens & {'facade', 'wall'}:
        return 'built_vertical_surface'
    if tokens & {'road', 'pavement', 'impervious', 'impervious surface'}:
        return 'transport_surface'
    if tokens & {'grass', 'low vegetation', 'low_vegetation', 'agricultural', 'agriculture'}:
        return 'low_vegetation_material'
    if tokens & {'tree', 'forest'}:
        return 'high_vegetation_canopy'
    if tokens & {'vegetation'}:
        return 'vegetation_generic'
    if tokens & {'water'}:
        return 'water_surface'
    if tokens & {'bareland', 'barren'}:
        return 'bare_surface'
    return 'unknown'


def focus_pair_name(dataset_name, gt_name, pred_name):
    dataset = (dataset_name or '').lower()
    gt = class_tokens(gt_name)
    pred = class_tokens(pred_name)
    if dataset == 'vdd' and 'roof' in gt and 'facade' in pred:
        return 'VDD roof -> facade'
    if dataset == 'potsdam' and 'tree' in gt and 'grass' in pred:
        return 'Potsdam tree -> grass'
    if dataset == 'vaihingen' and 'grass' in gt and 'tree' in pred:
        return 'Vaihingen grass -> tree'
    if dataset == 'openearthmap' and 'pavement' in gt and 'building' in pred:
        return 'OpenEarthMap pavement -> building'
    if (dataset == 'loveda'
            and bool({'forest', 'agricultural', 'agriculture'} & gt)
            and 'background' in pred):
        return f'LoveDA {gt_name} -> background'
    if dataset == 'udd5' and {'road', 'vegetation', 'building'} & gt and 'background' in pred:
        return f'UDD5 {gt_name} -> background'
    return None


def add_sum(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value


def add_mean(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_count'] += 1


PAIR_COUNT_KEYS = [
    'pixels',
    'gt_support_on_pair_pixels',
    'pred_support_on_pair_pixels',
    'gt_seed_on_pair_pixels',
    'pred_seed_on_pair_pixels',
    'gt_score_on_pair_sum',
    'pred_score_on_pair_sum',
    'gt_semantic_on_pair_sum',
    'pred_semantic_on_pair_sum',
    'gt_instance_on_pair_sum',
    'pred_instance_on_pair_sum',
]

GEOM_MEAN_KEYS = [
    'mask_ratio_gt_minus_pred',
    'bbox_fill_ratio_gt_minus_pred',
    'core_ratio_gt_minus_pred',
    'boundary_ratio_gt_minus_pred',
    'component_density_gt_minus_pred',
    'mean_component_area_gt_minus_pred',
    'max_component_area_gt_minus_pred',
    'score_mean_gt_minus_pred',
    'semantic_mean_gt_minus_pred',
    'instance_mean_gt_minus_pred',
    'gt_support_mask_ratio',
    'pred_support_mask_ratio',
    'gt_support_bbox_fill_ratio',
    'pred_support_bbox_fill_ratio',
    'gt_support_core_ratio',
    'pred_support_core_ratio',
    'gt_support_boundary_ratio',
    'pred_support_boundary_ratio',
    'gt_support_component_density',
    'pred_support_component_density',
    'gt_support_mean_component_area',
    'pred_support_mean_component_area',
    'gt_support_max_component_area',
    'pred_support_max_component_area',
    'gt_seed_mask_ratio',
    'pred_seed_mask_ratio',
    'gt_seed_core_ratio',
    'pred_seed_core_ratio',
]


CLASS_SUM_KEYS = [
    'gt_pixels',
    'support_pixels',
    'pred_pixels',
    'seed_pixels',
]

CLASS_MEAN_KEYS = [
    'support_precision',
    'support_recall',
    'pred_precision',
    'pred_recall',
    'seed_purity',
    'seed_recall',
    'support_mask_ratio',
    'support_bbox_fill_ratio',
    'support_core_ratio',
    'support_boundary_ratio',
    'support_component_density',
    'support_mean_component_area',
    'support_max_component_area',
    'pred_mask_ratio',
    'pred_bbox_fill_ratio',
    'pred_core_ratio',
    'pred_boundary_ratio',
    'seed_mask_ratio',
    'seed_bbox_fill_ratio',
    'seed_core_ratio',
    'seed_boundary_ratio',
]


def add_pair_row(bucket, row):
    bucket['rows'] += 1
    add_sum(bucket, row, PAIR_COUNT_KEYS)
    add_mean(bucket, row, GEOM_MEAN_KEYS)


def add_class_row(bucket, row):
    bucket['rows'] += 1
    add_sum(bucket, row, CLASS_SUM_KEYS)
    add_mean(bucket, row, CLASS_MEAN_KEYS)


def finalize_means(row):
    keys = [key[:-4] for key in list(row.keys()) if key.endswith('_sum')]
    for key in keys:
        count_key = f'{key}_count'
        sum_key = f'{key}_sum'
        if count_key in row:
            row[key] = safe_div(row.get(sum_key, 0.0), row.get(count_key, 0))
    return row


def finalize_pair(base, counts):
    row = finalize_means(dict(base, **counts))
    pixels = row.get('pixels', 0)
    row['gt_support_on_pair_ratio'] = safe_div(row.get('gt_support_on_pair_pixels', 0), pixels)
    row['pred_support_on_pair_ratio'] = safe_div(row.get('pred_support_on_pair_pixels', 0), pixels)
    row['gt_seed_on_pair_ratio'] = safe_div(row.get('gt_seed_on_pair_pixels', 0), pixels)
    row['pred_seed_on_pair_ratio'] = safe_div(row.get('pred_seed_on_pair_pixels', 0), pixels)
    row['mean_gt_score_on_pair'] = safe_div(row.get('gt_score_on_pair_sum', 0.0), pixels)
    row['mean_pred_score_on_pair'] = safe_div(row.get('pred_score_on_pair_sum', 0.0), pixels)
    row['mean_score_margin_on_pair'] = (
        None if pixels == 0 else
        (row.get('gt_score_on_pair_sum', 0.0) - row.get('pred_score_on_pair_sum', 0.0)) / pixels)
    row['mean_gt_semantic_on_pair'] = safe_div(row.get('gt_semantic_on_pair_sum', 0.0), pixels)
    row['mean_pred_semantic_on_pair'] = safe_div(row.get('pred_semantic_on_pair_sum', 0.0), pixels)
    row['mean_gt_instance_on_pair'] = safe_div(row.get('gt_instance_on_pair_sum', 0.0), pixels)
    row['mean_pred_instance_on_pair'] = safe_div(row.get('pred_instance_on_pair_sum', 0.0), pixels)
    focus = focus_pair_name(
        row.get('dataset_name'),
        row.get('gt_class_name'),
        row.get('base_pred_class_name'))
    row['focus_pair'] = bool(focus)
    row['focus_pair_name'] = focus
    return row


def finalize_class(base, counts):
    return finalize_means(dict(base, **counts))


def summarize(records):
    pair_counts = defaultdict(lambda: defaultdict(float))
    pair_meta = {}
    class_counts = defaultdict(lambda: defaultdict(float))
    class_meta = {}
    role_counts = defaultdict(lambda: defaultdict(float))
    role_meta = {}

    for record in records:
        stats = record.get('geometry_context_stats') or {}
        dataset = stats.get('dataset_name') or record.get('dataset_name') or 'unknown'
        params = (
            first([record.get('geometry_context_score_thd'), stats.get('geometry_context_score_thd')]),
            first([record.get('geometry_context_min_pixels'), stats.get('geometry_context_min_pixels')]),
            first([record.get('geometry_context_core_kernel'), stats.get('geometry_context_core_kernel')]),
        )
        for row in stats.get('pair_stats') or []:
            gt_name = row.get('gt_class_name')
            pred_name = row.get('base_pred_class_name')
            gt_role = infer_role(gt_name)
            pred_role = infer_role(pred_name)
            key = (
                dataset,
                *params,
                row.get('gt_class_index'),
                row.get('base_pred_class_index'),
            )
            add_pair_row(pair_counts[key], row)
            pair_meta[key] = dict(
                dataset_name=dataset,
                geometry_context_score_thd=params[0],
                geometry_context_min_pixels=params[1],
                geometry_context_core_kernel=params[2],
                gt_class_index=row.get('gt_class_index'),
                gt_class_name=gt_name,
                gt_role=gt_role,
                base_pred_class_index=row.get('base_pred_class_index'),
                base_pred_class_name=pred_name,
                pred_role=pred_role,
            )

            role_key = (dataset, *params, gt_role, pred_role)
            add_pair_row(role_counts[role_key], row)
            role_meta[role_key] = dict(
                dataset_name=dataset,
                geometry_context_score_thd=params[0],
                geometry_context_min_pixels=params[1],
                geometry_context_core_kernel=params[2],
                gt_role=gt_role,
                pred_role=pred_role,
            )

        for row in stats.get('class_stats') or []:
            class_name = row.get('class_name')
            key = (dataset, *params, row.get('class_index'))
            add_class_row(class_counts[key], row)
            class_meta[key] = dict(
                dataset_name=dataset,
                geometry_context_score_thd=params[0],
                geometry_context_min_pixels=params[1],
                geometry_context_core_kernel=params[2],
                class_index=row.get('class_index'),
                class_name=class_name,
                role=infer_role(class_name),
            )

    pair_rows = [finalize_pair(pair_meta[key], counts) for key, counts in pair_counts.items()]
    class_rows = [finalize_class(class_meta[key], counts) for key, counts in class_counts.items()]
    role_rows = [finalize_pair(role_meta[key], counts) for key, counts in role_counts.items()]

    pair_rows.sort(key=lambda row: (
        not row.get('focus_pair', False),
        row.get('dataset_name'),
        -(row.get('pixels') or 0)))
    class_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('class_index')))
    role_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('gt_role'),
        row.get('pred_role')))
    return pair_rows, class_rows, role_rows


def write_csv(path, rows, preferred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    fieldnames = [key for key in preferred if key in fieldnames or not rows] + [
        key for key in fieldnames if key not in preferred]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input JSONL files matched: {args.inputs}')
    pair_rows, class_rows, role_rows = summarize(iter_records(paths))

    pair_preferred = [
        'dataset_name', 'gt_class_index', 'gt_class_name', 'gt_role',
        'base_pred_class_index', 'base_pred_class_name', 'pred_role',
        'focus_pair', 'focus_pair_name', 'rows', 'pixels',
        'gt_support_on_pair_ratio', 'pred_support_on_pair_ratio',
        'gt_seed_on_pair_ratio', 'pred_seed_on_pair_ratio',
        'mean_score_margin_on_pair',
        'mean_gt_score_on_pair', 'mean_pred_score_on_pair',
        'mean_gt_semantic_on_pair', 'mean_pred_semantic_on_pair',
        'mean_gt_instance_on_pair', 'mean_pred_instance_on_pair',
        'mask_ratio_gt_minus_pred',
        'bbox_fill_ratio_gt_minus_pred',
        'core_ratio_gt_minus_pred',
        'boundary_ratio_gt_minus_pred',
        'component_density_gt_minus_pred',
        'mean_component_area_gt_minus_pred',
        'max_component_area_gt_minus_pred',
        'score_mean_gt_minus_pred',
        'semantic_mean_gt_minus_pred',
        'instance_mean_gt_minus_pred',
        'gt_support_bbox_fill_ratio', 'pred_support_bbox_fill_ratio',
        'gt_support_core_ratio', 'pred_support_core_ratio',
        'gt_support_boundary_ratio', 'pred_support_boundary_ratio',
        'gt_support_component_density', 'pred_support_component_density',
    ]
    class_preferred = [
        'dataset_name', 'class_index', 'class_name', 'role', 'rows',
        'gt_pixels', 'support_pixels', 'pred_pixels', 'seed_pixels',
        'support_precision', 'support_recall',
        'pred_precision', 'pred_recall',
        'seed_purity', 'seed_recall',
        'support_mask_ratio', 'support_bbox_fill_ratio',
        'support_core_ratio', 'support_boundary_ratio',
        'support_component_density', 'support_mean_component_area',
        'support_max_component_area',
    ]
    role_preferred = [
        'dataset_name', 'gt_role', 'pred_role', 'rows', 'pixels',
        'gt_support_on_pair_ratio', 'pred_support_on_pair_ratio',
        'gt_seed_on_pair_ratio', 'pred_seed_on_pair_ratio',
        'mean_score_margin_on_pair',
        'mask_ratio_gt_minus_pred',
        'bbox_fill_ratio_gt_minus_pred',
        'core_ratio_gt_minus_pred',
        'boundary_ratio_gt_minus_pred',
        'component_density_gt_minus_pred',
        'mean_component_area_gt_minus_pred',
    ]

    write_csv(os.path.join(args.out_dir, 'geometry_pair_summary.csv'), pair_rows, pair_preferred)
    write_csv(os.path.join(args.out_dir, 'geometry_class_summary.csv'), class_rows, class_preferred)
    write_csv(os.path.join(args.out_dir, 'geometry_role_summary.csv'), role_rows, role_preferred)
    print(f'Wrote {len(pair_rows)} pair rows, {len(class_rows)} class rows, '
          f'and {len(role_rows)} role rows to {args.out_dir}')


if __name__ == '__main__':
    main()
