import argparse
import csv
import glob
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize PE-layer seed separability from seed_pair_summary.csv files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='seed_pair_summary.csv files or glob patterns.')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for pe_layer_pair_summary.csv, '
             'pe_layer_class_summary.csv, and pe_layer_role_summary.csv.')
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


def to_float(value, default=0.0):
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if value in (None, ''):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_div(num, den):
    return None if den == 0 else num / den


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


def add_pair(bucket, row):
    pixels = to_int(row.get('pixels'))
    both = to_int(row.get('both_seed_available_pixels'))
    favors = to_int(row.get('seed_similarity_favors_gt_pixels'))
    margin_sum = to_float(row.get('similarity_margin_sum'))
    pred_sum = to_float(row.get('similarity_to_pred_seed_sum'))
    pred_pixels = to_int(row.get('similarity_to_pred_seed_pixels'))
    gt_sum = to_float(row.get('similarity_to_gt_seed_sum'))
    gt_pixels = to_int(row.get('similarity_to_gt_seed_pixels'))
    bucket['rows'] += 1
    bucket['pixels'] += pixels
    bucket['both_seed_available_pixels'] += both
    bucket['seed_similarity_favors_gt_pixels'] += favors
    bucket['similarity_margin_sum'] += margin_sum
    bucket['similarity_margin_pixels'] += to_int(row.get('similarity_margin_pixels'))
    bucket['similarity_to_pred_seed_sum'] += pred_sum
    bucket['similarity_to_pred_seed_pixels'] += pred_pixels
    bucket['similarity_to_gt_seed_sum'] += gt_sum
    bucket['similarity_to_gt_seed_pixels'] += gt_pixels


def finalize(base, counts):
    row = dict(base)
    row.update(counts)
    row['both_seed_available_ratio'] = safe_div(
        row.get('both_seed_available_pixels', 0),
        row.get('pixels', 0))
    row['mean_similarity_margin'] = safe_div(
        row.get('similarity_margin_sum', 0.0),
        row.get('similarity_margin_pixels', 0))
    row['mean_similarity_to_pred_seed'] = safe_div(
        row.get('similarity_to_pred_seed_sum', 0.0),
        row.get('similarity_to_pred_seed_pixels', 0))
    row['mean_similarity_to_gt_seed'] = safe_div(
        row.get('similarity_to_gt_seed_sum', 0.0),
        row.get('similarity_to_gt_seed_pixels', 0))
    row['seed_similarity_favors_gt_ratio'] = safe_div(
        row.get('seed_similarity_favors_gt_pixels', 0),
        row.get('both_seed_available_pixels', 0))
    return row


def read_rows(paths):
    for path in paths:
        with open(path, 'r', newline='') as f:
            for row in csv.DictReader(f):
                space = row.get('similarity_space') or ''
                if space == 'vision' or space.startswith('pe_layer_'):
                    row['_source_path'] = path
                    yield row


def summarize(rows):
    pair_counts = defaultdict(lambda: defaultdict(float))
    class_counts = defaultdict(lambda: defaultdict(float))
    role_counts = defaultdict(lambda: defaultdict(float))

    pair_meta = {}
    class_meta = {}
    role_meta = {}

    for row in rows:
        dataset = row.get('dataset_name') or 'unknown'
        rule_order = row.get('rule_order')
        rule_name = row.get('rule_name')
        space = row.get('similarity_space') or 'unknown'
        gt_idx = row.get('gt_class_index')
        pred_idx = row.get('base_pred_class_index')
        gt_name = row.get('gt_class_name') or ''
        pred_name = row.get('base_pred_class_name') or ''
        gt_role = infer_role(gt_name)
        pred_role = infer_role(pred_name)

        pair_key = (dataset, rule_order, rule_name, space, gt_idx, pred_idx)
        add_pair(pair_counts[pair_key], row)
        pair_meta[pair_key] = dict(
            dataset_name=dataset,
            rule_order=rule_order,
            rule_name=rule_name,
            similarity_space=space,
            gt_class_index=gt_idx,
            gt_class_name=gt_name,
            gt_role=gt_role,
            base_pred_class_index=pred_idx,
            base_pred_class_name=pred_name,
            pred_role=pred_role,
            focus_pair=row.get('focus_pair'),
            focus_pair_name=row.get('focus_pair_name'),
        )

        class_key = (dataset, rule_order, rule_name, space, gt_idx)
        add_pair(class_counts[class_key], row)
        class_meta[class_key] = dict(
            dataset_name=dataset,
            rule_order=rule_order,
            rule_name=rule_name,
            similarity_space=space,
            gt_class_index=gt_idx,
            gt_class_name=gt_name,
            gt_role=gt_role,
        )

        role_key = (dataset, rule_order, rule_name, space, gt_role, pred_role)
        add_pair(role_counts[role_key], row)
        role_meta[role_key] = dict(
            dataset_name=dataset,
            rule_order=rule_order,
            rule_name=rule_name,
            similarity_space=space,
            gt_role=gt_role,
            pred_role=pred_role,
        )

    pair_rows = [finalize(pair_meta[key], counts) for key, counts in pair_counts.items()]
    class_rows = [finalize(class_meta[key], counts) for key, counts in class_counts.items()]
    role_rows = [finalize(role_meta[key], counts) for key, counts in role_counts.items()]

    pair_rows.sort(key=lambda row: (
        str(row.get('focus_pair')).lower() != 'true',
        row.get('dataset_name'),
        row.get('rule_order'),
        row.get('similarity_space'),
        -(row.get('pixels') or 0)))
    class_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('rule_order'),
        row.get('gt_class_index'),
        row.get('similarity_space')))
    role_rows.sort(key=lambda row: (
        row.get('dataset_name'),
        row.get('rule_order'),
        row.get('gt_role'),
        row.get('pred_role'),
        row.get('similarity_space')))
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
        raise FileNotFoundError(f'No input CSV files matched: {args.inputs}')
    pair_rows, class_rows, role_rows = summarize(read_rows(paths))
    preferred = [
        'dataset_name', 'rule_order', 'rule_name', 'similarity_space',
        'gt_class_index', 'gt_class_name', 'gt_role',
        'base_pred_class_index', 'base_pred_class_name', 'pred_role',
        'focus_pair', 'focus_pair_name', 'rows', 'pixels',
        'both_seed_available_pixels', 'both_seed_available_ratio',
        'mean_similarity_to_pred_seed', 'mean_similarity_to_gt_seed',
        'mean_similarity_margin', 'seed_similarity_favors_gt_ratio',
        'seed_similarity_favors_gt_pixels',
    ]
    write_csv(
        os.path.join(args.out_dir, 'pe_layer_pair_summary.csv'),
        pair_rows,
        preferred)
    write_csv(
        os.path.join(args.out_dir, 'pe_layer_class_summary.csv'),
        class_rows,
        [
            'dataset_name', 'rule_order', 'rule_name', 'similarity_space',
            'gt_class_index', 'gt_class_name', 'gt_role', 'rows', 'pixels',
            'both_seed_available_pixels', 'both_seed_available_ratio',
            'mean_similarity_to_pred_seed', 'mean_similarity_to_gt_seed',
            'mean_similarity_margin', 'seed_similarity_favors_gt_ratio',
            'seed_similarity_favors_gt_pixels',
        ])
    write_csv(
        os.path.join(args.out_dir, 'pe_layer_role_summary.csv'),
        role_rows,
        [
            'dataset_name', 'rule_order', 'rule_name', 'similarity_space',
            'gt_role', 'pred_role', 'rows', 'pixels',
            'both_seed_available_pixels', 'both_seed_available_ratio',
            'mean_similarity_to_pred_seed', 'mean_similarity_to_gt_seed',
            'mean_similarity_margin', 'seed_similarity_favors_gt_ratio',
            'seed_similarity_favors_gt_pixels',
        ])
    print(f'Wrote {len(pair_rows)} pair rows, {len(class_rows)} class rows, '
          f'and {len(role_rows)} role rows to {args.out_dir}')


if __name__ == '__main__':
    main()
