import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SegEarth-OV3 clean-seed separability diagnostic JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/seed_diag/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for seed_rule_curve.csv, seed_class_summary.csv, '
             'seed_pair_summary.csv, and seed_region_summary.csv')
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
    if den == 0:
        return None
    return num / den


def first(values):
    for value in values:
        if value is not None:
            return value
    return None


def add_sum(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value


def add_rule_row(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'valid_pixels',
        'baseline_correct_pixels',
        'baseline_wrong_pixels',
        'seed_pixels',
        'seed_correct_pixels',
        'gt_present_class_count',
        'seed_class_count',
        'covered_gt_class_count',
        'wrong_gt_class_has_seed_pixels',
        'wrong_pred_class_has_seed_pixels',
        'wrong_both_seed_available_pixels',
        'wrong_similarity_to_pred_seed_sum',
        'wrong_similarity_to_pred_seed_pixels',
        'wrong_similarity_to_gt_seed_sum',
        'wrong_similarity_to_gt_seed_pixels',
        'wrong_similarity_margin_sum',
        'wrong_similarity_margin_pixels',
        'wrong_seed_similarity_favors_gt_pixels',
    ])


def finalize_rule_row(base, counts):
    row = dict(base)
    row.update(counts)
    valid = row.get('valid_pixels', 0)
    wrong = row.get('baseline_wrong_pixels', 0)
    seed = row.get('seed_pixels', 0)
    both = row.get('wrong_both_seed_available_pixels', 0)
    row['baseline_accuracy'] = safe_div(row.get('baseline_correct_pixels', 0), valid)
    row['seed_ratio'] = safe_div(seed, valid)
    row['seed_purity'] = safe_div(row.get('seed_correct_pixels', 0), seed)
    row['class_coverage_ratio'] = safe_div(
        row.get('covered_gt_class_count', 0),
        row.get('gt_present_class_count', 0))
    row['wrong_gt_class_has_seed_ratio'] = safe_div(
        row.get('wrong_gt_class_has_seed_pixels', 0), wrong)
    row['wrong_pred_class_has_seed_ratio'] = safe_div(
        row.get('wrong_pred_class_has_seed_pixels', 0), wrong)
    row['wrong_both_seed_available_ratio'] = safe_div(both, wrong)
    row['mean_wrong_similarity_to_pred_seed'] = safe_div(
        row.get('wrong_similarity_to_pred_seed_sum', 0.0),
        row.get('wrong_similarity_to_pred_seed_pixels', 0))
    row['mean_wrong_similarity_to_gt_seed'] = safe_div(
        row.get('wrong_similarity_to_gt_seed_sum', 0.0),
        row.get('wrong_similarity_to_gt_seed_pixels', 0))
    row['mean_wrong_similarity_margin'] = safe_div(
        row.get('wrong_similarity_margin_sum', 0.0),
        row.get('wrong_similarity_margin_pixels', 0))
    row['wrong_seed_similarity_favors_gt_ratio'] = safe_div(
        row.get('wrong_seed_similarity_favors_gt_pixels', 0), both)
    return row


def add_class_row(bucket, row, valid_pixels):
    bucket['images'] += 1
    bucket['valid_pixels'] += valid_pixels
    if (row.get('gt_pixels') or 0) > 0:
        bucket['gt_present_images'] += 1
    if (row.get('seed_pixels') or 0) > 0:
        bucket['seed_present_images'] += 1
    add_sum(bucket, row, [
        'gt_pixels',
        'pred_pixels',
        'seed_pixels',
        'seed_correct_pixels',
    ])


def finalize_class_row(base, counts):
    row = dict(base)
    row.update(counts)
    seed = row.get('seed_pixels', 0)
    pred = row.get('pred_pixels', 0)
    row['seed_ratio'] = safe_div(seed, row.get('valid_pixels', 0))
    row['seed_ratio_in_pred'] = safe_div(seed, pred)
    row['seed_purity'] = safe_div(row.get('seed_correct_pixels', 0), seed)
    row['gt_class_covered_image_ratio'] = safe_div(
        row.get('seed_present_images', 0),
        row.get('gt_present_images', 0))
    return row


def add_pair_row(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'pixels',
        'gt_class_has_seed_pixels',
        'pred_class_has_seed_pixels',
        'both_seed_available_pixels',
        'similarity_to_pred_seed_sum',
        'similarity_to_pred_seed_pixels',
        'similarity_to_gt_seed_sum',
        'similarity_to_gt_seed_pixels',
        'similarity_margin_sum',
        'similarity_margin_pixels',
        'seed_similarity_favors_gt_pixels',
    ])


def finalize_pair_row(base, counts):
    row = dict(base)
    row.update(counts)
    pixels = row.get('pixels', 0)
    both = row.get('both_seed_available_pixels', 0)
    row['gt_class_has_seed_ratio'] = safe_div(
        row.get('gt_class_has_seed_pixels', 0), pixels)
    row['pred_class_has_seed_ratio'] = safe_div(
        row.get('pred_class_has_seed_pixels', 0), pixels)
    row['both_seed_available_ratio'] = safe_div(both, pixels)
    row['mean_similarity_to_pred_seed'] = safe_div(
        row.get('similarity_to_pred_seed_sum', 0.0),
        row.get('similarity_to_pred_seed_pixels', 0))
    row['mean_similarity_to_gt_seed'] = safe_div(
        row.get('similarity_to_gt_seed_sum', 0.0),
        row.get('similarity_to_gt_seed_pixels', 0))
    row['mean_similarity_margin'] = safe_div(
        row.get('similarity_margin_sum', 0.0),
        row.get('similarity_margin_pixels', 0))
    row['seed_similarity_favors_gt_ratio'] = safe_div(
        row.get('seed_similarity_favors_gt_pixels', 0), both)
    focus_name = focus_pair_name(
        row.get('dataset_name'),
        row.get('gt_class_name'),
        row.get('base_pred_class_name'))
    row['focus_pair'] = bool(focus_name)
    row['focus_pair_name'] = focus_name
    return row


def add_region_row(bucket, row):
    bucket['images'] += 1
    add_sum(bucket, row, [
        'region_count',
        'region_pixels',
        'region_correct_pixels',
        'pure_region_count',
        'region_area_sum',
        'region_purity_sum',
    ])
    max_area = row.get('max_region_area')
    if isinstance(max_area, (int, float)):
        bucket['max_region_area'] = max(bucket.get('max_region_area', 0), max_area)
    min_area = row.get('min_region_area')
    if isinstance(min_area, (int, float)):
        current = bucket.get('min_region_area')
        bucket['min_region_area'] = min_area if current is None else min(current, min_area)


def finalize_region_row(base, counts):
    row = dict(base)
    row.update(counts)
    regions = row.get('region_count', 0)
    region_pixels = row.get('region_pixels', 0)
    row['region_purity'] = safe_div(row.get('region_correct_pixels', 0), region_pixels)
    row['pure_region_ratio'] = safe_div(row.get('pure_region_count', 0), regions)
    row['mean_region_area'] = safe_div(row.get('region_area_sum', 0), regions)
    row['mean_region_purity'] = safe_div(row.get('region_purity_sum', 0.0), regions)
    return row


def focus_pair_name(dataset_name, gt_name, pred_name):
    dataset = (dataset_name or '').lower()
    gt = (gt_name or '').lower()
    pred = (pred_name or '').lower()
    gt_tokens = class_name_tokens(gt)
    pred_tokens = class_name_tokens(pred)
    if dataset == 'vdd' and 'roof' in gt_tokens and 'facade' in pred_tokens:
        return 'VDD roof -> facade'
    if dataset == 'potsdam' and 'tree' in gt_tokens and 'grass' in pred_tokens:
        return 'Potsdam tree -> grass'
    if dataset == 'vaihingen' and 'grass' in gt_tokens and 'tree' in pred_tokens:
        return 'Vaihingen grass -> tree'
    if dataset == 'openearthmap' and 'pavement' in gt_tokens and 'building' in pred_tokens:
        return 'OpenEarthMap pavement -> building'
    if (dataset == 'loveda'
            and bool({'forest', 'agricultural', 'agriculture'} & gt_tokens)
            and 'background' in pred_tokens):
        return f'LoveDA {gt} -> background'
    return None


def class_name_tokens(name):
    tokens = set()
    for item in (name or '').replace('/', ',').split(','):
        item = item.strip().lower()
        if item:
            tokens.add(item)
    if name:
        tokens.add(name.strip().lower())
    return tokens


def summarize(records):
    rule_counts = defaultdict(lambda: defaultdict(float))
    class_counts = defaultdict(lambda: defaultdict(float))
    class_names = defaultdict(lambda: defaultdict(list))
    pair_counts = defaultdict(lambda: defaultdict(float))
    pair_names = defaultdict(lambda: defaultdict(list))
    region_counts = defaultdict(lambda: defaultdict(float))
    region_names = defaultdict(lambda: defaultdict(list))

    for record in records:
        stats = record.get('seed_separability_stats') or {}
        dataset_name = stats.get('dataset_name') or record.get('dataset_name') or 'unknown'
        valid_pixels = stats.get('valid_pixels') or 0
        run_params = dict(
            seed_final_score_thd=first([
                record.get('seed_final_score_thd'),
                stats.get('seed_final_score_thd'),
            ]),
            seed_margin_thd=first([
                record.get('seed_margin_thd'),
                stats.get('seed_margin_thd'),
            ]),
            seed_local_kernel=first([
                record.get('seed_local_kernel'),
                stats.get('seed_local_kernel'),
            ]),
            seed_local_consistency_thd=first([
                record.get('seed_local_consistency_thd'),
                stats.get('seed_local_consistency_thd'),
            ]),
            seed_core_kernel=first([
                record.get('seed_core_kernel'),
                stats.get('seed_core_kernel'),
            ]),
            seed_core_consistency_thd=first([
                record.get('seed_core_consistency_thd'),
                stats.get('seed_core_consistency_thd'),
            ]),
        )
        param_key = tuple(run_params.values())

        for row in stats.get('rule_stats') or []:
            key = (
                dataset_name,
                *param_key,
                row.get('rule_order'),
                row.get('rule_name'),
                row.get('similarity_space') or 'evidence',
            )
            add_rule_row(rule_counts[key], row)

        for row in stats.get('class_stats') or []:
            key = (
                dataset_name,
                *param_key,
                row.get('rule_order'),
                row.get('rule_name'),
                row.get('class_index'),
            )
            add_class_row(class_counts[key], row, valid_pixels)
            class_names[key]['class_name'].append(row.get('class_name'))

        for row in stats.get('pair_stats') or []:
            key = (
                dataset_name,
                *param_key,
                row.get('rule_order'),
                row.get('rule_name'),
                row.get('similarity_space') or 'evidence',
                row.get('gt_class_index'),
                row.get('base_pred_class_index'),
            )
            add_pair_row(pair_counts[key], row)
            pair_names[key]['gt_class_name'].append(row.get('gt_class_name'))
            pair_names[key]['base_pred_class_name'].append(row.get('base_pred_class_name'))

        for row in stats.get('region_stats') or []:
            key = (
                dataset_name,
                *param_key,
                row.get('rule_order'),
                row.get('rule_name'),
                row.get('class_index'),
            )
            add_region_row(region_counts[key], row)
            region_names[key]['class_name'].append(row.get('class_name'))

    rule_rows = []
    for (dataset_name, seed_final_score_thd, seed_margin_thd, seed_local_kernel,
         seed_local_consistency_thd, seed_core_kernel, seed_core_consistency_thd,
         rule_order, rule_name, similarity_space), counts in sorted(rule_counts.items()):
        rule_rows.append(finalize_rule_row(
            dict(
                dataset_name=dataset_name,
                seed_final_score_thd=seed_final_score_thd,
                seed_margin_thd=seed_margin_thd,
                seed_local_kernel=seed_local_kernel,
                seed_local_consistency_thd=seed_local_consistency_thd,
                seed_core_kernel=seed_core_kernel,
                seed_core_consistency_thd=seed_core_consistency_thd,
                rule_order=rule_order,
                rule_name=rule_name,
                similarity_space=similarity_space,
            ),
            counts))
    rule_rows.sort(key=lambda item: (
        item.get('dataset_name'), item.get('rule_order'), item.get('similarity_space')))

    class_rows = []
    for (dataset_name, seed_final_score_thd, seed_margin_thd, seed_local_kernel,
         seed_local_consistency_thd, seed_core_kernel, seed_core_consistency_thd,
         rule_order, rule_name, class_index), counts in sorted(class_counts.items()):
        name_key = (
            dataset_name,
            seed_final_score_thd,
            seed_margin_thd,
            seed_local_kernel,
            seed_local_consistency_thd,
            seed_core_kernel,
            seed_core_consistency_thd,
            rule_order,
            rule_name,
            class_index,
        )
        class_rows.append(finalize_class_row(
            dict(
                dataset_name=dataset_name,
                seed_final_score_thd=seed_final_score_thd,
                seed_margin_thd=seed_margin_thd,
                seed_local_kernel=seed_local_kernel,
                seed_local_consistency_thd=seed_local_consistency_thd,
                seed_core_kernel=seed_core_kernel,
                seed_core_consistency_thd=seed_core_consistency_thd,
                rule_order=rule_order,
                rule_name=rule_name,
                class_index=class_index,
                class_name=first(class_names[name_key]['class_name']),
            ),
            counts))
    class_rows.sort(key=lambda item: (
        item.get('dataset_name'), item.get('rule_order'), item.get('class_index')))

    pair_rows = []
    for (dataset_name, seed_final_score_thd, seed_margin_thd, seed_local_kernel,
         seed_local_consistency_thd, seed_core_kernel, seed_core_consistency_thd,
         rule_order, rule_name, similarity_space, gt_class, pred_class), counts in sorted(pair_counts.items()):
        name_key = (
            dataset_name,
            seed_final_score_thd,
            seed_margin_thd,
            seed_local_kernel,
            seed_local_consistency_thd,
            seed_core_kernel,
            seed_core_consistency_thd,
            rule_order,
            rule_name,
            similarity_space,
            gt_class,
            pred_class,
        )
        pair_rows.append(finalize_pair_row(
            dict(
                dataset_name=dataset_name,
                seed_final_score_thd=seed_final_score_thd,
                seed_margin_thd=seed_margin_thd,
                seed_local_kernel=seed_local_kernel,
                seed_local_consistency_thd=seed_local_consistency_thd,
                seed_core_kernel=seed_core_kernel,
                seed_core_consistency_thd=seed_core_consistency_thd,
                rule_order=rule_order,
                rule_name=rule_name,
                similarity_space=similarity_space,
                gt_class_index=gt_class,
                gt_class_name=first(pair_names[name_key]['gt_class_name']),
                base_pred_class_index=pred_class,
                base_pred_class_name=first(pair_names[name_key]['base_pred_class_name']),
            ),
            counts))
    pair_rows.sort(key=lambda item: (
        not item.get('focus_pair', False),
        item.get('dataset_name'),
        item.get('rule_order'),
        item.get('similarity_space'),
        -(item.get('pixels') or 0)))

    region_rows = []
    for (dataset_name, seed_final_score_thd, seed_margin_thd, seed_local_kernel,
         seed_local_consistency_thd, seed_core_kernel, seed_core_consistency_thd,
         rule_order, rule_name, class_index), counts in sorted(region_counts.items()):
        name_key = (
            dataset_name,
            seed_final_score_thd,
            seed_margin_thd,
            seed_local_kernel,
            seed_local_consistency_thd,
            seed_core_kernel,
            seed_core_consistency_thd,
            rule_order,
            rule_name,
            class_index,
        )
        region_rows.append(finalize_region_row(
            dict(
                dataset_name=dataset_name,
                seed_final_score_thd=seed_final_score_thd,
                seed_margin_thd=seed_margin_thd,
                seed_local_kernel=seed_local_kernel,
                seed_local_consistency_thd=seed_local_consistency_thd,
                seed_core_kernel=seed_core_kernel,
                seed_core_consistency_thd=seed_core_consistency_thd,
                rule_order=rule_order,
                rule_name=rule_name,
                class_index=class_index,
                class_name=first(region_names[name_key]['class_name']),
            ),
            counts))
    region_rows.sort(key=lambda item: (
        item.get('dataset_name'), item.get('rule_order'), item.get('class_index')))

    return rule_rows, class_rows, pair_rows, region_rows


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

    rule_rows, class_rows, pair_rows, region_rows = summarize(iter_records(paths))
    write_csv(
        os.path.join(args.out_dir, 'seed_rule_curve.csv'),
        rule_rows,
        [
            'dataset_name', 'rule_order', 'rule_name', 'similarity_space', 'images',
            'seed_final_score_thd', 'seed_margin_thd',
            'seed_local_kernel', 'seed_local_consistency_thd',
            'seed_core_kernel', 'seed_core_consistency_thd',
            'valid_pixels', 'baseline_correct_pixels', 'baseline_accuracy',
            'baseline_wrong_pixels', 'seed_pixels', 'seed_ratio',
            'seed_correct_pixels', 'seed_purity', 'gt_present_class_count',
            'seed_class_count', 'covered_gt_class_count', 'class_coverage_ratio',
            'wrong_gt_class_has_seed_ratio', 'wrong_both_seed_available_ratio',
            'mean_wrong_similarity_to_pred_seed',
            'mean_wrong_similarity_to_gt_seed',
            'mean_wrong_similarity_margin',
            'wrong_seed_similarity_favors_gt_ratio',
        ])
    write_csv(
        os.path.join(args.out_dir, 'seed_class_summary.csv'),
        class_rows,
        [
            'dataset_name', 'rule_order', 'rule_name', 'class_index',
            'class_name', 'seed_final_score_thd', 'seed_margin_thd',
            'seed_local_kernel', 'seed_local_consistency_thd',
            'seed_core_kernel', 'seed_core_consistency_thd',
            'images', 'valid_pixels', 'gt_pixels',
            'pred_pixels', 'seed_pixels', 'seed_ratio', 'seed_ratio_in_pred',
            'seed_correct_pixels', 'seed_purity', 'gt_present_images',
            'seed_present_images', 'gt_class_covered_image_ratio',
        ])
    write_csv(
        os.path.join(args.out_dir, 'seed_pair_summary.csv'),
        pair_rows,
        [
            'focus_pair', 'focus_pair_name', 'dataset_name', 'rule_order',
            'rule_name', 'similarity_space', 'gt_class_index', 'gt_class_name',
            'base_pred_class_index', 'base_pred_class_name', 'images',
            'seed_final_score_thd', 'seed_margin_thd',
            'seed_local_kernel', 'seed_local_consistency_thd',
            'seed_core_kernel', 'seed_core_consistency_thd',
            'pixels', 'gt_class_has_seed_ratio', 'pred_class_has_seed_ratio',
            'both_seed_available_ratio', 'mean_similarity_to_pred_seed',
            'mean_similarity_to_gt_seed', 'mean_similarity_margin',
            'seed_similarity_favors_gt_ratio',
        ])
    write_csv(
        os.path.join(args.out_dir, 'seed_region_summary.csv'),
        region_rows,
        [
            'dataset_name', 'rule_order', 'rule_name', 'class_index',
            'class_name', 'seed_final_score_thd', 'seed_margin_thd',
            'seed_local_kernel', 'seed_local_consistency_thd',
            'seed_core_kernel', 'seed_core_consistency_thd',
            'images', 'region_count', 'region_pixels',
            'region_correct_pixels', 'region_purity', 'pure_region_count',
            'pure_region_ratio', 'mean_region_area', 'max_region_area',
            'min_region_area', 'mean_region_purity',
        ])


if __name__ == '__main__':
    main()
