import argparse
import csv
import glob
import json
import os
from collections import defaultdict


FEATURE_KEYS = [
    'raw_score',
    'raw_presence_score',
    'mask_ratio',
    'core_ratio',
    'bbox_fill_ratio',
    'mask_prob_mean',
    'final_class_mean',
    'semantic_class_mean',
    'instance_class_mean',
    'evidence_mean',
    'final_margin_mean',
    'final_agreement_ratio',
    'semantic_agreement_ratio',
    'instance_agreement_ratio',
    'tri_agreement_ratio',
    'sem_final_agreement_ratio',
    'inst_final_agreement_ratio',
    'local_consistency_mean',
    'own_seed_overlap',
    'other_seed_overlap',
    'own_seed_coverage',
    'seed_clean_score',
    'evidence_consistency_score',
    'hybrid_quality_score',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SegEarth-OV3 raw candidate quality separability diagnostics.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/candidate_quality/vdd/quality_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for candidate_quality_feature_summary.csv, '
             'candidate_quality_pair_summary.csv, and candidate_quality_oracle_curve.csv')
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


def add_numeric(bucket, prefix, key, value):
    if isinstance(value, (int, float)):
        bucket[f'{prefix}{key}_sum'] += value
        bucket[f'{prefix}{key}_count'] += 1


def add_candidate_row(bucket, row):
    bucket['candidates'] += 1
    if row.get('kept') is True:
        bucket['kept_candidates'] += 1
    if row.get('is_clean') is True:
        bucket['clean_candidates'] += 1
        clean_prefix = 'clean_'
    else:
        clean_prefix = 'dirty_'
    for key in FEATURE_KEYS + ['purity', 'gt_recall', 'wrong_coverage']:
        value = row.get(key)
        add_numeric(bucket, '', key, value)
        add_numeric(bucket, clean_prefix, key, value)
    if isinstance(row.get('mask_pixels'), (int, float)):
        bucket['mask_pixels'] += row.get('mask_pixels')
    if isinstance(row.get('correct_pixels'), (int, float)):
        bucket['correct_pixels'] += row.get('correct_pixels')


def finalize_candidate_row(base, counts):
    row = dict(base)
    row.update(counts)
    row['kept_candidate_ratio'] = safe_div(
        row.get('kept_candidates', 0), row.get('candidates', 0))
    row['clean_candidate_ratio'] = safe_div(
        row.get('clean_candidates', 0), row.get('candidates', 0))
    row['pixel_weighted_purity'] = safe_div(
        row.get('correct_pixels', 0), row.get('mask_pixels', 0))
    for key in FEATURE_KEYS + ['purity', 'gt_recall', 'wrong_coverage']:
        row[f'mean_{key}'] = safe_div(
            row.get(f'{key}_sum', 0.0),
            row.get(f'{key}_count', 0))
        row[f'clean_mean_{key}'] = safe_div(
            row.get(f'clean_{key}_sum', 0.0),
            row.get(f'clean_{key}_count', 0))
        row[f'dirty_mean_{key}'] = safe_div(
            row.get(f'dirty_{key}_sum', 0.0),
            row.get(f'dirty_{key}_count', 0))
        if row[f'clean_mean_{key}'] is not None and row[f'dirty_mean_{key}'] is not None:
            row[f'clean_dirty_gap_{key}'] = row[f'clean_mean_{key}'] - row[f'dirty_mean_{key}']
        else:
            row[f'clean_dirty_gap_{key}'] = None
    return row


def add_pair_row(bucket, row):
    bucket['images'] += 1
    for key in [
            'pixels',
            'gt_selected_pair_cover_pixels',
            'pred_selected_pair_cover_pixels']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    for key in [
            'score_margin',
            'selected_purity_margin',
            'selected_pair_coverage_margin',
            'gt_selected_score',
            'pred_selected_score',
            'gt_selected_purity',
            'pred_selected_purity',
            'gt_selected_mask_ratio',
            'pred_selected_mask_ratio',
            'gt_selected_own_seed_overlap',
            'pred_selected_own_seed_overlap',
            'gt_selected_other_seed_overlap',
            'pred_selected_other_seed_overlap']:
        add_numeric(bucket, '', key, row.get(key))
    for key in [
            'score_favors_gt',
            'selected_purity_favors_gt',
            'gt_selected_clean',
            'pred_selected_clean',
            'gt_selected_kept',
            'pred_selected_kept']:
        value = row.get(key)
        if isinstance(value, bool):
            bucket[f'{key}_true'] += int(value)
            bucket[f'{key}_count'] += 1


def finalize_pair_row(base, counts):
    row = dict(base)
    row.update(counts)
    pixels = row.get('pixels', 0)
    row['gt_selected_pair_coverage'] = safe_div(
        row.get('gt_selected_pair_cover_pixels', 0), pixels)
    row['pred_selected_pair_coverage'] = safe_div(
        row.get('pred_selected_pair_cover_pixels', 0), pixels)
    row['selected_pair_coverage_margin_weighted'] = safe_div(
        row.get('gt_selected_pair_cover_pixels', 0)
        - row.get('pred_selected_pair_cover_pixels', 0),
        pixels)
    for key in [
            'score_margin',
            'selected_purity_margin',
            'selected_pair_coverage_margin',
            'gt_selected_score',
            'pred_selected_score',
            'gt_selected_purity',
            'pred_selected_purity',
            'gt_selected_mask_ratio',
            'pred_selected_mask_ratio',
            'gt_selected_own_seed_overlap',
            'pred_selected_own_seed_overlap',
            'gt_selected_other_seed_overlap',
            'pred_selected_other_seed_overlap']:
        row[f'mean_{key}'] = safe_div(
            row.get(f'{key}_sum', 0.0),
            row.get(f'{key}_count', 0))
    for key in [
            'score_favors_gt',
            'selected_purity_favors_gt',
            'gt_selected_clean',
            'pred_selected_clean',
            'gt_selected_kept',
            'pred_selected_kept']:
        row[f'{key}_ratio'] = safe_div(
            row.get(f'{key}_true', 0),
            row.get(f'{key}_count', 0))
    focus_name = focus_pair_name(
        row.get('dataset_name'),
        row.get('gt_class_name'),
        row.get('base_pred_class_name'))
    row['focus_pair'] = bool(focus_name)
    row['focus_pair_name'] = focus_name
    return row


def add_curve_row(bucket, row):
    add_pair_row(bucket, row)


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
    feature_buckets = defaultdict(lambda: defaultdict(int))
    pair_buckets = defaultdict(lambda: defaultdict(int))
    curve_buckets = defaultdict(lambda: defaultdict(int))

    for record in records:
        stats = record.get('candidate_quality_stats')
        if not stats:
            continue
        dataset_name = stats.get('dataset_name') or record.get('dataset_name')
        for row in stats.get('candidate_stats', []):
            key = (dataset_name, row.get('class_index'), row.get('class_name'))
            add_candidate_row(feature_buckets[key], row)
        for row in stats.get('pair_quality_stats', []):
            pair_key = (
                dataset_name,
                row.get('score_name'),
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('base_pred_class_index'),
                row.get('base_pred_class_name'),
            )
            add_pair_row(pair_buckets[pair_key], row)
            curve_key = (dataset_name, row.get('score_name'))
            add_curve_row(curve_buckets[curve_key], row)

    feature_rows = [
        finalize_candidate_row(
            dict(dataset_name=key[0], class_index=key[1], class_name=key[2]),
            counts)
        for key, counts in feature_buckets.items()
    ]
    pair_rows = [
        finalize_pair_row(
            dict(
                dataset_name=key[0],
                score_name=key[1],
                gt_class_index=key[2],
                gt_class_name=key[3],
                base_pred_class_index=key[4],
                base_pred_class_name=key[5]),
            counts)
        for key, counts in pair_buckets.items()
    ]
    curve_rows = [
        finalize_pair_row(
            dict(dataset_name=key[0], score_name=key[1]),
            counts)
        for key, counts in curve_buckets.items()
    ]

    feature_rows.sort(key=lambda item: (item.get('dataset_name') or '', item.get('class_index') or -1))
    pair_rows.sort(key=lambda item: (
        item.get('focus_pair') is not True,
        item.get('dataset_name') or '',
        item.get('score_name') or '',
        -(item.get('pixels') or 0)))
    curve_rows.sort(key=lambda item: (item.get('dataset_name') or '', item.get('score_name') or ''))
    return feature_rows, pair_rows, curve_rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    feature_rows, pair_rows, curve_rows = summarize(iter_records(paths))
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'candidate_quality_feature_summary.csv'), feature_rows)
    write_csv(os.path.join(args.out_dir, 'candidate_quality_pair_summary.csv'), pair_rows)
    write_csv(os.path.join(args.out_dir, 'candidate_quality_oracle_curve.csv'), curve_rows)


if __name__ == '__main__':
    main()
