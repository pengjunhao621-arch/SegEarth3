import argparse
import csv
import glob
import json
import os
from collections import defaultdict


PAIR_SUM_KEYS = [
    'valid_pixels',
    'conflict_pixels',
    'raw_conflict_count',
    'reverse_conflict_pixels',
    'target_score_sum',
    'competitor_score_sum',
    'target_vs_competitor_margin_sum',
    'semantic_target_favors_pixels',
    'instance_target_favors_pixels',
    'semantic_top1_target_pixels',
    'semantic_top1_competitor_pixels',
    'instance_top1_target_pixels',
    'instance_top1_competitor_pixels',
    'semantic_topk_target_pixels',
    'semantic_topk_competitor_pixels',
    'instance_topk_target_pixels',
    'instance_topk_competitor_pixels',
    'aux_both_favor_target_pixels',
    'aux_any_favor_target_pixels',
    'base_pred_target_pixels',
    'base_pred_competitor_pixels',
    'base_pred_bg_pixels',
    'gt_target_pixels',
    'gt_competitor_pixels',
    'gt_other_pixels',
    'gt_counterfactual_net_pixels',
    'gt_base_wrong_target_pixels',
    'gt_base_pred_competitor_pixels',
]

PAIR_MEAN_KEYS = [
    'mean_top1_score',
    'mean_top1_margin',
    'mean_target_score',
    'mean_competitor_score',
    'mean_target_vs_competitor_margin',
    'mean_semantic_margin',
    'mean_instance_margin',
    'mean_top1_local_consistency',
]

PROBE_SUM_KEYS = [
    'conflict_pixels',
    'reverse_conflict_pixels',
    'base_direct_margin_sum',
    'probe_margin_sum',
    'margin_gain_sum',
    'target_drop_sum',
    'competitor_drop_sum',
    'relative_stability_sum',
    'base_target_score_sum',
    'base_competitor_score_sum',
    'probe_target_score_sum',
    'probe_competitor_score_sum',
    'base_target_wins_pixels',
    'probe_target_wins_pixels',
    'base_target_support_pixels',
    'base_competitor_support_pixels',
    'probe_target_support_pixels',
    'probe_competitor_support_pixels',
    'probe_target_only_support_pixels',
    'probe_competitor_only_support_pixels',
    'probe_both_support_pixels',
    'probe_neither_support_pixels',
    'target_gain_support_pixels',
    'competitor_lost_support_pixels',
]

PRESENCE_KEYS = [
    'target_presence_base',
    'competitor_presence_base',
    'target_presence_probe',
    'competitor_presence_probe',
    'target_presence_drop',
    'competitor_presence_drop',
    'presence_relative_stability',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize top-k conflict bias diagnostic JSONL files.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--selector-support-threshold',
        default='0.1',
        help='Support threshold to merge into selector summary.')
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
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_div(num, den):
    return None if den == 0 else num / den


def as_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def new_bucket():
    bucket = defaultdict(float)
    bucket['rows'] = 0
    return bucket


def add_weighted_mean(bucket, row, keys, weight_key):
    weight = row.get(weight_key) or 0
    if not isinstance(weight, (int, float)) or weight <= 0:
        return
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_weighted_sum'] += value * weight
            bucket[f'{key}_weight'] += weight


def add_pair_row(bucket, row):
    bucket['rows'] += 1
    for key in PAIR_SUM_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    for key, value in row.items():
        if key.startswith('target_rank') and key.endswith('_pixels'):
            if isinstance(value, (int, float)):
                bucket[key] += value
    add_weighted_mean(bucket, row, PAIR_MEAN_KEYS, 'conflict_pixels')


def add_probe_row(bucket, row, include_support=False):
    bucket['rows'] += 1
    for key in PROBE_SUM_KEYS:
        if not include_support and 'support' in key:
            continue
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    add_weighted_mean(bucket, row, PRESENCE_KEYS, 'conflict_pixels')


def finalize_pair(meta, bucket):
    out = dict(meta)
    conflict = bucket.get('conflict_pixels', 0)
    out['rows'] = int(bucket.get('rows', 0))
    for key in PAIR_SUM_KEYS:
        out[key] = int(bucket.get(key, 0))
    for key, value in bucket.items():
        if key.startswith('target_rank') and key.endswith('_pixels'):
            out[key] = int(value)
            out[key.replace('_pixels', '_ratio')] = safe_div(value, conflict)
    for key in PAIR_MEAN_KEYS:
        out[key] = safe_div(
            bucket.get(f'{key}_weighted_sum', 0.0),
            bucket.get(f'{key}_weight', 0.0))
    out['conflict_ratio'] = safe_div(conflict, bucket.get('valid_pixels', 0))
    out['conflict_ratio_vs_reverse'] = safe_div(
        conflict, bucket.get('reverse_conflict_pixels', 0))
    if conflict is not None:
        out['log_conflict_ratio_vs_reverse'] = (
            None if bucket.get('reverse_conflict_pixels', 0) < 0
            else __import__('math').log(
                (conflict + 1.0)
                / (bucket.get('reverse_conflict_pixels', 0) + 1.0)))
    out['semantic_target_favors_ratio'] = safe_div(
        bucket.get('semantic_target_favors_pixels', 0), conflict)
    out['instance_target_favors_ratio'] = safe_div(
        bucket.get('instance_target_favors_pixels', 0), conflict)
    out['aux_both_favor_target_ratio'] = safe_div(
        bucket.get('aux_both_favor_target_pixels', 0), conflict)
    out['aux_any_favor_target_ratio'] = safe_div(
        bucket.get('aux_any_favor_target_pixels', 0), conflict)
    out['semantic_topk_target_ratio'] = safe_div(
        bucket.get('semantic_topk_target_pixels', 0), conflict)
    out['instance_topk_target_ratio'] = safe_div(
        bucket.get('instance_topk_target_pixels', 0), conflict)
    out['gt_target_ratio'] = safe_div(
        bucket.get('gt_target_pixels', 0), conflict)
    out['gt_competitor_ratio'] = safe_div(
        bucket.get('gt_competitor_pixels', 0), conflict)
    out['gt_other_ratio'] = safe_div(
        bucket.get('gt_other_pixels', 0), conflict)
    decided = (
        bucket.get('gt_target_pixels', 0)
        + bucket.get('gt_competitor_pixels', 0))
    out['gt_pair_precision'] = safe_div(
        bucket.get('gt_target_pixels', 0), decided)
    out['gt_counterfactual_net_ratio'] = safe_div(
        bucket.get('gt_counterfactual_net_pixels', 0), conflict)
    out['base_pred_competitor_ratio'] = safe_div(
        bucket.get('base_pred_competitor_pixels', 0), conflict)
    out['base_pred_bg_ratio'] = safe_div(
        bucket.get('base_pred_bg_pixels', 0), conflict)
    return out


def finalize_probe(meta, bucket, include_support=False):
    out = dict(meta)
    conflict = bucket.get('conflict_pixels', 0)
    out['rows'] = int(bucket.get('rows', 0))
    for key in PROBE_SUM_KEYS:
        if not include_support and 'support' in key:
            continue
        out[key] = int(bucket.get(key, 0))
    out['mean_base_direct_margin'] = safe_div(
        bucket.get('base_direct_margin_sum', 0), conflict)
    out['mean_probe_margin'] = safe_div(
        bucket.get('probe_margin_sum', 0), conflict)
    out['mean_margin_gain'] = safe_div(
        bucket.get('margin_gain_sum', 0), conflict)
    out['mean_target_drop'] = safe_div(
        bucket.get('target_drop_sum', 0), conflict)
    out['mean_competitor_drop'] = safe_div(
        bucket.get('competitor_drop_sum', 0), conflict)
    out['mean_relative_stability'] = safe_div(
        bucket.get('relative_stability_sum', 0), conflict)
    out['base_target_wins_ratio'] = safe_div(
        bucket.get('base_target_wins_pixels', 0), conflict)
    out['probe_target_wins_ratio'] = safe_div(
        bucket.get('probe_target_wins_pixels', 0), conflict)
    for key in PRESENCE_KEYS:
        out[f'mean_{key}'] = safe_div(
            bucket.get(f'{key}_weighted_sum', 0),
            bucket.get(f'{key}_weight', 0))
    if include_support:
        out['base_target_support_ratio'] = safe_div(
            bucket.get('base_target_support_pixels', 0), conflict)
        out['base_competitor_support_ratio'] = safe_div(
            bucket.get('base_competitor_support_pixels', 0), conflict)
        out['probe_target_support_ratio'] = safe_div(
            bucket.get('probe_target_support_pixels', 0), conflict)
        out['probe_competitor_support_ratio'] = safe_div(
            bucket.get('probe_competitor_support_pixels', 0), conflict)
        out['probe_target_only_support_ratio'] = safe_div(
            bucket.get('probe_target_only_support_pixels', 0), conflict)
        out['probe_competitor_only_support_ratio'] = safe_div(
            bucket.get('probe_competitor_only_support_pixels', 0), conflict)
        out['target_gain_support_ratio'] = safe_div(
            bucket.get('target_gain_support_pixels', 0), conflict)
        out['competitor_lost_support_ratio'] = safe_div(
            bucket.get('competitor_lost_support_pixels', 0), conflict)
    return out


def summarize(records):
    pair_buckets = {}
    probe_buckets = {}
    support_buckets = {}
    for record in records:
        stats = record.get('topk_conflict_bias_stats') or {}
        dataset = record.get('dataset_name') or stats.get('dataset_name') or ''
        topk = record.get('topk_conflict_topk') or stats.get('topk')
        for row in stats.get('pair_stats') or []:
            target = row.get('target_class_name')
            competitor = row.get('competitor_class_name')
            key = (dataset, target, competitor, topk)
            meta = dict(
                dataset_name=dataset,
                target_class_name=target,
                competitor_class_name=competitor,
                topk=topk,
            )
            pair_buckets.setdefault(key, (meta, new_bucket()))
            add_pair_row(pair_buckets[key][1], row)
        for row in stats.get('probe_stats') or []:
            target = row.get('target_class_name')
            competitor = row.get('competitor_class_name')
            support_threshold = row.get('support_threshold')
            common = dict(
                dataset_name=dataset,
                target_class_name=target,
                competitor_class_name=competitor,
                probe_name=row.get('probe_name'),
                head=row.get('head'),
                topk=topk,
            )
            base_key = (
                dataset, target, competitor, row.get('probe_name'),
                row.get('head'), topk)
            if support_threshold is None:
                probe_buckets.setdefault(base_key, (common, new_bucket()))
                add_probe_row(probe_buckets[base_key][1], row)
            else:
                support_meta = dict(common)
                support_meta['support_threshold'] = support_threshold
                support_key = base_key + (support_threshold,)
                support_buckets.setdefault(
                    support_key, (support_meta, new_bucket()))
                add_probe_row(
                    support_buckets[support_key][1],
                    row,
                    include_support=True)

    pair_rows = [
        finalize_pair(meta, bucket)
        for meta, bucket in pair_buckets.values()
    ]
    probe_rows = [
        finalize_probe(meta, bucket)
        for meta, bucket in probe_buckets.values()
    ]
    support_rows = [
        finalize_probe(meta, bucket, include_support=True)
        for meta, bucket in support_buckets.values()
    ]
    return pair_rows, probe_rows, support_rows


def selector_rows(pair_rows, probe_rows, support_rows, support_threshold):
    pair_by_key = {
        (row['dataset_name'], row['target_class_name'],
         row['competitor_class_name'], str(row.get('topk'))): row
        for row in pair_rows
    }
    probes_by_key = defaultdict(list)
    for row in probe_rows:
        if row.get('head') != 'final':
            continue
        key = (
            row['dataset_name'], row['target_class_name'],
            row['competitor_class_name'], str(row.get('topk')))
        probes_by_key[key].append(row)
    support_by_key = {}
    for row in support_rows:
        if row.get('head') != 'final':
            continue
        if str(row.get('support_threshold')) != str(support_threshold):
            continue
        key = (
            row['dataset_name'], row['target_class_name'],
            row['competitor_class_name'], str(row.get('topk')),
            row.get('probe_name'))
        support_by_key[key] = row

    rows = []
    for key, pair in pair_by_key.items():
        best = None
        for row in probes_by_key.get(key, []):
            if best is None or as_float(row.get('mean_relative_stability')) > as_float(
                    best.get('mean_relative_stability')):
                best = row
        reverse_key = (key[0], key[2], key[1], key[3])
        reverse_pair = pair_by_key.get(reverse_key)
        reverse_best = None
        for row in probes_by_key.get(reverse_key, []):
            if reverse_best is None or as_float(
                    row.get('mean_relative_stability')) > as_float(
                        reverse_best.get('mean_relative_stability')):
                reverse_best = row
        out = dict(pair)
        if best is not None:
            out.update({
                'best_probe_name': best.get('probe_name'),
                'best_relative_stability': best.get('mean_relative_stability'),
                'best_margin_gain': best.get('mean_margin_gain'),
                'best_probe_target_wins_ratio': best.get(
                    'probe_target_wins_ratio'),
                'best_target_drop': best.get('mean_target_drop'),
                'best_competitor_drop': best.get('mean_competitor_drop'),
            })
            support = support_by_key.get(key + (best.get('probe_name'),))
            if support is not None:
                for support_key in [
                        'probe_target_support_ratio',
                        'probe_competitor_support_ratio',
                        'probe_target_only_support_ratio',
                        'probe_competitor_only_support_ratio',
                        'target_gain_support_ratio',
                        'competitor_lost_support_ratio']:
                    out[support_key] = support.get(support_key)
        if reverse_pair is not None:
            out['reverse_conflict_pixels_from_summary'] = reverse_pair.get(
                'conflict_pixels')
            out['reverse_gt_pair_precision'] = reverse_pair.get(
                'gt_pair_precision')
        if reverse_best is not None and best is not None:
            out['reverse_best_probe_name'] = reverse_best.get('probe_name')
            out['reverse_best_relative_stability'] = reverse_best.get(
                'mean_relative_stability')
            out['reverse_best_probe_target_wins_ratio'] = reverse_best.get(
                'probe_target_wins_ratio')
            out['asymmetry_relative_stability'] = (
                as_float(best.get('mean_relative_stability'))
                - as_float(reverse_best.get('mean_relative_stability')))
            out['asymmetry_probe_win'] = (
                as_float(best.get('probe_target_wins_ratio'))
                - as_float(reverse_best.get('probe_target_wins_ratio')))
        rows.append(out)
    return rows


def write_csv(path, rows, preferred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for key in preferred:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    pair_rows, probe_rows, support_rows = summarize(iter_records(paths))
    pair_rows.sort(key=lambda row: (
        str(row.get('dataset_name')),
        -as_float(row.get('conflict_pixels')),
    ))
    probe_rows.sort(key=lambda row: (
        str(row.get('dataset_name')),
        str(row.get('target_class_name')),
        str(row.get('competitor_class_name')),
        str(row.get('head')),
        -as_float(row.get('mean_relative_stability')),
    ))
    support_rows.sort(key=lambda row: (
        str(row.get('dataset_name')),
        str(row.get('target_class_name')),
        str(row.get('competitor_class_name')),
        str(row.get('probe_name')),
        str(row.get('head')),
        as_float(row.get('support_threshold')),
    ))
    selectors = selector_rows(
        pair_rows, probe_rows, support_rows, args.selector_support_threshold)
    selectors.sort(key=lambda row: (
        str(row.get('dataset_name')),
        -as_float(row.get('conflict_pixels')),
    ))

    pair_preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'topk',
        'conflict_pixels',
        'reverse_conflict_pixels',
        'log_conflict_ratio_vs_reverse',
        'gt_target_ratio',
        'gt_competitor_ratio',
        'gt_other_ratio',
        'gt_pair_precision',
        'gt_counterfactual_net_pixels',
        'gt_counterfactual_net_ratio',
        'mean_target_vs_competitor_margin',
        'semantic_target_favors_ratio',
        'instance_target_favors_ratio',
        'aux_any_favor_target_ratio',
        'aux_both_favor_target_ratio',
        'semantic_topk_target_ratio',
        'instance_topk_target_ratio',
        'mean_top1_local_consistency',
        'base_pred_competitor_ratio',
        'base_pred_bg_ratio',
    ]
    probe_preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'probe_name',
        'head',
        'topk',
        'conflict_pixels',
        'mean_relative_stability',
        'mean_margin_gain',
        'mean_target_drop',
        'mean_competitor_drop',
        'base_target_wins_ratio',
        'probe_target_wins_ratio',
        'mean_presence_relative_stability',
    ]
    support_preferred = probe_preferred + [
        'support_threshold',
        'probe_target_support_ratio',
        'probe_competitor_support_ratio',
        'probe_target_only_support_ratio',
        'probe_competitor_only_support_ratio',
        'target_gain_support_ratio',
        'competitor_lost_support_ratio',
    ]
    selector_preferred = pair_preferred + [
        'best_probe_name',
        'best_relative_stability',
        'best_margin_gain',
        'best_probe_target_wins_ratio',
        'probe_target_only_support_ratio',
        'probe_competitor_only_support_ratio',
        'target_gain_support_ratio',
        'competitor_lost_support_ratio',
        'reverse_best_probe_name',
        'reverse_best_relative_stability',
        'reverse_best_probe_target_wins_ratio',
        'asymmetry_relative_stability',
        'asymmetry_probe_win',
        'reverse_gt_pair_precision',
    ]

    write_csv(
        os.path.join(args.out_dir, 'topk_conflict_pair_summary.csv'),
        pair_rows,
        pair_preferred)
    write_csv(
        os.path.join(args.out_dir, 'topk_conflict_probe_summary.csv'),
        probe_rows,
        probe_preferred)
    write_csv(
        os.path.join(args.out_dir, 'topk_conflict_support_summary.csv'),
        support_rows,
        support_preferred)
    write_csv(
        os.path.join(args.out_dir, 'topk_conflict_selector_summary.csv'),
        selectors,
        selector_preferred)


if __name__ == '__main__':
    main()
