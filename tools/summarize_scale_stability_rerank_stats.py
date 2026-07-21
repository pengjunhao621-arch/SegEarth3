import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize scale-stability rerank JSONL diagnostics.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/ssr/vaihingen/ssr_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for scale_stability_rerank_*_summary.csv files.')
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


OVERALL_KEYS = [
    'valid_pixels',
    'base_correct_pixels',
    'reranked_correct_pixels',
    'changed_pixels',
    'improved_pixels',
    'harmed_pixels',
    'net_improved_pixels',
]


PAIR_KEYS = [
    'target_in_topk_pixels',
    'scaled_choice_is_target_pixels',
    'candidate_pixels',
    'gate_pixels',
    'gate_valid_pixels',
    'pair_error_pixels',
    'pair_error_gate_pixels',
    'gate_true_target_pixels',
    'gate_true_competitor_pixels',
    'gate_true_other_pixels',
    'gate_counterfactual_improved_pixels',
    'gate_counterfactual_harmed_pixels',
    'gate_counterfactual_other_pixels',
    'gate_counterfactual_net_pixels',
    'changed_gate_pixels',
    'improved_gate_pixels',
    'harmed_gate_pixels',
    'net_improved_gate_pixels',
    'gate_base_margin_sum',
    'gate_scaled_margin_sum',
    'gate_margin_gain_sum',
    'gate_competitor_drop_sum',
    'gate_target_drop_sum',
    'gate_relative_stability_sum',
    'gate_choice_relative_stability_sum',
    'gate_target_semantic_sum',
    'gate_competitor_semantic_sum',
    'gate_target_instance_sum',
    'gate_competitor_instance_sum',
]


def new_bucket():
    bucket = defaultdict(float)
    bucket['rows'] = 0
    return bucket


def add_values(bucket, row, keys):
    bucket['rows'] += 1
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value
    for key in ['effective_scale_x', 'effective_scale_y']:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[f'{key}_sum'] += value
            bucket[f'{key}_count'] += 1


def finalize_overall(meta, bucket):
    valid = bucket.get('valid_pixels', 0)
    out = dict(meta)
    out['rows'] = int(bucket.get('rows', 0))
    for key in OVERALL_KEYS:
        out[key] = int(bucket.get(key, 0))
    out['base_accuracy'] = safe_div(
        bucket.get('base_correct_pixels', 0), valid)
    out['reranked_accuracy'] = safe_div(
        bucket.get('reranked_correct_pixels', 0), valid)
    out['accuracy_delta'] = (
        None if valid == 0 else
        safe_div(bucket.get('net_improved_pixels', 0), valid))
    out['changed_ratio'] = safe_div(bucket.get('changed_pixels', 0), valid)
    out['improved_ratio'] = safe_div(bucket.get('improved_pixels', 0), valid)
    out['harmed_ratio'] = safe_div(bucket.get('harmed_pixels', 0), valid)
    return out


def finalize_pair(meta, bucket):
    gate = bucket.get('gate_valid_pixels', 0)
    pair_error = bucket.get('pair_error_pixels', 0)
    candidate = bucket.get('candidate_pixels', 0)
    out = dict(meta)
    out['rows'] = int(bucket.get('rows', 0))
    for key in PAIR_KEYS:
        if key.endswith('_sum'):
            out[key] = bucket.get(key, 0)
        else:
            out[key] = int(bucket.get(key, 0))
    out['mean_effective_scale_x'] = safe_div(
        bucket.get('effective_scale_x_sum', 0),
        bucket.get('effective_scale_x_count', 0))
    out['mean_effective_scale_y'] = safe_div(
        bucket.get('effective_scale_y_sum', 0),
        bucket.get('effective_scale_y_count', 0))
    out['gate_ratio_in_candidate'] = safe_div(gate, candidate)
    out['pair_error_gate_coverage'] = safe_div(
        bucket.get('pair_error_gate_pixels', 0), pair_error)
    out['gate_true_target_ratio'] = safe_div(
        bucket.get('gate_true_target_pixels', 0), gate)
    out['gate_true_competitor_ratio'] = safe_div(
        bucket.get('gate_true_competitor_pixels', 0), gate)
    out['gate_true_other_ratio'] = safe_div(
        bucket.get('gate_true_other_pixels', 0), gate)
    oracle_decided = (
        bucket.get('gate_counterfactual_improved_pixels', 0)
        + bucket.get('gate_counterfactual_harmed_pixels', 0))
    out['gate_counterfactual_precision'] = safe_div(
        bucket.get('gate_counterfactual_improved_pixels', 0),
        oracle_decided)
    out['gate_counterfactual_net_ratio'] = safe_div(
        bucket.get('gate_counterfactual_net_pixels', 0), gate)
    out['changed_gate_ratio'] = safe_div(
        bucket.get('changed_gate_pixels', 0), gate)
    out['improved_gate_ratio'] = safe_div(
        bucket.get('improved_gate_pixels', 0), gate)
    out['harmed_gate_ratio'] = safe_div(
        bucket.get('harmed_gate_pixels', 0), gate)
    out['net_improved_gate_ratio'] = safe_div(
        bucket.get('net_improved_gate_pixels', 0), gate)
    for src, dst in [
            ('gate_base_margin_sum', 'mean_gate_base_margin'),
            ('gate_scaled_margin_sum', 'mean_gate_scaled_margin'),
            ('gate_margin_gain_sum', 'mean_gate_margin_gain'),
            ('gate_competitor_drop_sum', 'mean_gate_competitor_drop'),
            ('gate_target_drop_sum', 'mean_gate_target_drop'),
            ('gate_relative_stability_sum',
             'mean_gate_relative_stability'),
            ('gate_choice_relative_stability_sum',
             'mean_gate_choice_relative_stability'),
            ('gate_target_semantic_sum', 'mean_gate_target_semantic'),
            ('gate_competitor_semantic_sum',
             'mean_gate_competitor_semantic'),
            ('gate_target_instance_sum', 'mean_gate_target_instance'),
            ('gate_competitor_instance_sum',
             'mean_gate_competitor_instance')]:
        out[dst] = safe_div(bucket.get(src, 0), gate)
    return out


def write_csv(path, rows, preferred=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    keys = []
    for key in preferred or []:
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


def summarize(records):
    overall_buckets = {}
    pair_buckets = {}
    for record in records:
        stats = record.get('scale_stability_rerank_stats') or {}
        dataset = record.get('dataset_name') or stats.get('dataset_name') or ''
        scale = record.get('scale_stability_rerank_scale')
        gate = record.get('scale_stability_rerank_gate')
        threshold = record.get('scale_stability_rerank_drop_threshold')
        topk = record.get('scale_stability_rerank_topk')
        pair_mode = record.get('scale_stability_rerank_pair_mode')
        query_mode = record.get('scale_stability_rerank_query_mode')
        auto_min_pixels = record.get('scale_stability_rerank_auto_min_pixels')
        use_rerank = bool(record.get('use_scale_stability_rerank'))

        overall = stats.get('overall') or {}
        overall_key = (
            dataset, scale, topk, gate, threshold, pair_mode, query_mode,
            auto_min_pixels, use_rerank)
        overall_meta = dict(
            dataset_name=dataset,
            requested_scale=scale,
            scale_stability_rerank_topk=topk,
            gate_name=gate,
            drop_threshold=threshold,
            pair_mode=pair_mode,
            query_mode=query_mode,
            auto_min_pixels=auto_min_pixels,
            use_scale_stability_rerank=use_rerank,
        )
        overall_buckets.setdefault(overall_key, (overall_meta, new_bucket()))
        add_values(overall_buckets[overall_key][1], overall, OVERALL_KEYS)

        for row in stats.get('pair_stats') or []:
            target_name = row.get('target_class_name')
            competitor_name = row.get('competitor_class_name')
            pair_key = (
                dataset,
                target_name,
                competitor_name,
                row.get('requested_scale', scale),
                row.get('scale_stability_rerank_topk', topk),
                row.get('gate_name', gate),
                row.get('drop_threshold', threshold),
                row.get('pair_mode', pair_mode),
                row.get('query_mode', query_mode),
                row.get('auto_min_pixels', auto_min_pixels),
                row.get('queried_class_count'),
                use_rerank,
                row.get('min_base_margin'),
                row.get('max_base_margin'),
                row.get('min_scaled_margin'),
                row.get('max_target_rank'),
            )
            pair_meta = dict(
                dataset_name=dataset,
                target_class_name=target_name,
                competitor_class_name=competitor_name,
                requested_scale=row.get('requested_scale', scale),
                scale_stability_rerank_topk=row.get(
                    'scale_stability_rerank_topk', topk),
                gate_name=row.get('gate_name', gate),
                drop_threshold=row.get('drop_threshold', threshold),
                pair_mode=row.get('pair_mode', pair_mode),
                query_mode=row.get('query_mode', query_mode),
                auto_min_pixels=row.get('auto_min_pixels', auto_min_pixels),
                queried_class_count=row.get('queried_class_count'),
                use_scale_stability_rerank=use_rerank,
                min_base_margin=row.get('min_base_margin'),
                max_base_margin=row.get('max_base_margin'),
                min_scaled_margin=row.get('min_scaled_margin'),
                max_target_rank=row.get('max_target_rank'),
            )
            pair_buckets.setdefault(pair_key, (pair_meta, new_bucket()))
            add_values(pair_buckets[pair_key][1], row, PAIR_KEYS)
    overall_rows = [
        finalize_overall(meta, bucket)
        for meta, bucket in overall_buckets.values()
    ]
    pair_rows = [
        finalize_pair(meta, bucket)
        for meta, bucket in pair_buckets.values()
    ]
    return overall_rows, pair_rows


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    overall_rows, pair_rows = summarize(iter_records(paths))
    overall_preferred = [
        'dataset_name',
        'requested_scale',
        'scale_stability_rerank_topk',
        'gate_name',
        'drop_threshold',
        'pair_mode',
        'query_mode',
        'auto_min_pixels',
        'use_scale_stability_rerank',
        'rows',
        'valid_pixels',
        'base_accuracy',
        'reranked_accuracy',
        'accuracy_delta',
        'changed_pixels',
        'changed_ratio',
        'improved_pixels',
        'harmed_pixels',
        'net_improved_pixels',
    ]
    pair_preferred = [
        'dataset_name',
        'target_class_name',
        'competitor_class_name',
        'requested_scale',
        'scale_stability_rerank_topk',
        'gate_name',
        'drop_threshold',
        'pair_mode',
        'query_mode',
        'auto_min_pixels',
        'queried_class_count',
        'use_scale_stability_rerank',
        'min_base_margin',
        'max_base_margin',
        'min_scaled_margin',
        'max_target_rank',
        'rows',
        'candidate_pixels',
        'gate_valid_pixels',
        'gate_ratio_in_candidate',
        'pair_error_pixels',
        'pair_error_gate_pixels',
        'pair_error_gate_coverage',
        'gate_true_target_ratio',
        'gate_true_competitor_ratio',
        'gate_true_other_ratio',
        'gate_counterfactual_improved_pixels',
        'gate_counterfactual_harmed_pixels',
        'gate_counterfactual_net_pixels',
        'gate_counterfactual_precision',
        'gate_counterfactual_net_ratio',
        'changed_gate_pixels',
        'improved_gate_pixels',
        'harmed_gate_pixels',
        'net_improved_gate_pixels',
        'net_improved_gate_ratio',
        'mean_gate_base_margin',
        'mean_gate_scaled_margin',
        'mean_gate_margin_gain',
        'mean_gate_relative_stability',
    ]
    write_csv(
        os.path.join(args.out_dir,
                     'scale_stability_rerank_overall_summary.csv'),
        sorted(overall_rows, key=lambda item: (
            str(item.get('dataset_name')),
            float(item.get('drop_threshold') or 0),
        )),
        overall_preferred)
    write_csv(
        os.path.join(args.out_dir,
                     'scale_stability_rerank_pair_summary.csv'),
        sorted(pair_rows, key=lambda item: (
            str(item.get('dataset_name')),
            str(item.get('target_class_name')),
            str(item.get('competitor_class_name')),
            float(item.get('drop_threshold') or 0),
        )),
        pair_preferred)


if __name__ == '__main__':
    main()
