import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SAM3 internal evidence selection-gap diagnostics.')
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
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_div(num, den):
    return None if den in (None, 0) else num / den


def add_numeric(bucket, row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            bucket[key] += value


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')

    source_buckets = defaultdict(lambda: defaultdict(float))
    class_buckets = defaultdict(lambda: defaultdict(float))
    pair_buckets = defaultdict(lambda: defaultdict(float))
    selector_buckets = defaultdict(lambda: defaultdict(float))
    oracle_buckets = defaultdict(lambda: defaultdict(float))
    oracle_source_buckets = defaultdict(lambda: defaultdict(float))
    reliability_buckets = defaultdict(lambda: defaultdict(float))
    regime_buckets = defaultdict(lambda: defaultdict(float))

    for record in iter_records(paths):
        stats = record.get('internal_selection_gap_stats')
        if not stats:
            continue
        dataset = stats.get('dataset_name') or record.get('dataset_name') or 'unknown'

        for row in stats.get('source_stats', []):
            key = (dataset, row.get('source_name'))
            bucket = source_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'valid_pixels',
                'available_class_count',
                'top1_correct_pixels',
                'baseline_wrong_pixels',
                'wrong_top1_recovered_pixels',
                'wrong_gt_topk_pixels',
                'changed_pixels',
                'improved_pixels',
                'harmed_pixels',
                'net_improved_pixels',
                'top1_margin_sum',
            ])

        for row in stats.get('class_stats', []):
            key = (
                dataset,
                row.get('source_name'),
                row.get('class_index'),
                row.get('class_name'),
            )
            bucket = class_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'gt_pixels',
                'baseline_wrong_pixels',
                'source_top1_correct_pixels',
                'wrong_top1_recovered_pixels',
                'wrong_gt_topk_pixels',
            ])

        for row in stats.get('pair_stats', []):
            key = (
                dataset,
                row.get('source_name'),
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('base_pred_class_index'),
                row.get('base_pred_class_name'),
            )
            bucket = pair_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'pixels',
                'source_top1_recovers_gt_pixels',
                'source_topk_contains_gt_pixels',
                'gt_vs_base_pred_margin_sum',
                'gt_vs_base_pred_margin_pixels',
            ])

        for row in stats.get('selector_stats', []):
            key = (dataset, row.get('rule_name'))
            bucket = selector_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'valid_pixels',
                'baseline_correct_pixels',
                'selected_correct_pixels',
                'changed_pixels',
                'improved_pixels',
                'harmed_pixels',
                'net_improved_pixels',
            ])
            for source_name, count in (
                    row.get('source_choice_counts') or {}).items():
                bucket[f'choice_{source_name}_pixels'] += count

        oracle = stats.get('oracle_stats') or {}
        oracle_key = (dataset,)
        oracle_bucket = oracle_buckets[oracle_key]
        oracle_bucket['images'] += 1
        add_numeric(oracle_bucket, oracle, [
            'baseline_wrong_pixels',
            'head_any_top1_correct_pixels',
            'expanded_any_top1_correct_pixels',
            'expanded_incremental_over_heads_pixels',
            'oracle_positive_margin_pixels',
        ])

        for row in stats.get('oracle_source_stats', []):
            key = (dataset, row.get('source_name'))
            bucket = oracle_source_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, ['oracle_best_positive_pixels'])

        for row in stats.get('source_reliability_stats', []):
            key = (
                dataset,
                row.get('source_name'),
                row.get('class_index'),
                row.get('class_name'),
            )
            bucket = reliability_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'seed_pixels',
                'valid_seed_pixels',
                'seed_top1_agree_pixels',
                'seed_source_margin_sum',
            ])
            reliability = row.get('no_gt_reliability')
            weight = row.get('valid_seed_pixels') or 0
            if isinstance(reliability, (int, float)) and weight > 0:
                bucket['reliability_weighted_sum'] += reliability * weight
                bucket['reliability_weight'] += weight

        for row in stats.get('regime_stats', []):
            key = (dataset, row.get('regime_name'))
            bucket = regime_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'pixels',
                'head_any_top1_correct_pixels',
                'expanded_any_top1_correct_pixels',
                'expanded_incremental_pixels',
                'oracle_positive_margin_pixels',
                'conservative_recovers_gt_pixels',
            ])

    source_rows = []
    for (dataset, source_name), bucket in sorted(source_buckets.items()):
        valid = bucket['valid_pixels']
        wrong = bucket['baseline_wrong_pixels']
        changed = bucket['changed_pixels']
        source_rows.append(dict(
            dataset_name=dataset,
            source_name=source_name,
            images=int(bucket['images']),
            valid_pixels=int(valid),
            mean_available_class_count=safe_div(
                bucket['available_class_count'], bucket['images']),
            top1_accuracy=safe_div(bucket['top1_correct_pixels'], valid),
            wrong_top1_recovered_ratio=safe_div(
                bucket['wrong_top1_recovered_pixels'], wrong),
            wrong_gt_topk_ratio=safe_div(bucket['wrong_gt_topk_pixels'], wrong),
            changed_ratio=safe_div(changed, valid),
            changed_precision=safe_div(bucket['improved_pixels'], changed),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            net_improved_pixels=int(bucket['net_improved_pixels']),
            mean_top1_margin=safe_div(bucket['top1_margin_sum'], valid),
        ))

    class_rows = []
    for key, bucket in sorted(class_buckets.items()):
        dataset, source_name, class_index, class_name = key
        gt_pixels = bucket['gt_pixels']
        wrong = bucket['baseline_wrong_pixels']
        class_rows.append(dict(
            dataset_name=dataset,
            source_name=source_name,
            class_index=class_index,
            class_name=class_name,
            images=int(bucket['images']),
            gt_pixels=int(gt_pixels),
            baseline_wrong_pixels=int(wrong),
            source_class_recall=safe_div(
                bucket['source_top1_correct_pixels'], gt_pixels),
            wrong_top1_recovered_ratio=safe_div(
                bucket['wrong_top1_recovered_pixels'], wrong),
            wrong_gt_topk_ratio=safe_div(
                bucket['wrong_gt_topk_pixels'], wrong),
        ))

    pair_rows = []
    for key, bucket in sorted(pair_buckets.items()):
        dataset, source_name, gt_idx, gt_name, pred_idx, pred_name = key
        pixels = bucket['pixels']
        pair_rows.append(dict(
            dataset_name=dataset,
            source_name=source_name,
            gt_class_index=gt_idx,
            gt_class_name=gt_name,
            base_pred_class_index=pred_idx,
            base_pred_class_name=pred_name,
            images=int(bucket['images']),
            pixels=int(pixels),
            source_top1_recovers_gt_ratio=safe_div(
                bucket['source_top1_recovers_gt_pixels'], pixels),
            source_topk_contains_gt_ratio=safe_div(
                bucket['source_topk_contains_gt_pixels'], pixels),
            mean_gt_vs_base_pred_margin=safe_div(
                bucket['gt_vs_base_pred_margin_sum'],
                bucket['gt_vs_base_pred_margin_pixels']),
        ))

    selector_rows = []
    for (dataset, rule_name), bucket in sorted(selector_buckets.items()):
        valid = bucket['valid_pixels']
        changed = bucket['changed_pixels']
        row = dict(
            dataset_name=dataset,
            rule_name=rule_name,
            images=int(bucket['images']),
            valid_pixels=int(valid),
            baseline_accuracy=safe_div(
                bucket['baseline_correct_pixels'], valid),
            selected_accuracy=safe_div(
                bucket['selected_correct_pixels'], valid),
            accuracy_delta=safe_div(
                bucket['selected_correct_pixels']
                - bucket['baseline_correct_pixels'],
                valid),
            changed_pixels=int(changed),
            changed_ratio=safe_div(changed, valid),
            changed_precision=safe_div(bucket['improved_pixels'], changed),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            net_improved_pixels=int(bucket['net_improved_pixels']),
        )
        for key, value in bucket.items():
            if key.startswith('choice_'):
                row[key] = int(value)
        selector_rows.append(row)

    oracle_rows = []
    for (dataset,), bucket in sorted(oracle_buckets.items()):
        wrong = bucket['baseline_wrong_pixels']
        oracle_rows.append(dict(
            dataset_name=dataset,
            images=int(bucket['images']),
            baseline_wrong_pixels=int(wrong),
            head_any_top1_correct_ratio=safe_div(
                bucket['head_any_top1_correct_pixels'], wrong),
            expanded_any_top1_correct_ratio=safe_div(
                bucket['expanded_any_top1_correct_pixels'], wrong),
            expanded_incremental_over_heads_ratio=safe_div(
                bucket['expanded_incremental_over_heads_pixels'], wrong),
            oracle_positive_margin_ratio=safe_div(
                bucket['oracle_positive_margin_pixels'], wrong),
        ))

    oracle_source_rows = []
    for (dataset, source_name), bucket in sorted(
            oracle_source_buckets.items()):
        wrong = oracle_buckets[(dataset,)]['baseline_wrong_pixels']
        oracle_source_rows.append(dict(
            dataset_name=dataset,
            source_name=source_name,
            images=int(bucket['images']),
            oracle_best_positive_pixels=int(
                bucket['oracle_best_positive_pixels']),
            oracle_best_positive_ratio=safe_div(
                bucket['oracle_best_positive_pixels'], wrong),
        ))

    reliability_rows = []
    for key, bucket in sorted(reliability_buckets.items()):
        dataset, source_name, class_index, class_name = key
        valid_seed = bucket['valid_seed_pixels']
        reliability_rows.append(dict(
            dataset_name=dataset,
            source_name=source_name,
            class_index=class_index,
            class_name=class_name,
            images=int(bucket['images']),
            seed_pixels=int(bucket['seed_pixels']),
            valid_seed_pixels=int(valid_seed),
            seed_top1_agree_ratio=safe_div(
                bucket['seed_top1_agree_pixels'], valid_seed),
            mean_seed_source_margin=safe_div(
                bucket['seed_source_margin_sum'], valid_seed),
            mean_no_gt_reliability=safe_div(
                bucket['reliability_weighted_sum'],
                bucket['reliability_weight']),
        ))

    regime_rows = []
    for (dataset, regime_name), bucket in sorted(regime_buckets.items()):
        pixels = bucket['pixels']
        regime_rows.append(dict(
            dataset_name=dataset,
            regime_name=regime_name,
            images=int(bucket['images']),
            pixels=int(pixels),
            head_any_top1_correct_ratio=safe_div(
                bucket['head_any_top1_correct_pixels'], pixels),
            expanded_any_top1_correct_ratio=safe_div(
                bucket['expanded_any_top1_correct_pixels'], pixels),
            expanded_incremental_ratio=safe_div(
                bucket['expanded_incremental_pixels'], pixels),
            oracle_positive_margin_ratio=safe_div(
                bucket['oracle_positive_margin_pixels'], pixels),
            conservative_recovers_gt_ratio=safe_div(
                bucket['conservative_recovers_gt_pixels'], pixels),
        ))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'internal_source_summary.csv'), source_rows)
    write_csv(os.path.join(args.out_dir, 'internal_class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'internal_pair_summary.csv'), pair_rows)
    write_csv(os.path.join(args.out_dir, 'internal_selector_summary.csv'), selector_rows)
    write_csv(os.path.join(args.out_dir, 'internal_oracle_summary.csv'), oracle_rows)
    write_csv(
        os.path.join(args.out_dir, 'internal_oracle_source_summary.csv'),
        oracle_source_rows)
    write_csv(
        os.path.join(args.out_dir, 'internal_reliability_summary.csv'),
        reliability_rows)
    write_csv(os.path.join(args.out_dir, 'internal_regime_summary.csv'), regime_rows)

    print(f'Read {len(paths)} files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
