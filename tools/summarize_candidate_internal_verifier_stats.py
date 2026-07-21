import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize candidate-conditioned SAM3 internal evidence '
            'verifier diagnostics.'))
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


def safe_std(total, square_total, count):
    if count in (None, 0):
        return None
    mean = total / count
    return math.sqrt(max(0.0, square_total / count - mean * mean))


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

    candidate_buckets = defaultdict(lambda: defaultdict(float))
    feature_buckets = defaultdict(lambda: defaultdict(float))
    selector_buckets = defaultdict(lambda: defaultdict(float))
    pair_buckets = defaultdict(lambda: defaultdict(float))
    oracle_buckets = defaultdict(lambda: defaultdict(float))
    null_buckets = defaultdict(lambda: defaultdict(float))
    regime_buckets = defaultdict(lambda: defaultdict(float))
    family_sources = defaultdict(set)

    for record in iter_records(paths):
        stats = record.get('candidate_internal_verifier_stats')
        if not stats:
            continue
        dataset = (
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown'
        )

        for family_name, source_names in (
                stats.get('family_sources') or {}).items():
            family_sources[(dataset, family_name)].update(source_names)

        for row in stats.get('candidate_stats', []):
            key = (
                dataset,
                row.get('candidate_rank'),
                row.get('regime_name'),
            )
            bucket = candidate_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'candidate_pixels',
                'candidate_correct_pixels',
                'candidate_wrong_other_pixels',
                'candidate_on_baseline_correct_pixels',
            ])

        for row in stats.get('feature_stats', []):
            key = (
                dataset,
                row.get('family_name'),
                row.get('candidate_rank'),
                row.get('regime_name'),
            )
            bucket = feature_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'candidate_pixels',
                'positive_pixels',
                'negative_pixels',
                'positive_delta_sum',
                'positive_delta_sq_sum',
                'negative_delta_sum',
                'negative_delta_sq_sum',
                'positive_favored_pixels',
                'negative_favored_pixels',
            ])

        for row in stats.get('selector_stats', []):
            key = (dataset, row.get('rule_name'))
            bucket = selector_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'valid_pixels',
                'baseline_correct_pixels',
                'changed_pixels',
                'improved_pixels',
                'harmed_pixels',
                'neutral_wrong_to_wrong_pixels',
                'net_improved_pixels',
            ])

        for row in stats.get('pair_stats', []):
            key = (
                dataset,
                row.get('candidate_rank'),
                row.get('base_class_index'),
                row.get('base_class_name'),
                row.get('candidate_class_index'),
                row.get('candidate_class_name'),
            )
            bucket = pair_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'pixels',
                'candidate_correct_pixels',
                'baseline_correct_pixels',
            ])
            for family_name, family_row in (
                    row.get('family_stats') or {}).items():
                prefix = f'family_{family_name}_'
                bucket[prefix + 'valid_pixels'] += (
                    family_row.get('valid_pixels') or 0)
                bucket[prefix + 'delta_sum'] += (
                    family_row.get('delta_sum') or 0)
                bucket[prefix + 'favored_pixels'] += (
                    family_row.get('favored_pixels') or 0)

        oracle = stats.get('oracle_stats') or {}
        bucket = oracle_buckets[(dataset,)]
        bucket['images'] += 1
        add_numeric(bucket, oracle, [
            'valid_pixels',
            'baseline_wrong_pixels',
            'gt_in_candidate_pixels',
            'head_support_pixels',
            'readout_support_pixels',
            'latent_support_pixels',
            'expanded_support_pixels',
            'latent_incremental_pixels',
        ])

        for row in stats.get('null_stats', []):
            key = (
                dataset,
                row.get('control_name'),
                row.get('trial'),
            )
            bucket = null_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'gt_in_candidate_pixels',
                'latent_support_pixels',
                'latent_incremental_pixels',
            ])

        for row in stats.get('regime_oracle_stats', []):
            key = (dataset, row.get('regime_name'))
            bucket = regime_buckets[key]
            bucket['images'] += 1
            add_numeric(bucket, row, [
                'baseline_wrong_pixels',
                'gt_in_candidate_pixels',
                'head_support_pixels',
                'readout_support_pixels',
                'latent_support_pixels',
                'expanded_support_pixels',
                'latent_incremental_pixels',
            ])

    family_rows = [
        dict(
            dataset_name=dataset,
            family_name=family_name,
            source_names='|'.join(sorted(source_names)),
            source_count=len(source_names),
        )
        for (dataset, family_name), source_names
        in sorted(family_sources.items())
    ]

    candidate_rows = []
    for key, bucket in sorted(candidate_buckets.items()):
        dataset, candidate_rank, regime_name = key
        candidate_pixels = bucket['candidate_pixels']
        candidate_rows.append(dict(
            dataset_name=dataset,
            candidate_rank=candidate_rank,
            regime_name=regime_name,
            images=int(bucket['images']),
            candidate_pixels=int(candidate_pixels),
            candidate_correct_pixels=int(
                bucket['candidate_correct_pixels']),
            candidate_correct_ratio=safe_div(
                bucket['candidate_correct_pixels'], candidate_pixels),
            candidate_wrong_other_ratio=safe_div(
                bucket['candidate_wrong_other_pixels'], candidate_pixels),
            candidate_on_baseline_correct_ratio=safe_div(
                bucket['candidate_on_baseline_correct_pixels'],
                candidate_pixels),
        ))

    feature_rows = []
    for key, bucket in sorted(feature_buckets.items()):
        dataset, family_name, candidate_rank, regime_name = key
        positive = bucket['positive_pixels']
        negative = bucket['negative_pixels']
        positive_favored = bucket['positive_favored_pixels']
        negative_favored = bucket['negative_favored_pixels']
        positive_rate = safe_div(positive_favored, positive)
        negative_rate = safe_div(negative_favored, negative)
        feature_rows.append(dict(
            dataset_name=dataset,
            family_name=family_name,
            candidate_rank=candidate_rank,
            regime_name=regime_name,
            images=int(bucket['images']),
            candidate_pixels=int(bucket['candidate_pixels']),
            positive_pixels=int(positive),
            negative_pixels=int(negative),
            positive_mean_delta=safe_div(
                bucket['positive_delta_sum'], positive),
            positive_std_delta=safe_std(
                bucket['positive_delta_sum'],
                bucket['positive_delta_sq_sum'],
                positive),
            negative_mean_delta=safe_div(
                bucket['negative_delta_sum'], negative),
            negative_std_delta=safe_std(
                bucket['negative_delta_sum'],
                bucket['negative_delta_sq_sum'],
                negative),
            positive_favored_ratio=positive_rate,
            negative_favored_ratio=negative_rate,
            favored_ratio_lift=(
                None if positive_rate is None or negative_rate is None
                else positive_rate - negative_rate),
        ))

    selector_rows = []
    for (dataset, rule_name), bucket in sorted(selector_buckets.items()):
        valid = bucket['valid_pixels']
        changed = bucket['changed_pixels']
        baseline_correct = bucket['baseline_correct_pixels']
        net = bucket['net_improved_pixels']
        selector_rows.append(dict(
            dataset_name=dataset,
            rule_name=rule_name,
            images=int(bucket['images']),
            valid_pixels=int(valid),
            baseline_accuracy=safe_div(baseline_correct, valid),
            counterfactual_accuracy=safe_div(
                baseline_correct + net, valid),
            accuracy_delta=safe_div(net, valid),
            changed_pixels=int(changed),
            changed_ratio=safe_div(changed, valid),
            changed_precision=safe_div(
                bucket['improved_pixels'], changed),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            neutral_wrong_to_wrong_pixels=int(
                bucket['neutral_wrong_to_wrong_pixels']),
            net_improved_pixels=int(net),
        ))

    pair_rows = []
    for key, bucket in sorted(pair_buckets.items()):
        (dataset, candidate_rank, base_idx, base_name,
         candidate_idx, candidate_name) = key
        pixels = bucket['pixels']
        row = dict(
            dataset_name=dataset,
            candidate_rank=candidate_rank,
            base_class_index=base_idx,
            base_class_name=base_name,
            candidate_class_index=candidate_idx,
            candidate_class_name=candidate_name,
            images=int(bucket['images']),
            pixels=int(pixels),
            candidate_correct_pixels=int(
                bucket['candidate_correct_pixels']),
            candidate_correct_ratio=safe_div(
                bucket['candidate_correct_pixels'], pixels),
            baseline_correct_ratio=safe_div(
                bucket['baseline_correct_pixels'], pixels),
        )
        family_names = sorted({
            key_name[len('family_'):-len('_valid_pixels')]
            for key_name in bucket
            if (key_name.startswith('family_')
                and key_name.endswith('_valid_pixels'))
        })
        for family_name in family_names:
            prefix = f'family_{family_name}_'
            family_valid = bucket[prefix + 'valid_pixels']
            row[prefix + 'valid_pixels'] = int(family_valid)
            row[prefix + 'mean_delta'] = safe_div(
                bucket[prefix + 'delta_sum'], family_valid)
            row[prefix + 'favored_ratio'] = safe_div(
                bucket[prefix + 'favored_pixels'], family_valid)
        pair_rows.append(row)

    oracle_rows = []
    for (dataset,), bucket in sorted(oracle_buckets.items()):
        wrong = bucket['baseline_wrong_pixels']
        candidates = bucket['gt_in_candidate_pixels']
        oracle_rows.append(dict(
            dataset_name=dataset,
            images=int(bucket['images']),
            valid_pixels=int(bucket['valid_pixels']),
            baseline_wrong_pixels=int(wrong),
            gt_in_candidate_pixels=int(candidates),
            gt_in_candidate_ratio=safe_div(candidates, wrong),
            head_support_ratio=safe_div(
                bucket['head_support_pixels'], candidates),
            readout_support_ratio=safe_div(
                bucket['readout_support_pixels'], candidates),
            latent_support_ratio=safe_div(
                bucket['latent_support_pixels'], candidates),
            expanded_support_ratio=safe_div(
                bucket['expanded_support_pixels'], candidates),
            latent_incremental_ratio=safe_div(
                bucket['latent_incremental_pixels'], candidates),
            latent_incremental_over_all_errors_ratio=safe_div(
                bucket['latent_incremental_pixels'], wrong),
        ))

    null_rows = []
    for key, bucket in sorted(null_buckets.items()):
        dataset, control_name, trial = key
        candidates = bucket['gt_in_candidate_pixels']
        null_rows.append(dict(
            dataset_name=dataset,
            control_name=control_name,
            trial=trial,
            images=int(bucket['images']),
            gt_in_candidate_pixels=int(candidates),
            latent_support_pixels=int(bucket['latent_support_pixels']),
            latent_support_ratio=safe_div(
                bucket['latent_support_pixels'], candidates),
            latent_incremental_pixels=int(
                bucket['latent_incremental_pixels']),
            latent_incremental_ratio=safe_div(
                bucket['latent_incremental_pixels'], candidates),
        ))

    regime_rows = []
    for (dataset, regime_name), bucket in sorted(regime_buckets.items()):
        wrong = bucket['baseline_wrong_pixels']
        candidates = bucket['gt_in_candidate_pixels']
        regime_rows.append(dict(
            dataset_name=dataset,
            regime_name=regime_name,
            images=int(bucket['images']),
            baseline_wrong_pixels=int(wrong),
            gt_in_candidate_pixels=int(candidates),
            gt_in_candidate_ratio=safe_div(candidates, wrong),
            head_support_ratio=safe_div(
                bucket['head_support_pixels'], candidates),
            readout_support_ratio=safe_div(
                bucket['readout_support_pixels'], candidates),
            latent_support_ratio=safe_div(
                bucket['latent_support_pixels'], candidates),
            expanded_support_ratio=safe_div(
                bucket['expanded_support_pixels'], candidates),
            latent_incremental_ratio=safe_div(
                bucket['latent_incremental_pixels'], candidates),
        ))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(args.out_dir, 'candidate_family_sources.csv'),
        family_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_rank_summary.csv'),
        candidate_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_feature_summary.csv'),
        feature_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_selector_summary.csv'),
        selector_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_oracle_summary.csv'),
        oracle_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_null_summary.csv'),
        null_rows)
    write_csv(
        os.path.join(args.out_dir, 'candidate_regime_summary.csv'),
        regime_rows)

    print(f'Read {len(paths)} files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
