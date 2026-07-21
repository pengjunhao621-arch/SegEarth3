import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


PE_PATTERN = re.compile(r'^posbias_l(?P<layer>\d+)_pe_r(?P<rank>\d+)$')
CONTROL_PATTERN = re.compile(
    r'^posbias_l(?P<layer>\d+)_(?P<control>perm|rand)_'
    r'r(?P<rank>\d+)_t(?P<trial>\d+)$')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize SAM3 explicit-position-subspace energy, head '
            'alignment, and matched candidate-separability diagnostics.'))
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
    return numerator / denominator


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as file:
            file.write('')
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_weighted_bucket():
    return dict(
        images=set(),
        rows=0,
        weight=0,
        sums=defaultdict(float),
        valid_weights=defaultdict(float),
    )


def add_weighted(bucket, image_id, row, names):
    weight = max(1, int(row.get('spatial_pixels') or 1))
    bucket['images'].add(image_id)
    bucket['rows'] += 1
    bucket['weight'] += weight
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        bucket['sums'][name] += float(value) * weight
        bucket['valid_weights'][name] += weight


def weighted_row(bucket, names):
    row = dict(
        images=len(bucket['images']),
        rows=int(bucket['rows']),
        spatial_weight=int(bucket['weight']),
    )
    for name in names:
        row[f'mean_{name}'] = safe_div(
            bucket['sums'][name],
            bucket['valid_weights'][name],
        )
    return row


def new_metric_bucket():
    return dict(
        images=set(),
        strata=0,
        positive_pixels=0,
        negative_pixels=0,
        comparison_weight=0,
        auroc_weighted=0.0,
        ap_weight=0,
        auprc_weighted=0.0,
        auprc_lift_weighted=0.0,
    )


def add_metric(bucket, image_id, metric):
    positive = int(metric.get('positive_pixels') or 0)
    negative = int(metric.get('negative_pixels') or 0)
    comparison_weight = int(metric.get('comparison_weight') or 0)
    bucket['images'].add(image_id)
    bucket['strata'] += 1
    bucket['positive_pixels'] += positive
    bucket['negative_pixels'] += negative
    bucket['comparison_weight'] += comparison_weight
    if metric.get('auroc') is not None:
        bucket['auroc_weighted'] += (
            float(metric['auroc']) * comparison_weight)
    if metric.get('auprc') is not None:
        bucket['auprc_weighted'] += float(metric['auprc']) * positive
        bucket['auprc_lift_weighted'] += (
            float(metric.get('auprc_lift') or 0.0) * positive)
        bucket['ap_weight'] += positive


def metric_row(bucket):
    return dict(
        images=len(bucket['images']),
        matched_strata=int(bucket['strata']),
        positive_pixels=int(bucket['positive_pixels']),
        negative_pixels=int(bucket['negative_pixels']),
        matched_auroc=safe_div(
            bucket['auroc_weighted'],
            bucket['comparison_weight'],
        ),
        matched_auprc=safe_div(
            bucket['auprc_weighted'],
            bucket['ap_weight'],
        ),
        matched_auprc_lift=safe_div(
            bucket['auprc_lift_weighted'],
            bucket['ap_weight'],
        ),
    )


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')

    energy_names = (
        'total_feature_energy',
        'removed_feature_energy_ratio',
        'position_variance_explained_ratio',
    )
    alignment_names = (
        'mean_explained_ratio',
        'max_explained_ratio',
    )
    energy = defaultdict(new_weighted_bucket)
    alignment = defaultdict(new_weighted_bucket)
    matched = defaultdict(new_metric_bucket)
    source_inventory = defaultdict(lambda: dict(images=set(), count=0))

    for record_index, record in enumerate(iter_records(paths)):
        dataset = str(record.get('dataset_name') or 'unknown')
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{record_index}")
        for row in record.get('position_bias_feature_stats') or []:
            common = (
                dataset,
                row.get('layer_index'),
                row.get('rank'),
                row.get('variant'),
                row.get('trial'),
            )
            if row.get('stat_type') == 'feature_projection':
                add_weighted(
                    energy[common], image_id, row, energy_names)
            elif row.get('stat_type') == 'head_alignment':
                key = common + (row.get('head_name'),)
                add_weighted(
                    alignment[key], image_id, row, alignment_names)

        stats = record.get('position_bias_candidate_stats') or {}
        for family, sources in (
                stats.get('family_sources') or {}).items():
            if not str(family).startswith('posbias_'):
                continue
            key = (dataset, family, '|'.join(sources or []))
            source_inventory[key]['images'].add(image_id)
            source_inventory[key]['count'] += 1

        for row in stats.get('matched_stats') or []:
            family = str(row.get('family_name') or '')
            if not family.startswith('posbias_'):
                continue
            if row.get('control_name') != 'observed':
                continue
            for task_name, metric in (
                    row.get('task_metrics') or {}).items():
                common = (
                    dataset,
                    family,
                    row.get('candidate_rank'),
                    row.get('regime_name'),
                    row.get('margin_name'),
                    task_name,
                )
                add_metric(matched[common], image_id, metric)
                all_margin = common[:4] + (
                    'matched_all_margins', task_name)
                add_metric(matched[all_margin], image_id, metric)

    energy_rows = []
    for key, bucket in sorted(energy.items()):
        dataset, layer, rank, variant, trial = key
        row = dict(
            dataset_name=dataset,
            layer_index=layer,
            rank=rank,
            variant=variant,
            trial=trial,
        )
        row.update(weighted_row(bucket, energy_names))
        energy_rows.append(row)

    alignment_rows = []
    for key, bucket in sorted(alignment.items()):
        dataset, layer, rank, variant, trial, head = key
        row = dict(
            dataset_name=dataset,
            layer_index=layer,
            rank=rank,
            variant=variant,
            trial=trial,
            head_name=head,
        )
        row.update(weighted_row(bucket, alignment_names))
        alignment_rows.append(row)

    matched_rows = []
    matched_index = {}
    for key, bucket in sorted(matched.items()):
        dataset, family, rank, regime, margin_name, task = key
        row = dict(
            dataset_name=dataset,
            family_name=family,
            candidate_rank=rank,
            regime_name=regime,
            margin_name=margin_name,
            task_name=task,
        )
        row.update(metric_row(bucket))
        matched_rows.append(row)
        matched_index[key] = row

    gain_rows = []
    for key, row in sorted(matched_index.items()):
        dataset, family, candidate_rank, regime, margin_name, task = key
        match = PE_PATTERN.match(family)
        if match is None:
            continue
        layer = int(match.group('layer'))
        projection_rank = int(match.group('rank'))
        centered_family = f'posbias_l{layer}_centered'
        centered = matched_index.get((
            dataset,
            centered_family,
            candidate_rank,
            regime,
            margin_name,
            task,
        ))
        controls = []
        for other_key, other_row in matched_index.items():
            (
                other_dataset, other_family, other_candidate_rank,
                other_regime, other_margin, other_task,
            ) = other_key
            control_match = CONTROL_PATTERN.match(other_family)
            if control_match is None:
                continue
            if (
                    other_dataset == dataset
                    and other_candidate_rank == candidate_rank
                    and other_regime == regime
                    and other_margin == margin_name
                    and other_task == task
                    and int(control_match.group('layer')) == layer
                    and int(control_match.group('rank')) == projection_rank):
                controls.append(other_row)
        control_aucs = [
            item['matched_auroc'] for item in controls
            if item.get('matched_auroc') is not None
        ]
        control_ap_lifts = [
            item['matched_auprc_lift'] for item in controls
            if item.get('matched_auprc_lift') is not None
        ]
        gain_rows.append(dict(
            dataset_name=dataset,
            layer_index=layer,
            projection_rank=projection_rank,
            candidate_rank=candidate_rank,
            regime_name=regime,
            margin_name=margin_name,
            task_name=task,
            pe_auroc=row.get('matched_auroc'),
            centered_auroc=(
                None if centered is None
                else centered.get('matched_auroc')),
            pe_minus_centered_auroc=(
                None if centered is None
                or row.get('matched_auroc') is None
                or centered.get('matched_auroc') is None
                else row['matched_auroc']
                - centered['matched_auroc']),
            best_control_auroc=(
                max(control_aucs) if control_aucs else None),
            pe_minus_best_control_auroc=(
                None if not control_aucs
                or row.get('matched_auroc') is None
                else row['matched_auroc'] - max(control_aucs)),
            pe_auprc_lift=row.get('matched_auprc_lift'),
            centered_auprc_lift=(
                None if centered is None
                else centered.get('matched_auprc_lift')),
            best_control_auprc_lift=(
                max(control_ap_lifts)
                if control_ap_lifts else None),
        ))

    inventory_rows = []
    for key, bucket in sorted(source_inventory.items()):
        dataset, family, sources = key
        inventory_rows.append(dict(
            dataset_name=dataset,
            family_name=family,
            source_names=sources,
            images=len(bucket['images']),
            records=int(bucket['count']),
        ))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(args.out_dir, 'position_bias_energy_summary.csv'),
        energy_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'position_bias_head_alignment_summary.csv'),
        alignment_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'position_bias_matched_summary.csv'),
        matched_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'position_bias_gain_summary.csv'),
        gain_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'position_bias_source_inventory.csv'),
        inventory_rows)

    print(f'Read {len(paths)} files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
