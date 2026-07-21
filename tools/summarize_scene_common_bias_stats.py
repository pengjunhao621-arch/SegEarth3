import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict


FAMILY_PATTERN = re.compile(
    r'^scenecommon_l(?P<layer>\d+)_(?P<variant>.+)$')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize scene-common feature bias, candidate separability, '
            'and no-GT image-gating diagnostics.'))
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


def add_weighted(bucket, image_id, row, names, weight_name='spatial_pixels'):
    weight = max(1, int(row.get(weight_name) or 1))
    bucket['images'].add(image_id)
    bucket['rows'] += 1
    bucket['weight'] += weight
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        bucket['sums'][name] += float(value) * weight
        bucket['valid_weights'][name] += weight


def weighted_values(bucket, names):
    result = dict(
        images=len(bucket['images']),
        rows=int(bucket['rows']),
        spatial_weight=int(bucket['weight']),
    )
    for name in names:
        result[f'mean_{name}'] = safe_div(
            bucket['sums'][name],
            bucket['valid_weights'][name],
        )
    return result


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


def metric_values(bucket):
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


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def pearson(x_values, y_values):
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )
    x_energy = sum((x - x_mean) ** 2 for x in x_values)
    y_energy = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_energy * y_energy)
    return safe_div(numerator, denominator)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No input JSONL files matched.')

    feature_names = (
        'total_feature_energy',
        'mean_vector_norm',
        'mean_token_norm',
        'mean_norm_ratio',
        'common_energy_ratio',
        'mean_token_cosine',
        'effective_rank',
        'top1_variance_ratio',
        'top4_variance_ratio',
        'top8_variance_ratio',
        'removed_feature_energy_ratio',
        'residual_mean_norm_ratio',
        'seed_ratio',
        'seed_class_count',
        'mean_interclass_seed_cosine',
        'seed_prototype_dispersion',
        'robust_tokens',
        'balanced_seed_classes',
        'shared_layer_count',
        'local_kernel',
    )
    descriptor_names = (
        'mean_norm_ratio',
        'common_energy_ratio',
        'mean_token_cosine',
        'effective_rank',
        'top1_variance_ratio',
        'top4_variance_ratio',
        'top8_variance_ratio',
        'seed_ratio',
        'seed_class_count',
        'seed_prototype_dispersion',
    )

    feature_summary = defaultdict(new_weighted_bucket)
    image_descriptors = defaultdict(new_weighted_bucket)
    matched_summary = defaultdict(new_metric_bucket)
    image_metrics = defaultdict(new_metric_bucket)
    pair_summary = defaultdict(new_metric_bucket)
    source_inventory = defaultdict(lambda: dict(images=set(), records=0))

    for record_index, record in enumerate(iter_records(paths)):
        dataset = str(record.get('dataset_name') or 'unknown')
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{record_index}")
        for row in record.get('scene_common_feature_stats') or []:
            layer = row.get('layer_index')
            variant_name = str(row.get('variant_name') or '')
            match = FAMILY_PATTERN.match(variant_name)
            if match is None:
                continue
            variant = match.group('variant')
            stat_type = str(row.get('stat_type') or 'unknown')
            key = (
                dataset,
                layer,
                variant,
                stat_type,
                row.get('variant_kind'),
                row.get('strength'),
            )
            add_weighted(
                feature_summary[key],
                image_id,
                row,
                feature_names,
            )
            if stat_type == 'layer_descriptor' and variant == 'raw':
                add_weighted(
                    image_descriptors[(dataset, image_id, layer)],
                    image_id,
                    row,
                    descriptor_names,
                )

        stats = record.get('scene_common_candidate_stats') or {}
        for family, sources in (
                stats.get('family_sources') or {}).items():
            if not str(family).startswith('scenecommon_'):
                continue
            key = (dataset, family, '|'.join(sources or []))
            source_inventory[key]['images'].add(image_id)
            source_inventory[key]['records'] += 1

        for row in stats.get('matched_stats') or []:
            family = str(row.get('family_name') or '')
            match = FAMILY_PATTERN.match(family)
            if match is None or row.get('control_name') != 'observed':
                continue
            layer = int(match.group('layer'))
            variant = match.group('variant')
            candidate_rank = row.get('candidate_rank')
            regime = row.get('regime_name')
            margin = row.get('margin_name')
            pair = (
                row.get('base_class_name'),
                row.get('candidate_class_name'),
            )
            for task_name, metric in (
                    row.get('task_metrics') or {}).items():
                key = (
                    dataset,
                    layer,
                    variant,
                    candidate_rank,
                    regime,
                    margin,
                    task_name,
                )
                add_metric(matched_summary[key], image_id, metric)
                all_margin_key = (
                    dataset,
                    layer,
                    variant,
                    candidate_rank,
                    regime,
                    'matched_all_margins',
                    task_name,
                )
                add_metric(
                    matched_summary[all_margin_key], image_id, metric)
                image_key = (
                    dataset,
                    image_id,
                    layer,
                    variant,
                    candidate_rank,
                    regime,
                    task_name,
                )
                add_metric(image_metrics[image_key], image_id, metric)
                pair_key = (
                    dataset,
                    layer,
                    variant,
                    candidate_rank,
                    regime,
                    task_name,
                    pair[0],
                    pair[1],
                )
                add_metric(pair_summary[pair_key], image_id, metric)

    feature_rows = []
    for key, bucket in sorted(feature_summary.items()):
        (
            dataset, layer, variant, stat_type, variant_kind, strength,
        ) = key
        row = dict(
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            stat_type=stat_type,
            variant_kind=variant_kind,
            strength=strength,
        )
        row.update(weighted_values(bucket, feature_names))
        feature_rows.append(row)

    matched_rows = []
    matched_index = {}
    for key, bucket in sorted(matched_summary.items()):
        (
            dataset, layer, variant, candidate_rank, regime, margin, task,
        ) = key
        row = dict(
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            candidate_rank=candidate_rank,
            regime_name=regime,
            margin_name=margin,
            task_name=task,
        )
        row.update(metric_values(bucket))
        matched_rows.append(row)
        matched_index[key] = row

    gain_rows = []
    for key, row in sorted(matched_index.items()):
        (
            dataset, layer, variant, candidate_rank, regime, margin, task,
        ) = key
        if variant == 'raw':
            continue
        raw = matched_index.get((
            dataset,
            layer,
            'raw',
            candidate_rank,
            regime,
            margin,
            task,
        ))
        if raw is None:
            continue
        gain_rows.append(dict(
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            candidate_rank=candidate_rank,
            regime_name=regime,
            margin_name=margin,
            task_name=task,
            variant_auroc=row.get('matched_auroc'),
            raw_auroc=raw.get('matched_auroc'),
            variant_minus_raw_auroc=(
                None
                if row.get('matched_auroc') is None
                or raw.get('matched_auroc') is None
                else row['matched_auroc'] - raw['matched_auroc']
            ),
            variant_auprc_lift=row.get('matched_auprc_lift'),
            raw_auprc_lift=raw.get('matched_auprc_lift'),
            variant_minus_raw_auprc_lift=(
                None
                if row.get('matched_auprc_lift') is None
                or raw.get('matched_auprc_lift') is None
                else (
                    row['matched_auprc_lift']
                    - raw['matched_auprc_lift']
                )
            ),
        ))

    descriptor_index = {
        key: weighted_values(bucket, descriptor_names)
        for key, bucket in image_descriptors.items()
    }
    image_metric_index = {
        key: metric_values(bucket)
        for key, bucket in image_metrics.items()
    }
    image_rows = []
    for key, values in sorted(image_metric_index.items()):
        (
            dataset, image_id, layer, variant, candidate_rank, regime, task,
        ) = key
        if variant == 'raw':
            continue
        raw = image_metric_index.get((
            dataset,
            image_id,
            layer,
            'raw',
            candidate_rank,
            regime,
            task,
        ))
        descriptor = descriptor_index.get((dataset, image_id, layer))
        if raw is None or descriptor is None:
            continue
        row = dict(
            dataset_name=dataset,
            image_id=image_id,
            layer_index=layer,
            variant_name=variant,
            candidate_rank=candidate_rank,
            regime_name=regime,
            task_name=task,
        )
        for name in descriptor_names:
            row[name] = descriptor.get(f'mean_{name}')
        row.update(
            variant_auroc=values.get('matched_auroc'),
            raw_auroc=raw.get('matched_auroc'),
            variant_minus_raw_auroc=(
                None
                if values.get('matched_auroc') is None
                or raw.get('matched_auroc') is None
                else (
                    values['matched_auroc']
                    - raw['matched_auroc']
                )
            ),
            variant_auprc_lift=values.get('matched_auprc_lift'),
            raw_auprc_lift=raw.get('matched_auprc_lift'),
            variant_minus_raw_auprc_lift=(
                None
                if values.get('matched_auprc_lift') is None
                or raw.get('matched_auprc_lift') is None
                else (
                    values['matched_auprc_lift']
                    - raw['matched_auprc_lift']
                )
            ),
            positive_pixels=values.get('positive_pixels'),
            negative_pixels=values.get('negative_pixels'),
        )
        image_rows.append(row)

    gate_groups = defaultdict(list)
    for row in image_rows:
        base_group = (
            row['layer_index'],
            row['variant_name'],
            row['candidate_rank'],
            row['regime_name'],
            row['task_name'],
        )
        gate_groups[(row['dataset_name'],) + base_group].append(row)
        gate_groups[('all_datasets',) + base_group].append(row)

    gate_rows = []
    for group, rows in sorted(gate_groups.items()):
        (
            dataset, layer, variant, candidate_rank, regime, task,
        ) = group
        for descriptor_name in descriptor_names:
            usable = [
                row for row in rows
                if row.get(descriptor_name) is not None
                and row.get('variant_minus_raw_auroc') is not None
            ]
            if len(usable) < 4:
                continue
            x_values = [
                float(row[descriptor_name]) for row in usable]
            y_auc = [
                float(row['variant_minus_raw_auroc']) for row in usable]
            ap_usable = [
                row for row in usable
                if row.get('variant_minus_raw_auprc_lift') is not None
            ]
            x_ap = [
                float(row[descriptor_name]) for row in ap_usable]
            y_ap = [
                float(row['variant_minus_raw_auprc_lift'])
                for row in ap_usable
            ]
            gate_rows.append(dict(
                dataset_name=dataset,
                layer_index=layer,
                variant_name=variant,
                candidate_rank=candidate_rank,
                regime_name=regime,
                task_name=task,
                descriptor_name=descriptor_name,
                gate_side='correlation',
                quantile=None,
                threshold=None,
                selected_images=len(usable),
                selected_datasets=len({
                    row['dataset_name'] for row in usable}),
                mean_variant_minus_raw_auroc=(
                    sum(y_auc) / len(y_auc)),
                positive_auroc_gain_ratio=safe_div(
                    sum(value > 0 for value in y_auc),
                    len(y_auc),
                ),
                mean_variant_minus_raw_auprc_lift=(
                    None if not y_ap else sum(y_ap) / len(y_ap)),
                descriptor_gain_pearson=pearson(x_values, y_auc),
                descriptor_ap_gain_pearson=pearson(x_ap, y_ap),
            ))
            for probability in (0.25, 0.50, 0.75):
                threshold = quantile(x_values, probability)
                for side in ('low', 'high'):
                    if side == 'low':
                        selected = [
                            row for row in usable
                            if float(row[descriptor_name]) <= threshold]
                    else:
                        selected = [
                            row for row in usable
                            if float(row[descriptor_name]) >= threshold]
                    selected_auc = [
                        float(row['variant_minus_raw_auroc'])
                        for row in selected
                    ]
                    selected_ap = [
                        float(row['variant_minus_raw_auprc_lift'])
                        for row in selected
                        if row.get(
                            'variant_minus_raw_auprc_lift') is not None
                    ]
                    gate_rows.append(dict(
                        dataset_name=dataset,
                        layer_index=layer,
                        variant_name=variant,
                        candidate_rank=candidate_rank,
                        regime_name=regime,
                        task_name=task,
                        descriptor_name=descriptor_name,
                        gate_side=side,
                        quantile=probability,
                        threshold=threshold,
                        selected_images=len(selected),
                        selected_datasets=len({
                            row['dataset_name'] for row in selected}),
                        mean_variant_minus_raw_auroc=(
                            None
                            if not selected_auc
                            else sum(selected_auc) / len(selected_auc)
                        ),
                        positive_auroc_gain_ratio=safe_div(
                            sum(value > 0 for value in selected_auc),
                            len(selected_auc),
                        ),
                        mean_variant_minus_raw_auprc_lift=(
                            None
                            if not selected_ap
                            else sum(selected_ap) / len(selected_ap)
                        ),
                        descriptor_gain_pearson=None,
                        descriptor_ap_gain_pearson=None,
                    ))

    pair_rows = []
    for key, bucket in sorted(pair_summary.items()):
        (
            dataset, layer, variant, candidate_rank, regime, task,
            base_class, candidate_class,
        ) = key
        row = dict(
            dataset_name=dataset,
            layer_index=layer,
            variant_name=variant,
            candidate_rank=candidate_rank,
            regime_name=regime,
            task_name=task,
            base_class_name=base_class,
            candidate_class_name=candidate_class,
        )
        row.update(metric_values(bucket))
        pair_rows.append(row)

    inventory_rows = []
    for key, bucket in sorted(source_inventory.items()):
        dataset, family, sources = key
        inventory_rows.append(dict(
            dataset_name=dataset,
            family_name=family,
            source_names=sources,
            images=len(bucket['images']),
            records=int(bucket['records']),
        ))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_feature_summary.csv'),
        feature_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_matched_summary.csv'),
        matched_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_gain_summary.csv'),
        gain_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_image_summary.csv'),
        image_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_gate_summary.csv'),
        gate_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(
            args.out_dir, 'scene_common_source_inventory.csv'),
        inventory_rows)

    print(f'Read {len(paths)} files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
