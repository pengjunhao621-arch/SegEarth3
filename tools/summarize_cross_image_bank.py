#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def load_records(inputs):
    paths = []
    for pattern in inputs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def vec(values):
    return np.asarray(values, dtype=np.float64)


def normalize(x):
    if x is None:
        return None
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        return None
    return x / norm


def cosine(a, b):
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def transformed_vector(vector, entry, variant):
    if vector is None:
        return None
    if variant == 'raw':
        return vector
    if variant == 'image_centered':
        return vector - entry['image_mean']
    if variant == 'seed_centered':
        return vector - entry['seed_mean']
    if variant == 'class_balanced_centered':
        return vector - entry['class_balanced_seed_mean']
    raise ValueError(f'Unknown variant: {variant}')


def add_summary(bucket, pixels, margin, baseline_gap, image_id):
    bucket['pair_pixels'] += pixels
    bucket['weighted_margin_sum'] += margin * pixels
    bucket['weighted_baseline_gap_sum'] += baseline_gap * pixels
    bucket['positive_margin_pixels'] += pixels if margin > 0 else 0
    if 'images' in bucket:
        bucket['images'].add(image_id)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize cross-image clean-seed concept bank diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = defaultdict(lambda: defaultdict(list))
    class_coverage = defaultdict(lambda: dict(
        images=0,
        images_with_seed=0,
        seed_pixels=0,
        weighted_purity_sum=0.0,
    ))

    for record in load_records(args.inputs):
        stats = record.get('cross_image_bank_stats') or {}
        if not stats:
            continue
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        image_id = record.get('img_path') or f'{dataset}_{len(datasets[dataset])}'
        class_names = stats.get('class_names') or []
        for space_row in stats.get('spaces') or []:
            space = space_row.get('space')
            if not space:
                continue
            entry = dict(
                dataset=dataset,
                image_id=image_id,
                class_names=class_names,
                space=space,
                image_mean=vec(space_row.get('image_mean') or []),
                seed_mean=vec(space_row.get('seed_mean') or []),
                class_balanced_seed_mean=vec(
                    space_row.get('class_balanced_seed_mean') or []),
                prototypes={},
                prototype_weights={},
                prototype_purities={},
                pairs=[],
            )
            for proto_row in space_row.get('class_prototypes') or []:
                class_idx = int(proto_row.get('class_index'))
                proto = vec(proto_row.get('prototype') or [])
                if proto.size == 0:
                    continue
                weight = int(proto_row.get('seed_pixels', 0) or 0)
                purity = float(proto_row.get('seed_purity', 0.0) or 0.0)
                entry['prototypes'][class_idx] = proto
                entry['prototype_weights'][class_idx] = weight
                entry['prototype_purities'][class_idx] = purity
                cov = class_coverage[(dataset, space, class_idx)]
                cov['images'] += 1
                cov['images_with_seed'] += 1
                cov['seed_pixels'] += weight
                cov['weighted_purity_sum'] += purity * weight
                cov['class_name'] = proto_row.get('class_name', '')
            # Count images without seed for classes too, once class names are known.
            for class_idx, class_name in enumerate(class_names):
                cov = class_coverage[(dataset, space, class_idx)]
                cov.setdefault('class_name', class_name)
                if class_idx not in entry['prototypes']:
                    cov['images'] += 1

            for pair_row in space_row.get('pair_features') or []:
                pair_vec = vec(pair_row.get('pair_feature') or [])
                if pair_vec.size == 0:
                    continue
                entry['pairs'].append(dict(
                    gt_class_index=int(pair_row.get('gt_class_index')),
                    gt_class_name=pair_row.get('gt_class_name', ''),
                    pred_class_index=int(pair_row.get('pred_class_index')),
                    pred_class_name=pair_row.get('pred_class_name', ''),
                    pixels=int(pair_row.get('pixels', 0) or 0),
                    baseline_gap=float(
                        pair_row.get('mean_final_pred_gt_margin', 0.0) or 0.0),
                    pair_feature=pair_vec,
                ))
            datasets[dataset][space].append(entry)

    dataset_summary = defaultdict(lambda: dict(
        pair_pixels=0,
        covered_pixels=0,
        positive_margin_pixels=0,
        weighted_margin_sum=0.0,
        weighted_baseline_gap_sum=0.0,
        images=set(),
    ))
    pair_summary = defaultdict(lambda: dict(
        pair_pixels=0,
        covered_pixels=0,
        positive_margin_pixels=0,
        weighted_margin_sum=0.0,
        weighted_baseline_gap_sum=0.0,
        images=set(),
    ))
    image_summary = defaultdict(lambda: dict(
        pair_pixels=0,
        covered_pixels=0,
        positive_margin_pixels=0,
        weighted_margin_sum=0.0,
        weighted_baseline_gap_sum=0.0,
    ))

    transform_variants = [
        'raw',
        'image_centered',
        'seed_centered',
        'class_balanced_centered',
    ]
    bank_modes = ['self', 'leave_one']

    for dataset, by_space in datasets.items():
        for space, entries in by_space.items():
            totals = {
                variant: defaultdict(lambda: dict(sum=None, weight=0.0))
                for variant in transform_variants
            }
            contribs = {
                variant: defaultdict(
                    lambda: defaultdict(lambda: dict(sum=None, weight=0.0)))
                for variant in transform_variants
            }
            for entry in entries:
                image_id = entry['image_id']
                for class_idx, proto in entry['prototypes'].items():
                    weight = float(entry['prototype_weights'].get(class_idx, 0))
                    if weight <= 0:
                        continue
                    for variant in transform_variants:
                        transformed = transformed_vector(proto, entry, variant)
                        if transformed is None or transformed.size == 0:
                            continue
                        weighted = transformed * weight
                        total = totals[variant][class_idx]
                        total['sum'] = (
                            weighted.copy()
                            if total['sum'] is None
                            else total['sum'] + weighted)
                        total['weight'] += weight
                        contrib = contribs[variant][image_id][class_idx]
                        contrib['sum'] = (
                            weighted.copy()
                            if contrib['sum'] is None
                            else contrib['sum'] + weighted)
                        contrib['weight'] += weight

            for entry in entries:
                image_id = entry['image_id']
                for pair in entry['pairs']:
                    pixels = int(pair['pixels'])
                    if pixels <= 0:
                        continue
                    gt_idx = int(pair['gt_class_index'])
                    pred_idx = int(pair['pred_class_index'])
                    baseline_gap = float(pair['baseline_gap'])
                    for variant in transform_variants:
                        pair_feature = transformed_vector(
                            pair['pair_feature'], entry, variant)
                        if pair_feature is None or pair_feature.size == 0:
                            continue
                        for bank_mode in bank_modes:
                            if bank_mode == 'self':
                                gt_bank = transformed_vector(
                                    entry['prototypes'].get(gt_idx),
                                    entry,
                                    variant)
                                pred_bank = transformed_vector(
                                    entry['prototypes'].get(pred_idx),
                                    entry,
                                    variant)
                            else:
                                gt_total = totals[variant].get(gt_idx)
                                pred_total = totals[variant].get(pred_idx)
                                gt_contrib = contribs[variant][image_id].get(
                                    gt_idx, dict(sum=None, weight=0.0))
                                pred_contrib = contribs[variant][image_id].get(
                                    pred_idx, dict(sum=None, weight=0.0))
                                if (
                                        not gt_total or not pred_total
                                        or gt_total['sum'] is None
                                        or pred_total['sum'] is None):
                                    continue
                                gt_weight = gt_total['weight'] - gt_contrib['weight']
                                pred_weight = (
                                    pred_total['weight']
                                    - pred_contrib['weight'])
                                if gt_weight <= 0 or pred_weight <= 0:
                                    continue
                                gt_sum = gt_total['sum'].copy()
                                pred_sum = pred_total['sum'].copy()
                                if gt_contrib['sum'] is not None:
                                    gt_sum = gt_sum - gt_contrib['sum']
                                if pred_contrib['sum'] is not None:
                                    pred_sum = pred_sum - pred_contrib['sum']
                                gt_bank = gt_sum / gt_weight
                                pred_bank = pred_sum / pred_weight

                            if gt_bank is None or pred_bank is None:
                                continue
                            gt_sim = cosine(pair_feature, gt_bank)
                            pred_sim = cosine(pair_feature, pred_bank)
                            if gt_sim is None or pred_sim is None:
                                continue
                            margin = gt_sim - pred_sim
                            variant_name = f'{bank_mode}_{variant}'
                            ds_key = (dataset, space, variant_name)
                            pair_key = (
                                dataset, space, variant_name,
                                gt_idx, pair['gt_class_name'],
                                pred_idx, pair['pred_class_name'])
                            img_key = (dataset, space, variant_name, image_id)
                            dataset_summary[ds_key]['covered_pixels'] += pixels
                            pair_summary[pair_key]['covered_pixels'] += pixels
                            image_summary[img_key]['covered_pixels'] += pixels
                            add_summary(
                                dataset_summary[ds_key],
                                pixels,
                                margin,
                                baseline_gap,
                                image_id,
                            )
                            add_summary(
                                pair_summary[pair_key],
                                pixels,
                                margin,
                                baseline_gap,
                                image_id,
                            )
                            add_summary(
                                image_summary[img_key],
                                pixels,
                                margin,
                                baseline_gap,
                                image_id,
                            )

    dataset_path = os.path.join(
        args.output_dir, 'cross_image_bank_dataset_summary.csv')
    pair_path = os.path.join(
        args.output_dir, 'cross_image_bank_pair_summary.csv')
    class_path = os.path.join(
        args.output_dir, 'cross_image_bank_class_coverage.csv')
    image_path = os.path.join(
        args.output_dir, 'cross_image_bank_image_summary.csv')

    with open(dataset_path, 'w', newline='') as f:
        fields = [
            'dataset', 'space', 'bank_variant', 'images',
            'covered_pair_pixels', 'mean_bank_margin',
            'positive_margin_pixel_ratio', 'mean_baseline_pred_gt_margin',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, bucket in sorted(dataset_summary.items()):
            dataset, space, variant = key
            pixels = bucket['covered_pixels']
            writer.writerow(dict(
                dataset=dataset,
                space=space,
                bank_variant=variant,
                images=len(bucket['images']),
                covered_pair_pixels=pixels,
                mean_bank_margin=safe_div(
                    bucket['weighted_margin_sum'], pixels),
                positive_margin_pixel_ratio=safe_div(
                    bucket['positive_margin_pixels'], pixels),
                mean_baseline_pred_gt_margin=safe_div(
                    bucket['weighted_baseline_gap_sum'], pixels),
            ))

    with open(pair_path, 'w', newline='') as f:
        fields = [
            'dataset', 'space', 'bank_variant',
            'gt_class_index', 'gt_class_name',
            'pred_class_index', 'pred_class_name',
            'images', 'covered_pair_pixels', 'mean_bank_margin',
            'positive_margin_pixel_ratio', 'mean_baseline_pred_gt_margin',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, bucket in sorted(pair_summary.items()):
            (
                dataset, space, variant,
                gt_idx, gt_name, pred_idx, pred_name,
            ) = key
            pixels = bucket['covered_pixels']
            writer.writerow(dict(
                dataset=dataset,
                space=space,
                bank_variant=variant,
                gt_class_index=gt_idx,
                gt_class_name=gt_name,
                pred_class_index=pred_idx,
                pred_class_name=pred_name,
                images=len(bucket['images']),
                covered_pair_pixels=pixels,
                mean_bank_margin=safe_div(
                    bucket['weighted_margin_sum'], pixels),
                positive_margin_pixel_ratio=safe_div(
                    bucket['positive_margin_pixels'], pixels),
                mean_baseline_pred_gt_margin=safe_div(
                    bucket['weighted_baseline_gap_sum'], pixels),
            ))

    with open(class_path, 'w', newline='') as f:
        fields = [
            'dataset', 'space', 'class_index', 'class_name',
            'images', 'images_with_seed', 'image_seed_coverage',
            'seed_pixels', 'mean_seed_purity',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, bucket in sorted(class_coverage.items()):
            dataset, space, class_idx = key
            writer.writerow(dict(
                dataset=dataset,
                space=space,
                class_index=class_idx,
                class_name=bucket.get('class_name', ''),
                images=bucket['images'],
                images_with_seed=bucket['images_with_seed'],
                image_seed_coverage=safe_div(
                    bucket['images_with_seed'], bucket['images']),
                seed_pixels=bucket['seed_pixels'],
                mean_seed_purity=safe_div(
                    bucket['weighted_purity_sum'],
                    bucket['seed_pixels']),
            ))

    with open(image_path, 'w', newline='') as f:
        fields = [
            'dataset', 'space', 'bank_variant', 'image_id',
            'covered_pair_pixels', 'mean_bank_margin',
            'positive_margin_pixel_ratio', 'mean_baseline_pred_gt_margin',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, bucket in sorted(image_summary.items()):
            dataset, space, variant, image_id = key
            pixels = bucket['covered_pixels']
            writer.writerow(dict(
                dataset=dataset,
                space=space,
                bank_variant=variant,
                image_id=image_id,
                covered_pair_pixels=pixels,
                mean_bank_margin=safe_div(
                    bucket['weighted_margin_sum'], pixels),
                positive_margin_pixel_ratio=safe_div(
                    bucket['positive_margin_pixels'], pixels),
                mean_baseline_pred_gt_margin=safe_div(
                    bucket['weighted_baseline_gap_sum'], pixels),
            ))

    print(f'Wrote {dataset_path}')
    print(f'Wrote {pair_path}')
    print(f'Wrote {class_path}')
    print(f'Wrote {image_path}')


if __name__ == '__main__':
    main()
