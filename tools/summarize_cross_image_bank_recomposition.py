#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


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


def main():
    parser = argparse.ArgumentParser(
        description='Summarize cross-image bank recomposition stats.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    overall = defaultdict(lambda: dict(
        images=0,
        valid_pixels=0,
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
    ))
    pair_summary = defaultdict(lambda: dict(
        pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        wrong_to_wrong_pixels=0,
    ))

    for record in load_records(args.inputs):
        ctx = record.get('cross_image_bank_recomposition') or {}
        dataset = record.get('dataset_name') or 'unknown'
        key = (
            dataset,
            record.get('cross_image_bank_apply_space', ''),
            record.get('cross_image_bank_apply_variant', ''),
            str(record.get('cross_image_bank_apply_min_bank_margin', '')),
            str(record.get('cross_image_bank_apply_max_base_gap', '')),
        )
        bucket = overall[key]
        bucket['images'] += 1
        bucket['valid_pixels'] += int(ctx.get('valid_pixels', 0) or 0)
        bucket['changed_pixels'] += int(
            ctx.get('final_changed_pixels', 0) or 0)
        bucket['improved_pixels'] += int(ctx.get('improved_pixels', 0) or 0)
        bucket['harmed_pixels'] += int(ctx.get('harmed_pixels', 0) or 0)
        bucket['wrong_to_wrong_pixels'] += int(
            ctx.get('wrong_to_wrong_pixels', 0) or 0)
        for pair in ctx.get('changed_pairs') or []:
            pair_key = key + (
                pair.get('from_class_index'),
                pair.get('from_class_name'),
                pair.get('to_class_index'),
                pair.get('to_class_name'),
            )
            pair_bucket = pair_summary[pair_key]
            pair_bucket['pixels'] += int(pair.get('pixels', 0) or 0)
            pair_bucket['improved_pixels'] += int(
                pair.get('improved_pixels', 0) or 0)
            pair_bucket['harmed_pixels'] += int(
                pair.get('harmed_pixels', 0) or 0)
            pair_bucket['wrong_to_wrong_pixels'] += int(
                pair.get('wrong_to_wrong_pixels', 0) or 0)

    overall_path = os.path.join(
        args.output_dir, 'cross_image_bank_recompose_overall.csv')
    with open(overall_path, 'w', newline='') as f:
        fields = [
            'dataset', 'space', 'variant', 'min_bank_margin',
            'max_base_gap', 'images', 'valid_pixels', 'changed_pixels',
            'changed_ratio', 'improved_pixels', 'harmed_pixels',
            'wrong_to_wrong_pixels', 'net_pixels', 'correction_precision',
            'harm_rate',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, bucket in sorted(overall.items()):
            dataset, space, variant, min_margin, max_gap = key
            changed = bucket['changed_pixels']
            writer.writerow(dict(
                dataset=dataset,
                space=space,
                variant=variant,
                min_bank_margin=min_margin,
                max_base_gap=max_gap,
                images=bucket['images'],
                valid_pixels=bucket['valid_pixels'],
                changed_pixels=changed,
                changed_ratio=safe_div(changed, bucket['valid_pixels']),
                improved_pixels=bucket['improved_pixels'],
                harmed_pixels=bucket['harmed_pixels'],
                wrong_to_wrong_pixels=bucket['wrong_to_wrong_pixels'],
                net_pixels=(
                    bucket['improved_pixels'] - bucket['harmed_pixels']),
                correction_precision=safe_div(
                    bucket['improved_pixels'], changed),
                harm_rate=safe_div(bucket['harmed_pixels'], changed),
            ))

    pair_path = os.path.join(
        args.output_dir, 'cross_image_bank_recompose_pairs.csv')
    with open(pair_path, 'w', newline='') as f:
        fields = [
            'dataset', 'space', 'variant', 'min_bank_margin',
            'max_base_gap', 'from_class_index', 'from_class_name',
            'to_class_index', 'to_class_name', 'pixels',
            'improved_pixels', 'harmed_pixels', 'wrong_to_wrong_pixels',
            'net_pixels', 'correction_precision', 'harm_rate',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, bucket in sorted(pair_summary.items()):
            (
                dataset, space, variant, min_margin, max_gap,
                from_idx, from_name, to_idx, to_name,
            ) = key
            pixels = bucket['pixels']
            writer.writerow(dict(
                dataset=dataset,
                space=space,
                variant=variant,
                min_bank_margin=min_margin,
                max_base_gap=max_gap,
                from_class_index=from_idx,
                from_class_name=from_name,
                to_class_index=to_idx,
                to_class_name=to_name,
                pixels=pixels,
                improved_pixels=bucket['improved_pixels'],
                harmed_pixels=bucket['harmed_pixels'],
                wrong_to_wrong_pixels=bucket['wrong_to_wrong_pixels'],
                net_pixels=(
                    bucket['improved_pixels'] - bucket['harmed_pixels']),
                correction_precision=safe_div(
                    bucket['improved_pixels'], pixels),
                harm_rate=safe_div(bucket['harmed_pixels'], pixels),
            ))

    print(f'Wrote {overall_path}')
    print(f'Wrote {pair_path}')


if __name__ == '__main__':
    main()
