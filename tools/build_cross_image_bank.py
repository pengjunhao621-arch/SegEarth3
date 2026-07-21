#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict


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
    return [float(value) for value in values]


def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]


def sub_vec(a, b):
    return [x - y for x, y in zip(a, b)]


def mul_vec(a, scalar):
    return [x * scalar for x in a]


def div_vec(a, scalar):
    return [x / scalar for x in a]


def transform(vector, centers, variant):
    if variant == 'raw':
        return vector
    if variant == 'image_centered':
        return sub_vec(vector, centers['image_mean'])
    if variant == 'seed_centered':
        return sub_vec(vector, centers['seed_mean'])
    if variant == 'class_balanced_centered':
        return sub_vec(vector, centers['class_balanced_seed_mean'])
    raise ValueError(f'Unknown variant: {variant}')


def rounded(values, decimals):
    return [round(float(value), decimals) for value in values]


def main():
    parser = argparse.ArgumentParser(
        description='Build no-GT cross-image clean-seed concept bank.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--spaces',
        default='pe_layer_0',
        help='Comma-separated spaces to include, e.g. pe_layer_0,pe_layer_1.')
    parser.add_argument(
        '--variants',
        default='raw,image_centered,seed_centered,class_balanced_centered')
    parser.add_argument('--decimals', type=int, default=6)
    args = parser.parse_args()

    spaces = {item.strip() for item in args.spaces.split(',') if item.strip()}
    variants = [
        item.strip() for item in args.variants.split(',') if item.strip()]
    banks = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: dict(
                    class_names=None,
                    totals=defaultdict(lambda: dict(sum=None, weight=0.0)),
                    image_contribs=defaultdict(
                        lambda: defaultdict(
                            lambda: dict(sum=None, weight=0.0))),
                ))))

    for record in load_records(args.inputs):
        stats = record.get('cross_image_bank_stats') or {}
        if not stats:
            continue
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        image_id = str(record.get('img_path') or '')
        class_names = stats.get('class_names') or []
        for space_row in stats.get('spaces') or []:
            space = space_row.get('space')
            if space not in spaces:
                continue
            centers = dict(
                image_mean=vec(space_row.get('image_mean') or []),
                seed_mean=vec(space_row.get('seed_mean') or []),
                class_balanced_seed_mean=vec(
                    space_row.get('class_balanced_seed_mean') or []),
            )
            for proto_row in space_row.get('class_prototypes') or []:
                class_idx = int(proto_row.get('class_index'))
                weight = float(proto_row.get('seed_pixels', 0) or 0)
                if weight <= 0:
                    continue
                proto = vec(proto_row.get('prototype') or [])
                if not proto:
                    continue
                for variant in variants:
                    transformed = transform(proto, centers, variant)
                    weighted = mul_vec(transformed, weight)
                    bucket = banks[dataset][space][variant]
                    bucket['class_names'] = class_names
                    total = bucket['totals'][class_idx]
                    total['sum'] = (
                        list(weighted)
                        if total['sum'] is None
                        else add_vec(total['sum'], weighted))
                    total['weight'] += weight
                    contrib = bucket['image_contribs'][image_id][class_idx]
                    contrib['sum'] = (
                        list(weighted)
                        if contrib['sum'] is None
                        else add_vec(contrib['sum'], weighted))
                    contrib['weight'] += weight

    output = dict(version=1, banks={})
    for dataset, by_space in banks.items():
        output['banks'][dataset] = {}
        for space, by_variant in by_space.items():
            output['banks'][dataset][space] = {}
            for variant, bucket in by_variant.items():
                class_names = bucket['class_names'] or []
                classes = []
                for class_idx, total in sorted(bucket['totals'].items()):
                    weight = float(total['weight'])
                    if weight <= 0 or total['sum'] is None:
                        continue
                    proto = div_vec(total['sum'], weight)
                    classes.append(dict(
                        class_index=int(class_idx),
                        class_name=(
                            class_names[class_idx]
                            if class_idx < len(class_names)
                            else str(class_idx)),
                        weight=weight,
                        sum=rounded(total['sum'], args.decimals),
                        prototype=rounded(proto, args.decimals),
                    ))
                image_contribs = {}
                for image_id, by_class in bucket['image_contribs'].items():
                    class_rows = {}
                    for class_idx, contrib in by_class.items():
                        if contrib['sum'] is None or contrib['weight'] <= 0:
                            continue
                        class_rows[str(class_idx)] = dict(
                            weight=float(contrib['weight']),
                            sum=rounded(contrib['sum'], args.decimals),
                        )
                    if class_rows:
                        image_contribs[str(image_id)] = class_rows
                output['banks'][dataset][space][variant] = dict(
                    class_names=class_names,
                    classes=classes,
                    image_contribs=image_contribs,
                )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
