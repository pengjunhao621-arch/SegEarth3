#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def add_matrix(dst, src):
    if src is None:
        return dst
    if dst is None:
        return [list(map(int, row)) for row in src]
    for i, row in enumerate(src):
        for j, value in enumerate(row):
            dst[i][j] += int(value)
    return dst


def miou_from_confusion(matrix):
    if matrix is None:
        return None, []
    n = len(matrix)
    ious = []
    for i in range(n):
        tp = matrix[i][i]
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[r][i] for r in range(n))
        union = row_sum + col_sum - tp
        if union > 0:
            ious.append(tp / union)
        else:
            ious.append(None)
    valid = [value for value in ious if value is not None]
    if not valid:
        return None, ious
    return sum(valid) / len(valid), ious


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def read_records(paths):
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context = record.get('ontology_self_verification')
                if not context:
                    continue
                yield path, record, context


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No ontology self-verification JSONL found.')

    dataset_acc = {}
    class_acc = defaultdict(lambda: defaultdict(float))
    role_acc = defaultdict(lambda: defaultdict(float))
    pair_acc = defaultdict(lambda: defaultdict(float))

    for _, record, context in read_records(paths):
        dataset = (
            record.get('dataset_name')
            or context.get('dataset_name')
            or 'unknown')
        acc = dataset_acc.setdefault(dataset, dict(
            dataset=dataset,
            images=0,
            base_confusion=None,
            new_confusion=None,
            seed_pixels=0,
            seeded_classes_sum=0,
            selected_pixels=0,
            selected_ratio_sum=0.0,
            used_spaces=set(),
            reasons=defaultdict(int),
            use_ontology_self_verification=record.get(
                'use_ontology_self_verification'),
        ))
        acc['images'] += 1
        reason = context.get('reason', 'missing')
        acc['reasons'][reason] += 1
        if reason != 'ok':
            continue
        acc['base_confusion'] = add_matrix(
            acc['base_confusion'], context.get('base_confusion'))
        acc['new_confusion'] = add_matrix(
            acc['new_confusion'], context.get('new_confusion'))
        acc['seed_pixels'] += int(context.get('seed_pixels', 0) or 0)
        acc['seeded_classes_sum'] += int(
            context.get('seeded_classes', 0) or 0)
        acc['selected_pixels'] += int(
            context.get('selected_pixels', 0) or 0)
        acc['selected_ratio_sum'] += float(
            context.get('selected_ratio', 0.0) or 0.0)
        for space in context.get('used_spaces') or []:
            acc['used_spaces'].add(space)

        for row in context.get('class_stats') or []:
            key = (
                dataset,
                int(row.get('class_index', -1)),
                row.get('class_name'),
                row.get('role'),
            )
            bucket = class_acc[key]
            for name in [
                    'valid_pixels',
                    'seed_pixels',
                    'seed_correct_pixels',
                    'wrong_gt_pixels',
                    'wrong_gt_topk_contains_pixels',
                    'wrong_gt_seed_available_pixels',
                    'wrong_gt_similarity_favors_pixels',
                    'selected_gt_pixels',
                    'improved_pixels',
                    'harmed_from_class_pixels',
            ]:
                bucket[name] += int(row.get(name, 0) or 0)

        for row in context.get('role_stats') or []:
            key = (dataset, row.get('role'))
            bucket = role_acc[key]
            for name in [
                    'valid_pixels',
                    'base_wrong_pixels',
                    'gt_topk_pixels',
                    'gt_seed_available_pixels',
                    'gt_similarity_favors_pixels',
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels',
            ]:
                bucket[name] += int(row.get(name, 0) or 0)

        for row in context.get('changed_pairs') or []:
            key = (
                dataset,
                int(row.get('from_class_index', -1)),
                row.get('from_class_name'),
                row.get('from_role'),
                int(row.get('to_class_index', -1)),
                row.get('to_class_name'),
                row.get('to_role'),
            )
            bucket = pair_acc[key]
            for name in [
                    'pixels',
                    'improved_pixels',
                    'harmed_pixels',
                    'wrong_to_wrong_pixels',
            ]:
                bucket[name] += int(row.get(name, 0) or 0)

    summary_rows = []
    for dataset, acc in sorted(dataset_acc.items()):
        base_miou, _ = miou_from_confusion(acc['base_confusion'])
        new_miou, _ = miou_from_confusion(acc['new_confusion'])
        delta = None
        if base_miou is not None and new_miou is not None:
            delta = (new_miou - base_miou) * 100.0
        summary_rows.append(dict(
            dataset=dataset,
            images=acc['images'],
            use_ontology_self_verification=acc[
                'use_ontology_self_verification'],
            base_miou=None if base_miou is None else base_miou * 100.0,
            osv_miou=None if new_miou is None else new_miou * 100.0,
            delta_miou=delta,
            seed_pixels=int(acc['seed_pixels']),
            avg_seeded_classes=safe_div(
                acc['seeded_classes_sum'], acc['images']),
            selected_pixels=int(acc['selected_pixels']),
            avg_selected_ratio=safe_div(
                acc['selected_ratio_sum'], acc['images']),
            used_spaces='|'.join(sorted(acc['used_spaces'])),
            reasons=json.dumps(dict(acc['reasons']), sort_keys=True),
        ))

    class_rows = []
    for (dataset, class_idx, class_name, role), bucket in sorted(
            class_acc.items()):
        class_rows.append(dict(
            dataset=dataset,
            class_index=class_idx,
            class_name=class_name,
            role=role,
            valid_pixels=int(bucket['valid_pixels']),
            seed_pixels=int(bucket['seed_pixels']),
            seed_purity=safe_div(
                bucket['seed_correct_pixels'], bucket['seed_pixels']),
            wrong_gt_pixels=int(bucket['wrong_gt_pixels']),
            wrong_gt_topk_contains_ratio=safe_div(
                bucket['wrong_gt_topk_contains_pixels'],
                bucket['wrong_gt_pixels']),
            wrong_gt_seed_available_ratio=safe_div(
                bucket['wrong_gt_seed_available_pixels'],
                bucket['wrong_gt_pixels']),
            wrong_gt_similarity_favors_ratio=safe_div(
                bucket['wrong_gt_similarity_favors_pixels'],
                bucket['wrong_gt_pixels']),
            selected_gt_pixels=int(bucket['selected_gt_pixels']),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_from_class_pixels=int(bucket['harmed_from_class_pixels']),
            net_pixels=int(
                bucket['improved_pixels']
                - bucket['harmed_from_class_pixels']),
        ))

    role_rows = []
    for (dataset, role), bucket in sorted(role_acc.items()):
        role_rows.append(dict(
            dataset=dataset,
            role=role,
            valid_pixels=int(bucket['valid_pixels']),
            base_wrong_pixels=int(bucket['base_wrong_pixels']),
            gt_topk_ratio=safe_div(
                bucket['gt_topk_pixels'], bucket['base_wrong_pixels']),
            gt_seed_available_ratio=safe_div(
                bucket['gt_seed_available_pixels'],
                bucket['base_wrong_pixels']),
            gt_similarity_favors_ratio=safe_div(
                bucket['gt_similarity_favors_pixels'],
                bucket['base_wrong_pixels']),
            changed_pixels=int(bucket['changed_pixels']),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            correction_precision=safe_div(
                bucket['improved_pixels'], bucket['changed_pixels']),
            harmful_rate=safe_div(
                bucket['harmed_pixels'], bucket['changed_pixels']),
            net_pixels=int(
                bucket['improved_pixels'] - bucket['harmed_pixels']),
        ))

    pair_rows = []
    for key, bucket in sorted(
            pair_acc.items(),
            key=lambda item: item[1].get('pixels', 0),
            reverse=True):
        (
            dataset,
            from_class_index,
            from_class_name,
            from_role,
            to_class_index,
            to_class_name,
            to_role,
        ) = key
        pair_rows.append(dict(
            dataset=dataset,
            from_class_index=from_class_index,
            from_class_name=from_class_name,
            from_role=from_role,
            to_class_index=to_class_index,
            to_class_name=to_class_name,
            to_role=to_role,
            pixels=int(bucket['pixels']),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            wrong_to_wrong_pixels=int(bucket['wrong_to_wrong_pixels']),
            correction_precision=safe_div(
                bucket['improved_pixels'], bucket['pixels']),
            harmful_rate=safe_div(bucket['harmed_pixels'], bucket['pixels']),
            net_pixels=int(bucket['improved_pixels'] - bucket['harmed_pixels']),
        ))

    write_csv(
        os.path.join(
            args.output_dir,
            'ontology_self_verification_summary.csv',
        ),
        summary_rows,
        [
            'dataset',
            'images',
            'use_ontology_self_verification',
            'base_miou',
            'osv_miou',
            'delta_miou',
            'seed_pixels',
            'avg_seeded_classes',
            'selected_pixels',
            'avg_selected_ratio',
            'used_spaces',
            'reasons',
        ],
    )
    write_csv(
        os.path.join(
            args.output_dir,
            'ontology_self_verification_class_summary.csv',
        ),
        class_rows,
        [
            'dataset',
            'class_index',
            'class_name',
            'role',
            'valid_pixels',
            'seed_pixels',
            'seed_purity',
            'wrong_gt_pixels',
            'wrong_gt_topk_contains_ratio',
            'wrong_gt_seed_available_ratio',
            'wrong_gt_similarity_favors_ratio',
            'selected_gt_pixels',
            'improved_pixels',
            'harmed_from_class_pixels',
            'net_pixels',
        ],
    )
    write_csv(
        os.path.join(
            args.output_dir,
            'ontology_self_verification_role_summary.csv',
        ),
        role_rows,
        [
            'dataset',
            'role',
            'valid_pixels',
            'base_wrong_pixels',
            'gt_topk_ratio',
            'gt_seed_available_ratio',
            'gt_similarity_favors_ratio',
            'changed_pixels',
            'improved_pixels',
            'harmed_pixels',
            'correction_precision',
            'harmful_rate',
            'net_pixels',
        ],
    )
    write_csv(
        os.path.join(
            args.output_dir,
            'ontology_self_verification_pair_summary.csv',
        ),
        pair_rows,
        [
            'dataset',
            'from_class_index',
            'from_class_name',
            'from_role',
            'to_class_index',
            'to_class_name',
            'to_role',
            'pixels',
            'improved_pixels',
            'harmed_pixels',
            'wrong_to_wrong_pixels',
            'correction_precision',
            'harmful_rate',
            'net_pixels',
        ],
    )


if __name__ == '__main__':
    main()
