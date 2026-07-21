#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    return [path for path in paths if os.path.exists(path)]


def iter_records(paths):
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


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


def add_matrix(dst, matrix):
    if matrix is None:
        return
    if not dst:
        dst.extend([[0 for _ in row] for row in matrix])
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            dst[i][j] += int(value)


def confusion_metrics(matrix):
    if not matrix:
        return dict(miou=None, acc=None)
    n = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(n))
    ious = []
    for i in range(n):
        tp = matrix[i][i]
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[r][i] for r in range(n))
        den = row_sum + col_sum - tp
        if den > 0:
            ious.append(tp / float(den))
    return dict(
        miou=sum(ious) / len(ious) if ious else None,
        acc=correct / float(total) if total else None,
    )


def add_weighted(bucket, key, value, weight):
    if value is None:
        return
    bucket[f'{key}_sum'] += float(value) * weight
    bucket[f'{key}_weight'] += weight


def weighted(bucket, key):
    return safe_div(bucket.get(f'{key}_sum', 0.0),
                    bucket.get(f'{key}_weight', 0.0))


def main():
    parser = argparse.ArgumentParser(
        description='Summarize prompt winner attribution diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No prompt winner JSONL files matched.')

    action = defaultdict(lambda: dict(confusion=[], images=0))
    prompt = defaultdict(lambda: defaultdict(float))
    pair = defaultdict(lambda: defaultdict(float))
    inventory = defaultdict(lambda: dict(images=set(), records=0))

    for record_index, record in enumerate(iter_records(paths)):
        stats = record.get('prompt_winner_attribution_stats') or {}
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown')
        image_id = str(
            record.get('img_path')
            or f"{record.get('rank', 0)}:{record_index}")
        inventory[dataset]['images'].add(image_id)
        inventory[dataset]['records'] += 1

        for row in stats.get('action_stats') or []:
            key = (
                dataset,
                row.get('source_name'),
                row.get('action_name'),
            )
            action[key]['images'] += 1
            add_matrix(action[key]['confusion'], row.get('confusion'))
            for name in ['valid_pixels', 'oracle_correct_pixels',
                         'baseline_correct_pixels']:
                if row.get(name) is not None:
                    action[key][name] = action[key].get(name, 0) + int(row[name])

        for row in stats.get('prompt_stats') or []:
            key = (
                dataset,
                row.get('source_name'),
                row.get('class_index'),
                row.get('class_name'),
                row.get('role'),
                row.get('query_index'),
                row.get('query_word'),
            )
            b = prompt[key]
            b['images'] += 1
            for name in [
                    'class_winner_pixels',
                    'source_top1_prompt_pixels',
                    'baseline_pred_prompt_pixels',
                    'gt_class_prompt_pixels',
                    'correct_prompt_pixels',
                    'harmful_prompt_pixels',
                    'suppressed_gt_prompt_pixels',
            ]:
                b[name] += int(row.get(name, 0) or 0)
            add_weighted(
                b, 'winner_score_mean', row.get('winner_score_mean'),
                int(row.get('class_winner_pixels', 0) or 0))
            add_weighted(
                b, 'baseline_pred_score_mean',
                row.get('baseline_pred_score_mean'),
                int(row.get('baseline_pred_prompt_pixels', 0) or 0))

        for row in stats.get('pair_stats') or []:
            key = (
                dataset,
                row.get('source_name'),
                row.get('gt_class_index'),
                row.get('gt_class_name'),
                row.get('gt_role'),
                row.get('pred_class_index'),
                row.get('pred_class_name'),
                row.get('pred_role'),
            )
            b = pair[key]
            pixels = int(row.get('pixels', 0) or 0)
            b['images'] += 1
            b['pixels'] += pixels
            add_weighted(
                b, 'gt_minus_pred_score_mean',
                row.get('gt_minus_pred_score_mean'), pixels)
            add_weighted(b, 'gt_score_mean', row.get('gt_score_mean'), pixels)
            add_weighted(
                b, 'pred_score_mean', row.get('pred_score_mean'), pixels)
            for prefix in ('gt', 'pred'):
                word = row.get(f'{prefix}_winner_query_word')
                if word is not None:
                    b[f'{prefix}_winner::{word}'] += pixels
                ratio = row.get(f'{prefix}_winner_query_ratio')
                add_weighted(
                    b, f'{prefix}_winner_query_ratio', ratio, pixels)

    action_rows = []
    baselines = {}
    for key, bucket in action.items():
        dataset, source, name = key
        metrics = confusion_metrics(bucket['confusion'])
        row = dict(
            dataset=dataset,
            source_name=source,
            action_name=name,
            images=int(bucket['images']),
            valid_pixels=int(bucket.get('valid_pixels', 0)),
            miou=metrics['miou'],
            miou_points=(
                None if metrics['miou'] is None else metrics['miou'] * 100.0),
            acc=metrics['acc'],
            confusion=bucket['confusion'],
        )
        if name == 'baseline_threshold':
            baselines[dataset] = row
        action_rows.append(row)
    for row in action_rows:
        base = baselines.get(row['dataset'])
        if base and row.get('miou') is not None and base.get('miou') is not None:
            row['delta_miou_points'] = (
                (row['miou'] - base['miou']) * 100.0)
        else:
            row['delta_miou_points'] = None
    action_rows.sort(
        key=lambda row: (row['dataset'], row['source_name'], row['action_name']))

    prompt_rows = []
    for key, b in prompt.items():
        (
            dataset, source, class_index, class_name, role, query_index,
            query_word,
        ) = key
        baseline_pixels = int(b['baseline_pred_prompt_pixels'])
        gt_pixels = int(b['gt_class_prompt_pixels'])
        prompt_rows.append(dict(
            dataset=dataset,
            source_name=source,
            class_index=class_index,
            class_name=class_name,
            role=role,
            query_index=query_index,
            query_word=query_word,
            images=int(b['images']),
            class_winner_pixels=int(b['class_winner_pixels']),
            source_top1_prompt_pixels=int(b['source_top1_prompt_pixels']),
            baseline_pred_prompt_pixels=baseline_pixels,
            gt_class_prompt_pixels=gt_pixels,
            correct_prompt_pixels=int(b['correct_prompt_pixels']),
            harmful_prompt_pixels=int(b['harmful_prompt_pixels']),
            suppressed_gt_prompt_pixels=int(b['suppressed_gt_prompt_pixels']),
            baseline_prompt_purity=safe_div(
                b['correct_prompt_pixels'], baseline_pixels),
            harmful_ratio=safe_div(
                b['harmful_prompt_pixels'], baseline_pixels),
            suppressed_over_gt_prompt_ratio=safe_div(
                b['suppressed_gt_prompt_pixels'], gt_pixels),
            winner_score_mean=weighted(b, 'winner_score_mean'),
            baseline_pred_score_mean=weighted(b, 'baseline_pred_score_mean'),
        ))
    prompt_rows.sort(
        key=lambda row: (
            row['dataset'], row['source_name'], row['class_name'],
            -row['harmful_prompt_pixels']))

    pair_rows = []
    for key, b in pair.items():
        (
            dataset, source, gt_class_index, gt_class_name, gt_role,
            pred_class_index, pred_class_name, pred_role,
        ) = key
        pixels = int(b['pixels'])

        def top_word(prefix):
            items = [
                (name.split('::', 1)[1], count)
                for name, count in b.items()
                if name.startswith(f'{prefix}_winner::')
            ]
            if not items:
                return None, 0
            return max(items, key=lambda item: item[1])

        gt_word, gt_word_pixels = top_word('gt')
        pred_word, pred_word_pixels = top_word('pred')
        pair_rows.append(dict(
            dataset=dataset,
            source_name=source,
            gt_class_index=gt_class_index,
            gt_class_name=gt_class_name,
            gt_role=gt_role,
            pred_class_index=pred_class_index,
            pred_class_name=pred_class_name,
            pred_role=pred_role,
            images=int(b['images']),
            pixels=pixels,
            gt_score_mean=weighted(b, 'gt_score_mean'),
            pred_score_mean=weighted(b, 'pred_score_mean'),
            gt_minus_pred_score_mean=weighted(
                b, 'gt_minus_pred_score_mean'),
            gt_top_winner_query_word=gt_word,
            gt_top_winner_query_ratio=safe_div(gt_word_pixels, pixels),
            pred_top_winner_query_word=pred_word,
            pred_top_winner_query_ratio=safe_div(pred_word_pixels, pixels),
            gt_winner_query_ratio_mean=weighted(
                b, 'gt_winner_query_ratio'),
            pred_winner_query_ratio_mean=weighted(
                b, 'pred_winner_query_ratio'),
        ))
    pair_rows.sort(
        key=lambda row: (row['dataset'], row['source_name'], -row['pixels']))

    inventory_rows = [
        dict(dataset=dataset,
             images=len(bucket['images']),
             records=int(bucket['records']))
        for dataset, bucket in sorted(inventory.items())
    ]

    write_csv(
        os.path.join(args.output_dir, 'prompt_winner_action_summary.csv'),
        action_rows)
    write_csv(
        os.path.join(args.output_dir, 'prompt_winner_prompt_summary.csv'),
        prompt_rows)
    write_csv(
        os.path.join(args.output_dir, 'prompt_winner_pair_summary.csv'),
        pair_rows)
    write_csv(
        os.path.join(args.output_dir, 'prompt_winner_inventory.csv'),
        inventory_rows)


if __name__ == '__main__':
    main()
