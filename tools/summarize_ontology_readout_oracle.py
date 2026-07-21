#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict


BASELINE = 'baseline_thresholded'


def safe_div(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize ontology-role readout oracle diagnostics.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='Input JSONL paths or globs.')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for CSV summaries.')
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def iter_records(paths):
    for path in paths:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stats = record.get('ontology_readout_oracle_stats')
                if not stats:
                    continue
                yield record, stats


def add_counts(bucket, row):
    for key in (
            'gt_pixels', 'pred_pixels', 'tp_pixels', 'fp_pixels',
            'fn_pixels', 'baseline_correct_pixels',
            'baseline_wrong_pixels', 'wrong_recovered_pixels',
            'changed_pred_pixels'):
        bucket[key] += int(row.get(key) or 0)


def metric_row(meta, counts):
    gt = counts.get('gt_pixels', 0)
    pred = counts.get('pred_pixels', 0)
    tp = counts.get('tp_pixels', 0)
    fp = counts.get('fp_pixels', 0)
    fn = counts.get('fn_pixels', 0)
    wrong = counts.get('baseline_wrong_pixels', 0)
    recovered = counts.get('wrong_recovered_pixels', 0)
    row = dict(meta)
    row.update(dict(
        gt_pixels=int(gt),
        pred_pixels=int(pred),
        tp_pixels=int(tp),
        fp_pixels=int(fp),
        fn_pixels=int(fn),
        class_iou=safe_div(tp, tp + fp + fn),
        class_precision=safe_div(tp, pred),
        class_recall=safe_div(tp, gt),
        baseline_correct_pixels=int(
            counts.get('baseline_correct_pixels', 0)),
        baseline_wrong_pixels=int(wrong),
        wrong_recovered_pixels=int(recovered),
        wrong_recovered_ratio=safe_div(recovered, wrong),
        changed_pred_pixels=int(counts.get('changed_pred_pixels', 0)),
    ))
    return row


def mean(values):
    values = [
        float(value) for value in values
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path, rows, preferred=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(preferred or [])
        return
    fields = []
    for key in preferred or []:
        if key not in fields:
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_by(rows, key, candidates):
    best = None
    best_value = None
    for row in rows:
        if row.get('readout_name') not in candidates:
            continue
        value = row.get(key)
        if value is None:
            continue
        if best is None or value > best_value:
            best = row
            best_value = value
    return best


def summarize(records):
    class_counts = defaultdict(lambda: defaultdict(int))
    class_meta = {}
    image_oracle = defaultdict(lambda: defaultdict(float))
    pair_counts = defaultdict(lambda: defaultdict(int))
    pair_meta = {}
    requested = defaultdict(Counter)

    for record, stats in records:
        dataset = (
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown')
        for name in stats.get('requested_readouts') or []:
            requested[dataset][name] += 1
        for row in stats.get('class_stats') or []:
            key = (
                dataset,
                row.get('readout_name'),
                int(row.get('class_index')),
            )
            add_counts(class_counts[key], row)
            class_meta[key] = dict(
                dataset_name=dataset,
                readout_name=row.get('readout_name'),
                class_index=int(row.get('class_index')),
                class_name=row.get('class_name'),
                role=row.get('role'),
            )
        pixel_oracle = stats.get('pixel_oracle_stats') or {}
        for key, value in pixel_oracle.items():
            if isinstance(value, (int, float)):
                image_oracle[dataset][key] += float(value)
        for row in stats.get('pair_stats') or []:
            key = (
                dataset,
                row.get('readout_name'),
                int(row.get('gt_class_index')),
                int(row.get('base_pred_class_index')),
            )
            bucket = pair_counts[key]
            bucket['pixels'] += int(row.get('pixels') or 0)
            bucket['source_top1_recovers_gt_pixels'] += int(
                row.get('source_top1_recovers_gt_pixels') or 0)
            bucket['source_topk_contains_gt_pixels'] += int(
                row.get('source_topk_contains_gt_pixels') or 0)
            pair_meta[key] = dict(
                dataset_name=dataset,
                readout_name=row.get('readout_name'),
                gt_class_index=int(row.get('gt_class_index')),
                gt_class_name=row.get('gt_class_name'),
                gt_role=row.get('gt_role'),
                base_pred_class_index=int(row.get('base_pred_class_index')),
                base_pred_class_name=row.get('base_pred_class_name'),
                pred_role=row.get('pred_role'),
            )

    class_rows = [
        metric_row(class_meta[key], counts)
        for key, counts in class_counts.items()
    ]
    class_rows.sort(key=lambda row: (
        row['dataset_name'], row['class_index'], row['readout_name']))

    candidate_readouts = sorted({
        row['readout_name'] for row in class_rows
        if row['readout_name'] != BASELINE
    })

    by_dataset_class = defaultdict(list)
    for row in class_rows:
        by_dataset_class[(row['dataset_name'], row['class_index'])].append(row)
    for rows in by_dataset_class.values():
        baseline = next(
            (row for row in rows if row['readout_name'] == BASELINE),
            None)
        best = best_by(rows, 'class_iou', candidate_readouts)
        best_name = None if best is None else best['readout_name']
        best_iou = None if best is None else best['class_iou']
        baseline_iou = None if baseline is None else baseline['class_iou']
        for row in rows:
            row['baseline_class_iou'] = baseline_iou
            row['delta_iou_vs_baseline'] = (
                None if row['class_iou'] is None or baseline_iou is None
                else row['class_iou'] - baseline_iou)
            row['best_candidate_readout'] = best_name
            row['best_candidate_iou'] = best_iou
            row['is_best_candidate'] = bool(
                row['readout_name'] == best_name
                and row['readout_name'] != BASELINE)

    dataset_rows = []
    by_dataset_readout = defaultdict(list)
    for row in class_rows:
        by_dataset_readout[(row['dataset_name'], row['readout_name'])].append(row)
    for (dataset, readout), rows in sorted(by_dataset_readout.items()):
        present = [row for row in rows if row.get('gt_pixels', 0) > 0]
        dataset_rows.append(dict(
            dataset_name=dataset,
            readout_name=readout,
            class_count=len(present),
            gt_pixels=sum(int(row.get('gt_pixels') or 0) for row in present),
            pred_pixels=sum(int(row.get('pred_pixels') or 0) for row in present),
            tp_pixels=sum(int(row.get('tp_pixels') or 0) for row in present),
            mIoU=mean([row.get('class_iou') for row in present]),
            mAcc=mean([row.get('class_recall') for row in present]),
            mean_precision=mean([row.get('class_precision') for row in present]),
            baseline_wrong_pixels=sum(
                int(row.get('baseline_wrong_pixels') or 0) for row in present),
            wrong_recovered_pixels=sum(
                int(row.get('wrong_recovered_pixels') or 0) for row in present),
            wrong_recovered_ratio=safe_div(
                sum(int(row.get('wrong_recovered_pixels') or 0) for row in present),
                sum(int(row.get('baseline_wrong_pixels') or 0) for row in present)),
        ))

    role_rows = []
    by_dataset_role_readout = defaultdict(list)
    for row in class_rows:
        by_dataset_role_readout[
            (row['dataset_name'], row['role'], row['readout_name'])
        ].append(row)
    for (dataset, role, readout), rows in sorted(
            by_dataset_role_readout.items()):
        present = [row for row in rows if row.get('gt_pixels', 0) > 0]
        tp = sum(int(row.get('tp_pixels') or 0) for row in present)
        fp = sum(int(row.get('fp_pixels') or 0) for row in present)
        fn = sum(int(row.get('fn_pixels') or 0) for row in present)
        role_rows.append(dict(
            dataset_name=dataset,
            role=role,
            readout_name=readout,
            class_count=len(present),
            gt_pixels=sum(int(row.get('gt_pixels') or 0) for row in present),
            role_class_mIoU=mean([row.get('class_iou') for row in present]),
            role_class_mAcc=mean([row.get('class_recall') for row in present]),
            role_micro_iou=safe_div(tp, tp + fp + fn),
            baseline_wrong_pixels=sum(
                int(row.get('baseline_wrong_pixels') or 0) for row in present),
            wrong_recovered_pixels=sum(
                int(row.get('wrong_recovered_pixels') or 0) for row in present),
            wrong_recovered_ratio=safe_div(
                sum(int(row.get('wrong_recovered_pixels') or 0) for row in present),
                sum(int(row.get('baseline_wrong_pixels') or 0) for row in present)),
        ))

    by_dataset_role = defaultdict(list)
    for row in role_rows:
        by_dataset_role[(row['dataset_name'], row['role'])].append(row)
    for rows in by_dataset_role.values():
        baseline = next(
            (row for row in rows if row['readout_name'] == BASELINE),
            None)
        best = best_by(rows, 'role_class_mIoU', candidate_readouts)
        best_name = None if best is None else best['readout_name']
        best_value = None if best is None else best['role_class_mIoU']
        baseline_value = None if baseline is None else baseline['role_class_mIoU']
        for row in rows:
            row['baseline_role_class_mIoU'] = baseline_value
            row['delta_role_mIoU_vs_baseline'] = (
                None
                if row.get('role_class_mIoU') is None or baseline_value is None
                else row['role_class_mIoU'] - baseline_value)
            row['best_candidate_readout'] = best_name
            row['best_candidate_role_mIoU'] = best_value
            row['is_best_candidate'] = bool(
                row['readout_name'] == best_name
                and row['readout_name'] != BASELINE)

    stability_rows = []
    role_groups = defaultdict(list)
    for rows in by_dataset_role.values():
        best = best_by(rows, 'role_class_mIoU', candidate_readouts)
        if best is not None:
            role_groups[best['role']].append(best)
    for role, winners in sorted(role_groups.items()):
        win_counter = Counter(row['readout_name'] for row in winners)
        dataset_count = len({row['dataset_name'] for row in winners})
        for readout in candidate_readouts:
            readout_winners = [
                row for row in winners if row['readout_name'] == readout
            ]
            stability_rows.append(dict(
                role=role,
                readout_name=readout,
                dataset_role_win_count=len(readout_winners),
                dataset_role_win_fraction=safe_div(
                    len(readout_winners), dataset_count),
                winning_datasets='|'.join(sorted(
                    row['dataset_name'] for row in readout_winners)),
                total_role_datasets=dataset_count,
                mean_winning_role_mIoU=mean([
                    row.get('role_class_mIoU') for row in readout_winners]),
                mean_winning_delta_vs_baseline=mean([
                    row.get('delta_role_mIoU_vs_baseline')
                    for row in readout_winners]),
                is_role_majority_winner=bool(
                    win_counter and win_counter[readout] == max(
                        win_counter.values())),
            ))

    oracle_rows = []
    dataset_names = sorted({row['dataset_name'] for row in class_rows})
    dataset_readout_miou = {
        (row['dataset_name'], row['readout_name']): row.get('mIoU')
        for row in dataset_rows
    }
    global_role_best = {}
    for role in sorted({row['role'] for row in role_rows}):
        rows = [row for row in role_rows if row['role'] == role]
        scores = {}
        for readout in candidate_readouts:
            scores[readout] = mean([
                row.get('role_class_mIoU') for row in rows
                if row['readout_name'] == readout])
        global_role_best[role] = max(
            scores.items(),
            key=lambda item: -1e9 if item[1] is None else item[1],
        )[0] if scores else None

    for dataset in dataset_names:
        class_group = [
            rows for (ds, _), rows in by_dataset_class.items()
            if ds == dataset
        ]
        class_oracle = mean([
            (best_by(rows, 'class_iou', candidate_readouts) or {}).get(
                'class_iou')
            for rows in class_group
        ])
        role_group = [
            rows for (ds, _), rows in by_dataset_role.items()
            if ds == dataset
        ]
        role_oracle_values = []
        dataset_role_choice = {}
        for rows in role_group:
            best = best_by(rows, 'role_class_mIoU', candidate_readouts)
            if best is None:
                continue
            dataset_role_choice[best['role']] = best['readout_name']
            role_class_rows = [
                row for row in class_rows
                if row['dataset_name'] == dataset
                and row['role'] == best['role']
                and row['readout_name'] == best['readout_name']
                and row.get('gt_pixels', 0) > 0
            ]
            role_oracle_values.extend([
                row.get('class_iou') for row in role_class_rows])
        global_role_values = []
        for row in class_rows:
            if row['dataset_name'] != dataset or row.get('gt_pixels', 0) <= 0:
                continue
            if row['readout_name'] == global_role_best.get(row['role']):
                global_role_values.append(row.get('class_iou'))
        best_single = None
        best_single_readout = None
        for readout in candidate_readouts:
            value = dataset_readout_miou.get((dataset, readout))
            if value is not None and (best_single is None or value > best_single):
                best_single = value
                best_single_readout = readout
        oracle_rows.append(dict(
            dataset_name=dataset,
            baseline_mIoU=dataset_readout_miou.get((dataset, BASELINE)),
            best_single_readout=best_single_readout,
            best_single_readout_mIoU=best_single,
            class_oracle_mIoU=class_oracle,
            dataset_role_oracle_mIoU=mean(role_oracle_values),
            global_role_routed_mIoU=mean(global_role_values),
            dataset_role_choices='|'.join(
                f'{role}:{readout}'
                for role, readout in sorted(dataset_role_choice.items())),
            global_role_choices='|'.join(
                f'{role}:{readout}'
                for role, readout in sorted(global_role_best.items())),
            pixel_oracle_wrong_recovered_ratio=safe_div(
                image_oracle[dataset].get(
                    'baseline_wrong_any_readout_recovers_pixels', 0),
                image_oracle[dataset].get('baseline_wrong_pixels', 0)),
        ))

    pair_rows = []
    for key, counts in sorted(pair_counts.items()):
        row = dict(pair_meta[key])
        pixels = int(counts.get('pixels') or 0)
        row.update(dict(
            pixels=pixels,
            source_top1_recovers_gt_pixels=int(
                counts.get('source_top1_recovers_gt_pixels') or 0),
            source_top1_recovers_gt_ratio=safe_div(
                counts.get('source_top1_recovers_gt_pixels') or 0,
                pixels),
            source_topk_contains_gt_pixels=int(
                counts.get('source_topk_contains_gt_pixels') or 0),
            source_topk_contains_gt_ratio=safe_div(
                counts.get('source_topk_contains_gt_pixels') or 0,
                pixels),
        ))
        pair_rows.append(row)

    return dict(
        dataset_rows=dataset_rows,
        class_rows=class_rows,
        role_rows=role_rows,
        stability_rows=stability_rows,
        oracle_rows=oracle_rows,
        pair_rows=pair_rows,
    )


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError('No ontology readout oracle JSONL files found.')
    summaries = summarize(iter_records(paths))
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(
        os.path.join(args.out_dir, 'ontology_readout_dataset_summary.csv'),
        summaries['dataset_rows'],
        [
            'dataset_name', 'readout_name', 'class_count', 'gt_pixels',
            'pred_pixels', 'tp_pixels', 'mIoU', 'mAcc', 'mean_precision',
            'baseline_wrong_pixels', 'wrong_recovered_pixels',
            'wrong_recovered_ratio',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'ontology_readout_class_summary.csv'),
        summaries['class_rows'],
        [
            'dataset_name', 'class_index', 'class_name', 'role',
            'readout_name', 'gt_pixels', 'pred_pixels', 'tp_pixels',
            'fp_pixels', 'fn_pixels', 'class_iou', 'class_precision',
            'class_recall', 'baseline_class_iou',
            'delta_iou_vs_baseline', 'best_candidate_readout',
            'best_candidate_iou', 'is_best_candidate',
            'baseline_wrong_pixels', 'wrong_recovered_pixels',
            'wrong_recovered_ratio',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'ontology_readout_role_summary.csv'),
        summaries['role_rows'],
        [
            'dataset_name', 'role', 'readout_name', 'class_count',
            'gt_pixels', 'role_class_mIoU', 'role_micro_iou',
            'role_class_mAcc', 'baseline_role_class_mIoU',
            'delta_role_mIoU_vs_baseline', 'best_candidate_readout',
            'best_candidate_role_mIoU', 'is_best_candidate',
            'baseline_wrong_pixels', 'wrong_recovered_pixels',
            'wrong_recovered_ratio',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'ontology_readout_role_stability.csv'),
        summaries['stability_rows'],
        [
            'role', 'readout_name', 'dataset_role_win_count',
            'dataset_role_win_fraction', 'winning_datasets',
            'total_role_datasets', 'mean_winning_role_mIoU',
            'mean_winning_delta_vs_baseline', 'is_role_majority_winner',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'ontology_readout_oracle_summary.csv'),
        summaries['oracle_rows'],
        [
            'dataset_name', 'baseline_mIoU', 'best_single_readout',
            'best_single_readout_mIoU', 'class_oracle_mIoU',
            'dataset_role_oracle_mIoU', 'global_role_routed_mIoU',
            'pixel_oracle_wrong_recovered_ratio', 'dataset_role_choices',
            'global_role_choices',
        ],
    )
    write_csv(
        os.path.join(args.out_dir, 'ontology_readout_pair_summary.csv'),
        summaries['pair_rows'],
        [
            'dataset_name', 'readout_name', 'gt_class_index',
            'gt_class_name', 'gt_role', 'base_pred_class_index',
            'base_pred_class_name', 'pred_role', 'pixels',
            'source_top1_recovers_gt_pixels',
            'source_top1_recovers_gt_ratio',
            'source_topk_contains_gt_pixels',
            'source_topk_contains_gt_ratio',
        ],
    )
    print(
        f'Wrote ontology readout summaries from {len(paths)} files '
        f'to {args.out_dir}')


if __name__ == '__main__':
    main()
