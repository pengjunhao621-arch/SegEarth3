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


def add_matrix(dst, src):
    if not src:
        return
    if not dst:
        dst.extend([[0 for _ in row] for row in src])
    for i, row in enumerate(src):
        for j, value in enumerate(row):
            dst[i][j] += int(value)


def matrix_copy(matrix):
    return [[int(value) for value in row] for row in matrix]


def confusion_metrics(matrix):
    class_count = len(matrix)
    if class_count == 0:
        return dict(mIoU=0.0, aAcc=0.0, IoU=[], gt=[], pred=[], tp=[])
    tp = [int(matrix[i][i]) for i in range(class_count)]
    gt = [int(sum(matrix[i])) for i in range(class_count)]
    pred = [
        int(sum(matrix[i][j] for i in range(class_count)))
        for j in range(class_count)
    ]
    ious = []
    valid_ious = []
    for i in range(class_count):
        union = gt[i] + pred[i] - tp[i]
        iou = safe_div(tp[i], union)
        ious.append(iou)
        if gt[i] > 0:
            valid_ious.append(iou)
    return dict(
        mIoU=safe_div(sum(valid_ious), len(valid_ious)) * 100.0,
        aAcc=safe_div(sum(tp), sum(gt)) * 100.0,
        IoU=[value * 100.0 for value in ious],
        gt=gt,
        pred=pred,
        tp=tp,
    )


def apply_selected_matrix(baseline, selected, bg_idx):
    matrix = matrix_copy(baseline)
    if not selected:
        return matrix
    for gt_idx, row in enumerate(selected):
        moved = sum(int(value) for value in row)
        if moved == 0:
            continue
        matrix[gt_idx][bg_idx] -= moved
        if matrix[gt_idx][bg_idx] < 0:
            matrix[gt_idx][bg_idx] = 0
        for pred_idx, value in enumerate(row):
            matrix[gt_idx][pred_idx] += int(value)
    return matrix


def empty_bucket(hist_len):
    return dict(
        pixels=0,
        sum=0.0,
        sumsq=0.0,
        min=None,
        max=None,
        hist=[0 for _ in range(hist_len)],
    )


def add_bucket(dst, src):
    if not src:
        return
    if not dst:
        dst.update(empty_bucket(len(src.get('hist') or [])))
    dst['pixels'] += int(src.get('pixels', 0) or 0)
    dst['sum'] += float(src.get('sum', 0.0) or 0.0)
    dst['sumsq'] += float(src.get('sumsq', 0.0) or 0.0)
    src_min = src.get('min')
    src_max = src.get('max')
    if src_min is not None:
        dst['min'] = src_min if dst['min'] is None else min(dst['min'], src_min)
    if src_max is not None:
        dst['max'] = src_max if dst['max'] is None else max(dst['max'], src_max)
    for i, value in enumerate(src.get('hist') or []):
        dst['hist'][i] += int(value)


def hist_auc(pos_hist, neg_hist):
    pos_total = sum(pos_hist)
    neg_total = sum(neg_hist)
    if pos_total == 0 or neg_total == 0:
        return None
    neg_seen = 0
    score = 0.0
    for pos_count, neg_count in zip(pos_hist, neg_hist):
        score += pos_count * neg_seen
        score += 0.5 * pos_count * neg_count
        neg_seen += neg_count
    return score / float(pos_total * neg_total)


def bucket_mean(bucket):
    return safe_div(bucket.get('sum', 0.0), bucket.get('pixels', 0))


def bucket_std(bucket):
    count = bucket.get('pixels', 0)
    if count <= 1:
        return 0.0
    mean = bucket_mean(bucket)
    var = max(0.0, bucket.get('sumsq', 0.0) / count - mean * mean)
    return var ** 0.5


def row_ratio(row, key, den):
    return safe_div(float(row.get(key, 0) or 0), den)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize background foreground-separability diagnostics.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = {}
    for record in load_records(args.inputs):
        stats = record.get('background_separability_stats') or {}
        if not stats:
            continue
        dataset = (
            record.get('dataset_name')
            or stats.get('dataset_name')
            or 'unknown'
        )
        info = datasets.setdefault(dataset, dict(
            dataset=dataset,
            images=0,
            bg_idx=int(stats.get('bg_idx', 0) or 0),
            class_names=stats.get('class_names') or [],
            class_roles=stats.get('class_roles') or [],
            baseline=[],
            oracle=[],
            valid_pixels=0,
            baseline_bg_pixels=0,
            protected_background_pixels=0,
            recoverable_foreground_pixels=0,
            candidate_correct_pixels=0,
            threshold_route_pixels=0,
            weak_bg_route_pixels=0,
            features={},
            selectors={},
            class_stats=defaultdict(lambda: defaultdict(float)),
            pair_stats=defaultdict(lambda: defaultdict(float)),
        ))
        if not info['class_names'] and stats.get('class_names'):
            info['class_names'] = stats.get('class_names')
        if not info['class_roles'] and stats.get('class_roles'):
            info['class_roles'] = stats.get('class_roles')
        info['bg_idx'] = int(stats.get('bg_idx', info['bg_idx']) or 0)
        add_matrix(info['baseline'], stats.get('baseline_confusion'))
        add_matrix(info['oracle'], stats.get('candidate_oracle_matrix'))
        info['images'] += 1
        for key in [
                'valid_pixels',
                'baseline_bg_pixels',
                'protected_background_pixels',
                'recoverable_foreground_pixels',
                'candidate_correct_pixels',
                'threshold_route_pixels',
                'weak_bg_route_pixels',
        ]:
            info[key] += int(stats.get(key, 0) or 0)

        for feature in stats.get('feature_stats') or []:
            name = feature.get('feature')
            if not name:
                continue
            bucket = info['features'].setdefault(name, dict(
                bins=feature.get('bins') or [],
                protected={},
                recoverable={},
                candidate_correct={},
            ))
            add_bucket(bucket['protected'], feature.get('protected'))
            add_bucket(bucket['recoverable'], feature.get('recoverable'))
            add_bucket(
                bucket['candidate_correct'],
                feature.get('candidate_correct'),
            )

        for selector in stats.get('selector_stats') or []:
            name = selector.get('name')
            if not name:
                continue
            bucket = info['selectors'].setdefault(name, dict(
                name=name,
                selected_pixels=0,
                selected_recoverable_pixels=0,
                selected_protected_pixels=0,
                improved_pixels=0,
                harmed_pixels=0,
                wrong_to_wrong_pixels=0,
                matrix=[],
            ))
            for key in [
                    'selected_pixels',
                    'selected_recoverable_pixels',
                    'selected_protected_pixels',
                    'improved_pixels',
                    'harmed_pixels',
                    'wrong_to_wrong_pixels',
            ]:
                bucket[key] += int(selector.get(key, 0) or 0)
            add_matrix(bucket['matrix'], selector.get('selected_candidate_matrix'))

        for row in stats.get('class_stats') or []:
            class_idx = int(row.get('class_index', -1))
            if class_idx < 0:
                continue
            bucket = info['class_stats'][class_idx]
            for key, value in row.items():
                if key in ('class_index', 'class_name', 'role'):
                    continue
                bucket[key] += float(value or 0)

        for row in stats.get('pair_stats') or []:
            pair_key = (
                int(row.get('gt_class_index', -1)),
                int(row.get('candidate_class_index', -1)),
            )
            if pair_key[0] < 0 or pair_key[1] < 0:
                continue
            bucket = info['pair_stats'][pair_key]
            for key, value in row.items():
                if (
                        key.endswith('name')
                        or key.endswith('role')
                        or key in ('gt_class_index', 'candidate_class_index')):
                    continue
                bucket[key] += float(value or 0)

    summary_path = os.path.join(
        args.output_dir,
        'background_separability_summary.csv',
    )
    feature_path = os.path.join(
        args.output_dir,
        'background_feature_summary.csv',
    )
    selector_path = os.path.join(
        args.output_dir,
        'background_selector_summary.csv',
    )
    class_path = os.path.join(
        args.output_dir,
        'background_class_summary.csv',
    )
    pair_path = os.path.join(
        args.output_dir,
        'background_pair_summary.csv',
    )

    with open(summary_path, 'w', newline='') as f:
        fields = [
            'dataset', 'images', 'baseline_mIoU', 'candidate_oracle_mIoU',
            'candidate_oracle_delta_mIoU', 'baseline_aAcc',
            'candidate_oracle_aAcc', 'candidate_oracle_delta_aAcc',
            'baseline_bg_ratio', 'protected_background_ratio',
            'recoverable_foreground_ratio', 'candidate_correct_rate',
            'threshold_route_ratio', 'weak_bg_route_ratio',
            'best_selector', 'best_selector_mIoU',
            'best_selector_delta_mIoU', 'best_selector_selected_ratio',
            'best_selector_correction_precision',
            'best_selector_harmful_change_rate',
            'best_selector_wrong_to_wrong_rate',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            base = confusion_metrics(info['baseline'])
            oracle_matrix = apply_selected_matrix(
                info['baseline'],
                info['oracle'],
                info['bg_idx'],
            )
            oracle = confusion_metrics(oracle_matrix)
            best_name = ''
            best_metrics = None
            for selector in info['selectors'].values():
                selector_matrix = apply_selected_matrix(
                    info['baseline'],
                    selector['matrix'],
                    info['bg_idx'],
                )
                metrics = confusion_metrics(selector_matrix)
                if best_metrics is None or metrics['mIoU'] > best_metrics['mIoU']:
                    best_name = selector['name']
                    best_metrics = dict(metrics, selector=selector)
            selected = best_metrics['selector'] if best_metrics else {}
            selected_pixels = selected.get('selected_pixels', 0)
            writer.writerow(dict(
                dataset=dataset,
                images=info['images'],
                baseline_mIoU=base['mIoU'],
                candidate_oracle_mIoU=oracle['mIoU'],
                candidate_oracle_delta_mIoU=oracle['mIoU'] - base['mIoU'],
                baseline_aAcc=base['aAcc'],
                candidate_oracle_aAcc=oracle['aAcc'],
                candidate_oracle_delta_aAcc=oracle['aAcc'] - base['aAcc'],
                baseline_bg_ratio=safe_div(
                    info['baseline_bg_pixels'], info['valid_pixels']),
                protected_background_ratio=safe_div(
                    info['protected_background_pixels'], info['baseline_bg_pixels']),
                recoverable_foreground_ratio=safe_div(
                    info['recoverable_foreground_pixels'], info['baseline_bg_pixels']),
                candidate_correct_rate=safe_div(
                    info['candidate_correct_pixels'],
                    info['recoverable_foreground_pixels']),
                threshold_route_ratio=safe_div(
                    info['threshold_route_pixels'], info['baseline_bg_pixels']),
                weak_bg_route_ratio=safe_div(
                    info['weak_bg_route_pixels'], info['baseline_bg_pixels']),
                best_selector=best_name,
                best_selector_mIoU=(
                    best_metrics['mIoU'] if best_metrics else base['mIoU']),
                best_selector_delta_mIoU=(
                    best_metrics['mIoU'] - base['mIoU']
                    if best_metrics else 0.0),
                best_selector_selected_ratio=safe_div(
                    selected_pixels, info['valid_pixels']),
                best_selector_correction_precision=safe_div(
                    selected.get('improved_pixels', 0),
                    selected.get('improved_pixels', 0)
                    + selected.get('harmed_pixels', 0)),
                best_selector_harmful_change_rate=safe_div(
                    selected.get('harmed_pixels', 0), selected_pixels),
                best_selector_wrong_to_wrong_rate=safe_div(
                    selected.get('wrong_to_wrong_pixels', 0), selected_pixels),
            ))

    with open(feature_path, 'w', newline='') as f:
        fields = [
            'dataset', 'feature', 'auc_recoverable_gt_protected',
            'protected_pixels', 'recoverable_pixels',
            'candidate_correct_pixels', 'protected_mean',
            'recoverable_mean', 'candidate_correct_mean',
            'protected_std', 'recoverable_std',
            'candidate_correct_std', 'protected_min', 'protected_max',
            'recoverable_min', 'recoverable_max',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            for name, feature in sorted(info['features'].items()):
                protected = feature['protected']
                recoverable = feature['recoverable']
                candidate_correct = feature['candidate_correct']
                auc = hist_auc(
                    recoverable.get('hist', []),
                    protected.get('hist', []),
                )
                writer.writerow(dict(
                    dataset=dataset,
                    feature=name,
                    auc_recoverable_gt_protected=auc,
                    protected_pixels=protected.get('pixels', 0),
                    recoverable_pixels=recoverable.get('pixels', 0),
                    candidate_correct_pixels=candidate_correct.get('pixels', 0),
                    protected_mean=bucket_mean(protected),
                    recoverable_mean=bucket_mean(recoverable),
                    candidate_correct_mean=bucket_mean(candidate_correct),
                    protected_std=bucket_std(protected),
                    recoverable_std=bucket_std(recoverable),
                    candidate_correct_std=bucket_std(candidate_correct),
                    protected_min=protected.get('min'),
                    protected_max=protected.get('max'),
                    recoverable_min=recoverable.get('min'),
                    recoverable_max=recoverable.get('max'),
                ))

    with open(selector_path, 'w', newline='') as f:
        fields = [
            'dataset', 'selector', 'baseline_mIoU', 'mIoU', 'delta_mIoU',
            'baseline_aAcc', 'aAcc', 'delta_aAcc', 'selected_pixels',
            'selected_ratio', 'selected_recoverable_ratio',
            'selected_protected_ratio', 'improved_pixels', 'harmed_pixels',
            'wrong_to_wrong_pixels', 'correction_precision',
            'useful_precision', 'harmful_change_rate',
            'wrong_to_wrong_rate',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            base = confusion_metrics(info['baseline'])
            for name, selector in sorted(info['selectors'].items()):
                matrix = apply_selected_matrix(
                    info['baseline'],
                    selector['matrix'],
                    info['bg_idx'],
                )
                metrics = confusion_metrics(matrix)
                selected = selector.get('selected_pixels', 0)
                writer.writerow(dict(
                    dataset=dataset,
                    selector=name,
                    baseline_mIoU=base['mIoU'],
                    mIoU=metrics['mIoU'],
                    delta_mIoU=metrics['mIoU'] - base['mIoU'],
                    baseline_aAcc=base['aAcc'],
                    aAcc=metrics['aAcc'],
                    delta_aAcc=metrics['aAcc'] - base['aAcc'],
                    selected_pixels=selected,
                    selected_ratio=safe_div(selected, info['valid_pixels']),
                    selected_recoverable_ratio=safe_div(
                        selector.get('selected_recoverable_pixels', 0),
                        selected),
                    selected_protected_ratio=safe_div(
                        selector.get('selected_protected_pixels', 0),
                        selected),
                    improved_pixels=selector.get('improved_pixels', 0),
                    harmed_pixels=selector.get('harmed_pixels', 0),
                    wrong_to_wrong_pixels=selector.get(
                        'wrong_to_wrong_pixels', 0),
                    correction_precision=safe_div(
                        selector.get('improved_pixels', 0),
                        selector.get('improved_pixels', 0)
                        + selector.get('harmed_pixels', 0)),
                    useful_precision=safe_div(
                        selector.get('improved_pixels', 0),
                        selected),
                    harmful_change_rate=safe_div(
                        selector.get('harmed_pixels', 0), selected),
                    wrong_to_wrong_rate=safe_div(
                        selector.get('wrong_to_wrong_pixels', 0), selected),
                ))

    with open(class_path, 'w', newline='') as f:
        fields = [
            'dataset', 'class_index', 'class_name', 'role',
            'baseline_bg_pixels', 'candidate_correct_pixels',
            'candidate_correct_rate', 'threshold_route_ratio',
            'weak_bg_route_ratio', 'fg_score_mean',
            'fg_bg_margin_mean', 'reliability_margin_mean',
            'local_support_mean', 'head_votes_mean',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            class_names = info.get('class_names') or []
            class_roles = info.get('class_roles') or []
            for class_idx, row in sorted(info['class_stats'].items()):
                pixels = row.get('baseline_bg_pixels', 0)
                writer.writerow(dict(
                    dataset=dataset,
                    class_index=class_idx,
                    class_name=(
                        class_names[class_idx]
                        if class_idx < len(class_names) else class_idx),
                    role=(
                        class_roles[class_idx]
                        if class_idx < len(class_roles) else ''),
                    baseline_bg_pixels=int(pixels),
                    candidate_correct_pixels=int(
                        row.get('candidate_correct_pixels', 0)),
                    candidate_correct_rate=row_ratio(
                        row, 'candidate_correct_pixels', pixels),
                    threshold_route_ratio=row_ratio(
                        row, 'threshold_route_pixels', pixels),
                    weak_bg_route_ratio=row_ratio(
                        row, 'weak_bg_route_pixels', pixels),
                    fg_score_mean=row_ratio(row, 'fg_score_sum', pixels),
                    fg_bg_margin_mean=row_ratio(
                        row, 'fg_bg_margin_sum', pixels),
                    reliability_margin_mean=row_ratio(
                        row, 'reliability_margin_sum', pixels),
                    local_support_mean=row_ratio(
                        row, 'local_support_sum', pixels),
                    head_votes_mean=row_ratio(row, 'head_votes_sum', pixels),
                ))

    with open(pair_path, 'w', newline='') as f:
        fields = [
            'dataset', 'gt_class_index', 'gt_class_name', 'gt_role',
            'candidate_class_index', 'candidate_class_name',
            'candidate_role', 'pixels', 'correct_candidate_pixels',
            'correct_candidate_rate', 'threshold_route_ratio',
            'weak_bg_route_ratio', 'fg_score_mean',
            'fg_bg_margin_mean', 'reliability_margin_mean',
            'local_support_mean', 'head_votes_mean',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, info in sorted(datasets.items()):
            class_names = info.get('class_names') or []
            class_roles = info.get('class_roles') or []
            for (gt_idx, cand_idx), row in sorted(
                    info['pair_stats'].items(),
                    key=lambda item: (item[0][0], -item[1].get('pixels', 0))):
                pixels = row.get('pixels', 0)
                writer.writerow(dict(
                    dataset=dataset,
                    gt_class_index=gt_idx,
                    gt_class_name=(
                        class_names[gt_idx]
                        if gt_idx < len(class_names) else gt_idx),
                    gt_role=(
                        class_roles[gt_idx]
                        if gt_idx < len(class_roles) else ''),
                    candidate_class_index=cand_idx,
                    candidate_class_name=(
                        class_names[cand_idx]
                        if cand_idx < len(class_names) else cand_idx),
                    candidate_role=(
                        class_roles[cand_idx]
                        if cand_idx < len(class_roles) else ''),
                    pixels=int(pixels),
                    correct_candidate_pixels=int(
                        row.get('correct_candidate_pixels', 0)),
                    correct_candidate_rate=row_ratio(
                        row, 'correct_candidate_pixels', pixels),
                    threshold_route_ratio=row_ratio(
                        row, 'threshold_route_pixels', pixels),
                    weak_bg_route_ratio=row_ratio(
                        row, 'weak_bg_route_pixels', pixels),
                    fg_score_mean=row_ratio(row, 'fg_score_sum', pixels),
                    fg_bg_margin_mean=row_ratio(
                        row, 'fg_bg_margin_sum', pixels),
                    reliability_margin_mean=row_ratio(
                        row, 'reliability_margin_sum', pixels),
                    local_support_mean=row_ratio(
                        row, 'local_support_sum', pixels),
                    head_votes_mean=row_ratio(row, 'head_votes_sum', pixels),
                ))


if __name__ == '__main__':
    main()
