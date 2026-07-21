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
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_thresholds(text):
    if text is None:
        text = '-0.30,-0.20,-0.10,-0.05,0.00,0.05,0.10,0.20,0.30'
    out = []
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def iter_pair_rows(paths):
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
                stats = record.get('self_prompted_concept_verification')
                if not stats:
                    continue
                dataset = (
                    record.get('dataset_name')
                    or stats.get('dataset_name')
                    or 'unknown')
                image_id = record.get('img_path') or record.get('image_id')
                for pair in stats.get('pair_stats') or []:
                    if pair.get('error'):
                        continue
                    yield dataset, image_id, record, stats, pair


def new_acc(base=None):
    acc = defaultdict(float)
    acc['images'] = set()
    if base:
        acc.update(base)
    return acc


def add_weighted(acc, key, value, weight):
    if value is None or weight is None or weight <= 0:
        return
    acc[key] += float(value) * float(weight)


def update_acc(acc, image_id, pair):
    acc['images'].add(image_id)
    support = int(pair.get('pair_pixels') or pair.get('support_pixels') or 0)
    target_gt = int(pair.get('target_gt_pixels') or 0)
    pred_gt = int(pair.get('pred_gt_pixels') or 0)
    other_gt = int(pair.get('other_gt_pixels') or 0)
    help_pixels = int(pair.get('target_help_pixels') or 0)
    harm_pixels = int(pair.get('pred_harm_pixels') or 0)
    other_switch = int(pair.get('other_switch_pixels') or 0)

    acc['rows'] += 1
    acc['support_pixels'] += support
    acc['target_gt_pixels'] += target_gt
    acc['pred_gt_pixels'] += pred_gt
    acc['other_gt_pixels'] += other_gt
    acc['target_help_pixels'] += help_pixels
    acc['pred_harm_pixels'] += harm_pixels
    acc['other_switch_pixels'] += other_switch
    acc['oracle_net_pixels'] += target_gt - pred_gt
    acc['helpful_rows'] += 1 if target_gt > pred_gt else 0
    acc['harmful_rows'] += 1 if pred_gt >= target_gt else 0
    acc['target_requery_beats_pred_pixels'] += int(
        pair.get('target_requery_beats_pred_pixels') or 0)
    acc['pred_num_masks_sum'] += float(pair.get('pred_num_masks') or 0.0)
    acc['target_num_masks_sum'] += float(pair.get('target_num_masks') or 0.0)
    acc['pred_presence_sum'] += float(pair.get('pred_presence_score') or 0.0)
    acc['target_presence_sum'] += float(
        pair.get('target_presence_score') or 0.0)

    for src_key, dst_key in [
            ('mean_base_target_minus_pred', 'base_margin_sum'),
            ('mean_requery_target_minus_pred', 'requery_margin_sum'),
            ('mean_requery_margin_gain', 'margin_gain_sum'),
            ('mean_target_gain', 'target_gain_sum'),
            ('mean_pred_gain', 'pred_gain_sum'),
            ('mean_target_gain_minus_pred_gain', 'gain_diff_sum'),
            ('target_original_requery_iou', 'target_iou_sum'),
            ('pred_original_requery_iou', 'pred_iou_sum'),
            ('target_requery_support_ratio', 'target_support_ratio_sum'),
            ('pred_requery_support_ratio', 'pred_support_ratio_sum'),
    ]:
        add_weighted(acc, dst_key, pair.get(src_key), support)


def finalize_acc(acc):
    support = acc['support_pixels']
    rows = acc['rows']
    row = dict(acc)
    row['images'] = len(acc['images'])
    row['target_gt_ratio'] = safe_div(acc['target_gt_pixels'], support)
    row['pred_gt_ratio'] = safe_div(acc['pred_gt_pixels'], support)
    row['other_gt_ratio'] = safe_div(acc['other_gt_pixels'], support)
    row['direction_helpful_row_ratio'] = safe_div(acc['helpful_rows'], rows)
    row['oracle_net_ratio'] = safe_div(acc['oracle_net_pixels'], support)
    row['target_help_ratio'] = safe_div(
        acc['target_help_pixels'], acc['target_gt_pixels'])
    row['pred_harm_ratio'] = safe_div(
        acc['pred_harm_pixels'], acc['pred_gt_pixels'])
    row['requery_switch_ratio'] = safe_div(
        acc['target_requery_beats_pred_pixels'], support)
    row['useful_switch_precision'] = safe_div(
        acc['target_help_pixels'],
        acc['target_help_pixels'] + acc['pred_harm_pixels'])
    row['changed_pixel_precision'] = safe_div(
        acc['target_help_pixels'],
        acc['target_help_pixels']
        + acc['pred_harm_pixels']
        + acc['other_switch_pixels'])
    row['mean_base_target_minus_pred'] = safe_div(
        acc['base_margin_sum'], support)
    row['mean_requery_target_minus_pred'] = safe_div(
        acc['requery_margin_sum'], support)
    row['mean_requery_margin_gain'] = safe_div(
        acc['margin_gain_sum'], support)
    row['mean_target_gain'] = safe_div(acc['target_gain_sum'], support)
    row['mean_pred_gain'] = safe_div(acc['pred_gain_sum'], support)
    row['mean_target_gain_minus_pred_gain'] = safe_div(
        acc['gain_diff_sum'], support)
    row['mean_target_original_requery_iou'] = safe_div(
        acc['target_iou_sum'], support)
    row['mean_pred_original_requery_iou'] = safe_div(
        acc['pred_iou_sum'], support)
    row['mean_target_requery_support_ratio'] = safe_div(
        acc['target_support_ratio_sum'], support)
    row['mean_pred_requery_support_ratio'] = safe_div(
        acc['pred_support_ratio_sum'], support)
    row['mean_target_presence_score'] = safe_div(
        acc['target_presence_sum'], rows)
    row['mean_pred_presence_score'] = safe_div(acc['pred_presence_sum'], rows)
    row['mean_target_num_masks'] = safe_div(acc['target_num_masks_sum'], rows)
    row['mean_pred_num_masks'] = safe_div(acc['pred_num_masks_sum'], rows)
    return row


def feature_value(pair, name):
    if name == 'base_margin':
        return pair.get('mean_base_target_minus_pred')
    if name == 'requery_margin':
        return pair.get('mean_requery_target_minus_pred')
    if name == 'margin_gain':
        return pair.get('mean_requery_margin_gain')
    if name == 'gain_diff':
        return pair.get('mean_target_gain_minus_pred_gain')
    if name == 'presence_diff':
        if pair.get('target_presence_score') is None:
            return None
        return float(pair.get('target_presence_score') or 0.0) - float(
            pair.get('pred_presence_score') or 0.0)
    if name == 'mask_count_diff':
        return float(pair.get('target_num_masks') or 0.0) - float(
            pair.get('pred_num_masks') or 0.0)
    if name == 'support_ratio_diff':
        if pair.get('target_requery_support_ratio') is None:
            return None
        return float(pair.get('target_requery_support_ratio') or 0.0) - float(
            pair.get('pred_requery_support_ratio') or 0.0)
    if name == 'target_iou_minus_pred_iou':
        if pair.get('target_original_requery_iou') is None:
            return None
        return float(pair.get('target_original_requery_iou') or 0.0) - float(
            pair.get('pred_original_requery_iou') or 0.0)
    return pair.get(name)


def weighted_auc(labels, scores, weights):
    items = [
        (float(s), int(y), float(w))
        for y, s, w in zip(labels, scores, weights)
        if s is not None and w > 0
    ]
    if not items:
        return None
    pos = sum(w for _, y, w in items if y == 1)
    neg = sum(w for _, y, w in items if y == 0)
    if pos <= 0 or neg <= 0:
        return None
    items.sort(key=lambda x: x[0])
    neg_before = 0.0
    auc_num = 0.0
    i = 0
    while i < len(items):
        j = i
        score = items[i][0]
        pos_tie = 0.0
        neg_tie = 0.0
        while j < len(items) and items[j][0] == score:
            if items[j][1] == 1:
                pos_tie += items[j][2]
            else:
                neg_tie += items[j][2]
            j += 1
        auc_num += pos_tie * (neg_before + 0.5 * neg_tie)
        neg_before += neg_tie
        i = j
    return auc_num / (pos * neg)


def weighted_ap(labels, scores, weights):
    items = [
        (float(s), int(y), float(w))
        for y, s, w in zip(labels, scores, weights)
        if s is not None and w > 0
    ]
    if not items:
        return None
    pos = sum(w for _, y, w in items if y == 1)
    if pos <= 0:
        return None
    items.sort(key=lambda x: x[0], reverse=True)
    tp = 0.0
    total = 0.0
    ap = 0.0
    for _, y, w in items:
        total += w
        if y == 1:
            tp += w
            ap += (tp / total) * w
    return ap / pos


def summarize(paths, thresholds):
    dataset_acc = {}
    role_acc = {}
    pair_acc = {}
    feature_rows = []
    switch_rows = []
    flat_rows = []

    feature_names = [
        'base_margin',
        'requery_margin',
        'margin_gain',
        'gain_diff',
        'presence_diff',
        'mask_count_diff',
        'support_ratio_diff',
        'target_iou_minus_pred_iou',
    ]
    feature_store = defaultdict(lambda: defaultdict(lambda: {
        'labels': [], 'scores': [], 'weights': []}))
    switch_store = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for dataset, image_id, record, stats, pair in iter_pair_rows(paths):
        pred_name = pair.get('pred_class_name')
        target_name = pair.get('target_class_name')
        pred_role = pair.get('pred_role')
        target_role = pair.get('target_role')
        role_pair = f'{pred_role}->{target_role}'
        pair_name = f'{pred_name}->{target_name}'
        support = int(pair.get('pair_pixels') or pair.get('support_pixels') or 0)
        target_gt = int(pair.get('target_gt_pixels') or 0)
        pred_gt = int(pair.get('pred_gt_pixels') or 0)
        other_gt = int(pair.get('other_gt_pixels') or 0)
        label = 1 if target_gt > pred_gt else 0

        for table, key, base in [
                (dataset_acc, (dataset,), dict(dataset=dataset)),
                (role_acc, (dataset, role_pair),
                 dict(dataset=dataset, role_pair=role_pair,
                      pred_role=pred_role, target_role=target_role)),
                (pair_acc, (dataset, pair_name),
                 dict(dataset=dataset, pair=pair_name,
                      pred_class_name=pred_name,
                      target_class_name=target_name,
                      pred_role=pred_role, target_role=target_role)),
        ]:
            if key not in table:
                table[key] = new_acc(base)
            update_acc(table[key], image_id, pair)

        for feature in feature_names:
            value = feature_value(pair, feature)
            if value is None:
                continue
            for scope in ['all', dataset]:
                feature_store[scope][feature]['labels'].append(label)
                feature_store[scope][feature]['scores'].append(value)
                feature_store[scope][feature]['weights'].append(max(1, support))

        for feature in ['requery_margin', 'margin_gain', 'gain_diff']:
            value = feature_value(pair, feature)
            if value is None:
                continue
            for threshold in thresholds:
                if float(value) < float(threshold):
                    continue
                for scope in ['all', dataset]:
                    bucket = switch_store[scope][(feature, threshold)]
                    bucket['rows'] += 1
                    bucket['support_pixels'] += support
                    bucket['improve_pixels'] += target_gt
                    bucket['harm_pixels'] += pred_gt
                    bucket['wrong_to_wrong_pixels'] += other_gt
                    bucket['net_pixels'] += target_gt - pred_gt

        flat_rows.append(dict(
            dataset=dataset,
            image_id=image_id,
            pred_class_name=pred_name,
            target_class_name=target_name,
            pair=pair_name,
            role_pair=role_pair,
            support_pixels=support,
            target_gt_pixels=target_gt,
            pred_gt_pixels=pred_gt,
            other_gt_pixels=other_gt,
            target_direction_is_helpful=label,
            base_margin=feature_value(pair, 'base_margin'),
            requery_margin=feature_value(pair, 'requery_margin'),
            margin_gain=feature_value(pair, 'margin_gain'),
            gain_diff=feature_value(pair, 'gain_diff'),
            presence_diff=feature_value(pair, 'presence_diff'),
            support_ratio_diff=feature_value(pair, 'support_ratio_diff'),
            target_prompt=pair.get('target_prompt'),
            pred_prompt=pair.get('pred_prompt'),
        ))

    dataset_rows = [finalize_acc(acc) for acc in dataset_acc.values()]
    role_rows = [finalize_acc(acc) for acc in role_acc.values()]
    pair_rows = [finalize_acc(acc) for acc in pair_acc.values()]

    for scope, by_feature in sorted(feature_store.items()):
        for feature, data in sorted(by_feature.items()):
            labels = data['labels']
            scores = data['scores']
            weights = data['weights']
            feature_rows.append(dict(
                scope=scope,
                feature=feature,
                rows=len(labels),
                weighted_positive_rate=safe_div(
                    sum(w for y, w in zip(labels, weights) if y == 1),
                    sum(weights)),
                auroc=weighted_auc(labels, scores, weights),
                ap=weighted_ap(labels, scores, weights),
            ))

    for scope, by_key in sorted(switch_store.items()):
        for (feature, threshold), bucket in sorted(by_key.items()):
            improve = bucket['improve_pixels']
            harm = bucket['harm_pixels']
            wrong = bucket['wrong_to_wrong_pixels']
            support = bucket['support_pixels']
            switch_rows.append(dict(
                scope=scope,
                feature=feature,
                threshold=threshold,
                rows=int(bucket['rows']),
                changed_pixels=int(support),
                improve_pixels=int(improve),
                harm_pixels=int(harm),
                wrong_to_wrong_pixels=int(wrong),
                net_pixels=int(bucket['net_pixels']),
                net_ratio=safe_div(bucket['net_pixels'], support),
                useful_switch_precision=safe_div(improve, improve + harm),
                changed_pixel_precision=safe_div(improve, support),
            ))

    return dataset_rows, role_rows, pair_rows, feature_rows, switch_rows, flat_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--thresholds', default=None)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(
            'No self-prompted concept verification JSONL found.')

    tables = summarize(paths, parse_thresholds(args.thresholds))
    names = [
        'spcv_dataset_summary.csv',
        'spcv_role_pair_summary.csv',
        'spcv_pair_summary.csv',
        'spcv_feature_separability.csv',
        'spcv_switch_curve.csv',
        'spcv_pair_rows.csv',
    ]
    for name, rows in zip(names, tables):
        write_csv(os.path.join(args.out_dir, name), rows)
    print(f'Wrote self-prompted concept verification summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
