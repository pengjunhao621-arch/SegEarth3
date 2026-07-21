import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize counterfactual review JSONL files.')
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
    return float(numerator) / float(denominator)


def add_matrix(target, source):
    if source is None:
        return
    if not target:
        target.extend([[0 for _ in row] for row in source])
    for row_idx, row in enumerate(source):
        for col_idx, value in enumerate(row):
            target[row_idx][col_idx] += int(value)


def confusion_metrics(matrix):
    if not matrix:
        return dict(
            valid_pixels=0,
            pixel_accuracy=None,
            miou=None,
            ious=[],
        )
    class_count = len(matrix)
    intersections = [matrix[idx][idx] for idx in range(class_count)]
    gt_pixels = [sum(matrix[idx]) for idx in range(class_count)]
    pred_pixels = [
        sum(matrix[row_idx][col_idx] for row_idx in range(class_count))
        for col_idx in range(class_count)
    ]
    unions = [
        gt_pixels[idx] + pred_pixels[idx] - intersections[idx]
        for idx in range(class_count)
    ]
    ious = [
        safe_div(intersections[idx], unions[idx])
        if unions[idx] > 0 else None
        for idx in range(class_count)
    ]
    valid_ious = [value for value in ious if value is not None]
    total = sum(gt_pixels)
    return dict(
        valid_pixels=total,
        pixel_accuracy=safe_div(sum(intersections), total),
        miou=(
            sum(valid_ious) / len(valid_ious)
            if valid_ious else None),
        ious=ious,
        gt_pixels=gt_pixels,
        pred_pixels=pred_pixels,
        intersections=intersections,
        unions=unions,
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    datasets = defaultdict(lambda: dict(
        baseline_confusion=[],
        reviewed_confusion=[],
        images=0,
        valid_pixels=0,
        proposal_pixels=0,
        review_gate_pixels=0,
        accepted_pixels=0,
        changed_pixels=0,
        improved_pixels=0,
        harmed_pixels=0,
        risk_class_count_sum=0,
    ))
    action_groups = defaultdict(lambda: defaultdict(int))
    pair_groups = defaultdict(lambda: defaultdict(int))
    region_groups = defaultdict(lambda: defaultdict(float))
    class_names = defaultdict(dict)
    metadata = {}

    for record in records:
        stats = record.get('counterfactual_review_stats')
        if not stats:
            continue
        dataset = str(
            stats.get('dataset_name')
            or record.get('dataset_name')
            or 'unknown')
        group = datasets[dataset]
        group['images'] += 1
        add_matrix(
            group['baseline_confusion'],
            stats.get('baseline_confusion'))
        add_matrix(
            group['reviewed_confusion'],
            stats.get('reviewed_confusion'))
        for field in (
                'valid_pixels',
                'proposal_pixels',
                'review_gate_pixels',
                'accepted_pixels',
                'changed_pixels',
                'improved_pixels',
                'harmed_pixels'):
            group[field] += int(stats.get(field, 0) or 0)
        group['risk_class_count_sum'] += len(
            stats.get('risk_class_indices') or [])
        metadata[dataset] = dict(
            validation_mode=record.get(
                'counterfactual_review_validation_mode'),
            evidence_sources=record.get(
                'counterfactual_review_evidence_sources'),
            scale=record.get('counterfactual_review_scale'),
            topk=record.get('counterfactual_review_topk'),
            min_scaled_margin=record.get(
                'counterfactual_review_min_scaled_margin'),
            min_relative_stability=record.get(
                'counterfactual_review_min_relative_stability'),
            min_region_gate_ratio=record.get(
                'counterfactual_review_min_region_gate_ratio'),
            min_region_stability=record.get(
                'counterfactual_review_min_region_stability'),
            blend=record.get('counterfactual_review_blend'),
        )

        for row in stats.get('class_rows') or []:
            class_names[dataset][int(row['class_index'])] = (
                row.get('class_name'))

        for row in stats.get('action_rows') or []:
            key = (dataset, str(row['action_name']))
            bucket = action_groups[key]
            bucket['images'] += 1
            for field in (
                    'proposal_pixels',
                    'accepted_pixels',
                    'changed_pixels',
                    'improved_pixels',
                    'harmed_pixels'):
                bucket[field] += int(row.get(field, 0) or 0)

        for row in stats.get('pair_rows') or []:
            key = (
                dataset,
                int(row['gt_class_index']),
                str(row['gt_class_name']),
                int(row['base_pred_class_index']),
                str(row['base_pred_class_name']),
            )
            bucket = pair_groups[key]
            for field in (
                    'pixels',
                    'proposal_pixels',
                    'review_gate_pixels',
                    'accepted_pixels',
                    'changed_pixels',
                    'corrected_pixels'):
                bucket[field] += int(row.get(field, 0) or 0)

        for row in stats.get('region_rows') or []:
            key = (
                dataset,
                str(row['action_name']),
                int(row['target_class_index']),
                str(row['target_class_name']),
                int(row['competitor_class_index']),
                str(row['competitor_class_name']),
                bool(row.get('accepted')),
            )
            bucket = region_groups[key]
            bucket['regions'] += 1
            bucket['region_pixels'] += int(
                row.get('region_pixels', 0) or 0)
            bucket['review_gate_pixels'] += int(
                row.get('review_gate_pixels', 0) or 0)
            bucket['relative_stability_sum'] += float(
                row.get('mean_relative_stability', 0.0) or 0.0)
            bucket['scaled_margin_sum'] += float(
                row.get('mean_scaled_margin', 0.0) or 0.0)

    overall_rows = []
    class_rows = []
    for dataset, group in sorted(datasets.items()):
        baseline = confusion_metrics(group['baseline_confusion'])
        reviewed = confusion_metrics(group['reviewed_confusion'])
        overall_rows.append(dict(
            dataset_name=dataset,
            images=group['images'],
            valid_pixels=baseline['valid_pixels'],
            baseline_miou=baseline['miou'],
            reviewed_miou=reviewed['miou'],
            miou_delta=(
                reviewed['miou'] - baseline['miou']),
            baseline_pixel_accuracy=baseline['pixel_accuracy'],
            reviewed_pixel_accuracy=reviewed['pixel_accuracy'],
            pixel_accuracy_delta=(
                reviewed['pixel_accuracy']
                - baseline['pixel_accuracy']),
            proposal_pixels=group['proposal_pixels'],
            proposal_ratio=safe_div(
                group['proposal_pixels'], baseline['valid_pixels']),
            review_gate_pixels=group['review_gate_pixels'],
            review_gate_ratio=safe_div(
                group['review_gate_pixels'],
                group['proposal_pixels']),
            accepted_pixels=group['accepted_pixels'],
            accepted_ratio=safe_div(
                group['accepted_pixels'], baseline['valid_pixels']),
            changed_pixels=group['changed_pixels'],
            changed_ratio=safe_div(
                group['changed_pixels'], baseline['valid_pixels']),
            improved_pixels=group['improved_pixels'],
            harmed_pixels=group['harmed_pixels'],
            net_improved_pixels=(
                group['improved_pixels'] - group['harmed_pixels']),
            utility_precision=safe_div(
                group['improved_pixels'],
                group['improved_pixels'] + group['harmed_pixels']),
            mean_risk_class_count=safe_div(
                group['risk_class_count_sum'], group['images']),
            **metadata.get(dataset, {}),
        ))
        for class_idx, baseline_iou in enumerate(baseline['ious']):
            reviewed_iou = reviewed['ious'][class_idx]
            class_rows.append(dict(
                dataset_name=dataset,
                class_index=class_idx,
                class_name=class_names[dataset].get(class_idx),
                baseline_iou=baseline_iou,
                reviewed_iou=reviewed_iou,
                iou_delta=(
                    reviewed_iou - baseline_iou
                    if (
                        baseline_iou is not None
                        and reviewed_iou is not None)
                    else None),
                gt_pixels=baseline['gt_pixels'][class_idx],
                baseline_pred_pixels=baseline['pred_pixels'][class_idx],
                reviewed_pred_pixels=reviewed['pred_pixels'][class_idx],
            ))

    action_rows = []
    for (dataset, action_name), bucket in sorted(
            action_groups.items()):
        action_rows.append(dict(
            dataset_name=dataset,
            action_name=action_name,
            images=int(bucket['images']),
            proposal_pixels=int(bucket['proposal_pixels']),
            accepted_pixels=int(bucket['accepted_pixels']),
            acceptance_ratio=safe_div(
                bucket['accepted_pixels'],
                bucket['proposal_pixels']),
            changed_pixels=int(bucket['changed_pixels']),
            improved_pixels=int(bucket['improved_pixels']),
            harmed_pixels=int(bucket['harmed_pixels']),
            net_improved_pixels=(
                int(bucket['improved_pixels'])
                - int(bucket['harmed_pixels'])),
            utility_precision=safe_div(
                bucket['improved_pixels'],
                bucket['improved_pixels'] + bucket['harmed_pixels']),
        ))

    pair_rows = []
    for key, bucket in sorted(pair_groups.items()):
        (
            dataset,
            gt_idx,
            gt_name,
            pred_idx,
            pred_name,
        ) = key
        pair_rows.append(dict(
            dataset_name=dataset,
            gt_class_index=gt_idx,
            gt_class_name=gt_name,
            base_pred_class_index=pred_idx,
            base_pred_class_name=pred_name,
            pixels=int(bucket['pixels']),
            proposal_pixels=int(bucket['proposal_pixels']),
            proposal_coverage=safe_div(
                bucket['proposal_pixels'], bucket['pixels']),
            review_gate_pixels=int(bucket['review_gate_pixels']),
            accepted_pixels=int(bucket['accepted_pixels']),
            accepted_coverage=safe_div(
                bucket['accepted_pixels'], bucket['pixels']),
            changed_pixels=int(bucket['changed_pixels']),
            corrected_pixels=int(bucket['corrected_pixels']),
            correction_ratio=safe_div(
                bucket['corrected_pixels'], bucket['pixels']),
            correction_precision=safe_div(
                bucket['corrected_pixels'],
                bucket['changed_pixels']),
        ))
    pair_rows.sort(
        key=lambda row: (
            row['dataset_name'],
            -row['pixels'],
        ))

    region_rows = []
    for key, bucket in sorted(region_groups.items()):
        (
            dataset,
            action_name,
            target_idx,
            target_name,
            competitor_idx,
            competitor_name,
            accepted,
        ) = key
        regions = int(bucket['regions'])
        region_rows.append(dict(
            dataset_name=dataset,
            action_name=action_name,
            target_class_index=target_idx,
            target_class_name=target_name,
            competitor_class_index=competitor_idx,
            competitor_class_name=competitor_name,
            accepted=accepted,
            regions=regions,
            region_pixels=int(bucket['region_pixels']),
            mean_region_pixels=safe_div(
                bucket['region_pixels'], regions),
            mean_review_gate_ratio=safe_div(
                bucket['review_gate_pixels'],
                bucket['region_pixels']),
            mean_relative_stability=safe_div(
                bucket['relative_stability_sum'], regions),
            mean_scaled_margin=safe_div(
                bucket['scaled_margin_sum'], regions),
        ))

    return dict(
        overall=overall_rows,
        actions=action_rows,
        classes=class_rows,
        pairs=pair_rows,
        regions=region_rows,
    )


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(
            f'No JSONL inputs matched: {args.inputs}')
    tables = summarize(iter_records(paths))
    outputs = {
        'counterfactual_review_overall.csv': tables['overall'],
        'counterfactual_review_action_summary.csv': tables['actions'],
        'counterfactual_review_class_summary.csv': tables['classes'],
        'counterfactual_review_pair_summary.csv': tables['pairs'],
        'counterfactual_review_region_summary.csv': tables['regions'],
    }
    for name, rows in outputs.items():
        write_csv(os.path.join(args.out_dir, name), rows)
    print(f'Read {len(paths)} JSONL files.')
    print(f'Wrote summaries to {args.out_dir}')


if __name__ == '__main__':
    main()
