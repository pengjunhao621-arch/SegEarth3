import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize SegEarth-OV3 evidence JSONL files.')
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='JSONL files or glob patterns, e.g. logs/evidence/vdd_rank*.jsonl')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Directory for class_summary.csv and prompt_summary.csv')
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
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def iter_prompt_stats(evidence):
    if evidence is None:
        return
    if 'prompt_stats' in evidence:
        for item in evidence['prompt_stats']:
            yield item
    for crop in evidence.get('crop_stats', []):
        for item in crop.get('prompt_stats', []):
            yield item


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_div(num, den):
    if den == 0:
        return None
    return num / den


def add_mean_stat(bucket, key, value):
    if value is not None:
        bucket[key].append(value)


def summarize(records):
    class_buckets = defaultdict(lambda: defaultdict(list))
    class_counts = defaultdict(lambda: dict(tp=0, fp=0, fn=0, images=0))
    prompt_buckets = defaultdict(lambda: defaultdict(list))
    prompt_counts = defaultdict(lambda: dict(records=0))

    for record in records:
        for class_stat in record.get('class_stats', []):
            class_idx = class_stat['class_index']
            bucket = class_buckets[class_idx]
            count = class_counts[class_idx]
            count['images'] += 1
            count['tp'] += class_stat.get('tp') or 0
            count['fp'] += class_stat.get('fp') or 0
            count['fn'] += class_stat.get('fn') or 0
            bucket['class_name'].append(class_stat.get('class_name'))
            for key in [
                'logit_max',
                'logit_mean',
                'logit_area_prob_thd',
                'pred_area_ratio',
                'gt_area_ratio',
                'mean_winning_score',
                'iou',
                'precision',
                'recall',
            ]:
                add_mean_stat(bucket, key, class_stat.get(key))

        for prompt_stat in iter_prompt_stats(record.get('evidence')) or []:
            prompt_key = (prompt_stat['class_index'], prompt_stat['prompt'])
            bucket = prompt_buckets[prompt_key]
            count = prompt_counts[prompt_key]
            count['records'] += 1
            bucket['class_name'].append(prompt_stat.get('class_name'))
            for key in [
                'presence_score',
                'raw_candidate_count',
                'raw_keep_count',
                'kept_instance_count',
                'raw_keep_ratio',
                'raw_score_max',
                'raw_score_mean',
                'presence_score_max',
                'presence_score_mean',
                'kept_raw_score_max',
                'kept_raw_score_mean',
                'kept_presence_score_max',
                'kept_presence_score_mean',
                'instance_area_union_05',
                'instance_area_max_05',
                'sem_inst_iou_05',
                'fusion_semantic_win_ratio',
            ]:
                add_mean_stat(bucket, key, prompt_stat.get(key))
            for prefix in ['semantic', 'instance', 'final']:
                nested = prompt_stat.get(prefix) or {}
                for key in ['max', 'mean', 'area_ratio']:
                    add_mean_stat(bucket, f'{prefix}_{key}', nested.get(key))

    class_rows = []
    for class_idx, bucket in sorted(class_buckets.items()):
        count = class_counts[class_idx]
        tp, fp, fn = count['tp'], count['fp'], count['fn']
        class_rows.append(dict(
            class_index=class_idx,
            class_name=first_non_none(bucket['class_name']),
            images=count['images'],
            dataset_iou=safe_div(tp, tp + fp + fn),
            dataset_precision=safe_div(tp, tp + fp),
            dataset_recall=safe_div(tp, tp + fn),
            **{f'mean_{key}': mean(values)
               for key, values in bucket.items()
               if key != 'class_name'}
        ))

    prompt_rows = []
    for (class_idx, prompt), bucket in sorted(prompt_buckets.items()):
        prompt_rows.append(dict(
            class_index=class_idx,
            class_name=first_non_none(bucket['class_name']),
            prompt=prompt,
            records=prompt_counts[(class_idx, prompt)]['records'],
            **{f'mean_{key}': mean(values)
               for key, values in bucket.items()
               if key != 'class_name'}
        ))

    return class_rows, prompt_rows


def first_non_none(values):
    for value in values:
        if value is not None:
            return value
    return None


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        'class_index',
        'class_name',
        'prompt',
        'images',
        'records',
        'dataset_iou',
        'dataset_precision',
        'dataset_recall',
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f'No input files matched: {args.inputs}')
    os.makedirs(args.out_dir, exist_ok=True)
    class_rows, prompt_rows = summarize(iter_records(paths))
    write_csv(os.path.join(args.out_dir, 'class_summary.csv'), class_rows)
    write_csv(os.path.join(args.out_dir, 'prompt_summary.csv'), prompt_rows)
    print(f'Read {len(paths)} files')
    print(f'Wrote {len(class_rows)} class rows and {len(prompt_rows)} prompt rows to {args.out_dir}')


if __name__ == '__main__':
    main()
