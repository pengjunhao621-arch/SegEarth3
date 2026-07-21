#!/usr/bin/env python3
"""Summarize Prompt-SAM3 panel artifacts and optional intervention changes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic_dir", help="Directory containing rank*/image.npz")
    parser.add_argument("--comparison-dir", help="Matching static/no-component directory")
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_records(root):
    records = {}
    for path in sorted(Path(root).rglob("*.npz")):
        records[path.stem] = dict(np.load(path, allow_pickle=False))
    if not records:
        raise FileNotFoundError(f"No .npz diagnostics found below {root}")
    return records


def resize_label(label, size):
    height, width = size
    if label.shape == (height, width):
        return label
    image = Image.fromarray(label.astype(np.int32), mode="I")
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(image.resize((width, height), resample=resampling.NEAREST))


def confusion_matrix(prediction, target, num_classes, ignore_index):
    valid = target != ignore_index
    encoded = target[valid].astype(np.int64) * num_classes + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pred = confusion.sum(axis=0).astype(np.float64)
    union = gt + pred - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    recall = np.divide(true_positive, gt, out=np.full_like(true_positive, np.nan), where=gt > 0)
    return dict(
        mIoU=float(np.nanmean(iou)),
        mRecall=float(np.nanmean(recall)),
        per_class_iou=iou.tolist(),
        per_class_recall=recall.tolist(),
        confusion=confusion.tolist(),
    )


def summarize(records, diagnostic_dir, num_classes, ignore_index):
    confusion = {
        head: np.zeros((num_classes, num_classes), dtype=np.int64)
        for head in ("final", "semantic", "instance")
    }
    ce_values = []
    class_ce_sum = np.zeros(num_classes, dtype=np.float64)
    class_ce_count = np.zeros(num_classes, dtype=np.int64)
    class_error = np.zeros(num_classes, dtype=np.int64)
    class_pixels = np.zeros(num_classes, dtype=np.int64)
    prototype_cosines = []
    prototype_ranks = []
    attention_entropies = []
    attention_overlaps = []
    dynamic_vectors = []
    for stem, record in records.items():
        ground_truth = record["ground_truth"].astype(np.int64)
        for head in confusion:
            score = record[head].astype(np.float32)
            target = resize_label(ground_truth, score.shape[-2:])
            prediction = score.argmax(axis=0)
            confusion[head] += confusion_matrix(
                prediction, target, num_classes, ignore_index
            )
        score = record["final"].astype(np.float64)
        probability = np.maximum(score, 1e-6)
        probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-6)
        target = resize_label(ground_truth, score.shape[-2:])
        valid = target != ignore_index
        rows, cols = np.indices(target.shape)
        nll = -np.log(np.maximum(probability[target.clip(0, num_classes - 1), rows, cols], 1e-6))
        ce_values.append(nll[valid])
        prediction = probability.argmax(axis=0)
        for class_id in range(num_classes):
            selected = valid & (target == class_id)
            class_ce_sum[class_id] += nll[selected].sum()
            class_ce_count[class_id] += selected.sum()
            class_error[class_id] += (prediction[selected] != class_id).sum()
            class_pixels[class_id] += selected.sum()
        if "dynamic_inputs" in record:
            dynamic_vectors.append(
                record["dynamic_inputs"].astype(np.float32).mean(axis=(0, 1))
            )
        if "state_attention" in record:
            attention = record["state_attention"].astype(np.float32)
            attention = attention.reshape(-1, attention.shape[-2], attention.shape[-1])
            norm = np.linalg.norm(attention, axis=-1, keepdims=True)
            normalized = attention / np.maximum(norm, 1e-8)
            cosine = normalized @ normalized.transpose(0, 2, 1)
            k = cosine.shape[-1]
            if k > 1:
                off_diagonal = ~np.eye(k, dtype=bool)
                attention_overlaps.append(float(cosine[:, off_diagonal].mean()))

    for path in Path(diagnostic_dir).rglob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "prototype_pair_cosine" in item:
            prototype_cosines.append(item["prototype_pair_cosine"])
        if "prototype_effective_rank" in item:
            prototype_ranks.append(item["prototype_effective_rank"])
        if "attention_entropy" in item:
            attention_entropies.append(item["attention_entropy"])

    all_ce = np.concatenate(ce_values) if ce_values else np.array([], dtype=np.float64)
    ce_quantiles = {
        str(q): float(np.quantile(all_ce, q)) if all_ce.size else None
        for q in (0.5, 0.9, 0.95, 0.99)
    }
    if all_ce.size:
        hard_count = max(1, int(math.ceil(0.10 * all_ce.size)))
        hard_ce = np.partition(all_ce, all_ce.size - hard_count)[-hard_count:]
        hard_ce_share = float(hard_ce.sum() / max(all_ce.sum(), 1e-12))
    else:
        hard_ce_share = 0.0
    class_mean_ce = np.divide(
        class_ce_sum,
        class_ce_count,
        out=np.full(num_classes, np.nan),
        where=class_ce_count > 0,
    )
    class_error_rate = np.divide(
        class_error,
        class_pixels,
        out=np.full(num_classes, np.nan),
        where=class_pixels > 0,
    )
    return dict(
        images=len(records),
        heads={key: metrics_from_confusion(value) for key, value in confusion.items()},
        focal_evidence=dict(
            ce_quantiles=ce_quantiles,
            top10_percent_ce_share=hard_ce_share,
            per_class_mean_ce=class_mean_ce.tolist(),
            per_class_error_rate=class_error_rate.tolist(),
            per_class_pixels=class_pixels.tolist(),
        ),
        prototype_evidence=dict(
            mean_pair_cosine=float(np.mean(prototype_cosines)) if prototype_cosines else None,
            mean_effective_rank=float(np.mean(prototype_ranks)) if prototype_ranks else None,
            mean_attention_entropy=float(np.mean(attention_entropies)) if attention_entropies else None,
            mean_attention_overlap=float(np.mean(attention_overlaps)) if attention_overlaps else None,
            cross_image_prompt_variance=(
                float(np.var(np.stack(dynamic_vectors), axis=0).mean())
                if len(dynamic_vectors) > 1
                else None
            ),
            collapse_signal=(
                bool(np.mean(prototype_cosines) > 0.9)
                if prototype_cosines
                else None
            ),
        ),
    )


def compare(full, other, ignore_index):
    common = sorted(set(full) & set(other))
    improved = harmed = unchanged = valid_total = 0
    for stem in common:
        gt = full[stem]["ground_truth"].astype(np.int64)
        full_pred = full[stem]["prediction"].astype(np.int64)
        other_pred = other[stem]["prediction"].astype(np.int64)
        if gt.shape != full_pred.shape:
            gt = resize_label(gt, full_pred.shape)
        if other_pred.shape != full_pred.shape:
            other_pred = resize_label(other_pred, full_pred.shape)
        valid = gt != ignore_index
        full_correct = full_pred == gt
        other_correct = other_pred == gt
        improved += np.sum(valid & full_correct & ~other_correct)
        harmed += np.sum(valid & ~full_correct & other_correct)
        unchanged += np.sum(valid & (full_correct == other_correct))
        valid_total += np.sum(valid)
    return dict(
        matched_images=len(common),
        corrected_pixels=int(improved),
        harmed_pixels=int(harmed),
        unchanged_pixels=int(unchanged),
        corrected_ratio=float(improved / max(valid_total, 1)),
        harmed_ratio=float(harmed / max(valid_total, 1)),
        net_corrected_pixels=int(improved - harmed),
    )


if __name__ == "__main__":
    args = parse_args()
    records = load_records(args.diagnostic_dir)
    report = summarize(
        records, args.diagnostic_dir, args.num_classes, args.ignore_index
    )
    if args.comparison_dir:
        comparison = load_records(args.comparison_dir)
        report["full_vs_comparison"] = compare(records, comparison, args.ignore_index)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
