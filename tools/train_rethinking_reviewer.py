#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rethinking_reviewer import (
    InternalTrajectoryReviewer,
    REVIEWER_SOURCE_NAMES,
    reviewer_variant_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a frozen-SAM3 internal-state reviewer.')
    parser.add_argument(
        '--train-cache', nargs='+', required=True,
        help='One or more reviewer cache roots for training.')
    parser.add_argument(
        '--val-cache', nargs='+',
        help='Optional separate calibration cache roots.')
    parser.add_argument(
        '--validation-fraction', type=float, default=0.10,
        help='Per-cache-root train/calibration split when --val-cache is '
        'not provided.')
    parser.add_argument(
        '--variant',
        choices=('output_only', 'three_head', 'internal_trajectory'),
        required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--source-layers', type=int, default=2)
    parser.add_argument('--candidate-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--risk-weight', type=float, default=0.5)
    parser.add_argument('--recoverable-weight', type=float, default=0.5)
    parser.add_argument('--preserve-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--gate-thresholds',
        default='0.40,0.50,0.60,0.70,0.80,0.90,1.01')
    parser.add_argument(
        '--action-margins', default='0.00,0.05,0.10,0.20')
    parser.add_argument('--max-train-shards', type=int, default=0)
    parser.add_argument('--max-val-shards', type=int, default=0)
    parser.add_argument(
        '--protocol-name',
        default='source_supervised_candidate_validity')
    parser.add_argument('--source-dataset', default='')
    return parser.parse_args()


def parse_float_list(value):
    return [float(item) for item in str(value).split(',') if item.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_shards(roots, limit=0):
    files = []
    for root in roots:
        files.extend(Path(root).rglob('*.pt'))
    files = sorted(set(files))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(
            f'No .pt reviewer cache shards found under {roots}.')
    return files


def split_train_calibration(roots, fraction, seed, limit=0):
    train_files = []
    calibration_files = []
    for root_id, root in enumerate(roots):
        files = collect_shards([root], limit=0)
        generator = random.Random(int(seed) + root_id * 1009)
        generator.shuffle(files)
        calibration_count = max(
            1, int(round(len(files) * float(fraction))))
        calibration_count = min(
            calibration_count, max(1, len(files) - 1))
        calibration_files.extend(files[:calibration_count])
        train_files.extend(files[calibration_count:])
    if limit > 0:
        train_files = train_files[:limit]
    return train_files, calibration_files


def load_shard(path):
    record = torch.load(path, map_location='cpu')
    source_names = tuple(record.get('source_names') or ())
    if source_names != REVIEWER_SOURCE_NAMES:
        raise ValueError(
            f'{path} has incompatible source names: {source_names}.')
    return record


def iter_batches(files, batch_size, shuffle, seed):
    files = list(files)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    if shuffle:
        order = torch.randperm(len(files), generator=generator).tolist()
        files = [files[index] for index in order]
    for path in files:
        record = load_shard(path)
        row_count = int(record['source_features'].shape[0])
        if row_count <= 0:
            continue
        if shuffle:
            row_order = torch.randperm(
                row_count, generator=generator)
        else:
            row_order = torch.arange(row_count)
        for start in range(0, row_count, batch_size):
            index = row_order[start:start + batch_size]
            batch = {
                key: value[index]
                for key, value in record.items()
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == row_count
            }
            batch['_dataset_name'] = str(
                record.get('dataset_name') or 'unknown')
            batch['_num_classes'] = int(record['num_classes'])
            yield batch


def move_batch(batch, device):
    return {
        key: (
            (
                value.to(device, non_blocking=True).float()
                if key in ('source_features', 'candidate_aux')
                else value.to(device, non_blocking=True)
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def compute_loss(
        output, batch, risk_weight, recoverable_weight,
        preserve_weight):
    action_target = batch['action_target'].long()
    risk_target = batch['risk_target'].float()
    recoverable_target = batch['recoverable_target'].float()
    valid_action = action_target >= 0

    if valid_action.any():
        action_loss = nn.functional.cross_entropy(
            output['action_logits'][valid_action],
            action_target[valid_action],
        )
    else:
        action_loss = output['action_logits'].sum() * 0.0
    risk_loss = nn.functional.binary_cross_entropy_with_logits(
        output['risk_logits'], risk_target)
    recoverable_loss = nn.functional.binary_cross_entropy_with_logits(
        output['recoverable_logits'], recoverable_target)

    correct = batch['baseline_correct'].bool()
    if correct.any():
        preserve_target = torch.zeros(
            int(correct.sum().item()),
            dtype=torch.long,
            device=action_target.device,
        )
        preserve_loss = nn.functional.cross_entropy(
            output['action_logits'][correct],
            preserve_target,
        )
    else:
        preserve_loss = output['action_logits'].sum() * 0.0

    total = (
        action_loss
        + float(risk_weight) * risk_loss
        + float(recoverable_weight) * recoverable_loss
        + float(preserve_weight) * preserve_loss
    )
    return total, {
        'action': float(action_loss.detach().item()),
        'risk': float(risk_loss.detach().item()),
        'recoverable': float(recoverable_loss.detach().item()),
        'preserve': float(preserve_loss.detach().item()),
    }


def train_epoch(
        model, files, optimizer, scheduler, scaler, variant_mask,
        args, epoch):
    model.train()
    totals = {
        'loss': 0.0,
        'action': 0.0,
        'risk': 0.0,
        'recoverable': 0.0,
        'preserve': 0.0,
        'batches': 0,
    }
    for batch in iter_batches(
            files, args.batch_size, True, args.seed + epoch):
        batch = move_batch(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = str(args.device).startswith('cuda')
        with torch.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=amp_enabled):
            output = model(
                batch['source_features'],
                batch['source_mask'],
                batch['candidate_aux'],
                variant_mask,
            )
            loss, pieces = compute_loss(
                output,
                batch,
                args.risk_weight,
                args.recoverable_weight,
                args.preserve_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        totals['loss'] += float(loss.detach().item())
        for key, value in pieces.items():
            totals[key] += value
        totals['batches'] += 1
    denominator = max(1, totals.pop('batches'))
    return {key: value / denominator for key, value in totals.items()}


def empty_gate_stats():
    return {
        'changed': 0.0,
        'improved': 0.0,
        'harmed': 0.0,
        'wrong_to_wrong': 0.0,
        'rows': 0.0,
    }


def update_weighted_confusion(
        confusion, gt, prediction, sample_weight, num_classes):
    valid = (
        (gt >= 0)
        & (gt < int(num_classes))
        & (prediction >= 0)
        & (prediction < int(num_classes))
    )
    if not valid.any():
        return confusion
    indices = (
        gt[valid].long() * int(num_classes)
        + prediction[valid].long()
    )
    counts = torch.bincount(
        indices,
        weights=sample_weight[valid].double(),
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))
    counts = counts.detach().cpu()
    if confusion is None:
        return counts
    if tuple(confusion.shape) != tuple(counts.shape):
        raise ValueError(
            f'Confusion shape changed from {tuple(confusion.shape)} '
            f'to {tuple(counts.shape)}.')
    return confusion + counts


def confusion_miou(confusion):
    if confusion is None:
        return 0.0
    confusion = confusion.double()
    true_positive = torch.diag(confusion)
    denominator = (
        confusion.sum(dim=1)
        + confusion.sum(dim=0)
        - true_positive
    )
    valid = denominator > 0
    if not valid.any():
        return 0.0
    return float(
        (true_positive[valid] / denominator[valid]).mean().item())


def evaluate(
        model, files, variant_mask, args, gate_thresholds,
        action_margins):
    model.eval()
    grid = {
        (gate, margin): empty_gate_stats()
        for gate in gate_thresholds
        for margin in action_margins
    }
    action_correct = 0
    action_total = 0
    risk_correct = 0
    risk_total = 0
    dataset_num_classes = {}
    baseline_confusions = {}
    reviewed_confusions = {
        key: {} for key in grid
    }
    with torch.no_grad():
        for batch in iter_batches(
                files, args.batch_size, False, args.seed):
            batch = move_batch(batch, args.device)
            dataset_name = batch['_dataset_name']
            num_classes = int(batch['_num_classes'])
            previous_num_classes = dataset_num_classes.setdefault(
                dataset_name, num_classes)
            if previous_num_classes != num_classes:
                raise ValueError(
                    f'Dataset {dataset_name!r} changed from '
                    f'{previous_num_classes} to {num_classes} classes.')
            output = model(
                batch['source_features'],
                batch['source_mask'],
                batch['candidate_aux'],
                variant_mask,
            )
            action_probability = torch.softmax(
                output['action_logits'], dim=1)
            selected_slot = action_probability.argmax(dim=1)
            selected_probability = torch.gather(
                action_probability,
                1,
                selected_slot.unsqueeze(1),
            ).squeeze(1)
            keep_probability = action_probability[:, 0]
            risk = torch.sigmoid(output['risk_logits'])
            recoverable = torch.sigmoid(
                output['recoverable_logits'])
            gate_score = risk * recoverable

            action_target = batch['action_target'].long()
            valid_action = action_target >= 0
            action_correct += int(
                (selected_slot[valid_action]
                 == action_target[valid_action]).sum().item())
            action_total += int(valid_action.sum().item())
            risk_target = batch['risk_target'] > 0.5
            risk_correct += int(
                ((risk >= 0.5) == risk_target).sum().item())
            risk_total += int(risk_target.numel())

            candidates = batch['candidates'].long()
            selected_class = torch.gather(
                candidates,
                1,
                selected_slot.unsqueeze(1),
            ).squeeze(1)
            base_pred = batch['base_pred'].long()
            gt = batch['gt'].long()
            baseline_correct = batch['baseline_correct'].bool()
            sample_weight = batch.get('sample_weight')
            if sample_weight is None:
                sample_weight = torch.ones_like(
                    risk, dtype=torch.float32)
            else:
                sample_weight = sample_weight.float()
            baseline_confusions[dataset_name] = update_weighted_confusion(
                baseline_confusions.get(dataset_name),
                gt,
                base_pred,
                sample_weight,
                num_classes,
            )
            for gate_threshold in gate_thresholds:
                for action_margin in action_margins:
                    change = (
                        (selected_slot > 0)
                        & (gate_score >= gate_threshold)
                        & (
                            selected_probability - keep_probability
                            >= action_margin)
                        & (selected_class != base_pred)
                    )
                    stats = grid[(gate_threshold, action_margin)]
                    stats['rows'] += float(sample_weight.sum().item())
                    stats['changed'] += float(
                        sample_weight[change].sum().item())
                    improved_mask = (
                        change & ~baseline_correct
                        & (selected_class == gt))
                    harmed_mask = (
                        change & baseline_correct
                        & (selected_class != gt))
                    wrong_mask = (
                        change & ~baseline_correct
                        & (selected_class != gt))
                    stats['improved'] += float(
                        sample_weight[improved_mask].sum().item())
                    stats['harmed'] += float(
                        sample_weight[harmed_mask].sum().item())
                    stats['wrong_to_wrong'] += float(
                        sample_weight[wrong_mask].sum().item())
                    key = (gate_threshold, action_margin)
                    reviewed_prediction = torch.where(
                        change, selected_class, base_pred)
                    dataset_confusions = reviewed_confusions[key]
                    dataset_confusions[dataset_name] = (
                        update_weighted_confusion(
                            dataset_confusions.get(dataset_name),
                            gt,
                            reviewed_prediction,
                            sample_weight,
                            num_classes,
                        )
                    )

    rows = []
    for (gate_threshold, action_margin), stats in grid.items():
        changed = max(1, stats['changed'])
        net = stats['improved'] - stats['harmed']
        dataset_miou = {}
        for dataset_name in sorted(dataset_num_classes):
            baseline_miou = confusion_miou(
                baseline_confusions.get(dataset_name))
            reviewed_miou = confusion_miou(
                reviewed_confusions[
                    (gate_threshold, action_margin)].get(dataset_name))
            dataset_miou[dataset_name] = {
                'baseline_miou': baseline_miou,
                'reviewed_miou': reviewed_miou,
                'delta_miou': reviewed_miou - baseline_miou,
            }
        mean_baseline_miou = (
            sum(
                values['baseline_miou']
                for values in dataset_miou.values())
            / max(1, len(dataset_miou))
        )
        mean_reviewed_miou = (
            sum(
                values['reviewed_miou']
                for values in dataset_miou.values())
            / max(1, len(dataset_miou))
        )
        rows.append({
            'gate_threshold': gate_threshold,
            'action_margin': action_margin,
            **stats,
            'net': net,
            'net_ratio': net / max(1, stats['rows']),
            'correction_precision': (
                stats['improved']
                / max(1, stats['improved'] + stats['harmed'])),
            'harmful_change_rate': stats['harmed'] / changed,
            'mean_baseline_miou': mean_baseline_miou,
            'mean_reviewed_miou': mean_reviewed_miou,
            'mean_delta_miou': (
                mean_reviewed_miou - mean_baseline_miou),
            'dataset_miou': dataset_miou,
        })
    rows.sort(
        key=lambda row: (
            row['mean_delta_miou'],
            row['mean_reviewed_miou'],
            row['correction_precision'],
            -row['changed'],
        ),
        reverse=True,
    )
    return {
        'action_accuracy': action_correct / max(1, action_total),
        'risk_accuracy': risk_correct / max(1, risk_total),
        'best_gate': rows[0],
        'gate_sweep': rows,
    }


def count_batches(files, batch_size):
    total = 0
    for path in files:
        record = load_shard(path)
        total += math.ceil(
            int(record['source_features'].shape[0]) / batch_size)
    return max(1, total)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.val_cache:
        train_files = collect_shards(
            args.train_cache, args.max_train_shards)
        val_files = collect_shards(
            args.val_cache, args.max_val_shards)
    else:
        train_files, val_files = split_train_calibration(
            args.train_cache,
            args.validation_fraction,
            args.seed,
            args.max_train_shards,
        )
        if args.max_val_shards > 0:
            val_files = val_files[:args.max_val_shards]
    first = load_shard(train_files[0])
    topk = int(first['source_features'].shape[1])

    model_kwargs = {
        'topk': topk,
        'source_feature_dim': int(
            first['source_features'].shape[-1]),
        'candidate_aux_dim': int(
            first['candidate_aux'].shape[-1]),
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'source_layers': args.source_layers,
        'candidate_layers': args.candidate_layers,
        'dropout': args.dropout,
    }
    model = InternalTrajectoryReviewer(**model_kwargs).to(args.device)
    variant_mask = reviewer_variant_mask(
        args.variant, device=args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * count_batches(
        train_files, args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps))
    scaler = torch.cuda.amp.GradScaler(
        enabled=str(args.device).startswith('cuda'))
    gate_thresholds = parse_float_list(args.gate_thresholds)
    action_margins = parse_float_list(args.action_margins)

    history = []
    best_score = float('-inf')
    best_path = os.path.join(args.output_dir, 'best.pth')
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model,
            train_files,
            optimizer,
            scheduler,
            scaler,
            variant_mask,
            args,
            epoch,
        )
        val_metrics = evaluate(
            model,
            val_files,
            variant_mask,
            args,
            gate_thresholds,
            action_margins,
        )
        best_gate = val_metrics['best_gate']
        epoch_record = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'validation': val_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        score = float(best_gate['mean_delta_miou'])
        if score > best_score:
            best_score = score
            checkpoint = {
                'format_version': 1,
                'variant': args.variant,
                'source_names': REVIEWER_SOURCE_NAMES,
                'model_kwargs': model_kwargs,
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'gate_threshold': float(
                    best_gate['gate_threshold']),
                'action_margin': float(best_gate['action_margin']),
                'selection_metric': 'macro_dataset_mIoU_delta',
                'protocol_name': args.protocol_name,
                'source_dataset': args.source_dataset,
                'validation': val_metrics,
                'train_cache': args.train_cache,
                'val_cache': args.val_cache,
                'seed': args.seed,
            }
            torch.save(checkpoint, best_path)

    with open(
            os.path.join(args.output_dir, 'history.json'),
            'w',
            encoding='utf-8') as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        'best_checkpoint': best_path,
        'selection_metric': 'macro_dataset_mIoU_delta',
        'best_validation_mean_delta_miou': best_score,
        'train_shards': len(train_files),
        'val_shards': len(val_files),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
