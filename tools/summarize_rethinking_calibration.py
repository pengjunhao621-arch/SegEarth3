#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize mIoU-selected reviewer checkpoints.')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    dataset_names = set()
    records = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        best_gate = (
            checkpoint.get('validation', {}).get('best_gate', {}))
        dataset_miou = best_gate.get('dataset_miou', {})
        dataset_names.update(dataset_miou)
        records.append((checkpoint_path, checkpoint, best_gate, dataset_miou))

    for checkpoint_path, checkpoint, best_gate, dataset_miou in records:
        row = {
            'variant': checkpoint.get('variant'),
            'epoch': checkpoint.get('epoch'),
            'selection_metric': checkpoint.get('selection_metric'),
            'gate_threshold': checkpoint.get('gate_threshold'),
            'action_margin': checkpoint.get('action_margin'),
            'mean_baseline_miou': best_gate.get('mean_baseline_miou'),
            'mean_reviewed_miou': best_gate.get('mean_reviewed_miou'),
            'mean_delta_miou': best_gate.get('mean_delta_miou'),
            'net_ratio': best_gate.get('net_ratio'),
            'correction_precision': best_gate.get(
                'correction_precision'),
            'harmful_change_rate': best_gate.get(
                'harmful_change_rate'),
            'checkpoint': str(checkpoint_path),
        }
        for dataset_name in sorted(dataset_names):
            values = dataset_miou.get(dataset_name, {})
            row[f'{dataset_name}_baseline_miou'] = values.get(
                'baseline_miou')
            row[f'{dataset_name}_reviewed_miou'] = values.get(
                'reviewed_miou')
            row[f'{dataset_name}_delta_miou'] = values.get(
                'delta_miou')
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ['variant', 'checkpoint']
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output}.')


if __name__ == '__main__':
    main()
