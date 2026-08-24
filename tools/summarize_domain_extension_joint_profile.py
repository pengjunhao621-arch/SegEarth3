#!/usr/bin/env python3
"""Build the main natural-domain/iSAID Role-View discovery report."""

import argparse
import csv
import json
import os


PAPER_BASELINES = {
    'voc20': 96.8,
    'cityscapes': 69.7,
    'isaid': 27.6,
}

ROLE_SLOT_MEANINGS = {
    'P0': 'official Anchor',
    'P1': 'canonical visual existence / equivalent noun',
    'P2': 'multiplicity or area-level existence',
    'S0': 'official Anchor',
    'S1': 'complete dense extent / spatial continuity',
    'S2': 'silhouette, surface or coherent appearance',
    'S3': 'dataset-view-specific geometry / visual structure',
    'I0': 'official Anchor',
    'I1': 'individual object / connected component',
    'I2': 'separated or bounded complete region',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--baseline-tolerance', type=float, default=0.05)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def metric_value(record, suffix):
    direct = record.get(suffix)
    if isinstance(direct, (int, float)):
        return float(direct)
    matches = [
        float(value) for key, value in record.items()
        if str(key).lower().endswith(suffix.lower())
        and isinstance(value, (int, float))]
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely find {suffix} in result keys.')
    return matches[0]


def load_baseline(root, dataset):
    path = os.path.join(root, 'baseline', dataset, 'results.json')
    values = read_json(path)
    if isinstance(values, dict):
        values = [values]
    if not values:
        raise ValueError(f'Empty baseline result: {path}')
    return metric_value(values[-1], 'mIoU')


def profile_lookup(summary):
    return {
        (str(row['dataset']).lower(), row['profile']): row
        for row in summary['profiles']}


def selected_prompts(root, dataset, slots):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'prompt_banks', 'role_functional_text_v3',
        f'{dataset}.json')
    bank = read_json(path)
    fields = (
        ('presence', 'presence_candidates'),
        ('semantic', 'semantic_candidates'),
        ('instance', 'instance_candidates'),
    )
    rows = []
    for item in bank['classes']:
        value = {
            'class_id': item['id'], 'class_name': item['name'],
            'official_anchor': item['official_prompts']}
        for (role, field), slot in zip(fields, slots):
            value[role] = (
                item[field][slot - 1] if slot else item['official_prompts'])
        rows.append(value)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f'{float(value):.3f}'


def ranked_profiles(summary, dataset, role_only):
    rows = [
        row for row in summary['variants']
        if str(row['dataset']).lower() == dataset
        and row['candidate_id'] != 'anchor']
    if role_only:
        endpoint = next(
            row['operator'] for row in summary['profiles']
            if str(row['dataset']).lower() == dataset
            and row['profile'] == 'official')
        rows = [row for row in rows if row['operator'] == endpoint]
    return sorted(rows, key=lambda row: float(row['miou']), reverse=True)[:5]


def main():
    args = parse_args()
    summary_path = os.path.join(args.root, 'summary', 'summary.json')
    summary = read_json(summary_path)
    profiles = profile_lookup(summary)
    class_rows = summary['classes']
    main_rows = []
    prompt_payload = {}
    lines = [
        '# Natural-domain and iSAID Joint Role-View discovery', '',
        '## Main results', '',
        '| Dataset | Paper | Standalone baseline | Screen baseline | '
        'Best Role | Best View | Best Joint | Joint gain | Joint profile |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for dataset in args.datasets:
        official = profiles[(dataset, 'official')]
        role = profiles[(dataset, 'role_only_best')]
        view = profiles[(dataset, 'view_only_best')]
        joint = profiles[(dataset, 'joint_role_view_best')]
        standalone = load_baseline(args.root, dataset)
        baseline_delta = standalone - float(official['miou'])
        if abs(baseline_delta) > args.baseline_tolerance:
            raise ValueError(
                f'{dataset}: standalone baseline and protected screen '
                f'baseline differ by {baseline_delta:.4f} mIoU.')
        row = {
            'dataset': dataset,
            'paper_baseline_miou': PAPER_BASELINES[dataset],
            'standalone_baseline_miou': standalone,
            'screen_official_miou': official['miou'],
            'baseline_identity_delta': baseline_delta,
            'best_role_miou': role['miou'],
            'best_role_slots': role['slots'],
            'best_view_miou': view['miou'],
            'best_view_operator': view['operator'],
            'best_joint_miou': joint['miou'],
            'joint_gain_to_baseline': (
                float(joint['miou']) - float(official['miou'])),
            'joint_gain_to_best_role': (
                float(joint['miou']) - float(role['miou'])),
            'joint_slots': joint['slots'],
            'joint_admission': joint['admission'],
            'joint_operator': joint['operator'],
            'joint_global_weight': joint['global_weight'],
            'joint_rho': joint['rho'],
        }
        main_rows.append(row)
        slots = [int(value) for value in str(joint['slots']).split('+')]
        prompt_payload[dataset] = {
            'best_role_profile': role,
            'best_joint_profile': joint,
            'best_role_class_prompts': selected_prompts(
                args.root, dataset,
                [int(value) for value in str(role['slots']).split('+')]),
            'best_joint_class_prompts': selected_prompts(
                args.root, dataset, slots),
        }
        profile = (
            f"P/S/I={joint['slots']}; {joint['admission']}; "
            f"{joint['operator']}")
        lines.append(
            f"| {dataset} | {fmt(PAPER_BASELINES[dataset])} | "
            f"{fmt(standalone)} | {fmt(official['miou'])} | "
            f"{fmt(role['miou'])} | {fmt(view['miou'])} | "
            f"{fmt(joint['miou'])} | "
            f"{fmt(row['joint_gain_to_baseline'])} | {profile} |")

    lines.extend(['', '## Frozen candidate semantics', ''])
    for key, value in ROLE_SLOT_MEANINGS.items():
        lines.append(f'- `{key}`: {value}.')

    lines.extend(['', '## Top-5 profiles from full validation evaluation', ''])
    for dataset in args.datasets:
        lines.extend([
            f'### {dataset}', '',
            '| Rank | Role-only profile | mIoU | Joint profile | mIoU |',
            '|---:|---|---:|---|---:|'])
        role_rows = ranked_profiles(summary, dataset, role_only=True)
        joint_rows = ranked_profiles(summary, dataset, role_only=False)
        for rank, (role_row, joint_row) in enumerate(
                zip(role_rows, joint_rows), 1):
            lines.append(
                f"| {rank} | {role_row['slots']} / "
                f"{role_row['operator']} | {fmt(role_row['miou'])} | "
                f"{joint_row['slots']} / {joint_row['operator']} | "
                f"{fmt(joint_row['miou'])} |")

    lines.extend(['', '## Most informative class changes', ''])
    for dataset in args.datasets:
        by_profile = {}
        for row in class_rows:
            if str(row['dataset']).lower() != dataset:
                continue
            by_profile.setdefault(row['profile'], {})[row['class_name']] = row
        official = by_profile['official']
        joint = by_profile['joint_role_view_best']
        changes = sorted(
            ((name, float(joint[name]['iou']) - float(value['iou']))
             for name, value in official.items()
             if value['iou'] is not None and joint[name]['iou'] is not None),
            key=lambda value: value[1], reverse=True)
        top = ', '.join(f'{name} {delta:+.2f}' for name, delta in changes[:5])
        bottom = ', '.join(
            f'{name} {delta:+.2f}' for name, delta in changes[-5:])
        lines.extend([
            f'### {dataset}', '',
            f'- Largest gains: {top}',
            f'- Largest losses: {bottom}',
            '',
        ])

    os.makedirs(os.path.join(args.root, 'summary'), exist_ok=True)
    write_csv(os.path.join(
        args.root, 'summary', 'domain_extension_main_table.csv'), main_rows)
    with open(os.path.join(
            args.root, 'summary', 'selected_prompt_profiles.json'),
            'w', encoding='utf-8') as handle:
        json.dump(prompt_payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    with open(os.path.join(
            args.root, 'summary', 'domain_extension_report.md'),
            'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
