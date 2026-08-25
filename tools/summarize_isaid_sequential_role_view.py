#!/usr/bin/env python3
"""Summarize the full-validation iSAID Global-only Role screen."""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    JOINT_ROLE_VIEW_PROTOCOL,
    JOINT_ROLE_VIEW_SCHEMA_VERSION,
)
from tools.summarize_role_functional_text_screen import (
    add_matrix,
    confusion_metrics,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dataset', default='isaid')
    return parser.parse_args()


def expected_variants(payload):
    specs = payload.get('operator_specs', [])
    if [value.get('name') for value in specs] != ['global']:
        raise ValueError('Role-only records must expose only Global.')
    anchor = payload['anchor_candidate']
    names = ['jrv_official', 'jrv_anchor__global']
    names.extend(
        f'jrv_{value["id"]}__global'
        for value in payload['role_candidates']
        if value['id'] != anchor)
    return tuple(names)


def load_records(patterns, dataset):
    paths = sorted(set(
        path for pattern in patterns
        for path in (glob.glob(pattern) or [pattern])
        if os.path.isfile(path)))
    records, seen, duplicates = [], {}, 0
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = record.get('joint_role_view')
                if str(record.get('dataset_name', '')).lower() != dataset:
                    continue
                if (payload is None
                        or int(payload.get('schema_version', -1))
                        != JOINT_ROLE_VIEW_SCHEMA_VERSION
                        or payload.get('protocol') != JOINT_ROLE_VIEW_PROTOCOL
                        or payload.get('execution') != 'role_only'
                        or tuple(payload.get('variants', {}))
                        != expected_variants(payload)):
                    raise ValueError(
                        f'{path}:{line_number}: iSAID Role-only contract mismatch.')
                key = (dataset, str(record.get('img_path')))
                if key in seen:
                    if seen[key]['joint_role_view'] != payload:
                        raise ValueError(f'Inconsistent DDP duplicate: {key}.')
                    duplicates += 1
                    continue
                seen[key] = record
                records.append(record)
    if not records:
        raise ValueError('No iSAID sequential Role-only records found.')
    return records, duplicates


def summarize(records):
    first = records[0]['joint_role_view']
    variant_names = tuple(first['variants'])
    matrices = {name: None for name in variant_names}
    counters = {name: defaultdict(int) for name in variant_names}
    contract = {
        key: first[key] for key in (
            'role_candidates', 'anchor_candidate', 'operator_specs',
            'reference_endpoint', 'global_source')}
    for record in records:
        payload = record['joint_role_view']
        for key, value in contract.items():
            if payload[key] != value:
                raise ValueError(f'iSAID Role-only metadata drifted at {key}.')
        for name, stats in payload['variants'].items():
            matrices[name] = add_matrix(
                matrices[name], stats['confusion']['matrix'])
            for field in ('changed_pixels', 'improved_pixels',
                          'harmed_pixels', 'help_minus_harm'):
                counters[name][field] += int(stats[field])

    metrics = {
        name: confusion_metrics(matrix) for name, matrix in matrices.items()}
    metadata = first['variants']
    official_name = 'jrv_official'
    anchor_name = 'jrv_anchor__global'
    anchor_id = first['anchor_candidate']
    role_names = [
        name for name in variant_names
        if name not in (official_name, anchor_name)
        and metadata[name]['candidate_id'] != anchor_id]
    best_name = max(role_names, key=lambda name: metrics[name]['miou'])
    official = metrics[official_name]

    variants = []
    for name in variant_names:
        source = metadata[name]
        metric = metrics[name]
        variants.append(dict(
            dataset='isaid', variant=name,
            candidate_id=source['candidate_id'],
            slots='+'.join(str(value) for value in source['slots']),
            admission=source['admission'], operator=source['operator'],
            miou=metric['miou'], aacc=metric['aacc'],
            delta_to_official=metric['miou'] - official['miou'],
            images=len(records),
            **dict(counters[name]),
        ))

    profiles = []
    for profile, name in (
            ('official', official_name),
            ('role_only_anchor', anchor_name),
            ('role_only_best', best_name)):
        source = metadata[name]
        metric = metrics[name]
        profiles.append(dict(
            dataset='isaid', profile=profile, variant=name,
            candidate_id=source['candidate_id'],
            slots='+'.join(str(value) for value in source['slots']),
            admission=source['admission'], operator=source['operator'],
            miou=metric['miou'], aacc=metric['aacc'],
            delta_to_official=metric['miou'] - official['miou'],
        ))

    classes = []
    for profile, name in (
            ('official', official_name), ('role_only_best', best_name)):
        metric = metrics[name]
        for index, class_name in enumerate(records[0]['class_names']):
            classes.append(dict(
                dataset='isaid', profile=profile, variant=name,
                class_index=index, class_name=class_name,
                iou=metric['iou'][index], gt_pixels=metric['gt'][index],
                predicted_pixels=metric['pred'][index]))

    return dict(
        dataset='isaid', images=len(records), variants=variants,
        profiles=profiles, classes=classes,
        native_prompt_parity_max_abs=max(float(
            record['joint_role_view']['native_prompt_parity_max_abs'])
            for record in records),
    )


def write_report(path, result, duplicates):
    profiles = {value['profile']: value for value in result['profiles']}
    official = profiles['official']
    selected = profiles['role_only_best']
    lines = [
        '# iSAID Sequential Role-only Screen', '',
        f'- Images: `{result["images"]}`',
        f'- DDP padding duplicates removed: `{duplicates}`',
        f'- Protected official: `{official["miou"]:.3f}` mIoU',
        f'- Best non-anchor Role: `{selected["candidate_id"]}` '
        f'(`{selected["slots"]}`)',
        f'- Selected Role: `{selected["miou"]:.3f}` mIoU '
        f'(`{selected["delta_to_official"]:+.3f}`)',
        f'- Native prompt parity max abs: '
        f'`{result["native_prompt_parity_max_abs"]:.3e}`', '',
        '## Top-10 non-anchor Role profiles', '',
        '| Rank | Candidate | P/S/I | mIoU | Delta |',
        '|---:|---|---|---:|---:|',
    ]
    rows = sorted(
        (value for value in result['variants']
         if value['candidate_id'] != 'anchor'),
        key=lambda value: value['miou'], reverse=True)[:10]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f'| {rank} | {row["candidate_id"]} | {row["slots"]} | '
            f'{row["miou"]:.3f} | {row["delta_to_official"]:+.3f} |')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    dataset = str(args.dataset).lower()
    if dataset != 'isaid':
        raise ValueError('This sequential summarizer is intentionally iSAID-only.')
    records, duplicates = load_records(args.inputs, dataset)
    result = summarize(records)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'isaid_role_only_variants.csv'),
              result['variants'])
    write_csv(os.path.join(args.out_dir, 'isaid_role_only_classes.csv'),
              result['classes'])
    with open(os.path.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(dict(duplicate_count=duplicates, **result), handle,
                  ensure_ascii=False, indent=2)
        handle.write('\n')
    write_report(os.path.join(args.out_dir, 'isaid_role_only.md'),
                 result, duplicates)


if __name__ == '__main__':
    main()
