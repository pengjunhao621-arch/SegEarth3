#!/usr/bin/env python3
"""Compile the full-validation best Role into a reduced Joint registry."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    JOINT_ROLE_VIEW_PROTOCOL,
    JOINT_ROLE_VIEW_SCHEMA_VERSION,
    load_joint_role_view_registry,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--source-registry', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def selected_rows(summary, dataset):
    rows = [
        value for value in summary['profiles']
        if str(value['dataset']).lower() == dataset]
    by_profile = {value['profile']: value for value in rows}
    for key in ('official', 'role_only_best'):
        if key not in by_profile:
            raise ValueError(f'{dataset}: summary has no {key!r} profile.')
    return by_profile['official'], by_profile['role_only_best']


def build_registry(summary, source_path, dataset):
    source = load_joint_role_view_registry(source_path, dataset)
    official, selected = selected_rows(summary, dataset)
    selected_id = str(selected['candidate_id'])
    candidates = {value['id']: value for value in source['role_candidates']}
    if selected_id == source['anchor_candidate'] or selected_id not in candidates:
        raise ValueError(
            f'{dataset}: invalid selected non-anchor Role {selected_id!r}.')

    def export(identifier, source_miou):
        value = candidates[identifier]
        return dict(
            id=value['id'],
            slots=list(value['slots']),
            admission=value['admission'],
            source_miou=float(source_miou),
        )

    official_miou = float(official['miou'])
    selected_miou = float(selected['miou'])
    return {
        'schema_version': JOINT_ROLE_VIEW_SCHEMA_VERSION,
        'protocol': JOINT_ROLE_VIEW_PROTOCOL,
        'selection': {
            'dataset': dataset,
            'rule': 'full-validation role-only best non-anchor profile',
            'official_miou': official_miou,
            'selected_role_miou': selected_miou,
            'selected_gain': selected_miou - official_miou,
            'selected_candidate': selected_id,
            'selected_slots': list(candidates[selected_id]['slots']),
        },
        'datasets': {
            dataset: {
                'anchor_candidate': source['anchor_candidate'],
                'current_role_candidate': selected_id,
                'prior_view_operator': 'global',
                'prior_view_miou': selected_miou,
                'role_candidates': [
                    export(source['anchor_candidate'], official_miou),
                    export(selected_id, selected_miou),
                ],
            }
        },
    }


def main():
    args = parse_args()
    dataset = str(args.dataset).lower()
    payload = build_registry(
        read_json(args.summary), args.source_registry, dataset)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    selection = payload['selection']
    print(
        f"Selected {dataset}: {selection['selected_candidate']} "
        f"({selection['selected_role_miou']:.3f} mIoU, "
        f"{selection['selected_gain']:+.3f} vs official).")
    print(f'Reduced Joint registry: {output}')


if __name__ == '__main__':
    main()
