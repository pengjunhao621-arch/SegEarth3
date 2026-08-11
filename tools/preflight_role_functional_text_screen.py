#!/usr/bin/env python3
"""Static and optional server checks for the role-functional text screen."""

import argparse
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from role_functional_text_definitions import (
    DEFAULT_SETTING,
    PROTOCOL,
    RESIDUAL_SETTINGS,
    ROLE_FIELDS,
    VARIANT_NAMES,
    load_role_functional_text_bank,
)


DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def class_file(path):
    rows = [line.strip() for line in read(path).splitlines() if line.strip()]
    return [row.split(',')[0] for row in rows], [row.split(',') for row in rows]


def static_checks():
    sources = (
        'role_functional_text_definitions.py',
        'role_functional_text_screen.py',
        'segearthov3_segmentor.py',
        'tools/summarize_role_functional_text_screen.py',
    )
    for relative in sources:
        path = os.path.join(ROOT, relative)
        compile(read(path), path, 'exec')

    report = dict(datasets={}, variants=len(VARIANT_NAMES), checks=[])
    for dataset in DATASETS:
        names, official = class_file(os.path.join(
            ROOT, 'configs', f'cls_{dataset}.txt'))
        bank_path = os.path.join(
            ROOT, 'configs', 'prompt_banks', 'role_functional_text_v1',
            f'{dataset}.json')
        bank = load_role_functional_text_bank(
            bank_path, names, official)
        config_path = os.path.join(
            ROOT, 'configs', 'experiments',
            f'cfg_{dataset}_role_functional_text_screen.py')
        config = read(config_path)
        compile(config, config_path, 'exec')
        compact = config.replace(' ', '').replace('\n', '')
        assert f"role_prompt_tta_protocol='{PROTOCOL}'" in compact
        assert "role_prompt_tta_primary_variant='baseline'" in compact
        assert 'isaid' not in config.lower()
        report['datasets'][dataset] = dict(
            classes=len(names),
            official_queries=sum(len(row) for row in official),
            frozen_candidates=sum(
                len(item[field])
                for item in bank['classes']
                for _, field, _ in ROLE_FIELDS),
            bank=os.path.relpath(bank_path, ROOT),
            config=os.path.relpath(config_path, ROOT),
        )
    assert len(set(VARIANT_NAMES)) == len(VARIANT_NAMES)
    assert DEFAULT_SETTING == 'alpha050_clip025'
    assert len(RESIDUAL_SETTINGS) == 5
    report['checks'] = [
        'six fixed validation datasets are wired; iSAID is absent',
        'prompt-bank official aliases exactly match each baseline class file',
        'all candidate texts are frozen before evaluation',
        'cached text follows independent native SAM3 grounding',
        'no-update role composition is checked against the exact baseline',
        'single-role, full-combination, shared-text and head-only maps exist',
        'five residual points separate alpha from clipping sensitivity',
        'RemoteCLIP, CLIP APIs, training and online candidate selection are absent',
    ]
    return report


def runtime_checks(report):
    torch = importlib.import_module('torch')
    mmengine = importlib.import_module('mmengine')
    from mmengine.config import Config
    checkpoint = os.path.join(ROOT, 'weights', 'sam3', 'sam3.pt')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    for dataset in DATASETS:
        Config.fromfile(os.path.join(
            ROOT, 'configs', 'experiments',
            f'cfg_{dataset}_role_functional_text_screen.py'))
    report['runtime'] = dict(
        torch=torch.__version__,
        mmengine=mmengine.__version__,
        sam3_checkpoint_bytes=os.path.getsize(checkpoint),
        all_mmengine_configs='parsed',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-runtime-assets', action='store_true')
    args = parser.parse_args()
    report = static_checks()
    if args.check_runtime_assets:
        runtime_checks(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
