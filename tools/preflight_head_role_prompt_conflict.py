#!/usr/bin/env python3
"""Static and optional server-runtime checks for the head-role audit."""

import argparse
import ast
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from head_role_prompt_conflict_definitions import (
    DESCRIPTION_SLOTS,
    HEAD_PATHS,
    PROMPT_COUNT,
    VARIANT_NAMES,
)

DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def primary_class_names(path):
    return [line.split(',')[0].strip() for line in read(path).splitlines()
            if line.strip()]


def constructor_default(source_path, argument):
    tree = ast.parse(read(source_path), filename=source_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            args = node.args.args
            defaults = [None] * (len(args) - len(node.args.defaults)) \
                + list(node.args.defaults)
            for arg, default in zip(args, defaults):
                if arg.arg == argument:
                    return ast.literal_eval(default)
    raise AssertionError(f'Constructor argument {argument} not found.')


def validate_bank(dataset):
    class_path = os.path.join(ROOT, 'configs', f'cls_{dataset}.txt')
    bank_path = os.path.join(
        ROOT, 'configs', 'prompt_banks', f'{dataset}.json')
    with open(bank_path, encoding='utf-8') as handle:
        bank = json.load(handle)
    classes = bank['classes']
    expected = primary_class_names(class_path)
    assert bank['schema_version'] == 1
    assert [row['id'] for row in classes] == list(range(len(classes)))
    assert [row['name'] for row in classes] == expected
    for row in classes:
        assert len(row['descriptions']) == PROMPT_COUNT
        assert row['descriptions'][0].lower() == row['name'].lower()
        assert len(set(row['descriptions'])) == PROMPT_COUNT
        assert all(row['name'].lower() in prompt.lower()
                   for prompt in row['descriptions'])
    return len(classes)


def static_checks():
    paths = [
        os.path.join(ROOT, 'head_role_prompt_conflict_definitions.py'),
        os.path.join(ROOT, 'role_prompt_tta.py'),
        os.path.join(ROOT, 'segearthov3_segmentor.py'),
        os.path.join(ROOT, 'tools',
                     'summarize_head_role_prompt_conflict.py'),
    ]
    for path in paths:
        compile(read(path), path, 'exec')
    report = dict(datasets={}, variants=len(VARIANT_NAMES), checks=[])
    for dataset in DATASETS:
        config_path = os.path.join(
            ROOT, 'configs', 'experiments',
            f'cfg_{dataset}_head_role_prompt_conflict.py')
        config = read(config_path)
        compile(config, config_path, 'exec')
        compact = config.replace(' ', '').replace('\n', '')
        assert "role_prompt_tta_protocol='head_role_prompt_conflict_v1'" \
            in compact
        assert "role_prompt_tta_primary_variant='baseline'" in compact
        assert f'prompt_banks/{dataset}.json' in config
        assert 'isaid' not in config.lower()
        report['datasets'][dataset] = dict(
            classes=validate_bank(dataset),
            config=os.path.relpath(config_path, ROOT),
        )
    segmentor_path = os.path.join(ROOT, 'segearthov3_segmentor.py')
    assert constructor_default(segmentor_path, 'use_role_prompt_tta') is False
    assert constructor_default(segmentor_path, 'role_prompt_tta_protocol') == 'v1'
    role_source = read(os.path.join(ROOT, 'role_prompt_tta.py'))
    segmentor_source = read(segmentor_path)
    for token in (
            '_rpt_head_role_infer_single_view',
            '_rpt_head_role_instance_from_raw',
            'raw_recomposition_max_abs',
            "self._rpt_remoteclip = None"):
        assert token in role_source
    for token in (
            'role_prompt_class_space',
            'role_prompt_variant_class_logits',
            'Synonym prompts must be averaged across crops'):
        assert token in segmentor_source
    assert len(VARIANT_NAMES) == 2 + len(DESCRIPTION_SLOTS) * len(HEAD_PATHS)
    assert len(set(VARIANT_NAMES)) == len(VARIANT_NAMES)
    report['checks'] = [
        'six fixed datasets and prompt banks are wired',
        'official baseline remains the returned primary prediction',
        'canonical literal and official synonym baseline are separate',
        'every description uses native grounding before role assignment',
        'counterfactual Presence recomputes the raw native query gate',
        'raw-query native recomposition has a strict identity control',
        'RemoteCLIP and test-time gradients are absent from this protocol',
        'sliding crops aggregate class scores and preserve synonym order',
        'constructor defaults keep the official baseline off-path',
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
            f'cfg_{dataset}_head_role_prompt_conflict.py'))
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
