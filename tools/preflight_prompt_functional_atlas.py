#!/usr/bin/env python3
"""Preflight for the UDD5 SAM3 prompt functional atlas."""

import argparse
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preflight_role_prompt_tta import runtime_checks


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


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


def static_checks():
    from prompt_functional_atlas_definitions import (
        PROMPT_COUNT,
        PROMPT_SLOTS,
        REGROUND_VARIANTS,
        VARIANT_NAMES,
    )

    paths = [
        os.path.join(ROOT, 'role_prompt_tta.py'),
        os.path.join(ROOT, 'segearthov3_segmentor.py'),
        os.path.join(ROOT, 'prompt_functional_atlas_definitions.py'),
        os.path.join(ROOT, 'tools', 'summarize_prompt_functional_atlas.py'),
    ]
    for path in paths:
        compile(read(path), path, 'exec')
    config_path = os.path.join(
        ROOT, 'configs', 'experiments',
        'cfg_udd5_prompt_functional_atlas.py')
    config = read(config_path)
    compile(config, config_path, 'exec')
    assert "role_prompt_tta_protocol='prompt_functional_atlas_v2'" \
        in config.replace(' ', '')
    assert "role_prompt_tta_primary_variant='baseline'" \
        in config.replace(' ', '')
    assert 'isaid' not in config.lower()

    bank_path = os.path.join(
        ROOT, 'configs', 'prompt_banks', 'atlas_v2', 'udd5.json')
    with open(bank_path, encoding='utf-8') as handle:
        bank = json.load(handle)
    classes = bank['classes']
    class_names = [
        line.strip() for line in read(
            os.path.join(ROOT, 'configs', 'cls_udd5.txt')).splitlines()
        if line.strip()
    ]
    assert bank['schema_version'] == 1
    assert bank['prompt_roles'] == [role for _, role in PROMPT_SLOTS]
    assert [row['name'] for row in classes] == class_names
    assert all(len(row['descriptions']) == PROMPT_COUNT for row in classes)
    for row in classes:
        name = row['name'].lower()
        assert row['descriptions'][0].lower() == name
        assert all(name in prompt.lower() for prompt in row['descriptions'])
        assert len(set(row['descriptions'])) == PROMPT_COUNT

    segmentor = os.path.join(ROOT, 'segearthov3_segmentor.py')
    assert constructor_default(segmentor, 'role_prompt_tta_protocol') == 'v1'
    assert constructor_default(
        segmentor, 'role_prompt_tta_e2e_measure_post') is False
    assert len(set(VARIANT_NAMES)) == len(VARIANT_NAMES)
    assert VARIANT_NAMES[0] == 'baseline'
    assert set(REGROUND_VARIANTS).issubset(VARIANT_NAMES)
    role_source = read(os.path.join(ROOT, 'role_prompt_tta.py'))
    for token in (
            '_rpt_atlas_infer_single_view',
            '_rpt_reground_anchor_residual',
            'literal_reground_parity_max_abs',
            'full_bg_literal_output',
            'presence_selected_output',
            'e2e_bg_literal_output',
            'post_update_loss'):
        assert token in role_source
    segmentor_source = read(segmentor)
    assert 'for name in self._rpt_variant_names()' in segmentor_source
    assert 'for variant_name in self._rpt_variant_names()' in segmentor_source
    return dict(
        dataset='udd5',
        classes=len(classes),
        prompts_per_class=PROMPT_COUNT,
        variants=len(VARIANT_NAMES),
        checks=[
            'protected constructor defaults remain v1/off',
            'atlas returns baseline as primary prediction',
            'controlled prompt roles and class order are exact',
            'literal identity, output, Presence, E2E and residual controls wired',
            'sliding aggregation uses the active protocol variant contract',
            'iSAID is absent',
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-runtime-assets', action='store_true')
    args = parser.parse_args()
    report = static_checks()
    if args.check_runtime_assets:
        runtime_report = {}
        runtime_checks(runtime_report)
        from mmengine.config import Config
        Config.fromfile(os.path.join(
            ROOT, 'configs', 'experiments',
            'cfg_udd5_prompt_functional_atlas.py'))
        report['runtime'] = runtime_report['runtime']
        report['runtime']['atlas_config'] = 'parsed'
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
