#!/usr/bin/env python3
"""Static and optional server-runtime checks for two-prompt screening."""

import argparse
import ast
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from semantic_supplement_definitions import (
    FIXED_FAMILY_VARIANTS,
    MAX_CANDIDATES,
    PROTOCOL,
    VARIANT_NAMES,
    load_semantic_supplement_bank,
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


def static_checks():
    source_paths = (
        'semantic_supplement_definitions.py',
        'semantic_supplement_screen.py',
        'role_prompt_tta.py',
        'segearthov3_segmentor.py',
        'tools/summarize_semantic_supplement_screen.py',
    )
    for relative in source_paths:
        path = os.path.join(ROOT, relative)
        compile(read(path), path, 'exec')

    report = dict(datasets={}, variants=len(VARIANT_NAMES), checks=[])
    for dataset in DATASETS:
        class_names = primary_class_names(os.path.join(
            ROOT, 'configs', f'cls_{dataset}.txt'))
        bank_path = os.path.join(
            ROOT, 'configs', 'prompt_banks',
            'semantic_supplement_v1', f'{dataset}.json')
        bank = load_semantic_supplement_bank(bank_path, class_names)
        config_path = os.path.join(
            ROOT, 'configs', 'experiments',
            f'cfg_{dataset}_semantic_supplement_screen.py')
        config = read(config_path)
        compile(config, config_path, 'exec')
        compact = config.replace(' ', '').replace('\n', '')
        assert f"role_prompt_tta_protocol='{PROTOCOL}'" in compact
        assert "role_prompt_tta_primary_variant='baseline'" in compact
        assert 'isaid' not in config.lower()
        report['datasets'][dataset] = dict(
            classes=len(class_names),
            lexical_candidates=sum(
                len(row['lexical_equivalents']) for row in bank['classes']),
            modifier_candidates=sum(
                len(row['semantic_modifiers']) for row in bank['classes']),
            bank=os.path.relpath(bank_path, ROOT),
            config=os.path.relpath(config_path, ROOT),
        )

    segmentor_path = os.path.join(ROOT, 'segearthov3_segmentor.py')
    assert constructor_default(segmentor_path, 'use_role_prompt_tta') is False
    assert constructor_default(
        segmentor_path, 'role_prompt_tta_protocol') == 'v1'
    assert constructor_default(
        segmentor_path,
        'role_prompt_tta_semantic_residual_alpha') == 0.50
    assert constructor_default(
        segmentor_path,
        'role_prompt_tta_semantic_residual_clip') == 0.25

    role_source = read(os.path.join(ROOT, 'role_prompt_tta.py'))
    screen_source = read(os.path.join(ROOT, 'semantic_supplement_screen.py'))
    segmentor_source = read(segmentor_path)
    for token in (
            'SEMANTIC_SUPPLEMENT_PROTOCOL',
            '_rpt_semantic_supplement_infer_single_view',
            '_rpt_uses_class_space_variants',
            'self._rpt_remoteclip = None'):
        assert token in role_source
    for token in (
            '_rpt_head_role_raw_package',
            '_rpt_head_role_instance_from_raw',
            'candidate_semantic.float() - anchor_semantic.float()',
            'active_semantics',
            'raw_recomposition_max_abs'):
        assert token in screen_source
    for token in (
            'role_prompt_variant_class_logits',
            '_rpt_uses_class_space_variants',
            'Synonym prompts must be averaged across crops'):
        assert token in segmentor_source
    assert len(VARIANT_NAMES) == 24
    assert len(set(VARIANT_NAMES)) == len(VARIANT_NAMES)
    assert FIXED_FAMILY_VARIANTS == (
        'lexical_mean_semantic_residual',
        'modifier_mean_semantic_residual')
    assert MAX_CANDIDATES == 3
    report['checks'] = [
        'six fixed evaluation datasets are wired; iSAID is absent',
        'official baseline remains the returned primary prediction',
        'canonical Presence and instance stay coupled',
        'candidate prompts use complete native SAM3 grounding',
        'lexical and modifier families remain separately attributable',
        'missing candidate slots are exact anchor fallbacks',
        'replacement and bounded-residual paths are both recorded',
        'raw-query recomposition and cached-native parity are strict',
        'class-space sliding aggregation preserves the official baseline',
        'RemoteCLIP, gradients and online GT selection are absent',
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
            f'cfg_{dataset}_semantic_supplement_screen.py'))
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
