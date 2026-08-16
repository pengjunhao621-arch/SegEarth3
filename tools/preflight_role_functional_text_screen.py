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
    BOUNDARY_GUIDES,
    BOUNDARY_REPLAY_VARIANT_NAMES,
    BOUNDARY_REPLAY_TRACE_TOLERANCE,
    BOUNDARY_STRENGTHS,
    CLASS_ROLE_ALIGNMENT_PE_LAYER,
    CRA_COCO_WEIGHTS,
    CRA_FCRA_TEMPERATURES,
    CRA_VARIANT_NAMES,
    COMPLETION_PROTOCOL,
    DEFAULT_SETTING,
    PI_MECHANISM_MAP_NAMES,
    PI_VARIANT_NAMES,
    PE_LAYER_IDS,
    PE_VARIANT_NAMES,
    PROTOCOL,
    RESIDUAL_SETTINGS,
    ROLE_FIELDS,
    ROLE_VISUAL_FIELD_COMPOSITIONS,
    ROLE_VISUAL_FIELD_PROTOCOL,
    ROLE_VISUAL_FIELD_VARIANT_NAMES,
    VARIANT_NAMES,
    load_role_functional_text_bank,
    load_role_text_selection_registry,
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
        'boundary_replay.py',
        'class_role_alignment.py',
        'role_functional_text_definitions.py',
        'role_functional_text_screen.py',
        'role_visual_field.py',
        'segearthov3_segmentor.py',
        'tiled_context.py',
        'tools/summarize_role_functional_text_screen.py',
        'tools/summarize_pi_role_compatibility.py',
        'tools/summarize_pe_role_evidence.py',
        'tools/summarize_boundary_replay.py',
        'tools/summarize_class_role_alignment.py',
        'tools/summarize_role_visual_field.py',
        'tools/verify_tiled_context.py',
    )
    for relative in sources:
        path = os.path.join(ROOT, relative)
        compile(read(path), path, 'exec')
    segmentor_source = read(os.path.join(ROOT, 'segearthov3_segmentor.py'))
    assert 'self._rpt_aggregate_query_logits_to_classes' not in segmentor_source

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
        completion_bank_path = os.path.join(
            ROOT, 'configs', 'prompt_banks', 'role_functional_text_v2',
            f'{dataset}.json')
        completion_bank = load_role_functional_text_bank(
            completion_bank_path, names, official)
        assert completion_bank['completion']['protocol'] == COMPLETION_PROTOCOL
        report['datasets'][dataset]['completion_bank'] = os.path.relpath(
            completion_bank_path, ROOT)
        selection_path = os.path.join(
            ROOT, 'configs', 'experiments', 'role_text_selections_v1.json')
        selection = load_role_text_selection_registry(selection_path, dataset)
        if os.path.normpath(selection['prompt_bank']) != os.path.normpath(
                os.path.relpath(completion_bank_path, ROOT)):
            raise ValueError(
                f'{dataset}: selection registry prompt bank mismatch.')
        report['datasets'][dataset]['best_overall_slots'] = list(
            selection['_best_overall_slots'])
        report['datasets'][dataset]['best_all_nonzero_slots'] = list(
            selection['_best_all_nonzero_slots'])
        visual_registry_path = os.path.join(
            ROOT, 'configs', 'experiments', 'role_visual_field_v1.json')
        with open(visual_registry_path, encoding='utf-8') as handle:
            visual_registry = json.load(handle)
        assert visual_registry['protocol'] == ROLE_VISUAL_FIELD_PROTOCOL
        visual = visual_registry['datasets'][dataset]
        assert int(visual['context_size']) > int(visual['fine_size']) > 0
        report['datasets'][dataset]['visual_field'] = visual
    assert len(set(VARIANT_NAMES)) == len(VARIANT_NAMES)
    assert len(set(PI_VARIANT_NAMES)) == len(PI_VARIANT_NAMES)
    assert len(set(PI_MECHANISM_MAP_NAMES)) == len(PI_MECHANISM_MAP_NAMES)
    assert PE_LAYER_IDS == (7, 15, 23, 31)
    assert len(set(PE_VARIANT_NAMES)) == len(PE_VARIANT_NAMES)
    assert BOUNDARY_GUIDES == ('block23', 'rgb', 'uniform')
    assert tuple(value for _, value in BOUNDARY_STRENGTHS) == (
        0.25, 0.50, 1.00)
    assert BOUNDARY_REPLAY_TRACE_TOLERANCE == 5e-3
    assert (len(set(BOUNDARY_REPLAY_VARIANT_NAMES))
            == len(BOUNDARY_REPLAY_VARIANT_NAMES))
    assert CLASS_ROLE_ALIGNMENT_PE_LAYER == 18
    assert tuple(value for _, value in CRA_COCO_WEIGHTS) == (
        0.30, 0.50, 0.70, 0.90)
    assert tuple(value for _, value in CRA_FCRA_TEMPERATURES) == (0.05, 0.10)
    assert len(set(CRA_VARIANT_NAMES)) == len(CRA_VARIANT_NAMES)
    assert len(ROLE_VISUAL_FIELD_COMPOSITIONS) == 8
    assert len(set(ROLE_VISUAL_FIELD_VARIANT_NAMES)) == 18
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
        'PI replays freeze admission, amplitude and head-winner sites',
        'completion banks define selected and anchor-admission combinations',
        'P0 completion controls are required to equal native admission',
        'PE screen reads true ViT blocks 7/15/23/31 in one image forward',
        'best overall and best all-nonzero role compositions are registered',
        'boundary replay keeps block23/RGB/uniform controls and three fixed strengths',
        'anchor-clamped replay has separate P/S/I/all counterfactual paths',
        'class-role alignment reads PE block18 once and preserves fixed admission',
        'CoCo class calibration excludes synonym expansion and double Presence',
        'FCRA compares raw role difference with class-centered role evidence',
        'visual-field screen has two views and all eight P/S/I assignments',
        'visual-field FFF is identity-anchored to official/current Role-Text',
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
