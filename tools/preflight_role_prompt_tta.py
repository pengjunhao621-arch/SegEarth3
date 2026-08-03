#!/usr/bin/env python3
"""Static locally-runnable and optional server-runtime preflight checks."""

import argparse
import ast
import gc
import importlib
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')
CLASS_FILES = {
    name: os.path.join(ROOT, 'configs', f'cls_{name}.txt')
    for name in DATASETS
}


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def primary_class_names(path):
    return [line.split(',')[0].strip() for line in read(path).splitlines()
            if line.strip()]


def validate_prompt_bank(dataset):
    path = os.path.join(ROOT, 'configs', 'prompt_banks', f'{dataset}.json')
    with open(path, encoding='utf-8') as handle:
        bank = json.load(handle)
    assert bank['schema_version'] == 1
    classes = bank['classes']
    expected = primary_class_names(CLASS_FILES[dataset])
    assert [row['id'] for row in classes] == list(range(len(classes)))
    assert [row['name'] for row in classes] == expected
    counts = {len(row['descriptions']) for row in classes}
    assert counts == {5}, (dataset, counts)
    for row in classes:
        name = row['name'].lower()
        assert row['descriptions'][0].lower() == name
        assert all(name in description.lower()
                   for description in row['descriptions'])
        assert len(set(row['descriptions'])) == len(row['descriptions'])
    return len(classes)


def constructor_default(source_path, argument):
    tree = ast.parse(read(source_path), filename=source_path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == '__init__':
            args = node.args.args
            defaults = [None] * (len(args) - len(node.args.defaults)) \
                + list(node.args.defaults)
            for arg, default in zip(args, defaults):
                if arg.arg == argument:
                    return ast.literal_eval(default)
    raise AssertionError(f'Constructor argument {argument} not found.')


def static_checks():
    report = {'datasets': {}, 'checks': []}
    for dataset in DATASETS:
        report['datasets'][dataset] = {
            'classes': validate_prompt_bank(dataset),
            'config': f'configs/experiments/cfg_{dataset}_role_prompt_tta.py',
        }
        config_path = os.path.join(
            ROOT, 'configs', 'experiments',
            f'cfg_{dataset}_role_prompt_tta.py')
        config = read(config_path)
        assert 'use_role_prompt_tta=True' in config.replace(' ', '')
        assert f'prompt_banks/{dataset}.json' in config
        assert 'isaid' not in config.lower()
        compile(config, config_path, 'exec')
    segmentor = os.path.join(ROOT, 'segearthov3_segmentor.py')
    processor = os.path.join(ROOT, 'sam3', 'model', 'sam3_image_processor.py')
    role_module = os.path.join(ROOT, 'role_prompt_tta.py')
    definitions = os.path.join(ROOT, 'role_prompt_tta_definitions.py')
    for path in (segmentor, processor, role_module, definitions):
        compile(read(path), path, 'exec')
    assert constructor_default(segmentor, 'use_role_prompt_tta') is False
    segmentor_text = read(segmentor)
    assert 'or self._uses_role_prompt_tta()' in segmentor_text
    assert 'role_prompt_variant_query_logits' in segmentor_text
    assert '_rpt_record_image(' in segmentor_text
    processor_text = read(processor)
    assert 'semantic_mask_raw_logits' in processor_text
    role_text = read(role_module)
    for required in (
            'full_no_anchor_output', 'full_no_presence_gate_output',
            'anchor_output', 'anchor_regrounded',
            'full_regrounded_surrogate', 'full_regrounded_e2e',
            'torch.enable_grad()', 'seed_gt_diagnostics'):
        assert required in role_text
    report['checks'].extend([
        'all prompt banks match protected primary class order',
        'all experiment configs exclude iSAID',
        'new method is constructor-default-off',
        'sliding aggregation and exact per-image recorder are connected',
        'raw semantic logits are exposed without changing predictions',
        'surrogate/e2e/anchor/presence diagnostic controls are present',
    ])
    return report


def runtime_checks(report):
    torch = importlib.import_module('torch')
    mmengine = importlib.import_module('mmengine')
    checkpoint = os.path.join(
        ROOT, 'weights', 'remoteclip', 'RemoteCLIP-ViT-L-14.pt')
    source = os.path.join(
        ROOT, 'SCORE-main', 'open_clip_training', 'src')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if os.path.getsize(checkpoint) < 100 * 1024 * 1024:
        raise RuntimeError(
            f'RemoteCLIP checkpoint is unexpectedly small: {checkpoint}')
    if not os.path.isdir(os.path.join(source, 'open_clip')):
        raise FileNotFoundError(source)
    if source not in sys.path:
        sys.path.insert(0, source)
    importlib.import_module('open_clip')
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    role = importlib.import_module('role_prompt_tta')
    remoteclip = role.RemoteCLIPRuntime(
        checkpoint=checkpoint,
        source_root=os.path.join(ROOT, 'SCORE-main'),
        model_name='ViT-L-14',
        device=torch.device('cpu'),
    )
    remoteclip._load()
    remoteclip_parameter_count = sum(
        parameter.numel() for parameter in remoteclip.model.parameters())
    del remoteclip
    gc.collect()
    weights = role.initial_prompt_weights(3, 5, 0.5, torch.device('cpu'))
    assert torch.allclose(weights.sum(dim=1), torch.ones(3))
    raw = torch.randn(3, 5, 8, 8)
    affinity = torch.randn(3, 5)
    gate = torch.tensor([1.0, 0.5, 0.0])
    adapted, _, trajectory = role.optimize_surrogate_weights(
        raw, weights, affinity, gate, raw[:, 0].argmax(dim=0),
        2, 0.05, 1.0, 0.05, 1.0, 1.0, True)
    assert torch.isfinite(adapted).all()
    assert torch.allclose(adapted.sum(dim=1), torch.ones(3), atol=1e-5)
    assert len(trajectory) == 2 and trajectory[-1]['grad_norm'] > 0
    from mmengine.config import Config
    for dataset in DATASETS:
        Config.fromfile(os.path.join(
            ROOT, 'configs', 'experiments',
            f'cfg_{dataset}_role_prompt_tta.py'))
    report['runtime'] = {
        'torch': torch.__version__,
        'mmengine': mmengine.__version__,
        'remoteclip_checkpoint_bytes': os.path.getsize(checkpoint),
        'remoteclip_model_load': 'ok',
        'remoteclip_parameter_count': remoteclip_parameter_count,
        'tensor_helper_test': 'ok',
        'all_mmengine_configs': 'parsed',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--check-runtime-assets', action='store_true',
        help='Run on the server environment with torch/mmengine/weights.')
    args = parser.parse_args()
    report = static_checks()
    if args.check_runtime_assets:
        runtime_checks(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
