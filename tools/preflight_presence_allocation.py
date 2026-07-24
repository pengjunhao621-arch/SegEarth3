#!/usr/bin/env python3
"""Server-side preflight for Presence Allocation Structural Audit v1."""

import ast
import os
import sys
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from mmengine.config import Config

from presence_allocation_diagnostic import (
    PresenceAllocationDiagnosticMixin,
    VARIANT_NAMES,
)
from tools.summarize_presence_allocation import summarize


DATASETS = (
    'udd5',
    'vdd',
    'vaihingen',
    'potsdam',
    'openearthmap',
    'loveda',
)


class _DummyDiagnostic(PresenceAllocationDiagnosticMixin):
    device = torch.device('cpu')
    use_sem_seg = True
    use_transformer_decoder = True
    use_presence_score = True
    instance_score_type = 'presence'
    use_reject_aware_calibration = False
    confidence_threshold = 0.3
    prob_thd = 0.25
    presence_allocation_raw_gate_threshold = None
    presence_allocation_support_threshold = 0.05
    presence_allocation_chunk_size = 1
    presence_allocation_logit_eps = 1e-4

    @staticmethod
    def _interpolate_float32(tensor, shape):
        return F.interpolate(
            tensor.float(),
            size=shape,
            mode='bilinear',
            align_corners=False,
        )


def _logit(probability):
    probability = probability.clamp(1e-4, 1 - 1e-4)
    return torch.log(probability / (1 - probability))


def _check_formula():
    diagnostic = _DummyDiagnostic()
    semantic = torch.tensor(
        [[0.10, 0.70], [0.50, 0.20]], dtype=torch.float32)
    first_mask = torch.tensor(
        [[0.90, 0.20], [0.10, 0.80]], dtype=torch.float32)
    second_mask = torch.tensor(
        [[0.10, 0.80], [0.90, 0.10]], dtype=torch.float32)
    raw_scores = torch.tensor([0.8, 0.4], dtype=torch.float32)
    presence = torch.tensor(0.5, dtype=torch.float32)
    native_keep = raw_scores * presence > diagnostic.confidence_threshold
    native_instance = first_mask * raw_scores[0] * presence
    baseline = presence * torch.maximum(semantic, native_instance)
    variants, stats = diagnostic._pa_build_prompt_variants(
        dict(
            presence_score=presence,
            raw_masks_logits_lowres=torch.stack(
                [_logit(first_mask), _logit(second_mask)]),
            raw_object_score=raw_scores,
            raw_keep_mask=native_keep,
        ),
        semantic,
        native_instance,
        baseline,
        output_shape=(2, 2),
    )
    raw_gate_instance = torch.maximum(
        first_mask * raw_scores[0],
        second_mask * raw_scores[1],
    )
    expected = dict(
        p1_branch_once=torch.maximum(
            semantic * presence, native_instance),
        p2_delayed_gate=presence * torch.maximum(
            semantic, raw_gate_instance),
        p3_logit_prior=torch.sigmoid(
            diagnostic._pa_logit(
                torch.maximum(semantic, raw_gate_instance), 1e-4)
            + diagnostic._pa_logit(presence, 1e-4)),
        p4_instance_scope=torch.maximum(semantic, native_instance),
    )
    for name in VARIANT_NAMES:
        torch.testing.assert_close(variants[name], expected[name])
    assert stats['presence_deleted_count'] == 1
    assert stats['baseline_reconstruction_max_abs'] == 0.0


def _constructor_default(name):
    path = os.path.join(ROOT, 'segearthov3_segmentor.py')
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if (
                isinstance(node, ast.ClassDef)
                and node.name == 'SegEarthOV3Segmentation'):
            for child in node.body:
                if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == '__init__'):
                    arguments = child.args.args
                    defaults = child.args.defaults
                    offset = len(arguments) - len(defaults)
                    names = [argument.arg for argument in arguments]
                    argument_index = names.index(name)
                    default_index = argument_index - offset
                    if default_index < 0:
                        raise AssertionError(
                            f'Constructor argument {name!r} has no default.')
                    return ast.literal_eval(defaults[default_index])
    raise AssertionError(f'Could not find constructor default {name!r}.')


def _check_configs():
    assert _constructor_default('dump_presence_allocation_stats') is False
    for dataset in DATASETS:
        path = os.path.join(
            ROOT,
            'configs',
            'experiments',
            f'cfg_{dataset}_presence_allocation.py',
        )
        cfg = Config.fromfile(path)
        assert cfg.model['dump_presence_allocation_stats'] is True
        assert cfg.model['presence_allocation_dataset_name'] == dataset
        assert cfg.model['presence_allocation_raw_gate_threshold'] is None
        assert cfg.model['presence_allocation_strict_integrity'] is True


def _summary_record(dataset):
    baseline = [[8, 2], [2, 8]]
    better = [[9, 1], [1, 9]]
    variant_stats = {}
    for name in ('p0_baseline', *VARIANT_NAMES):
        improved = name == 'p1_branch_once'
        variant_stats[name] = dict(
            confusion=better if improved else baseline,
            changed_pixels=2 if improved else 0,
            improved_pixels=2 if improved else 0,
            harmed_pixels=0,
            wrong_to_wrong_pixels=0,
            net_correct_pixels=2 if improved else 0,
        )
    return dict(
        schema_version='presence-allocation-v1',
        dataset_name=dataset,
        img_path=f'/synthetic/{dataset}.png',
        rank=0,
        valid_pixels=20,
        class_names=['background', 'object'],
        integrity=dict(
            class_logit_max_abs=0.0,
            class_logit_mae=0.0,
            prompt_reconstruction_max_abs=0.0,
            baseline_prediction_mismatch_pixels=0,
        ),
        variant_stats=variant_stats,
        contrast_stats=[],
        prompt_stats=[],
        class_stats=[],
        artifact_path=None,
    )


def _check_summary():
    args = SimpleNamespace(
        expected_datasets=list(DATASETS),
        integrity_tolerance=1e-5,
    )
    result = summarize(
        [_summary_record(dataset) for dataset in DATASETS],
        args,
    )
    feasibility = result[-1]
    assert feasibility['complete']
    assert feasibility['integrity_pass']
    assert feasibility['universal_positive']['p1_branch_once']
    assert not feasibility['universal_positive']['p2_delayed_gate']


def main():
    _check_formula()
    _check_configs()
    _check_summary()
    print(
        'Presence Allocation v1 preflight: PASS '
        '(formula, baseline default, six configs, summary gate).')


if __name__ == '__main__':
    main()
