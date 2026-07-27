#!/usr/bin/env python3
"""Server-side preflight for the same-forward dual-head fusion bank."""

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

from dual_head_fusion_diagnostic import (
    DualHeadFusionDiagnosticMixin,
    SCHEMA_VERSION,
    VARIANT_NAMES,
)
from tools.summarize_dual_head_fusion import summarize


DATASETS = (
    'udd5',
    'vdd',
    'vaihingen',
    'potsdam',
    'openearthmap',
    'loveda',
)


class _DummyDiagnostic(DualHeadFusionDiagnosticMixin):
    device = torch.device('cpu')
    use_sem_seg = True
    use_transformer_decoder = True
    use_presence_score = True
    instance_score_type = 'presence'
    use_reject_aware_calibration = False
    confidence_threshold = 0.3
    prob_thd = 0.25
    dual_head_fusion_chunk_size = 1
    dual_head_fusion_ring_kernel = 3
    dual_head_fusion_mask_threshold = 0.5
    dual_head_fusion_support_threshold = 0.05
    dual_head_fusion_agreement_gamma = 0.5
    dual_head_fusion_eta = 0.0
    dual_head_fusion_beta = 0.25
    dual_head_fusion_eps = 1e-6
    dual_head_fusion_integrity_tolerance = 1e-5

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


def _check_formulas():
    diagnostic = _DummyDiagnostic()
    semantic = torch.tensor(
        [[0.10, 0.70], [0.50, 0.20]], dtype=torch.float32)
    mask = torch.tensor(
        [[0.90, 0.20], [0.10, 0.80]], dtype=torch.float32)
    other = 1.0 - mask
    raw_scores = torch.tensor([0.8, 0.4], dtype=torch.float32)
    presence = torch.tensor(0.5, dtype=torch.float32)
    keep = raw_scores * presence > diagnostic.confidence_threshold
    native_instance = mask * raw_scores[0] * presence
    baseline = presence * torch.maximum(
        semantic, native_instance)
    variants, stats = diagnostic._dhf_build_prompt_variants(
        dict(
            presence_score=presence,
            raw_masks_logits_lowres=torch.stack(
                [_logit(mask), _logit(other)]),
            raw_object_score=raw_scores,
            raw_keep_mask=keep,
        ),
        semantic,
        native_instance,
        baseline,
        output_shape=(2, 2),
    )
    object_instance = mask * raw_scores[0]
    expected_p1 = presence * torch.maximum(
        semantic, object_instance)
    torch.testing.assert_close(
        variants['p1_role_once'], expected_p1)
    proc_base = torch.maximum(
        semantic, native_instance)
    proc_agreement = 1.0 - (
        semantic - proc_base).abs()
    proc_expected = presence * (
        semantic
        + presence * (presence * raw_scores[0])
        * proc_agreement
        * (proc_base - semantic).clamp_min(0.0)
    ).clamp(1e-4, 1 - 1e-4)
    torch.testing.assert_close(
        variants['proc_pgrf'], proc_expected)
    assert stats['native_keep_count'] == 1
    assert stats['baseline_reconstruction_max_abs'] == 0.0
    assert set(variants) == set(VARIANT_NAMES)

    empty_variants, empty_stats = (
        diagnostic._dhf_build_prompt_variants(
            dict(
                presence_score=presence,
                raw_masks_logits_lowres=torch.stack(
                    [_logit(mask), _logit(other)]),
                raw_object_score=raw_scores,
                raw_keep_mask=torch.zeros(
                    2, dtype=torch.bool),
            ),
            semantic,
            torch.zeros_like(semantic),
            presence * semantic,
            output_shape=(2, 2),
        )
    )
    assert empty_stats['native_keep_count'] == 0
    protected = presence * semantic
    for name in (
            'p1_role_once', 'winner_score_once',
            'uni_rcrf_floor', 'uni_rcrf_region',
            'uni_rcrf_unfloored', 'bi_s2i_only',
            'bi_i2s_only', 'bi_full', 'soft_or',
            'boundary_residual', 'interior_residual',
            'raw_mask_residual'):
        torch.testing.assert_close(
            empty_variants[name], protected)
    proc_empty = presence * semantic.clamp(
        1e-4, 1 - 1e-4)
    for name in (
            'proc_pgrf', 'proc_gate_debiased',
            'proc_role_once'):
        torch.testing.assert_close(
            empty_variants[name], proc_empty)


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
                        isinstance(
                            child,
                            (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == '__init__'):
                    arguments = child.args.args
                    defaults = child.args.defaults
                    offset = len(arguments) - len(defaults)
                    names = [
                        argument.arg for argument in arguments]
                    index = names.index(name) - offset
                    if index < 0:
                        raise AssertionError(
                            f'Constructor argument {name!r} has no default.')
                    return ast.literal_eval(defaults[index])
    raise AssertionError(
        f'Could not find constructor default {name!r}.')


def _check_query_reduction_order():
    path = os.path.join(ROOT, 'segearthov3_segmentor.py')
    with open(path) as handle:
        source = handle.read()
    required_fragments = (
        "components['dual_head_fusion_query_logits']",
        "'dual_head_fusion_class_logits'",
        'self._aggregate_query_logits_to_classes(\n'
        '                                query_logits)',
        '(self.num_queries, h_img, w_img)',
    )
    missing = [
        fragment for fragment in required_fragments
        if fragment not in source
    ]
    if missing:
        raise AssertionError(
            'Dual-head fusion must merge sliding crops in query space '
            'before synonym/class aggregation; missing source contracts: '
            f'{missing}')


def _check_configs():
    assert _constructor_default(
        'dump_dual_head_fusion_stats') is False
    _check_query_reduction_order()
    for dataset in DATASETS:
        path = os.path.join(
            ROOT,
            'configs',
            'experiments',
            f'cfg_{dataset}_dual_head_fusion.py',
        )
        cfg = Config.fromfile(path)
        model = cfg.model
        assert model['dump_dual_head_fusion_stats'] is True
        assert model['dual_head_fusion_dataset_name'] == dataset
        assert model['instance_score_type'] == 'presence'
        assert model['use_presence_score'] is True
        assert model['use_sem_seg'] is True
        assert model['use_transformer_decoder'] is True
        assert model['dual_head_fusion_strict_integrity'] is True


def _summary_record(dataset):
    baseline = [[8, 2], [2, 8]]
    better = [[9, 1], [1, 9]]
    variants = {}
    for name in ('p0_baseline',) + VARIANT_NAMES:
        improved = name == 'p1_role_once'
        variants[name] = dict(
            confusion=better if improved else baseline,
            valid_pixels=20,
            changed_pixels=2 if improved else 0,
            improved_pixels=2 if improved else 0,
            harmed_pixels=0,
            wrong_to_wrong_pixels=0,
            net_correct_pixels=2 if improved else 0,
        )
    return dict(
        schema_version=SCHEMA_VERSION,
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
        variant_stats=variants,
        contrast_stats=[],
        high_agreement_stats=[],
        prompt_stats=[],
        class_stats=[],
        artifact_path=None,
    )


def _check_summary():
    args = SimpleNamespace(
        expected_datasets=list(DATASETS),
        integrity_tolerance=1e-5,
        positive_dataset_gate=4,
    )
    result = summarize(
        [_summary_record(dataset) for dataset in DATASETS],
        args,
    )
    decision = result[-1]
    assert decision['complete']
    assert decision['integrity_pass']
    assert 'p1_role_once' in decision[
        'variants_passing_positive_gate']


def main():
    _check_formulas()
    _check_configs()
    _check_summary()
    print(
        'Dual-head fusion v1 preflight: PASS '
        '(formula bank, exact ProC control, baseline default, '
        'query-first slide reduction, six configs, summary gate).')


if __name__ == '__main__':
    main()
