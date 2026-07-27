import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dual_head_fusion_diagnostic import (
    DualHeadFusionDiagnosticMixin,
    SCHEMA_VERSION,
    VARIANT_NAMES,
)
from tools.summarize_dual_head_fusion import summarize


class DummyDiagnostic(DualHeadFusionDiagnosticMixin):
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


def _inputs(with_candidate=True):
    semantic = torch.tensor(
        [[0.10, 0.70], [0.50, 0.20]], dtype=torch.float32)
    first_mask = torch.tensor(
        [[0.90, 0.20], [0.10, 0.80]], dtype=torch.float32)
    second_mask = torch.tensor(
        [[0.10, 0.80], [0.90, 0.10]], dtype=torch.float32)
    raw_scores = torch.tensor([0.8, 0.4], dtype=torch.float32)
    presence = torch.tensor(0.5, dtype=torch.float32)
    keep = (
        raw_scores * presence > 0.3
        if with_candidate
        else torch.zeros(2, dtype=torch.bool))
    native_instance = (
        first_mask * raw_scores[0] * presence
        if with_candidate
        else torch.zeros_like(semantic))
    baseline = presence * torch.maximum(
        semantic, native_instance)
    state = dict(
        presence_score=presence,
        raw_masks_logits_lowres=torch.stack(
            [_logit(first_mask), _logit(second_mask)]),
        raw_object_score=raw_scores,
        raw_keep_mask=keep,
    )
    return (
        semantic, first_mask, raw_scores,
        presence, native_instance, baseline, state,
    )


def test_formula_bank_matches_predeclared_role_and_proc_graphs():
    diagnostic = DummyDiagnostic()
    (
        semantic, first_mask, raw_scores,
        presence, native_instance, baseline, state,
    ) = _inputs()
    variants, stats = diagnostic._dhf_build_prompt_variants(
        state,
        semantic,
        native_instance,
        baseline,
        output_shape=(2, 2),
    )
    object_instance = first_mask * raw_scores[0]
    agreement_floor = 1.0 - 0.5 * (
        semantic - first_mask).abs()
    object_advantage = (
        object_instance - semantic).clamp_min(0.0)
    proc_base = torch.maximum(semantic, native_instance)
    proc_agreement = 1.0 - (
        semantic - proc_base).abs()
    proc_advantage = (
        proc_base - semantic).clamp_min(0.0)
    proc_gate = (
        presence * (presence * raw_scores[0])
        * proc_agreement)

    p1 = presence * torch.maximum(
        semantic, object_instance)
    proc = presence * (
        semantic + proc_gate * proc_advantage
    ).clamp(1e-4, 1 - 1e-4)
    proc_gate_debiased = presence * (
        semantic
        + raw_scores[0] * proc_agreement * proc_advantage
    ).clamp(1e-4, 1 - 1e-4)
    proc_role_base = torch.maximum(
        semantic, object_instance)
    proc_role_agreement = 1.0 - (
        semantic - proc_role_base).abs()
    proc_once = presence * (
        semantic
        + raw_scores[0]
        * proc_role_agreement
        * (proc_role_base - semantic).clamp_min(0.0)
    ).clamp(1e-4, 1 - 1e-4)
    uni = presence * (
        semantic + agreement_floor * object_advantage)
    boundary = 4.0 * first_mask * (1.0 - first_mask)
    semantic_i2s = (
        semantic
        + 0.25 * boundary * (first_mask - semantic)
    ).clamp(0.0, 1.0)
    i2s_agreement = 1.0 - 0.5 * (
        semantic_i2s - first_mask).abs()
    bi_i2s = presence * (
        semantic_i2s
        + i2s_agreement
        * (object_instance - semantic_i2s).clamp_min(0.0))

    torch.testing.assert_close(
        variants['p1_role_once'], p1)
    torch.testing.assert_close(
        variants['winner_score_once'], p1)
    torch.testing.assert_close(
        variants['proc_pgrf'], proc)
    torch.testing.assert_close(
        variants['proc_gate_debiased'],
        proc_gate_debiased)
    torch.testing.assert_close(
        variants['proc_role_once'], proc_once)
    torch.testing.assert_close(
        variants['uni_rcrf_floor'], uni)
    torch.testing.assert_close(
        variants['bi_s2i_only'], uni)
    torch.testing.assert_close(
        variants['bi_i2s_only'], bi_i2s)
    torch.testing.assert_close(
        variants['bi_full'], bi_i2s)
    assert stats['native_keep_count'] == 1
    assert stats['baseline_reconstruction_max_abs'] == 0.0


def test_no_candidate_path_is_explicit_and_finite():
    diagnostic = DummyDiagnostic()
    (
        semantic, _, _, presence,
        native_instance, baseline, state,
    ) = _inputs(with_candidate=False)
    variants, stats = diagnostic._dhf_build_prompt_variants(
        state,
        semantic,
        native_instance,
        baseline,
        output_shape=(2, 2),
    )
    protected = presence * semantic
    for name in (
            'semantic_only', 'instance_p0_path',
            'p1_role_once', 'winner_score_once',
            'uni_rcrf_floor', 'uni_rcrf_region',
            'uni_rcrf_unfloored',
            'bi_s2i_only', 'bi_i2s_only', 'bi_full',
            'soft_or', 'boundary_residual',
            'interior_residual', 'raw_mask_residual'):
        expected = (
            torch.zeros_like(protected)
            if name == 'instance_p0_path' else protected)
        torch.testing.assert_close(variants[name], expected)
    proc_empty = presence * semantic.clamp(
        1e-4, 1 - 1e-4)
    for name in (
            'proc_pgrf', 'proc_gate_debiased',
            'proc_role_once'):
        torch.testing.assert_close(
            variants[name], proc_empty)
    torch.testing.assert_close(
        variants['convex_25'], 0.75 * protected)
    assert stats['native_keep_count'] == 0
    assert all(
        torch.isfinite(value).all()
        for value in variants.values())


def _record(dataset, improved_variant='p1_role_once'):
    baseline = [[8, 2], [2, 8]]
    better = [[9, 1], [1, 9]]
    variants = {}
    for name in ('p0_baseline',) + VARIANT_NAMES:
        improved = name == improved_variant
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
        high_agreement_stats=[dict(
            gap_threshold=0.1,
            evidence_threshold=0.25,
            selected_pixels=4,
            wrong_pixels=1,
            correct_pixels=3,
            mean_wrong_presence=0.8,
            mean_wrong_object_score=0.7,
            mean_wrong_margin=0.2,
            low_presence_wrong_pixels=0,
            low_object_wrong_pixels=0,
        )],
        prompt_stats=[dict(
            raw_candidate_count=2,
            native_keep_count=1,
            presence=0.5,
            semantic_contrast_mean=0.1,
        )],
        class_stats=[],
        artifact_path=None,
    )


def test_summary_applies_four_of_six_prioritization_gate():
    datasets = [
        'udd5', 'vdd', 'vaihingen',
        'potsdam', 'openearthmap', 'loveda']
    records = []
    for index, dataset in enumerate(datasets):
        records.append(_record(
            dataset,
            improved_variant=(
                'p1_role_once' if index < 4 else 'semantic_only'),
        ))
    args = SimpleNamespace(
        expected_datasets=datasets,
        integrity_tolerance=1e-5,
        positive_dataset_gate=4,
    )
    result = summarize(records, args)
    ranking = {
        row['variant']: row for row in result[5]}
    decision = result[-1]
    assert decision['complete']
    assert decision['integrity_pass']
    assert ranking['p1_role_once']['positive_datasets'] == 4
    assert ranking['p1_role_once']['positive_gate_pass']


def test_summary_removes_only_consistent_cross_rank_padding_duplicate():
    first = _record('toy')
    duplicate = copy.deepcopy(first)
    duplicate['rank'] = 1
    args = SimpleNamespace(
        expected_datasets=['toy'],
        integrity_tolerance=1e-5,
        positive_dataset_gate=1,
    )
    result = summarize([first, duplicate], args)
    assert all(row['images'] == 1 for row in result[0])
    assert result[-1][
        'distributed_padding_duplicates_removed'] == {'toy': 1}
