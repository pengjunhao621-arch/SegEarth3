import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from presence_allocation_diagnostic import (
    PresenceAllocationDiagnosticMixin,
    VARIANT_NAMES,
)
from tools.summarize_presence_allocation import summarize


class DummyDiagnostic(PresenceAllocationDiagnosticMixin):
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


def test_presence_variants_isolate_predeclared_graph_changes():
    diagnostic = DummyDiagnostic()
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
    state = dict(
        presence_score=presence,
        raw_masks_logits_lowres=torch.stack(
            [_logit(first_mask), _logit(second_mask)]),
        raw_object_score=raw_scores,
        raw_keep_mask=native_keep,
    )

    variants, stats = diagnostic._pa_build_prompt_variants(
        state,
        semantic,
        native_instance,
        baseline,
        output_shape=(2, 2),
    )

    raw_gate_instance = torch.maximum(
        first_mask * raw_scores[0],
        second_mask * raw_scores[1],
    )
    p1_expected = torch.maximum(
        semantic * presence, native_instance)
    p2_expected = presence * torch.maximum(
        semantic, raw_gate_instance)
    p3_expected = torch.sigmoid(
        diagnostic._pa_logit(
            torch.maximum(semantic, raw_gate_instance), 1e-4)
        + diagnostic._pa_logit(presence, 1e-4)
    )
    p4_expected = torch.maximum(semantic, native_instance)

    torch.testing.assert_close(variants['p1_branch_once'], p1_expected)
    torch.testing.assert_close(variants['p2_delayed_gate'], p2_expected)
    torch.testing.assert_close(variants['p3_logit_prior'], p3_expected)
    torch.testing.assert_close(variants['p4_instance_scope'], p4_expected)
    assert stats['native_keep_count'] == 1
    assert stats['raw_gate_keep_count'] == 2
    assert stats['presence_deleted_count'] == 1
    assert stats['baseline_reconstruction_max_abs'] == 0.0


def _record(dataset):
    baseline = [[8, 2], [2, 8]]
    better = [[9, 1], [1, 9]]
    variants = {}
    for name in ('p0_baseline', *VARIANT_NAMES):
        confusion = better if name == 'p1_branch_once' else baseline
        variants[name] = dict(
            confusion=confusion,
            changed_pixels=2 if name == 'p1_branch_once' else 0,
            improved_pixels=2 if name == 'p1_branch_once' else 0,
            harmed_pixels=0,
            wrong_to_wrong_pixels=0,
            net_correct_pixels=(
                2 if name == 'p1_branch_once' else 0),
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
        variant_stats=variants,
        contrast_stats=[dict(
            reference='p0_baseline',
            candidate='p1_branch_once',
            hypothesis='remove_duplicate_instance_presence',
            changed_pixels=2,
            improved_pixels=2,
            harmed_pixels=0,
            wrong_to_wrong_pixels=0,
            net_correct_pixels=2,
        )],
        prompt_stats=[dict(
            raw_candidate_count=2,
            native_keep_count=1,
            raw_gate_keep_count=2,
            presence_deleted_count=1,
            presence=0.5,
            deleted_raw_instance_area=0.1,
            semantic_threshold_suppressed_area=0.2,
            instance_double_suppressed_area=0.3,
        )],
        class_stats=[
            dict(
                class_index=0,
                gt_pixels=10,
                semantic_threshold_suppressed_gt_pixels=1,
                instance_double_suppressed_gt_pixels=2,
            ),
            dict(
                class_index=1,
                gt_pixels=10,
                semantic_threshold_suppressed_gt_pixels=1,
                instance_double_suppressed_gt_pixels=2,
            ),
        ],
        artifact_path=None,
    )


def test_summary_requires_same_fixed_variant_to_improve_every_dataset():
    args = SimpleNamespace(
        expected_datasets=['toy_a', 'toy_b'],
        integrity_tolerance=1e-5,
    )
    (
        summary_rows,
        _,
        _,
        mechanism_rows,
        integrity_rows,
        feasibility,
    ) = summarize([_record('toy_a'), _record('toy_b')], args)

    assert feasibility['complete']
    assert feasibility['integrity_pass']
    assert feasibility['universal_positive']['p1_branch_once']
    assert not feasibility['universal_positive']['p2_delayed_gate']
    assert all(row['integrity_pass'] for row in integrity_rows)
    assert all(
        row['presence_deleted_candidate_ratio'] == 0.5
        for row in mechanism_rows
    )
    assert all(
        row['delta_miou'] > 0
        for row in summary_rows
        if row['variant'] == 'p1_branch_once'
    )


def test_summary_removes_only_consistent_cross_rank_padding_duplicate():
    first = _record('toy')
    duplicate = copy.deepcopy(first)
    duplicate['rank'] = 1
    args = SimpleNamespace(
        expected_datasets=['toy'],
        integrity_tolerance=1e-5,
    )

    result = summarize([first, duplicate], args)
    summary_rows = result[0]
    feasibility = result[-1]

    assert all(row['images'] == 1 for row in summary_rows)
    assert feasibility[
        'distributed_padding_duplicates_removed'] == {'toy': 1}
