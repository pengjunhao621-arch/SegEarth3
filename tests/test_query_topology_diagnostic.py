import torch
import torch.nn.functional as F

from query_topology_diagnostic import QueryTopologyDiagnosticMixin


class DummyDiagnostic(QueryTopologyDiagnosticMixin):
    device = torch.device('cpu')
    instance_score_type = 'presence'
    use_presence_score = True

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


def test_collect_sources_reconstructs_prompt_local_max_and_keeps_provenance():
    diagnostic = DummyDiagnostic()
    first = torch.tensor(
        [[0.9, 0.8], [0.1, 0.1]], dtype=torch.float32)
    second = torch.tensor(
        [[0.1, 0.2], [0.8, 0.9]], dtype=torch.float32)
    candidates = [dict(
        query_index=0,
        class_index=1,
        query_word='building',
        view_id='full_image',
        crop_box=None,
        selected_indices=torch.tensor([7, 11]),
        raw_masks_lowres=torch.stack([_logit(first), _logit(second)]),
        raw_scores=torch.tensor([0.8, 0.6]),
        raw_presence_scores=torch.tensor([0.4, 0.3]),
        raw_keep_mask=torch.tensor([True, True]),
        raw_candidate_count=2,
        raw_kept_count=2,
    )]

    result = diagnostic._qtd_collect_sources(
        candidates, (2, 2), (2, 2), query_count=1)

    expected = torch.maximum(first * 0.4, second * 0.3)
    torch.testing.assert_close(
        result['reconstructed_query_instance'][0], expected)
    assert result['kept_selection_complete']
    assert result['selected_kept_count'] == 2
    assert [
        row['object_query_index'] for row in result['source_metadata']
    ] == [7, 11]
    # Atomic scores include the later baseline presence multiplication and
    # retain only pixels where that object query wins the prompt-local max.
    first_wins = first * 0.4 >= second * 0.3
    torch.testing.assert_close(
        result['source_scores'][0],
        first * 0.4 * 0.5 * first_wins.float(),
    )


def test_topology_separates_one_coherent_source_from_fragmented_sources():
    region = torch.ones((4, 4), dtype=torch.bool)
    coherent = torch.ones((1, 4, 4), dtype=torch.float32)
    fragmented = torch.zeros((4, 4, 4), dtype=torch.float32)
    fragmented[0, :2, :2] = 1.0
    fragmented[1, :2, 2:] = 1.0
    fragmented[2, 2:, :2] = 1.0
    fragmented[3, 2:, 2:] = 1.0

    coherent_stats = DummyDiagnostic._qtd_topology_from_scores(
        coherent, region, threshold=0.5)
    fragmented_stats = DummyDiagnostic._qtd_topology_from_scores(
        fragmented, region, threshold=0.5)

    assert coherent_stats['dominant_source_coverage'] == 1.0
    assert coherent_stats['effective_source_count'] == 1.0
    assert fragmented_stats['dominant_source_coverage'] == 0.25
    assert fragmented_stats['effective_source_count'] == 4.0
    assert (
        fragmented_stats['switch_boundary_density']
        > coherent_stats['switch_boundary_density']
    )


def test_spatial_roll_control_is_deterministic_and_preserves_source_mass():
    scores = torch.arange(
        3 * 5 * 7, dtype=torch.float32).view(3, 5, 7)
    first = DummyDiagnostic._qtd_roll_control(scores)
    second = DummyDiagnostic._qtd_roll_control(scores)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first.flatten(1).sum(dim=1),
        scores.flatten(1).sum(dim=1),
    )
