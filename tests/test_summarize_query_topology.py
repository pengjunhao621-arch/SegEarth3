from tools.summarize_query_topology import (
    build_conditional_rows,
    build_gate_rows,
)


def _region(index):
    coherent = index % 2 == 0
    quality = 0.9 if coherent else 0.2
    return dict(
        dataset='toy',
        class_index=1,
        is_background=False,
        area_pixels=64,
        final_score_mean=0.6,
        instance_head_fraction=0.8,
        gt_purity=quality,
        baseline_local_iou=quality,
        best_action_delta=0.01 if coherent else 0.15,
        dominant_source_coverage=0.9 if coherent else 0.2,
        effective_source_count=1.1 if coherent else 4.0,
        switch_boundary_density=0.02 if coherent else 0.5,
        consensus_fraction=0.8 if coherent else 0.1,
        control_roll_dominant_source_coverage=(index % 3) / 3,
        control_roll_effective_source_count=2.0 + (index % 3),
        control_roll_switch_boundary_density=(index % 4) / 4,
        control_roll_consensus_fraction=(index % 5) / 5,
    )


def test_conditional_summary_and_gate_require_incremental_signal():
    regions = [_region(index) for index in range(60)]
    conditional = build_conditional_rows(regions)
    assert any(
        row['target'] == 'gt_purity'
        and row['metric'] == 'dominant_source_coverage'
        and row['directional_lift'] > 0
        for row in conditional
    )

    images = [dict(
        dataset='toy',
        kept_selection_complete=True,
        raw_record_complete=True,
        reconstruction_mae=0.01,
    )]
    action_rows = [dict(
        dataset='toy',
        oracle_improvable_rate=0.5,
        mean_positive_oracle_gain=0.08,
    )]

    class Args:
        max_reconstruction_mae = 0.05
        min_foreground_regions = 50
        min_improvable_rate = 0.05
        min_mean_positive_oracle_gain = 0.01
        min_conditional_lift = 0.02

    gates = build_gate_rows(
        images,
        regions,
        conditional,
        action_rows,
        ['toy'],
        Args(),
    )
    assert gates[0]['passed']
