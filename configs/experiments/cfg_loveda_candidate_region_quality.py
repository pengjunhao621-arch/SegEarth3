_base_ = '../cfg_loveda.py'

model = dict(
    seed_dataset_name='loveda',
    dump_candidate_region_quality_stats=True,
    candidate_region_quality_stats_path=(
        'logs/candidate_region_quality/loveda/regions.jsonl'),
    raw_mask_oracle_topk=10,
    region_readout_max_regions=192,
    region_readout_masks_per_prompt=8,
    region_readout_mask_threshold=0.50,
    region_readout_min_pixels=16,
    region_readout_dedup_iou=0.90,
    candidate_region_quality_sources=(
        'raw_instance,grouped_instance,semantic_cc,hybrid'),
    candidate_region_quality_max_side=256,
    candidate_region_quality_max_regions=384,
    candidate_region_quality_min_pixels=16,
    candidate_region_quality_gt_min_pixels=16,
    candidate_region_quality_high_purity=0.80,
    candidate_region_quality_group_iou=0.35,
    candidate_region_quality_group_containment=0.75,
    candidate_region_quality_semantic_threshold=0.35,
    candidate_region_quality_semantic_max_per_class=32,
    candidate_region_quality_hybrid_dedup_iou=0.90,
    candidate_region_quality_fragment_coverage=0.10,
)
