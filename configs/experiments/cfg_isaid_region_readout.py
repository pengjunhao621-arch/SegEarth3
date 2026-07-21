_base_ = '../cfg_iSAID.py'

model = dict(
    seed_dataset_name='isaid',
    use_region_contrastive_readout=False,
    dump_region_contrastive_readout_stats=True,
    region_contrastive_readout_stats_path=(
        'logs/region_readout_v1/isaid/region.jsonl'),
    region_readout_max_side=256,
    region_readout_max_regions=192,
    region_readout_masks_per_prompt=8,
    region_readout_mask_threshold=0.50,
    region_readout_min_pixels=16,
    region_readout_dedup_iou=0.90,
    region_readout_ring_kernel=7,
    region_readout_top_fraction=0.25,
    region_readout_presence_weight=0.10,
    region_readout_topk=3,
    region_readout_blends='0.25,0.50,1.00',
    region_readout_mix_sources=(
        'true_zcontrast_top,true_zcontrast_top_presence,'
        'rect_zcontrast_top,shift_zcontrast_top,'
        'classperm_zcontrast_top,pixel_semantic'),
    region_readout_apply_source='true_zcontrast_top_presence',
    region_readout_apply_blend=0.50,
    region_readout_apply_topk=3,
    region_readout_min_margin=0.10,
    region_readout_min_pair_pixels=16,
    region_readout_high_purity_threshold=0.80,
)
