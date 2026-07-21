_base_ = './cfg_vaihingen_region_readout.py'

model = dict(
    use_region_contrastive_readout=False,
    dump_region_contrastive_readout_stats=True,
    region_contrastive_readout_stats_path=(
        'logs/region_route_v2/diagnostic/vaihingen/region.jsonl'),
    region_readout_projection='region_winner',
    region_readout_route_variant='foreground_reject_contrast_mean',
    region_readout_foreground_blend=0.25,
    region_readout_reject_blend=0.25,
    region_readout_background_blend=0.25,
    region_readout_formula_agreement=1,
    region_readout_agreement_sources=(
        'true_inside_mean,true_inside_top,'
        'true_contrast_top,true_zcontrast_top'),
    region_readout_mix_sources=(
        'true_inside_mean,true_inside_top,true_contrast_top,'
        'true_zcontrast_top,true_zcontrast_top_presence,'
        'pixel_semantic'),
    region_readout_blends='0.25',
)
