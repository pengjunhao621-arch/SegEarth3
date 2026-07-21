_base_ = '../cfg_udd5.py'

model = dict(
    seed_dataset_name='udd5',
    dump_scene_common_bias_stats=False,
    dump_candidate_residual_miou_stats=False,
)
