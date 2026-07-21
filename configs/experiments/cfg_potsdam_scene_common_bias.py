_base_ = '../cfg_potsdam.py'

model = dict(
    seed_dataset_name='potsdam',
    dump_scene_common_bias_stats=False,
    dump_candidate_residual_miou_stats=False,
)
