_base_ = '../cfg_vaihingen.py'

model = dict(
    seed_dataset_name='vaihingen',
    dump_cross_image_bank_stats=False,
    cross_image_bank_spaces='vision,pe_layer_0,pe_layer_1,pe_layer_2',
    cross_image_bank_seed_rule='final_score_margin_sem_inst_final_agree_local_core',
    cross_image_bank_min_seed_pixels=4,
    cross_image_bank_min_pair_pixels=4,
    cross_image_bank_feature_max_side=1024,
)
