_base_ = '../cfg_vdd.py'

model = dict(
    use_sem_seg=True,
    use_transformer_decoder=True,
    use_presence_score=True,
    instance_score_type='presence',
    use_reject_aware_calibration=False,
    use_role_prompt_tta=True,
    dump_role_prompt_tta_stats=True,
    role_prompt_tta_protocol='role_functional_text_screen_v1',
    role_prompt_tta_dataset_name='vdd',
    role_prompt_tta_prompt_bank=(
        'configs/prompt_banks/role_functional_text_v1/vdd.json'),
    role_prompt_tta_stats_path=(
        'logs/role_functional_text_screen_v1/vdd/screen.jsonl'),
    role_prompt_tta_primary_variant='baseline',
    role_prompt_tta_strict_integrity=True,
    role_prompt_tta_integrity_tolerance=1e-5,
    role_prompt_tta_save_npz=False,
)
