_base_ = '../cfg_udd5.py'

model = dict(
    use_sem_seg=True,
    use_transformer_decoder=True,
    use_presence_score=True,
    instance_score_type='presence',
    use_reject_aware_calibration=False,
    use_role_prompt_tta=True,
    dump_role_prompt_tta_stats=True,
    role_prompt_tta_protocol='head_role_prompt_conflict_v1',
    role_prompt_tta_dataset_name='udd5',
    role_prompt_tta_prompt_bank='configs/prompt_banks/udd5.json',
    role_prompt_tta_stats_path=(
        'logs/head_role_prompt_conflict_v1/udd5/head_role_conflict.jsonl'),
    role_prompt_tta_artifact_dir=(
        'logs/head_role_prompt_conflict_v1/udd5/artifacts'),
    role_prompt_tta_primary_variant='baseline',
    role_prompt_tta_strict_integrity=True,
    role_prompt_tta_integrity_tolerance=1e-5,
    role_prompt_tta_save_npz=True,
    role_prompt_tta_max_saved_images=8,
)
