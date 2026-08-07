_base_ = '../cfg_openearthmap.py'

model = dict(
    use_sem_seg=True,
    use_transformer_decoder=True,
    use_presence_score=True,
    instance_score_type='presence',
    use_reject_aware_calibration=False,
    use_role_prompt_tta=True,
    dump_role_prompt_tta_stats=True,
    role_prompt_tta_protocol='semantic_supplement_screen_v1',
    role_prompt_tta_dataset_name='openearthmap',
    role_prompt_tta_prompt_bank=(
        'configs/prompt_banks/semantic_supplement_v1/openearthmap.json'),
    role_prompt_tta_stats_path=(
        'logs/semantic_supplement_screen_v1/openearthmap/screen.jsonl'),
    role_prompt_tta_artifact_dir=(
        'logs/semantic_supplement_screen_v1/openearthmap/artifacts'),
    role_prompt_tta_primary_variant='baseline',
    role_prompt_tta_semantic_residual_alpha=0.50,
    role_prompt_tta_semantic_residual_clip=0.25,
    role_prompt_tta_strict_integrity=True,
    role_prompt_tta_integrity_tolerance=1e-5,
    role_prompt_tta_save_npz=True,
    role_prompt_tta_max_saved_images=8,
)
