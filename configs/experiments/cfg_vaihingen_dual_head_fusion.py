_base_ = '../cfg_vaihingen.py'

model = dict(
    seed_dataset_name='vaihingen',
    use_sem_seg=True,
    use_transformer_decoder=True,
    use_presence_score=True,
    instance_score_type='presence',
    use_reject_aware_calibration=False,
    dump_dual_head_fusion_stats=True,
    dual_head_fusion_dataset_name='vaihingen',
    dual_head_fusion_stats_path=(
        'logs/dual_head_fusion/vaihingen/dual_head_fusion.jsonl'),
    dual_head_fusion_chunk_size=8,
    dual_head_fusion_ring_kernel=7,
    dual_head_fusion_mask_threshold=0.50,
    dual_head_fusion_support_threshold=0.05,
    dual_head_fusion_agreement_gamma=0.50,
    dual_head_fusion_eta=0.25,
    dual_head_fusion_beta=0.25,
    dual_head_fusion_high_agreement_evidence=0.25,
    dual_head_fusion_high_agreement_gaps='0.05,0.10,0.20',
    dual_head_fusion_strict_integrity=True,
    dual_head_fusion_integrity_tolerance=1e-5,
    dual_head_fusion_save_npz=True,
    dual_head_fusion_artifact_dir=(
        'logs/dual_head_fusion/vaihingen/artifacts'),
    dual_head_fusion_artifact_max_side=128,
    dual_head_fusion_max_saved_images=8,
)
