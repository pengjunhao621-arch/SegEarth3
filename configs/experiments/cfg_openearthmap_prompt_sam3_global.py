_base_ = [
    '../base_config.py',
    '../_base_/datasets/openearthmap_prompt.py',
]

model = dict(
    _delete_=True,
    type='PromptSegEarthOV3Segmentation',
    model_type='Prompt-SAM3-sam3-global',
    classname_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
    bg_idx=0,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
    use_sem_seg=True,
    use_presence_score=True,
    use_transformer_decoder=True,
    instance_score_type='presence',
    global_feature_source='sam3_global',
    global_feature_dim=256,
    num_shared_tokens=4,
    num_state_prototypes=4,
    num_global_tokens=1,
    prompt_residual_scale=0.05,
    prompt_initial_gate=0.10,
    prompt_attention_temperature=0.07,
    prompt_component_mode='full',
    prompt_batch_size=1,
    use_grounding_checkpoint=True,
    loss_ce_weight=1.0,
    loss_dice_weight=1.0,
    loss_anchor_weight=0.05,
    ignore_index=255,
    diagnostic_dir='work_dirs/prompt_sam3_sam3_global/diagnostics',
    diagnostic_panel_file=(
        'work_dirs/prompt_diagnostic_panels/openearthmap_val32.txt'),
    diagnostic_max_images=32,
    diagnostic_max_side=128,
)

# Primary command uses 2 GPUs.  With one image/GPU and two-way accumulation,
# 12,000 runner iterations equal 6,000 optimizer updates and effective batch 4.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=2,
    loss_scale='dynamic',
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'shared_context': dict(decay_mult=0.0),
            'state_slots': dict(decay_mult=0.0),
            'position_logits': dict(decay_mult=0.0),
            'gate_logits': dict(decay_mult=0.0),
        },
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=600),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=False,
        begin=600,
        end=12000),
]

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore']),
    dict(type='PromptValidationMetric', ignore_index=255, num_bins=15),
]
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=7),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1),
)

custom_hooks = [
    dict(type='PromptValidationArtifactHook', subdirectory='diagnostics'),
    dict(
        type='PromptGradientDiagnosticHook',
        interval=20,
        path='work_dirs/prompt_sam3_sam3_global/gradient_diagnostics.jsonl'),
    dict(
        type='PromptBestCheckpointHook',
        filename='best_prompt_source_val.pth',
        record_file='best_prompt_source_val.json',
        miou_key='mIoU',
        nll_key='prompt/validation_objective'),
]

randomness = dict(seed=0, deterministic=True)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
work_dir = 'work_dirs/prompt_sam3_sam3_global'
