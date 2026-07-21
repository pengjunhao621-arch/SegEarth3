_base_ = './cfg_openearthmap_prompt_sam3_global.py'

model = dict(
    model_type='Prompt-SAM3-remoteclip-global',
    global_feature_source='remoteclip_global',
    global_feature_dim=768,
    remoteclip_checkpoint='weights/remoteclip/RemoteCLIP-ViT-L-14.pt',
    remoteclip_source_root='SCORE-main/open_clip_training/src',
    diagnostic_dir='work_dirs/prompt_sam3_remoteclip_global/diagnostics',
)

custom_hooks = [
    dict(type='PromptValidationArtifactHook', subdirectory='diagnostics'),
    dict(
        type='PromptGradientDiagnosticHook',
        interval=20,
        path='work_dirs/prompt_sam3_remoteclip_global/gradient_diagnostics.jsonl'),
    dict(
        type='PromptBestCheckpointHook',
        filename='best_prompt_source_val.pth',
        record_file='best_prompt_source_val.json',
        miou_key='mIoU',
        nll_key='prompt/validation_objective'),
]

work_dir = 'work_dirs/prompt_sam3_remoteclip_global'
