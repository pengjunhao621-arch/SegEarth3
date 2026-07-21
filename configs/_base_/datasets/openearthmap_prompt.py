"""OpenEarthMap data fragment for the prompt-synthesis training variant.

This is intentionally data-only.  It should be inherited by the future
prompt-training experiment config after the trainable segmentor is added.  The
official evaluation config ``configs/cfg_openearthmap.py`` remains unchanged.
"""

dataset_type = 'OpenEarthMapDataset'
data_root = '/home/PengJunhao/workspace/data/OpenEarthMap'

# The Prompt-SAM3 segmentor consumes this transformed tensor directly.  The
# same augmented RGB view is then passed to SAM3 and, when selected, RemoteCLIP.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomDiscreteRotate90', choices=(0, 1, 2, 3)),
    dict(
        type='MildBrightnessContrastSaturation',
        prob=0.8,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2)),
    dict(type='PackSegInputs'),
]

eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/train',
            seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=eval_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
