_base_ = './base_config.py'
# python eval.py ./configs/cfg_vaihingen.py

# model settings
model = dict(
    classname_path='./configs/cls_vaihingen.txt',
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
)

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = '/home/PengJunhao/workspace/data/vaihingen'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
# _base_ = './base_config.py'

# custom_imports = dict(
#     imports=['segearthov3_segmentor', 'custom_datasets', 'cognitive_metric'],
#     allow_failed_imports=False)

# # model settings
# model = dict(
#     classname_path='./configs/cls_vaihingen.txt',
#     prob_thd=0.1,
#     bg_idx=5,
#     confidence_threshold=0.4,
# )

# # dataset settings
# dataset_type = 'ISPRSDataset'
# data_root = '/home/PengJunhao/workspace/data/vaihingen'

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='PackSegInputs')
# ]

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='img_dir/val',
#             seg_map_path='ann_dir/val'),
#         pipeline=test_pipeline))

# # 原有的 IoU 评估器 + 认知状态评估器
# val_evaluator = None
# test_evaluator = [
#     dict(type='IoUMetric', metrics=['mIoU', 'mAcc', 'aAcc']),  # ✅ 修正参数名
#     dict(type='CognitiveStateMetric', num_classes=6, n_bins=50, output_dir='./cognitive_plots')
# ]