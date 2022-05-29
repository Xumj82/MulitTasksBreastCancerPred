from sys import prefix


_base_ = [
    '../mmcls/_base_/models/resnext50_32x4d.py',
    # 'configs/_base_/datasets/imagenet_bs32_pil_resize.py',
    '../mmcls/_base_/schedules/imagenet_bs256.py', 
    '../mmcls/_base_/default_runtime.py'
]


pretrained = 'checkpoints/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone')),
    head=dict(
            type='LinearClsHead',
            num_classes=3,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1,),
        )
    )

# dataset settings
dataset_type = 'DdsmPatch'
img_norm_cfg = dict(
    mean=[0.229, 0.224, 0.225], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(type='ElasticTransform',alpha=500),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
# train_pipeline = [
#     dict(type='LoadMMImageFromFile'),
#     dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 15)),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='RandomFlip', flip_ratio=0.5,direction='vertical'),
#     dict(
#         type='Translate',
#         magnitude_key='magnitude',
#         magnitude_range=(0, 0.2),
#         direction='horizontal'),
#     dict(
#         type='Translate',
#         magnitude_key='magnitude',
#         magnitude_range=(0, 0.2),
#         direction='vertical'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]

# test_pipeline = [
#     dict(type='LoadMMImageFromFile'),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        classes = ('bkg','calc','mass'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/patch_set/img_dir/train',
        ann_file='/home/xumingjie/dataset/patch_set/img_dir/train_meta.csv',
        pipeline=train_pipeline),
    val=dict(
        classes = ('bkg','calc','mass'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/patch_set/img_dir/test',
        ann_file='/home/xumingjie/dataset/patch_set/img_dir/test_meta.csv',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        classes = ('bkg','calc','mass'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/patch_set/img_dir/test',
        ann_file='/home/xumingjie/dataset/patch_set/img_dir/test_meta.csv',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='accuracy')