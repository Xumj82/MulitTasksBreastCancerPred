_base_ = [
    'configs/mmcls/_base_/models/resnext50_32x4d.py',
    # 'configs/_base_/datasets/imagenet_bs32_pil_resize.py',
    'configs/mmcls/_base_/schedules/imagenet_bs256.py', 
    'configs/mmcls/_base_/default_runtime.py'
]


pretrained = 'checkpoints/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ResNecks',
        depth=2,
        num_stages=2,
        out_indices=(1, ),
        style='pytorch'), 
    head=dict(
            type='LinearClsHead',
            num_classes=2,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1,),
        )
    )

# dataset settings
dataset_type = 'DdsmBreast'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(type='RandomResizedCrop', size=512),
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

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=6,
    train=dict(
        classes = ('benign','malignant'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/ddsm_coco/ddsm_coco/train',
        ann_file= '/home/xumingjie/dataset/ddsm_coco/ddsm_coco/train/annotation_coco_5cls.json',
        pipeline=train_pipeline),
    val=dict(
        classes = ('benign','malignant'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/ddsm_coco/ddsm_coco/test/',
        ann_file='/home/xumingjie/dataset/ddsm_coco/ddsm_coco/test/annotation_coco_5cls.json',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        classes = ('benign','malignant'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/ddsm_coco/ddsm_coco/test/',
        ann_file='/home/xumingjie/dataset/ddsm_coco/ddsm_coco/test/annotation_coco_5cls.json',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='accuracy')