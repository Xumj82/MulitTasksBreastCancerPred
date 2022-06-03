from sys import prefix


_base_ = [
    '../mmcls/_base_/models/resnet50.py',
    # 'configs/_base_/datasets/imagenet_bs32_pil_resize.py',
    # '../mmcls/_base_/schedules/imagenet_bs256.py', 
    # '../mmcls/_base_/default_runtime.py'
]

# pretrained = 'checkpoints/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'
model = dict(
    backbone=dict(
        frozen_stages = -1,
        init_cfg=dict(type='Pretrained', checkpoint='logs/ddsm_patch_resnet50/epoch_10.pth',prefix='backbone')),
    head=dict(
            type='LinearClsHead',
            num_classes=5,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1,),
            init_cfg=dict(type='Pretrained', checkpoint='logs/ddsm_patch_resnet50/epoch_10.pth',prefix='head')
        )
    )

# dataset settings
dataset_type = 'DdsmPatch'
img_shape = (224, 224)
train_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(type='ElasticTransform',alpha=500),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='LinearNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadMMImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='LinearNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=12,
    train=dict(
        classes = ('bkg','be_calc','be_mass','ma_calc','ma_mass'),
        type=dataset_type,
        img_shape = img_shape,
        split = True,
        data_prefix='/root/autodl-tmp/patch_set_1150_224/patch_set',
        ann_file='/root/autodl-tmp/patch_set_1150_224/train_meta.csv',
        pipeline=train_pipeline),
    val=dict(
        classes = ('bkg','be_calc','be_mass','ma_calc','ma_mass'),
        type=dataset_type,
        img_shape = img_shape,
        split = True,
        data_prefix='/root/autodl-tmp/patch_set_1150_224/patch_set',
        ann_file='/root/autodl-tmp/patch_set_1150_224/train_meta.csv',
        pipeline=test_pipeline),
    test=dict(
        img_shape = img_shape,
        # replace `data/val` with `data/test` for standard test
        classes = ('bkg','be_calc','be_mass','ma_calc','ma_mass'),
        type=dataset_type,
        data_prefix='/root/autodl-tmp/patch_set_1150_224/patch_set',
        ann_file='/root/autodl-tmp/patch_set_1150_224/test_meta.csv',
        pipeline=test_pipeline))
# hooks
# custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=3)]
# checkpoint saving
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=1e-6, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3,6])
runner = dict(type='EpochBasedRunner', max_epochs=10)
work_dir = 'logs/ddsm_patch_resnet50'
evaluation = dict(interval=1, metric='accuracy',metric_options = dict(topk=(1,)))