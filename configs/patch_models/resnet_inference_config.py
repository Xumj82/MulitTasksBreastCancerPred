from sys import prefix


_base_ = [
    '../mmcls/_base_/models/resnet50.py',
    # 'configs/_base_/datasets/imagenet_bs32_pil_resize.py',
    # '../mmcls/_base_/schedules/imagenet_bs256.py', 
    # '../mmcls/_base_/default_runtime.py'
]

# pretrained = 'checkpoints/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type = 'ResNetNoPool',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50_patch_epoch_8.pth',prefix='backbone'),
        style='pytorch'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50_patch_epoch_8.pth',prefix='head'),
        topk=(1, 5),
    ))
# dataset settings
dataset_type = 'DdsmPatch'
img_shape = (224, 224)

test_pipeline = [
    # dict(type='LoadMMImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='LinearNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=12,
    test=dict(
        img_shape = img_shape,
        # replace `data/val` with `data/test` for standard test
        classes = ('bkg','be_calc','be_mass','ma_calc','ma_mass'),
        type=dataset_type,
        data_prefix='/root/autodl-tmp/patch_set_1150_224/patch_set',
        ann_file='/root/autodl-tmp/patch_set_1150_224/test_meta.csv',
        pipeline=test_pipeline))
