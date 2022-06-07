classes = ('no_cancer','cancer')
pretrained = 'checkpoints/resnet50_patch_epoch_8.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type = 'ResNetMultiView',
        frozen_stages = 4,
        num_views=2,
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained,prefix='backbone')
        ),
    neck=dict(type='ResNecks',depth=2,in_channels = 4096),
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=512,
        loss=dict(type='FocalLoss'),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained,prefix='head')
    ))


# dataset settings
dataset_type = 'CsawBreast'
img_shape = (1120, 896)
train_pipeline = [
    dict(type='LoadBreastImageFromFile',rep_dim = -1),
    dict(type='ElasticTransform',alpha=120,num_img = 2,elas_prob=1),
    dict(type='LinearNormalize',max_val=65535,),
    dict(type='DuoViewImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadBreastImageFromFile',rep_dim = -1),
    dict(type='LinearNormalize',max_val=65535,),
    dict(type='DuoViewImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# classes = ('lv1','lv2','lv3','lv4')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        classes = classes,
        # classes = ('lv1','lv2','lv3','lv4'),
        type=dataset_type,
        img_shape = img_shape,
        data_prefix='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA_BREAST/csaw_breast_lv_set/',
        ann_file='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA_BREAST/breast_lv_train_set.csv',
        pipeline=train_pipeline),
    val=dict(
        classes = classes,
        type=dataset_type,
        img_shape = img_shape,
        data_prefix='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA_BREAST/csaw_breast_lv_set/',
        ann_file='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA_BREAST/breast_lv_test_set.csv',
        pipeline=test_pipeline),
    test=dict(
        img_shape = img_shape,
        # replace `data/val` with `data/test` for standard test
        classes = classes,
        type=dataset_type,
        data_prefix='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA_BREAST/csaw_breast_lv_set/',
        ann_file='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA_BREAST/breast_lv_val_set.csv',
        pipeline=test_pipeline))
# hooks
custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=10)]
# checkpoint saving
checkpoint_config = dict(interval=10)
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
optimizer = dict(type='Adam', lr=1e-5, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[5,10,20,40,60,80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
work_dir = 'logs/csaw_breast_resnet50'
evaluation = dict(interval=1, metric='accuracy',metric_options = dict(topk=(1,)))