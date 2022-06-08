classes = ('no_cancer','cancer')
pretrained = 'checkpoints/epoch_6.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained,prefix='backbone')
        ),
    neck=dict(
        type='ViVit',
        dim = 768
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    # train_cfg=dict(augments=[
    #     dict(type='BatchMixup', alpha=0.8, num_classes=2, prob=0.5),
    #     dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=0.5)
    # ])
    )



# dataset settings
dataset_type = 'CsawSeq'
img_shape = (1120, 896)
train_pipeline = [
    dict(type='LoadSeqImageFromFile',rep_dim = -1,resize=(512, 512)),
    # dict(type='ElasticTransform',alpha=120,num_img = 2,elas_prob=1),
    dict(type='LinearNormalize',max_val=65535,),
    dict(type='DuoViewImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadSeqImageFromFile',rep_dim = -1, resize=(512, 512)),
    dict(type='LinearNormalize',max_val=65535,),
    dict(type='DuoViewImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# classes = ('lv1','lv2','lv3','lv4')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes = classes,
        # classes = ('lv1','lv2','lv3','lv4'),
        type=dataset_type,
        img_shape = img_shape,
        data_prefix='/mnt/h/datasets/csaw_seq_lv_set/',
        ann_file='/mnt/h/datasets/csaw_seq_lv_set/seq_lv_train_set_.csv',
        pipeline=train_pipeline),
    val=dict(
        classes = classes,
        type=dataset_type,
        img_shape = img_shape,
        data_prefix='/mnt/h/datasets/csaw_seq_lv_set/',
        ann_file='/mnt/h/datasets/csaw_seq_lv_set/seq_lv_val_set_.csv',
        pipeline=test_pipeline),
    test=dict(
        img_shape = img_shape,
        # replace `data/val` with `data/test` for standard test
        classes = classes,
        type=dataset_type,
        data_prefix='/mnt/h/datasets/csaw_seq_lv_set/',
        ann_file='/mnt/h/datasets/csaw_seq_lv_set/seq_lv_test_set_.csv',
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
work_dir = 'logs/csaw_seq_swin'
evaluation = dict(interval=1, metric='accuracy',metric_options = dict(topk=(1,)))