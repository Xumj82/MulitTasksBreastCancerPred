classes = ('lv1', 'lv2', 'lv3', 'lv4') # lv1 : <60 days, lv2 : 60~729 days, 3 : >729 days, 4: no cancer
img_shape = (1120, 896)
pretrained = 'checkpoints/swin_tiny_epoch_6.pth'
dataset_type = 'CsawSeq'
data_root = '/home/xumingjie/Desktop/CSAW_SEQ/'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        img_size=1024,
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone')),
    neck=dict(type='ViVit', dim=768),
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=768,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

train_pipeline = [
    dict(type='LoadSeqImageFromFile', rep_dim=-1, resize=(1024, 1024)),
    dict(type='LinearNormalize', max_val=65535),
    dict(type='DuoViewImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadSeqImageFromFile', rep_dim=-1, resize=(1024, 1024)),
    dict(type='LinearNormalize', max_val=65535),
    dict(type='DuoViewImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=14,
    train=dict(
        classes=classes,
        type=dataset_type,
        img_shape=(1120, 896),
        data_prefix=data_root+'csaw_seq_lv_set/',
        ann_file=data_root+'seq_lv_train_set_.csv',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        img_shape=(1120, 896),
        data_prefix=data_root+'csaw_seq_lv_set/',
        ann_file=data_root+'seq_lv_val_set_.csv',
        pipeline=test_pipeline),
    test=dict(
        img_shape=(1120, 896),
        classes=classes, 
        type=dataset_type,
        data_prefix=data_root+'csaw_seq_lv_set/',
        ann_file=data_root+'seq_lv_test_set_.csv',
        pipeline=test_pipeline))
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='Adam', lr=1e-05, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[5, 10, 20, 40, 60, 80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
work_dir = 'logs/csaw_seq_swin'
evaluation = dict(
    interval=1, metric='accuracy', metric_options=dict(topk=(1, ))
    )
