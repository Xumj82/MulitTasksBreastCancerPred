_base_ = '../mmdet/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2,
                aligned=False)),
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_roi_extractor=dict(
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=2,
                aligned=False))))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_img_label','gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
classes = ('lesion',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file= '/mnt/c/Users/11351/Desktop/datasets/csaw_coco/train/annotation_coco.json',
        img_prefix='/mnt/c/Users/11351/Desktop/datasets/csaw_coco/train/'),
    val=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/mnt/c/Users/11351/Desktop/datasets/csaw_coco/val/annotation_coco.json',
        img_prefix='/mnt/c/Users/11351/Desktop/datasets/csaw_coco/val/'),
    test=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/mnt/c/Users/11351/Desktop/datasets/csaw_coco/val/annotation_coco.json',
        img_prefix='/mnt/c/Users/11351/Desktop/datasets/csaw_coco/val/')
    )