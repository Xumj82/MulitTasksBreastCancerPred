from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import mmcv
import os.path as osp
import os
from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg = Config.fromfile('configs/breast_models/csaw_breast_mask_rcnn_config.py')
cfg.dataset_type = 'COCODataset'
cfg.data.samples_per_gpu=2
cfg.data.workers_per_gpu=1
cfg.data.test.ann_file = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/csaw_coco/test/annotation_coco.json'
cfg.data.test.img_prefix = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/csaw_coco/test/'
cfg.data.test.classes = ('lesion')

cfg.data.train.ann_file = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/csaw_coco/train/annotation_coco.json'
cfg.data.train.img_prefix = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/csaw_coco/train/'
cfg.data.test.classes = ('lesion')


cfg.data.val.ann_file = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/csaw_coco/val/annotation_coco.json'
cfg.data.val.img_prefix = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/csaw_coco/val/'
cfg.data.test.classes = ('lesion')

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = 'logs/csaw_mrcnn'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = [1]
cfg.device='cuda'
# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')



# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)