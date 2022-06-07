# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import imp
import itertools
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction, ProgressBar

from mmcls.core import visualization as vis
from mmcls.datasets.builder import PIPELINES, build_dataset, build_from_cfg
from mmcls.models.utils import to_2tuple

bright_style, reset_style = '\x1b[1m', '\x1b[0m'
red_text, blue_text = '\x1b[31m', '\x1b[34m'
white_background = '\x1b[107m'

def convert_to_8bit(img):
    if np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    return (img * 255).astype(np.float32)

def retrieve_data_cfg(config_path, skip_type, cfg_options, phase):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    data_cfg = cfg.data[phase]
    while 'dataset' in data_cfg:
        data_cfg = data_cfg['dataset']
    data_cfg['pipeline'] = [
        x for x in data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def build_dataset_pipelines(cfg, phase):
    """build dataset and pipeline from config.

    Separate the pipeline except 'LoadImageFromFile' step if
    'LoadImageFromFile' in the pipeline.
    """
    data_cfg = cfg.data[phase]
    loadimage_pipeline = []
    if len(data_cfg.pipeline
           ) != 0 and data_cfg.pipeline[0]['type'] == 'LoadMMImageFromFile':
        loadimage_pipeline.append(data_cfg.pipeline.pop(0))
    origin_pipeline = data_cfg.pipeline
    data_cfg.pipeline = loadimage_pipeline
    dataset = build_dataset(data_cfg)
    pipelines = {
        pipeline_cfg['type']: build_from_cfg(pipeline_cfg, PIPELINES)
        for pipeline_cfg in origin_pipeline
    }

    return dataset, pipelines


def prepare_imgs(args, imgs: List[np.ndarray], steps=None):
    """prepare the showing picture."""
    for i, img in enumerate(imgs):
        img = np.concatenate((img[0],img[1]), axis=-2)
        # img = np.transpose(img,(1,2,0))
        img = convert_to_8bit(img)
        if img.shape[-1] == 1 or len(img.shape)==2:
            img = np.repeat(img, 3, axis=-1)
        imgs[i] = img
    ori_shapes = [img.shape for img in imgs]
    # adaptive adjustment to rescale pictures
    if args.adaptive:
        for i, img in enumerate(imgs):
            imgs[i] = adaptive_size(img, args.min_edge_length,
                                    args.max_edge_length)
    else:
        # if src image is too large or too small,
        # warning a "--adaptive" message.
        for ori_h, ori_w, _ in ori_shapes:
            if (args.min_edge_length > ori_h or args.min_edge_length > ori_w
                    or args.max_edge_length < ori_h
                    or args.max_edge_length < ori_w):
                msg = red_text
                msg += 'The visualization picture is too small or too large to'
                msg += ' put text information on it, please add '
                msg += bright_style + red_text + white_background
                msg += '"--adaptive"'
                msg += reset_style + red_text
                msg += ' to adaptively rescale the showing pictures'
                msg += reset_style
                warnings.warn(msg)

    if len(imgs) == 1:
        return imgs[0]
    else:
        return concat_imgs(imgs, steps, ori_shapes)


def concat_imgs(imgs, steps, ori_shapes):
    """Concat list of pictures into a single big picture, align height here."""
    show_shapes = [img.shape for img in imgs]
    show_heights = [shape[0] for shape in show_shapes]
    show_widths = [shape[1] for shape in show_shapes]

    max_height = max(show_heights)
    text_height = 20
    font_size = 0.5
    pic_horizontal_gap = min(show_widths) // 10
    for i, img in enumerate(imgs):
        cur_height = show_heights[i]
        pad_height = max_height - cur_height
        pad_top, pad_bottom = to_2tuple(pad_height // 2)
        # handle instance that the pad_height is an odd number
        if pad_height % 2 == 1:
            pad_top = pad_top + 1
        pad_bottom += text_height * 3  # keep pxs to put step information text
        pad_left, pad_right = to_2tuple(pic_horizontal_gap)
        # make border
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255))
        # put transform phase information in the bottom
        imgs[i] = cv2.putText(
            img=img,
            text=steps[i],
            org=(pic_horizontal_gap, max_height + text_height // 2),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=font_size,
            color=(255, 0, 0),
            lineType=1)
        # put image size information in the bottom
        imgs[i] = cv2.putText(
            img=img,
            text=str(ori_shapes[i]),
            org=(pic_horizontal_gap, max_height + int(text_height * 1.5)),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=font_size,
            color=(255, 0, 0),
            lineType=1)

    # Height alignment for concatenating
    board = np.concatenate(imgs, axis=1)
    return board


def adaptive_size(image, min_edge_length, max_edge_length, src_shape=None):
    """rescale image if image is too small to put text like cifar."""
    assert min_edge_length >= 0 and max_edge_length >= 0
    assert max_edge_length >= min_edge_length
    src_shape = image.shape if src_shape is None else src_shape
    image_h, image_w, _ = src_shape

    if image_h < min_edge_length or image_w < min_edge_length:
        image = mmcv.imrescale(
            image, min(min_edge_length / image_h, min_edge_length / image_h))
    if image_h > max_edge_length or image_w > max_edge_length:
        image = mmcv.imrescale(
            image, max(max_edge_length / image_h, max_edge_length / image_w))
    return image


def get_display_img(args, item, pipelines):
    """get image to display."""
    # srcs picture could be in RGB or BGR order due to different backends.
    # if args.gray2rgb:
    #     item['img_info']['cc_view'] = mmcv.gray2rgb(item['img_info']['cc_view'])
    #     item['img_info']['mlo_view'] = mmcv.gray2rgb(item['img_info']['mlo_view'])
    src_image_cc = item['img_info']['cc_view']
    src_image_mlo = item['img_info']['mlo_view']
    pipeline_images = []

    # get intermediate images through pipelines
    if args.mode in ['transformed', 'concat', 'pipeline']:
        for pipeline in pipelines.values():
            item = pipeline(item)
            trans_image = copy.deepcopy(item['img'])
            # trans_image = np.ascontiguousarray(trans_image, dtype=np.uint16)
            pipeline_images.append(trans_image)

    # concatenate images to be showed according to mode
    if args.mode == 'original':
        image = prepare_imgs(args, [src_image_cc,src_image_mlo], ['src'])
    elif args.mode == 'transformed':
        image = prepare_imgs(args, [pipeline_images[-1]], ['transformed'])
    elif args.mode == 'concat':
        steps = ['src', 'transformed']
        image = prepare_imgs(args, [pipeline_images[0], pipeline_images[-1]],
                             steps)
    elif args.mode == 'pipeline':
        steps = list(pipelines.keys())
        image = prepare_imgs(args, pipeline_images, steps)

    return image