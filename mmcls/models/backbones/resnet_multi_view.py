# Copyright (c) OpenMMLab. All rights reserved.

from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import ResNet

@BACKBONES.register_module()
class ResNetMultiView(ResNet):
    """ResNetV1c backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv
    in the input stem with three 3x3 convs.
    """

    def __init__(self,num_views=2,**kwargs):
        self.num_views = num_views
        super(ResNetMultiView, self).__init__(
            deep_stem=False, avg_down=False, **kwargs)

    def forward(self, x:Tensor):
        in_shape = x.shape
        x = x.reshape((-1,in_shape[2],in_shape[3],in_shape[4]))
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # x = x.reshape((in_shape[0],in_shape[1]*x.shape[1],x.shape[2],x.shape[3]))
            if i in self.out_indices:
                outs.append(x)
        
        outs = [x.reshape((in_shape[0],in_shape[1]*x.shape[1],x.shape[2],x.shape[3])) for x in outs]
        # for x in outs:
        #     x = x.reshape((in_shape[0],in_shape[1]*x.shape[1],x.shape[2],x.shape[3]))

        return tuple(outs)

# @BACKBONES.register_module()
# class ResNetV1d(ResNet):
#     """ResNetV1d backbone.

#     This variant is described in `Bag of Tricks.
#     <https://arxiv.org/pdf/1812.01187.pdf>`_.

#     Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
#     the input stem with three 3x3 convs. And in the downsampling block, a 2x2
#     avg_pool with stride 2 is added before conv, whose stride is changed to 1.
#     """

#     def __init__(self, **kwargs):
#         super(ResNetV1d, self).__init__(
#             deep_stem=True, avg_down=True, **kwargs)
