import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchmetrics
import warnings

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
# upernet
class UperNet(nn.Module):
    def __init__(self, nr_classes={
                            'pathology':1,
                            'lesion': 2
                        }, 
                fc_dim=2048,
                use_softmax=False, pool_scales=(1, 2, 3, 6),
                fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UperNet, self).__init__()
        self.use_softmax = use_softmax
        

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            # self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.))
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale)) ## PrRoIPool2D is not available in pytorch 1.10
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                # SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                # SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        # background included. if ignore in loss, output channel 0 will not be trained.
        self.nr_pathology_class, self.nr_lesion_class, = \
            nr_classes['pathology'], nr_classes['lesion']

        # input: PPM out, input_dim: fpn_dim
        self.pathology_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fpn_dim, self.nr_pathology_class, kernel_size=1, bias=True)
        )
        # input: Fusion out, input_dim: fpn_dim
        self.lesion_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_lesion_class, kernel_size=1, bias=True)
        )

    def forward(self, conv_out,orig_feed,output_switch = dict(pathology = True,lesion = True)):
        seg_size= orig_feed['seg_lesion'].shape[-2:]
        output_dict = dict(pathology = None,lesion = None )

        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = [] # fake rois, just used for pooling
        for i in range(input_size[0]): # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1)) # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            pool_scale_res = pool_scale(conv5)
            up_sample_res = F.interpolate(pool_scale_res,(input_size[2], input_size[3]),mode='bilinear', align_corners=False)
            pool_conv_res = pool_conv(up_sample_res)
            ppm_out.append(pool_conv_res)
            # ppm_out.append(pool_conv(F.interpolate(
            #     # pool_scale(conv5, roi.detach()),
            #     pool_scale(conv5),
            #     (input_size[2], input_size[3]),
            #     mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        if output_switch['pathology']: # pathology
            output_dict['pathology'] = self.pathology_head(f)

        if output_switch['lesion']:
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x) # lateral branch

                f = F.interpolate(
                    f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse() # [P2 - P5]

            output_size = fpn_feature_list[0].size()[2:]
            fusion_list = [fpn_feature_list[0]]

            for i in range(1, len(fpn_feature_list)):
                fusion_list.append(F.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear', align_corners=False))

            fusion_out = torch.cat(fusion_list, 1)

            # for i in range(1, len(fpn_feature_list)):
            #     fusion_list.append(F.interpolate(
            #         fpn_feature_list[i],
            #         output_size,
            #         mode='bilinear', align_corners=False))

            # fusion_out = torch.cat(fusion_list, 1)

            x = self.conv_fusion(fusion_out)
            output_dict['lesion'] = self.lesion_head(x)

        if self.use_softmax:  # is True during inference
            # inference pathology
            x = output_dict['pathology']
            x = x.squeeze(3).squeeze(2)
            x = F.softmax(x, dim=1)
            output_dict['pathology'] = x
            # inference lesion
            x = output_dict['lesion']
            x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
            x = F.softmax(x, dim=1)
            output_dict['lesion'] = x

        else:   # Training
            for k in ['pathology', 'lesion']:
                if output_dict[k] is None:
                    continue
                x = output_dict[k]
                # x = F.log_softmax(x, dim=1)
                if k == "pathology":  # for scene
                    x = x.squeeze(3).squeeze(2)
                output_dict[k] = x

        return output_dict
