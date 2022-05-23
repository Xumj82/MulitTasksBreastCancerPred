from asyncio import trsock
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
class Mlp(BaseModule):
    def __init__(self,num_classes=3,in_channel=768,freeze=False,pretrained=None):
        super().__init__()
        self.fc = nn.Linear(in_channel, num_classes)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if freeze:
            for param in self.fc.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x[0]
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x, -3)
        x = self.fc(x)
        out = dict(
            lesion = x
        )
        return out