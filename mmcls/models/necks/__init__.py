# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .res_necks import ResNecks
from .vi_vit import ViVit

__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales','ResNecks','ViVit']
