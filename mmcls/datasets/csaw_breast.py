import mmcv
import numpy as np
import pandas as pd
import random
from typing import Optional, Sequence, Union

from torch import rand
from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register_module()
class CsawBreast(BaseDataset):
    # IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def __init__(self,
                 img_shape:tuple(),
                 data_prefix: str,
                 pipeline: Sequence = (),
                 seed:int=32,
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        self.img_shape = img_shape
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode)

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        df = pd.read_csv(self.ann_file)
        for idx,row in df.iterrows():
            cc_view = row['id']
            mlo_view = row['id']
            rad_time = row['rad_time']
            info = {'img_prefix': self.data_prefix}
            info['img_id'] = row['id']
            info['img_info'] = {'cc_view':cc_view,'mlo_view':mlo_view}
            if self.CLASSES == 4:
                info['gt_label'] = np.array(rad_time-1, dtype=np.int64)
            else:
                info['gt_label'] = 1 if rad_time <=3 else 0
            info['img_shape'] = self.img_shape
            data_infos.append(info)
        return data_infos