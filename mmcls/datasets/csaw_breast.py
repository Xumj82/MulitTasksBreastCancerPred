import mmcv
import numpy as np
import pandas as pd
import random
from typing import Optional, Sequence, Union

from torch import rand
from .builder import DATASETS
from .base_dataset import BaseDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval

@DATASETS.register_module()
class CsawBreast(BaseDataset):
    # IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 seed:int=32,
                 sample_rate: float=0.01,
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        self.data_infos = []
        self.sample_rate = sample_rate
        random.seed(seed)
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
        patients = df['anon_patientid'].unique()[0:100]
        patients = random.sample(patients, int(len(patients)*self.sample_rate))
        for p in patients:
            p_rows = df[df['anon_patientid']==p]

            years = p_rows['exam_year'].unique()
            for y in years:
                y_rows = p_rows[p_rows['exam_year']==y]

                sides = ['Right','Left']
                for s in sides:

                    cc_row = y_rows[(y_rows['imagelaterality']==s)&(y_rows['viewposition']=='CC')]
                    mlo_row = y_rows[(y_rows['imagelaterality']==s)&(y_rows['viewposition']=='MLO')]
                    gt_label = y_rows['rad_timing'].max() if y_rows['x_cancer_laterality'].iloc[0]==s else 4

                    files = [cc_row.iloc[0]['anon_filename'], mlo_row.iloc[0]['anon_filename']]

                    info = {'img_prefix': self.data_prefix}
                    info['img_info'] = {'filenames':files}
                    info['gt_label'] = np.array(gt_label, dtype=np.int64)
                    data_infos.append(info)
        return data_infos