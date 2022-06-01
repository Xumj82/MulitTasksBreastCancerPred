import mmcv
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .builder import DATASETS
from .base_dataset import BaseDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval
from sklearn.model_selection import train_test_split

@DATASETS.register_module()
class CsawBreast(BaseDataset):
    # IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def __init__(self,
                 data_prefix: str,
                 mode:  str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode)
        
        p_infos = []
        y_infos = []
        s_infos = []

        if mode == 'patient_level':
            self.data_list = p_info
        if mode == 'exam_level':
            self.data_list = y_info
        if mode == 'breast_level':
            self.data_list = s_info
        ann_df = pd.read_csv(ann_file)

        patients = ann_df['anon_patientid'].unique()
        for p in patients:
            p_row = ann_df[ann_df['anon_patientid']==p]
            p_info = dict(
                id = p,
                label =  p_row['x_case'].max(),
                img_files = p_row['anon_filename'].tolist()
            )
            p_infos.append(p_info)

            years = p_row['exam_year'].unique()
            for y in years:
                y_row = p_row[p_row['exam_year']==y]
                y_info = dict(
                    id = p+'_'+y,
                    label = y_row['rad_time'].max()-1,
                    img_files = y_row['anon_filename'].tolist()
                )
                y_infos.append(y_info)

                sides = ['Right','Left',]
                for s in sides:
                    s_row = y_row[y_row['imagelaterality']==s]
                    s_row.sort_values(by=['viewposition'])
                    s_info = dict(
                        id = p+'_'+y+'_'+s,
                        label = s_row['rad_time'].max()-1 if s_row['x_cancer_laterality'].iloc[0]==s else 3,
                        img_files = s_row['anon_filename'].tolist()
                    )
                    s_infos.append(s_info)
        
        
                



    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        for data in self.data_list:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filenames':data['img_files']}
            info['gt_label'] = np.array(data['label'], dtype=np.int64)
            data_infos.append(info)
        return data_infos