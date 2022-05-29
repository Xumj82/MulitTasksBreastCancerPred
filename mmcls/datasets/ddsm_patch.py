import mmcv
import numpy as np
from typing import Optional, Sequence, Union
from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class DdsmPatch(BaseDataset):
    # IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def __init__(self,
                 data_prefix: str,
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


    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
            samples.pop(0)
            if len(self.CLASSES) == 3:
                for img_id,patch_id,type,pathology,full_img,ROI_img in samples:
                    info = {'img_prefix': self.data_prefix}
                    info['img_info'] = {'filename': patch_id+'.png'}
                    info['gt_label'] = np.array(0, dtype=np.int64)                
                    if type == 'bkg':
                        info['gt_label'] = np.array(0, dtype=np.int64)
                    if type == 'calc':
                        info['gt_label'] = np.array(1, dtype=np.int64)
                    if type == 'mass':
                        info['gt_label'] = np.array(2, dtype=np.int64)
                    data_infos.append(info)
            if len(self.CLASSES) == 5:
                for img_id,patch_id,type,pathology,full_img,ROI_img in samples:
                    info = {'img_prefix': self.data_prefix}
                    info['img_info'] = {'filename': patch_id+'.png'}
                    
                    if type == 'bkg':
                        info['gt_label'] = np.array(0, dtype=np.int64)
                    if type == 'calc' and pathology!='MALIGNANT':
                        info['gt_label'] = np.array(1, dtype=np.int64)
                    if type == 'mass' and pathology!='MALIGNANT':
                        info['gt_label'] = np.array(2, dtype=np.int64)                        
                    if type == 'calc'and pathology=='MALIGNANT':
                        info['gt_label'] = np.array(3, dtype=np.int64)
                    if type == 'mass'and pathology=='MALIGNANT':
                        info['gt_label'] = np.array(4, dtype=np.int64)
                    data_infos.append(info)
            return data_infos
