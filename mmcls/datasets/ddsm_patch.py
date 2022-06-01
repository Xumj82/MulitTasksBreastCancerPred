import mmcv
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
from .builder import DATASETS
from .base_dataset import BaseDataset
from sklearn.model_selection import train_test_split

@DATASETS.register_module()
class DdsmPatch(BaseDataset):
    # IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def __init__(self,
                 data_prefix: str,
                 img_shape: tuple,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 split: bool = False,
                 val_size: float = 0.1,
                 test_mode: bool = False,
                 random_state: int = 32,):
        self.split = split
        self.img_shape = img_shape
        self.val_size = val_size
        self.random_state = random_state
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode)

    @staticmethod
    def train_test_split_on_patient(data_list ,test_size, random_state):
        patients = []
        for index, row in data_list.iterrows():
            patientid = row['img_id'].split('_')[1] + row['img_id'].split('_')[2]
            patients.append(patientid)
        data_list['PatientID'] = patients

        patientdict = list( dict.fromkeys(patients) )
        traindict, testdict = train_test_split(patientdict, test_size=test_size, random_state=random_state)
        traindict = pd.DataFrame(data=traindict, columns=["PatientID"])
        testdict = pd.DataFrame(data=testdict, columns=["PatientID"])
        data_train = pd.merge(traindict, data_list,  on ='PatientID', how ='left')
        data_test = pd.merge(testdict, data_list,  on ='PatientID', how ='left')
        return data_train,data_test

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        # with open(self.ann_file) as f:
        #     samples = [x.strip().split(',') for x in f.readlines()]
        #     samples.pop(0)
        samples = pd.read_csv(self.ann_file)
        if self.split:
            train_sp, val_sp = self.train_test_split_on_patient(samples, test_size=self.val_size, random_state=self.random_state)
            samples = val_sp if self.test_mode else train_sp
        info = {'img_prefix': self.data_prefix}
        info['img_shape'] = self.img_shape
        if len(self.CLASSES) == 3:
            for patientid,img_id,patch_id,type,pathology,full_img,ROI_img in samples.values.tolist():
                info['img_info'] = {'filename': patch_id}
                info['gt_label'] = np.array(0, dtype=np.int64)                
                if type == 'bkg':
                    info['gt_label'] = np.array(0, dtype=np.int64)
                if type == 'calcification':
                    info['gt_label'] = np.array(1, dtype=np.int64)
                if type == 'mass':
                    info['gt_label'] = np.array(2, dtype=np.int64)
                data_infos.append(info)
        if len(self.CLASSES) == 5:
            for patientid,img_id,patch_id,type,pathology,full_img,ROI_img in samples.values.tolist():
                info['img_info'] = {'filename': patch_id}
                if type == 'bkg':
                    info['gt_label'] = np.array(0, dtype=np.int64)
                if type == 'calcification' and pathology!='MALIGNANT':
                    info['gt_label'] = np.array(1, dtype=np.int64)
                if type == 'mass' and pathology!='MALIGNANT':
                    info['gt_label'] = np.array(2, dtype=np.int64)                        
                if type == 'calcification'and pathology=='MALIGNANT':
                    info['gt_label'] = np.array(3, dtype=np.int64)
                if type == 'mass'and pathology=='MALIGNANT':
                    info['gt_label'] = np.array(4, dtype=np.int64)
                data_infos.append(info)
        return data_infos
