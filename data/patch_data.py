import json
import os

import cv2
import os.path as op
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torchvision import transforms

import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split

class PatchData(data.Dataset):
    def __init__(self, 
                data_dir,
                stage = 'fit',
                class_num = 5,
                train=True,
                repeat_channel = False,
                elastic_param = 300,
                aug_prob=0.5,
                img_mean=(0.485),
                img_std=(0.229)):
        # self.__dict__.update(locals())
        self.train = train
        self.img_mean = img_mean
        self.img_std = img_std
        self.data_dir = data_dir
        self.stage = stage
        self.aug_prob = aug_prob
        self.elastic_param = elastic_param
        self.class_num = class_num
        self.repeat_channel = repeat_channel
        if self.stage=='fit':
            self.csv_file = data_dir+'/train_meta.csv'
        else:
            self.csv_file = data_dir+'/test_meta.csv'

        self.check_files()

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

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        file_list_path = op.join(self.data_dir, self.csv_file)
        df = pd.read_csv(file_list_path)
        if self.stage=='fit':
            fl_train, fl_val = self.train_test_split_on_patient(df, test_size=0.1, random_state=32)
            self.data_list = fl_train if self.train else fl_val
        else:
            self.data_list = df

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num , dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        
        # path = os.path.join(self.data_dir.format('img_dir'),'train',self.data_list.iloc[index]['img_id']+'.png')
        img_id = self.data_list.iloc[idx]['patch_id']
        

        if self.stage=='fit':
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/train',self.data_list.iloc[idx]['patch_id']+'.png'),cv2.IMREAD_ANYDEPTH)         
        else:
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/test',self.data_list.iloc[idx]['patch_id']+'.png'),cv2.IMREAD_ANYDEPTH)

        # if self.train:
        #     img = elasticdeform.deform_random_grid(img,sigma=15, points=3,axis=(0,1))


        if self.class_num == 3:
            if str(self.data_list.iloc[idx]['pathology']) == 'NORMAL':
                label = 0
            elif str(self.data_list.iloc[idx]['pathology']) == 'BENIGN' or str(self.data_list.iloc[idx]['pathology']) == 'BENIGN_WITHOUT_CALLBACK':
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 2
            else:
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 2

        if self.class_num ==5:
            if str(self.data_list.iloc[idx]['pathology']) == 'NORMAL':
                label = 0
            elif str(self.data_list.iloc[idx]['pathology']) == 'BENIGN' or str(self.data_list.iloc[idx]['pathology']) == 'BENIGN_WITHOUT_CALLBACK':
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 3
            else:
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 4
                
        aug = A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Normalize(mean=self.img_mean, std=self.img_std),
            A.VerticalFlip(p=self.aug_prob),              
            A.RandomRotate90(p=self.aug_prob),
            A.ElasticTransform(alpha=self.elastic_param, sigma=self.elastic_param * 0.05, alpha_affine=self.elastic_param * 0.03, p=self.aug_prob),    
            ]
        ) if self.train else A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Normalize(mean=self.img_mean, std=self.img_std),
        ])

        img_auged = aug(image=img)["image"]
        img_tensor = torch.tensor(img_auged)
        if self.repeat_channel:
            img_tensor = img_tensor.repeat(3,1,1)
        lesion_tensor = torch.tensor(label)

        output = dict(
            input = img_tensor,
            lesion = lesion_tensor,
        )
        return output

    def __len__(self):
        # return 10
        return len(self.data_list)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass