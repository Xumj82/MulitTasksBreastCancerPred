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

class SegemData(data.Dataset):
    def __init__(self, 
                data_dir='/mnt/hdd/datasets/ddsm_dataset/{}/train',
                stage = 'fit',
                pic_size =(1152,896),
                classes = 3,
                train=True,
                aug_prob=0.5,
                img_mean=(0.485),
                img_std=(0.229),
                aug = True
                ):
        # self.__dict__.update(locals())
        self.train = train
        self.img_mean = img_mean
        self.img_std = img_std
        self.data_dir = data_dir
        self.stage = stage
        self.aug = aug
        if self.stage=='fit':
            self.csv_file = data_dir+'/train_meta.csv'
        else:
            self.csv_file = data_dir+'/test_meta.csv'
        self.pic_size = pic_size
        self.aug_prob = aug_prob
        self.classes = classes
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        file_list_path = op.join(self.data_dir, self.csv_file)
        df = pd.read_csv(file_list_path)
        if self.stage=='fit':
            fl_train, fl_val = train_test_split(df, test_size=0.1, random_state=32)
            self.data_list = fl_train if self.train else fl_val
        else:
            self.data_list = df

    def __getitem__(self, index):
        
        # path = os.path.join(self.data_dir.format('img_dir'),'train',self.data_list.iloc[index]['img_id']+'.png')
        # path = os.path.join(self.data_dir.format('img_dir'),'train',self.data_list.iloc[index]['img_id']+'.png')
        if self.stage=='fit':
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/train',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_ANYDEPTH)         
            ann = cv2.imread(os.path.join(self.data_dir,'ann_dir/train',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/test',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_ANYDEPTH)         
            ann = cv2.imread(os.path.join(self.data_dir,'ann_dir/test',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_GRAYSCALE)
        # ann = np.where(ann > 4,0,ann)

        if self.train:
            [img, ann] = elasticdeform.deform_random_grid([img,ann],sigma=15, points=3,axis=(0,1),)
        ann = np.where(ann>0, 1, 0)
        pathology_label = 1 if  self.data_list.iloc[index]['pathology'] else 0

        # if self.classes == 3:
        #     ann = np.where(ann == 1*50, 1, ann)
        #     ann = np.where(ann == 2*50, 2, ann)
        #     ann = np.where(ann == 3*50, 1, ann)
        #     ann = np.where(ann == 4*50, 2, ann)
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

        pathology_tensor = torch.tensor(pathology_label).long()
        output = dict(
            input=img_tensor,
            seg_lesion=ann_tensor,
            pathology=pathology_tensor,
        )
        return output

    def __len__(self):
        return len(self.data_list)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass