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

class CsawData(data.Dataset):
    def __init__(self, 
                data_dir='/mnt/hdd/datasets/ddsm_dataset/{}/train',
                stage = 'fit',
                pic_size =(896,1152),
                classes = 2,
                train=True,
                aug_prob=0.5,
                elastic_param=300,
                img_mean=(0.485),
                img_std=(0.229),
                aug = True,
                repeat_channel = True
                ):
        # self.__dict__.update(locals())
        self.train = train
        self.img_mean = img_mean
        self.img_std = img_std
        self.data_dir = data_dir
        self.stage = stage
        self.elastic_param = elastic_param
        self.aug = aug
        self.repeat_channel = repeat_channel
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
        
        img = cv2.resize(img, self.pic_size)
        ann = cv2.resize(ann, self.pic_size)

        ann = np.where(ann>0, 1, 0)
        pathology_label = 1 if  self.data_list.iloc[index]['pathology'] else 0
        rad_time = self.data_list.iloc[index]['rad_time']
        
        aug = A.Compose([
            A.ToFloat(max_value=4095.0),
            A.Normalize(mean=self.img_mean, std=self.img_std),
            A.VerticalFlip(p=self.aug_prob),              
            A.SafeRotate(p=self.aug),
            A.ElasticTransform(alpha=self.elastic_param, sigma=self.elastic_param * 0.05, alpha_affine=self.elastic_param * 0.03, p=self.aug_prob),    
            ]
        ) if self.train else A.Compose([
            A.ToFloat(max_value=4095.0),
            A.Normalize(mean=self.img_mean, std=self.img_std),
        ])
        augmented = aug(image=img, mask=ann)
        image_auged = augmented['image']
        mask_auged = augmented['mask']

        mask_tensor = torch.tensor(mask_auged).long()
        img_tensor = torch.unsqueeze(torch.tensor(image_auged),0)
        if self.repeat_channel:
            img_tensor = img_tensor.repeat(3,1,1).float()

        rad_time_tensor = torch.tensor(rad_time).float()
        pathology_tensor = torch.tensor(pathology_label).float()

        output = dict(
            input=img_tensor,
            lesion=mask_tensor,
            pathology=pathology_tensor,
            rad_time=rad_time_tensor
        )

        return output

    def __len__(self):
        return len(self.data_list)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass