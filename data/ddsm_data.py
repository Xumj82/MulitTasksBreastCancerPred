import json
import os
from turtle import st


import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os.path as op
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split

class DdsmData(data.Dataset):
    def __init__(self, 
                data_dir,
                stage = 'fit',
                crop_size = 224,
                crop_weight = [0.9, 0.1, 0],
                num_classes = 3,
                positive_thr = 0.75,
                elastic_param = 12,
                train=True,
                aug_prob=0.5,
                img_mean=(0.485),
                img_std=(0.229),
                repeat_channel = True
                ):
        # self.__dict__.update(locals())
        self.train = train
        self.img_mean = img_mean
        self.img_std = img_std
        self.data_dir = data_dir
        self.stage = stage
        if self.stage=='fit':
            self.csv_file = data_dir+'/train_meta.csv'
        else:
            self.csv_file = data_dir+'/test_meta.csv'
        self.crop_size = crop_size
        self.crop_weight = crop_weight
        self.positive_thr = positive_thr
        self.elastic_param = elastic_param
        self.aug_prob = aug_prob
        self.num_classes = num_classes
        self.repeat_channel = repeat_channel
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

    def random_choose_crop_point(self, img, ann):
        masked_indexes = np.argwhere(ann > 0)
        breast_indexes = np.argwhere(img > 0)
        black_indexes = np.argwhere(img > -1)

        index = np.random.choice(3,1, p=self.crop_weight)
        if index == 0 :
            point = masked_indexes[np.random.randint(len(masked_indexes))]
        if index == 1 :
            point = breast_indexes[np.random.randint(len(breast_indexes))]
        if index == 2 :
            point = black_indexes[np.random.randint(len(black_indexes))]

        point = [max(self.crop_size,min(img.shape[0]-self.crop_size, point[0])) ,max(self.crop_size,min(img.shape[1]-self.crop_size, point[1]))  ]
        
        return point


    def __getitem__(self, index):
        
        # path = os.path.join(self.data_dir.format('img_dir'),'train',self.data_list.iloc[index]['img_id']+'.png')
        # path = os.path.join(self.data_dir.format('img_dir'),'train',self.data_list.iloc[index]['img_id']+'.png')
        if self.stage=='fit':
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/train',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_ANYDEPTH)         
            ann = cv2.imread(os.path.join(self.data_dir,'ann_dir/train',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/test',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_ANYDEPTH)         
            ann = cv2.imread(os.path.join(self.data_dir,'ann_dir/test',self.data_list.iloc[index]['img_id']+'.png'),cv2.IMREAD_GRAYSCALE)
        crop_point = self.random_choose_crop_point(img, ann)

        class_labels = []
        masks = []
        for label in np.unique(ann):
            if label:
                zeros = np.zeros(ann.shape)
                mask = np.where(ann==label,1,zeros)
                masks.append(mask)
                class_labels.append(label)
                
        aug = A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Normalize(mean=self.img_mean, std=self.img_std),
            A.Crop(x_min=crop_point[1]-self.crop_size//2, 
            y_min=crop_point[0]-self.crop_size//2, 
            x_max=crop_point[1]+self.crop_size//2, 
            y_max=crop_point[0]+self.crop_size//2, 
            p=1),
            A.VerticalFlip(p=self.aug_prob),              
            A.RandomRotate90(p=self.aug_prob),
            A.ElasticTransform(alpha=self.elastic_param, sigma=self.elastic_param * 0.05, alpha_affine=self.elastic_param * 0.03, p=self.aug_prob),
            
            ]
        ) if self.train else A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Normalize(mean=self.img_mean, std=self.img_std),
            A.Crop(x_min=crop_point[1]-self.crop_size//2, 
            y_min=crop_point[0]-self.crop_size//2, 
            x_max=crop_point[1]+self.crop_size//2, 
            y_max=crop_point[0]+self.crop_size//2, 
            p=1)
        ])


        augmented = aug(image=img, masks=masks)
        image_auged = augmented['image']
        masks_auged = augmented['masks']
        mask_auged = np.zeros([self.crop_size, self.crop_size])

        pathology = 0
        for label, mask in zip(class_labels, masks_auged):
            if np.sum(mask) > self.positive_thr * self.crop_size**2:
                pathology = label
                break
        
        for label, mask in zip(class_labels, masks_auged):
            mask_auged = np.where(mask>0,label,0)

        if self.repeat_channel:
            image_tensor =torch.unsqueeze(torch.tensor(image_auged),0).repeat(3,1,1)
        else:
            image_tensor =torch.unsqueeze(torch.tensor(image_auged),0)

        if self.num_classes == 3:
            if pathology == 3:
                pathology = 1
            if pathology == 4:
                pathology = 2

        output = dict(
            input=image_tensor,
            seg_lesion=torch.unsqueeze(torch.tensor(mask_auged),0),
            lesion=torch.tensor(pathology).long(),
            # crop_point = torch.tensor(crop_point)
        )
        return output

    def __len__(self):
        return len(self.data_list)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass
