from glob import glob
import os
import cv2
import random

import pydicom
import numpy as np
import pandas as pd
import torch
from os import path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from tqdm import tqdm
from lib.preprocess_utils import read_resize_img,segment_breast,crop_img,add_img_margins,get_max_connected_area,get_candidate_patch,draw_rect,overlap_patch_roi
from lib.preprocess_utils import convert_to_8bit, show_img_cv, clahe, crop_borders
from sklearn.model_selection import train_test_split
class CsawSet(Dataset):
    def __init__(self, 
                data_dir,
                train=True,
                target_size = None,
                gs_255 = False,
                CLAHE = False,
                split = False,
                ):
        self.data_dir = data_dir
        mask_dir = path.join(data_dir, "anon_annotations_nonhidden")
        csv_file = path.join(data_dir, "anon_dataset_nonhidden_211125.csv")
        self.target_size = target_size
        self.CLAHE = CLAHE
        self.mask_files = glob(mask_dir+"/*.png")
        self.csaw_meta = pd.read_csv(csv_file)

        patients = self.csaw_meta['anon_patientid'].unique()

        self.items = []

        if split:
            fl_train, fl_val = train_test_split(self.mask_files, test_size=0.1, random_state=32)
            self.data_list = fl_train if train else fl_val
        else:
            self.data_list = self.mask_files

        for mask_file in self.data_list:
            mask_file_name = os.path.basename(mask_file)
            mmimg_file_name = mask_file_name.replace('_mask','').replace('png','dcm')
            mmimg_rows = self.csaw_meta[self.csaw_meta.anon_filename==mmimg_file_name]
            if len(mmimg_rows) >0:
                mmimg_row = mmimg_rows.iloc[0]
                mmimg_path = glob(self.data_dir+'/*/'+mmimg_file_name)
                if len(mmimg_path) >0:
                    mmimg_detail = dict(
                        mmimg_file_name = mmimg_file_name,
                        mmimg_file = mmimg_path[0],
                        mask_file = mask_file,
                        pathology = mmimg_row['x_case'],
                        rad_timing = mmimg_row['rad_timing'] 
                    )
                    self.items.append(mmimg_detail)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        # if self.items[idx]['mmimg_file_name'] == "01106_20990909_L_MLO_2.dcm":
        #     t = self.items[idx]['mmimg_file_name']
        mm_img = pydicom.read_file(self.items[idx]['mmimg_file']).pixel_array
        mm_img = crop_borders(mm_img,border_size=(0.01,0.01,0.01,0.01))
        mask_img = cv2.imread(self.items[idx]['mask_file'], cv2.IMREAD_ANYDEPTH)
        mask_img = crop_borders(mask_img,border_size=(0.01,0.01,0.01,0.01))

        if mask_img.shape[0] > mm_img.shape[0] or mask_img.shape[1] > mm_img.shape[1]:
            return self.items[idx]['mmimg_file_name'],None,None,None,None

        mm_img_cropped,bbox = segment_breast(mm_img, erosion= True)
        mask_img_cropped = crop_img(mask_img,bbox)

        if mm_img_cropped.shape != mask_img_cropped.shape:
            if mm_img_cropped.shape == mask_img_cropped.shape:
                mask_img_cropped  = mask_img
            else:
                return self.items[idx]['mmimg_file_name'],None,None,None,None

        if self.CLAHE:
            mm_img_cropped = clahe(mm_img_cropped,4095)

        if self.target_size is not None:
            mm_img_cropped = cv2.resize(mm_img_cropped.astype(np.float32),self.target_size)
            mask_img_cropped = cv2.resize(mask_img_cropped,self.target_size)

        mask_img_cropped = np.where(mask_img_cropped > 0, 1,mask_img_cropped)
        
        pathology = self.items[idx]['pathology']
        rad_timing = self.items[idx]['rad_timing']


        return self.items[idx]['mmimg_file_name'],mm_img_cropped,mask_img_cropped,pathology,rad_timing