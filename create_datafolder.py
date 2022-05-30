import pandas as pd
from os import path
import os
import cv2
from tqdm import tqdm
import numpy as np
from lib.preprocess_utils import segment_breast, read_resize_img

train_df = pd.read_csv("/mnt/f/datasets/csv/train_roi.csv")
test_df = pd.read_csv("/mnt/f/datasets/csv/test_roi.csv")

output_dir = '/mnt/c/Users/11351/Desktop/dou_view/'
pic_size = (2000, 3000)
df = pd.concat([train_df, test_df])

patients = df.patient_id.unique()

print(len(patients))

with tqdm(total=len(patients)) as pbar:
    for p in patients:
        
        try:
            img_l_cc_p = df[(df['patient_id']==p) & (df['left or right breast']=='LEFT') & (df['image view']=='CC')].iloc[0]['image file path']
            img_l_mlo_p = df[(df['patient_id']==p) & (df['left or right breast']=='LEFT') & (df['image view']=='MLO')].iloc[0]['image file path']

            img_l_cc,_ = segment_breast(cv2.imread(path.join('/mnt/f/datasets/cbis-ddsm-png',img_l_cc_p), cv2.IMREAD_ANYDEPTH))
            img_l_mlo,_ = segment_breast(cv2.imread(path.join('/mnt/f/datasets/cbis-ddsm-png',img_l_mlo_p), cv2.IMREAD_ANYDEPTH))
            img_l_cc = cv2.resize(img_l_cc,dsize= pic_size, interpolation=cv2.INTER_CUBIC)
            img_l_mlo = cv2.resize(img_l_mlo,dsize= pic_size, interpolation=cv2.INTER_CUBIC)

            p_l_folder = path.join(output_dir, p+'_left')
            os.mkdir(p_l_folder)
            cv2.imwrite(path.join(p_l_folder,'left_cc.png'),img_l_cc)
            cv2.imwrite(path.join(p_l_folder,'left_mlo.png'),img_l_mlo)
        except Exception as e:
            print(p,e)

        try:
            img_r_cc_p = df[(df['patient_id']==p) & (df['left or right breast']=='RIGHT') & (df['image view']=='CC')].iloc[0]['image file path']
            img_r_mlo_p = df[(df['patient_id']==p) & (df['left or right breast']=='RIGHT') & (df['image view']=='MLO')].iloc[0]['image file path']

            img_r_cc,_ = segment_breast(cv2.imread(path.join('/mnt/f/datasets/cbis-ddsm-png',img_r_cc_p), cv2.IMREAD_ANYDEPTH))
            img_r_mlo,_ = segment_breast(cv2.imread(path.join('/mnt/f/datasets/cbis-ddsm-png',img_r_mlo_p), cv2.IMREAD_ANYDEPTH))
            img_r_cc = cv2.resize(img_r_cc,dsize= pic_size, interpolation=cv2.INTER_CUBIC)
            img_r_mlo = cv2.resize(img_r_mlo,dsize= pic_size, interpolation=cv2.INTER_CUBIC)

            p_r_folder = path.join(output_dir, p+'_right')
            os.mkdir(p_r_folder)
            cv2.imwrite(path.join(p_r_folder,'right_cc.png'),img_r_cc)
            cv2.imwrite(path.join(p_r_folder,'right_mlo.png'),img_r_mlo)
        except Exception as e:
            print(p,e)

        pbar.update(1)

