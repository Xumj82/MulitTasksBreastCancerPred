import os
import pandas as pd
from os import path
import pydicom
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import PIL 
from tqdm import tqdm
from IPython.display import clear_output
from pathlib import Path
from multiprocessing import Pool

## put you dir here 
os.chdir('/mnt/f/datasets/')


def update_file_path(df):
  for index, row in df.iterrows():
    old_full_path_id = row['image file path'].split('/')[0]
    old_crop_path_id = row['cropped image file path'].split('/')[0]

    rep = meta_data.loc[(meta_data['Subject ID'] == old_full_path_id) & (meta_data['Series Description'] == 'full mammogram images')]
    new_full_path = rep['File Location'].iloc[0] if rep.shape[0] else np.nan

    rep = meta_data.loc[(meta_data['Subject ID'] == old_crop_path_id) & (meta_data['Series Description'] == 'cropped images')]
    new_crop_path = rep['File Location'].iloc[0] if rep.shape[0] else np.nan

    rep = meta_data.loc[(meta_data['Subject ID'] == old_crop_path_id) & (meta_data['Series Description'] == 'ROI mask images')]
    new_roi_path = rep['File Location'].iloc[0] if rep.shape[0] else np.nan

    df.loc[index,['image file path']] = new_full_path
    df.loc[index,['cropped image file path']] = new_crop_path
    df.loc[index,['ROI mask file path']] = new_roi_path
  # df.dropna(subset=['image file path','cropped image file path','ROI mask file path'])
  return df.dropna(subset=['image file path'])


if __name__ == '__main__':

    calc_train = pd.read_csv('csv/calc_case_description_train_set.csv')
    mass_train = pd.read_csv('csv/mass_case_description_train_set.csv')

    calc_test = pd.read_csv('csv/calc_case_description_test_set.csv')
    mass_test = pd.read_csv('csv/mass_case_description_test_set.csv')

    meta_data = pd.read_csv('cbis-ddsm-png/reconstruct_meta.csv')

    meta_data['Subject ID'] =  meta_data['Subject ID'].str.strip()
    meta_data['Series Description'] =  meta_data['Series Description'].str.strip()

    calc_train_ = update_file_path(calc_train)
    calc_test_ = update_file_path(calc_test)
    mass_train_ = update_file_path(mass_train)
    mass_test_ = update_file_path(mass_test)


    calc_train_roi = calc_train_.dropna(subset=['ROI mask file path']).drop(columns=['breast density','calc type','calc distribution'])
    calc_test_roi = calc_test_.dropna(subset=['ROI mask file path']).drop(columns=['breast density','calc type','calc distribution'])
    mass_train_roi = mass_train_.dropna(subset=['ROI mask file path']).drop(columns=['breast_density','mass shape','mass margins'])
    mass_test_roi = mass_test_.dropna(subset=['ROI mask file path']).drop(columns=['breast_density','mass shape','mass margins'])

    train_roi = pd.concat([calc_train_roi,mass_train_roi],ignore_index=True).sample(frac=1)
    test_roi = pd.concat([calc_test_roi,mass_test_roi],ignore_index=True).sample(frac=1)

    train_roi.to_csv('csv/train_roi.csv',index=False)
    test_roi.to_csv('csv/test_roi.csv',index=False)