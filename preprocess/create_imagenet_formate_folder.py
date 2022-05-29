import imp
import os
import csv 
import numpy as np
import pandas as pd
import cv2
import mmcv
import inspect
import importlib
from argparse import ArgumentParser

from tqdm import tqdm
from data.csaw_set import CsawSet
from data.ddsm_set import DdsmSet
from lib.preprocess_utils import get_max_connected_area, read_resize_img,segment_breast

def get_img(img_file):
    img = read_resize_img(img_file, target_height=4096,gs_255=False)
    full_img_segment,bbox = segment_breast(img)
    full_img_segment = cv2.resize(full_img_segment, (2000,3000))
    return full_img_segment

def main(csv_file_path, out_dir):
    df = pd.read_csv(csv_file_path)
    patients = df['patient_id'].unique()
    with tqdm(total=len(patients)) as pbar:
        for p in patients:
            breast_l = df[(df['patient_id']==p)&(df['left or right breast']=='LEFT')]
            breast_r = df[(df['patient_id']==p)&(df['left or right breast']=='RIGHT')]
            try:
                if len(breast_l)==2:
                    type ='MALIGNANT' if breast_l.iloc[0]['abnormality type'] =='MALIGNANT' or breast_l.iloc[1]['abnormality type'] == 'MALIGNANT' else 'BENIGN'
                    img_0 =get_img(os.path.join(args.data_root,'cbis-ddsm-png',breast_l[breast_l['image view']=='CC'].iloc[0]['image file path']))
                    img_1 =get_img(os.path.join(args.data_root,'cbis-ddsm-png',breast_l[breast_l['image view']=='MLO'].iloc[0]['image file path']))
                    combined = np.concatenate([np.expand_dims(img_0, axis=0),np.expand_dims(img_1, axis=0)])
                    # cv2.imwrite(os.path.join(out_dir,type,'{}_left.png'.format(p)),)
                    mmcv.dump(combined,os.path.join(out_dir,type,'{}_left.pkl'.format(p)))

                if len(breast_r)==2:
                    type ='MALIGNANT' if breast_r.iloc[0]['abnormality type'] =='MALIGNANT' or breast_r.iloc[1]['abnormality type'] == 'MALIGNANT' else 'BENIGN'
                    img_0 =get_img(os.path.join(args.data_root,'cbis-ddsm-png',breast_r[breast_r['image view']=='CC'].iloc[0]['image file path']))
                    img_1 =get_img(os.path.join(args.data_root,'cbis-ddsm-png',breast_r[breast_r['image view']=='MLO'].iloc[0]['image file path']))
                    combined = np.concatenate([np.expand_dims(img_0, axis=0),np.expand_dims(img_1, axis=0)])
                    mmcv.dump(combined,os.path.join(out_dir,type,'{}_right.pkl'.format(p)))
            except Exception as e:
                print(e)

            pbar.update(1)

if __name__ == '__main__':
   
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--data_name',default='ddsm',type=str)
    parser.add_argument('--output_dir',default='/home/xumingjie/dataset/ddsm_breast',type=str)
    parser.add_argument('--data_root',default='/mnt/hdd/datasets/',type=str)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.environ['data_root']
    print('DATA_ROOT:',args.data_root)
    
    # args.data_root = os.path.expanduser(args.data_root)
    main('/mnt/hdd/datasets/csv/train_roi.csv','/home/xumingjie/dataset/ddsm_breast/train')
    main('/mnt/hdd/datasets/csv/test_roi.csv','/home/xumingjie/dataset/ddsm_breast/test')
