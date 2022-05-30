from ctypes import sizeof
from lib2to3.pgen2.pgen import DFAState
from argparse import ArgumentParser
import os
import shutil
import pydicom
import pandas as pd
from glob import glob
from os import link, path
import pydicom
import numpy as np
import random
import cv2
from tqdm import tqdm


def get_dcm_type(dcm_path):
    try:
        # Read dicom.
        ds = pydicom.dcmread(dcm_path)
        img_type = ds.SeriesDescription
        return img_type
    except Exception as ex:
        # print(ex)
        return None

def generate_new_list():

    def contrast_assign(type_A, dcm_path_A, dcm_path_B):
        if type_A == 'ROI mask images':
            link_list.append([subjecr_id,'ROI mask images', dcm_path_A])
            link_list.append([subjecr_id,'cropped images',dcm_path_B])
        if type_A == 'cropped images':
            link_list.append([subjecr_id,'cropped images', dcm_path_A])
            link_list.append([subjecr_id,'ROI mask images',  dcm_path_B])

    meta_path = path.join(top,'metadata.csv')
    metadata = pd.read_csv(meta_path)
    link_list = []
    files_count = len(glob(top+"\\*\\*\\*\\*\\*.dcm"))
    for index, row in metadata.iterrows():
        subjecr_id = row['Subject ID']
        file_location = row['File Location'].replace('.\\',top+'\\')
        num_images = row['Number of Images']
        series_desc = row['Series Description']

        if num_images >2 or num_images < 1:
            print(subjecr_id)

        dcm_files = glob(file_location+"\\*dcm")

        if len(dcm_files) != num_images:
            print(subjecr_id)

        if num_images == 1:
            link_list.append([subjecr_id,series_desc, dcm_files[0]])
        if num_images == 2:
            type_0 = get_dcm_type(dcm_files[0])
            type_1 = get_dcm_type(dcm_files[1])
            if type_0 == None and type_1 == None:
                print(subjecr_id)
            elif type_0 != None and type_1 != None:
                link_list.append([subjecr_id,type_0,dcm_files[0]])
                link_list.append([subjecr_id,type_1,dcm_files[1]]) 
            elif type_0 != None:
                contrast_assign(type_0, dcm_files[0],dcm_files[1])
            else:
                contrast_assign(type_1, dcm_files[1],dcm_files[0])

    df = pd.DataFrame(data=link_list, columns=["Subject ID", "Series Description","File Location"])

    for index, row in df.iterrows():
        old_image_path = row['File Location']
        new_image_path = old_image_path.replace("C:\\Users\\11351\\Desktop\\manifest-ZkhPvrLo5216730872708713142",".")
        df['File Location'].iloc[index] = new_image_path

    df.to_csv('F:\\datasets\\cbis-ddsm-breast-cancer-image-dataset\\csv\\re_meta.csv')
    if files_count != len(link_list):
        print('{}(real) != {}(modified) '.format(files_count,len(link_list)))

def generate_png(df : pd.DataFrame, output_dir):
    file_location = []
    with tqdm(total=len(df)) as pbar:
        for index , row in df.iterrows():
            dcm_path = path.join(row['Image Path'])
            output_sub = path.join(row['Series Description'].replace(" ","_"),row['Subject ID']+'.png')
            output_path = path.join(output_dir,output_sub)
            file_location.append(output_sub.replace("\\","/"))
            img = pydicom.read_file(dcm_path).pixel_array
            if os.path.exists(output_path):
                print('Errors in :', output_path)
            else: 
                cv2.imwrite(output_path,img)
            pbar.update(1)
    df['File Location'] = file_location

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--dcm_dir', default='manifest-ZkhPvrLo5216730872708713142', type=str)
    parser.add_argument('--output_dir', default='../cbis-ddsm-png/', type=str)
    args = parser.parse_args()

    top = args.dcm_dir
    out_dir  = args.output_dir
    df = pd.read_csv(path.join(top,'reconstruct_meta.csv'))
    df = df[df['Series Description']=='ROI mask images']
    print(len(df))
    generate_png(df, out_dir)
    df = df.drop(['Index','Image Path'], axis=1)
    print(df.duplicated(subset=['File Location'],keep=False).head())
    df.to_csv(path.join(out_dir,'reconstruct_meta.csv'),index=False)





