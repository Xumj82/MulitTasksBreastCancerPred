import lmdb
import cv2
import random
import os
import csv
import mmcv
import numpy as np
import pandas as pd
import torch
from os import path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from tqdm import tqdm
from lib.preprocess_utils import (read_resize_img,
                                  segment_breast,
                                  crop_img,
                                  add_img_margins,
                                  get_max_connected_area,
                                  get_candidate_patch,
                                  overlap_patch_roi,
                                  clahe,
                                  crop_borders,
                                  max_pix_val,
                                  remove_pectoral,
                                  horizontal_flip,
                                  convert_to_16bit,
                                  convert_to_8bit)

class ImageGenerator(object):
    def __init__(self,
                img_dir, 
                img_list, 
                img_resize = (3000,2000),
                img_formats = ['dcm','png'],
                max_pixel_val = 65535,
                crop_borders_size = (0,0,0.04,0.04),
                data_type = np.uint16
                ):
        
        self.img_dir = img_dir
        self.img_list = img_list
        self.img_resize = img_resize
        self.img_formats =img_formats
        self.max_pixel_val = max_pixel_val
        self.data_type = data_type
        self.crop_borders_size = crop_borders_size
        self.mean = 0
        self.std = 0

    def __len__(self):
        return len(self.img_list)

    def process_pipeline(self,img_file_path):
        target_height, target_width = self.img_resize
        img = read_resize_img(img_file_path)
        img = crop_borders(img,border_size=(0,0,0.04,0.04))
        img = np.clip(img,0.1*self.max_pixel_val, 0.9*self.max_pixel_val)
        img = convert_to_16bit(img)
        img_segment,crop_rect,breast_mask = segment_breast(img)
        breast_mask = crop_img(breast_mask, crop_rect)
        # img_segment,_ = remove_pectoral(img=img_segment,breast_mask=breast_mask,high_int_threshold=0.9,sm_kn_size=5)
        img_filped = horizontal_flip(img_segment,breast_mask)
        img_out = clahe(img_filped.astype(np.uint16), self.max_pixel_val)
        if self.img_resize:
            img_out = cv2.resize(img_filped,dsize=(target_width, target_height), 
                    interpolation=cv2.INTER_CUBIC)
        img_out = img_out.astype(self.data_type)
        # img_resized = mmcv.image.imnormalize(img_resized,mean=np.array([123.675]),std=np.array([58.395]),to_rgb=False)
        return img_out

    # def get_stat(self):
    #     total_pval = 0
    #     valid_pnum = 0
    #     for i in range(len(self)):
    #         _,img = self[i]
    #         total_pval += img.sum()
    #         valid_pnum += (img>0).sum()

    def __getitem__(self, idx):
        full_img_path = self.img_list[idx]
        img_id = full_img_path
        for f in self.img_formats:
            img_id = img_id.replace(f,'')
        full_img = self.process_pipeline(path.join(self.img_dir,full_img_path))
        return img_id,full_img

class PatchGenerator(ImageGenerator):
    def __init__(self, img_dir, roi_df, 
                out_dir=None,
                out_csv=None,
                img_resize = (3000,2000),
                patch_size=224,
                jitter=10,
                positiv_overlap = 0.8, 
                negative_overlap = 0.2, 
                number_positive = 10,
                number_negative = 10,
                number_hard_bkg = 10,
                gs_255 = False
                ):
        self.full_images = roi_df['image file path'].copy().drop_duplicates().to_numpy()
        super(PatchGenerator,self).__init__(
            img_dir=img_dir,img_list=self.full_images,img_resize=img_resize
        )
        self.img_dir = img_dir
        self.roi_df = roi_df
        self.img_resize = img_resize
        self.patch_size = patch_size
        self.jitter = jitter
        self.positiv_overlap = positiv_overlap
        self.negative_overlap = negative_overlap
        self.number_positive = number_positive
        self.number_negative = number_negative
        self.number_hard_bkg = number_hard_bkg
        self.sample_records = []
        self.gs_255 = gs_255
        self.out_dir = out_dir
        self.out_csv = out_csv

    def __len__(self):
        return len(self.full_images)

    def get_img_with_lesion(self, idx):
        full_img_path = self.full_images[idx]
        img_id = full_img_path.split('/')[1][:-4]
        lesions = self.roi_df[self.roi_df['image file path']==full_img_path]
        full_img = read_resize_img(path.join(self.img_dir,full_img_path), target_size =self.img_resize,gs_255=self.gs_255)
        full_img,bbox,_ = segment_breast(full_img)
        full_img = clahe(full_img.astype(np.uint16), 65535)
        img = add_img_margins(full_img, self.patch_size//2+self.jitter)
        roi_areas = []
        
        for idx,lesion in lesions.iterrows():
            mask = lesion['ROI mask file path']
            type = lesion['abnormality type']
            pathology = lesion['pathology']

            roi_area = {}

            mask_img =read_resize_img(path.join(self.img_dir,mask), target_size =self.img_resize,gs_255=True)
            mask_img = crop_img(mask_img,bbox) 
            roi_mask = add_img_margins(mask_img, self.patch_size//2+self.jitter)

            idx,cont_areas,contours = get_max_connected_area(roi_mask)
            rx,ry,rw,rh = cv2.boundingRect(contours[idx])
            roi_area_with_margin = img[ry-self.patch_size//2-self.jitter:ry+rh+self.patch_size//2+self.jitter, 
                                        rx-self.patch_size//2-self.jitter:rx+rw+self.patch_size//2+self.jitter]
            roi_mask_with_margin = roi_mask[ry-self.patch_size//2-self.jitter:ry+rh+self.patch_size//2+self.jitter, 
                                        rx-self.patch_size//2-self.jitter:rx+rw+self.patch_size//2+self.jitter]
            
            
            roi_area['rect'] = (rx,ry,rw,rh)
            roi_area['mask'] = mask_img
            roi_area['roi_crop'] = roi_area_with_margin
            roi_area['roi_mask'] = roi_mask_with_margin
            roi_area['roi_path'] = mask
            roi_areas.append(roi_area)
        
        return img_id,type,pathology,roi_areas,img
        

    def get_all_patches(self, verbose=False):
        env = lmdb.open(os.path.join(self.out_dir,'patch_set'), map_size=1099511627776)
        if verbose:
            os.makedirs(os.path.join(self.out_dir,'bkg'),exist_ok=True)
            os.makedirs(os.path.join(self.out_dir,'calcification'),exist_ok=True)
            os.makedirs(os.path.join(self.out_dir,'mass'),exist_ok=True)
        fields=['img_id','patch_id','type','pathology','full_img','ROI_img']
        with open(self.out_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        with open(self.out_csv, 'a') as f:
            writer = csv.writer(f)
            with tqdm(total=len(self)) as pbar:
                for i in range(len(self)):
                    txn = env.begin(write=True)   
                    try:                
                        img_id,type,pathology,roi_areas,img = self[i]
                        patches = self.sample_patches(img_id,type,pathology,img,roi_areas)
                        for idx, patch in enumerate(patches):
                            patch_id = img_id+'_'+patch['type']+'_'+str(idx)
                            txn.put(str(patch_id).encode(), patch['data'].tobytes(order='C'))
                            if verbose:
                                verbose_img = convert_to_8bit(patch['data'])
                                cv2.imwrite(os.path.join(self.out_dir,patch['type'],patch_id+".png"),verbose_img)
                            # self.txn.put(patch_id.encode("ascii"), patch['data'])
                            writer.writerow([img_id, patch_id,patch['type'],patch['pathology'],self.full_images[i],patch['roi_path']])
                    except Exception as e:
                        print("{} : {}".format(i,e))
                    txn.commit()
                    pbar.update(1)
        
        env.close()
        # records = pd.DataFrame(self.sample_records,columns=['img_id','patch_id','type','pathology','full_img','ROI_img'])
        # records.to_csv(self.out_csv,index=False)

    def sample_patches(self, img_id,lesion_type,pathology,img,roi_areas):

        patch_size =self.patch_size
        JITTER_HEIGHT = self.jitter
        JITTER_WIDTH = self.jitter
        neg_cutoff = self.negative_overlap
        lesion_patches = []
        roi_masks = []
        # img_id,lesion_type,pathology,img,roi_areas =  self[idx]
        lesion_patches = []

        if self.gs_255:
            img_gs = np.uint8
        else:
            img_gs = np.uint16

        for roi_area in roi_areas:
            pos_cutoff = self.positiv_overlap
            nb_abn = self.number_positive
            def overlap_patch_roi_(patch):
                roi_area = (mask_1D>0).sum()
                patch_area = (patch>0).sum()
                inter_area = (mask_1D*patch>0).sum()
                return (inter_area/roi_area > self.positiv_overlap or inter_area/patch_area > self.positiv_overlap) 

            rng = np.random.RandomState(12345)
            roi_path = roi_area['roi_path']
            (rx,ry,rw,rh) = roi_area['rect']
            roi_masks.append(roi_area['mask'])
            roi_area_with_margin = roi_area['roi_crop']
            roi_mask_with_margin = roi_area['roi_mask']

            patches, tl = get_candidate_patch(roi_mask_with_margin.shape, 
                                                rw, rh, patch_size, 
                                                rx=patch_size//2, ry = patch_size//2,
                                                JITTER_HEIGHT=JITTER_HEIGHT, JITTER_WIDTH=JITTER_WIDTH
                                                )
            
            sel_centers = np.array([], dtype=np.int32).reshape(0,2)
            while True:
                patches_1D = patches.reshape((tl.shape[0], roi_mask_with_margin.shape[0]*roi_mask_with_margin.shape[1]))
                mask_1D = roi_mask_with_margin.reshape((roi_mask_with_margin.shape[0]*roi_mask_with_margin.shape[1]))

                overlap_idx = np.apply_along_axis(overlap_patch_roi_, 1, patches_1D)
                overlap_idx = np.argwhere(overlap_idx).squeeze(-1)
                if overlap_idx.shape[0] <= 1:
                    pos_cutoff -= 0.05
                    print('img:{}  new cutoff:{}'.format(img_id,pos_cutoff))
                    if pos_cutoff <= neg_cutoff:
                        print('img:{}  less than:{}'.format(img_id,pos_cutoff))
                        return
                    continue
                sel_centers = np.concatenate((sel_centers , tl[overlap_idx]))
                if sel_centers.shape[0] < nb_abn:
                    sel_centers_t = np.transpose(sel_centers)
                    
                    # img_rect = draw_rect(cv2.equalizeHist(roi_area_with_margin.astype(np.uint8)), tl[overlap_idx], patch_size)
                    patches, tl = get_candidate_patch(
                        roi_mask_with_margin.shape,
                        max(sel_centers_t[0])-min(sel_centers_t[0]), 
                        max(sel_centers_t[1])-min(sel_centers_t[1]), 
                        patch_size,
                        rx= min(sel_centers_t[0]),
                        ry= min(sel_centers_t[1])
                        )       
                else:
                    break
            sel_idx = np.random.randint(sel_centers.shape[0], size=nb_abn)
            sel_centers =sel_centers[sel_idx]
            sel_centers = sel_centers


            # img_rect = draw_rect(cv2.equalizeHist(roi_area_with_margin.astype(np.uint8)), sel_centers, patch_size)

            # cv2.imwrite('patches/{}.jpg'.format(img_id), img_rect) 
            for idx,[x, y] in enumerate(sel_centers):
                patch_img =roi_area_with_margin[y - patch_size//2:y + patch_size//2, x - patch_size//2:x + patch_size//2]     
                # save_patch(patch_img, roi_out, idx)
                patch = {
                    'type':lesion_type,
                    'pathology':pathology,
                    'data':patch_img.astype(img_gs),
                    'roi_path': roi_path
                }
                lesion_patches.append(patch)

        def get_bkg(i):
            x = rng.randint(patch_size//2, img.shape[1] - patch_size//2)
            y = rng.randint(patch_size//2, img.shape[0] - patch_size//2)
            patch_img = img[int(y - patch_size//2):y + int(patch_size//2), 
                        int(x - patch_size//2):x + int(patch_size//2)]
            if (patch_img>0).sum() > 0:
                for roi_mask in roi_masks: 
                    if overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
                        return i
                patch = {
                    'type':'bkg',
                    'pathology':'NORMAL',
                    'data': patch_img.astype(img_gs),
                    'roi_path':None
                }
                lesion_patches.append(patch)
                return i-1
            return i
        
        def get_hard_bkg(keypoints):

            x, y = map(int,random.choice(keypoints).pt)
            # x = rng.randint(patch_size//2, img.shape[1] - patch_size//2)
            # y = rng.randint(patch_size//2, img.shape[0] - patch_size//2)
            patch_img = img[int(y - patch_size//2):y + int(patch_size//2), 
                        int(x - patch_size//2):x + int(patch_size//2)]
            if patch_img.shape[0]*patch_img.shape[1] != patch_size*patch_size:
                return False

            for roi_mask in roi_masks: 
                if overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
                    return False
            patch = {
                'type':'bkg',
                'pathology':'NORMAL',
                'data': patch_img.astype(img_gs),
                'roi_path':None
            }
            lesion_patches.append(patch)
            return True

        i = int(self.number_negative*len(roi_areas))
        while i > 0 :
            i = get_bkg(i)
        
        hard_try = 0

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 2

        detector = cv2.SimpleBlobDetector_create(params)
        img_8bit = convert_to_8bit(img).astype(np.uint8)
        keypoints = detector.detect(img_8bit)

        i = int(self.number_hard_bkg*len(roi_areas))
        while i > 0 :
            if get_hard_bkg(keypoints):
                i -= 1
                hard_try = 0
            else:
                hard_try += 1
                if hard_try == 10:
                    print('hard bkg num:{}/{}'.format(self.number_hard_bkg*len(roi_areas)-i,self.number_hard_bkg*len(roi_areas)))
                    break

        return lesion_patches