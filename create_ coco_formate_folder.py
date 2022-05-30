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
import sys
# sys.path.append("..")
from data import CsawSet
from data import DdsmSet
from lib.preprocess_utils import get_max_connected_area


def main():

    if args.data_name == 'ddsm':
        # dataset = DdsmSet(args.data_root,dataset_stat = True)
        # generate_img_mask_pairs(DdsmSet(args.data_root,train=True),args.output_dir,train=True)
        ddsm_generate_img_mask_pairs(DdsmSet(args.data_root,train=False),args.output_dir,train=False)
    if args.data_name == 'csaw':
        dataset = CsawSet(args.data_root,train=True, split= True)
        csaw_generate_img_mask_pairs(dataset,args.output_dir,type='trian')
        dataset = CsawSet(args.data_root,train=False, split= True)
        csaw_generate_img_mask_pairs(dataset,args.output_dir,type='val')
        dataset = CsawSet(args.data_root, split= False)
        csaw_generate_img_mask_pairs(dataset,args.output_dir,type='test')

def get_lesions(mask):
    indexes = np.unique(mask)
    bboxes = []
    masks = []
    for idx in indexes:
        if idx != 0:
            ann_img = np.where(mask==idx,255,0)
            contours,_ = cv2.findContours(
            ann_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0])
            mask_pixels = np.argwhere(ann_img > 0)
            bboxes.append((x,y,w,h))
            masks.append(mask_pixels)
    return bboxes,masks

def get_category_name(id,categories):
    for item in categories:
        if item['id']==id:
            return item['name']
    raise "Not found {} in {}".format(id, categories)

def csaw_generate_img_mask_pairs(data_set,dataset_name,type = 'train', verbose=True):

    # roi_file = pd.read_csv(os.path.join(args.data_root,'csv/'+type+'_roi.csv'))

    # data_set = DDSMSet(os.path.join(args.data_root,'cbis-ddsm-png/'),roi_file,target_size=(2000,3000))
    # data_set_module = load_data_module(dataset_name)
    # data_set = data_set_module(args.data_root)
    annotations_ptype = []
    annotations_lonly = []
    images = []

    categories_lonly=[{'id':1, 'name': 'lesion'}]
    obj_count = 0
    if not os.path.exists(os.path.join(dataset_name,type)):
        os.makedirs(os.path.join(dataset_name,type))

    if verbose:
        if not os.path.exists(os.path.join(dataset_name,type+'_verbose')):
            os.makedirs(os.path.join(dataset_name,type+'_verbose'))  

    with tqdm(total=len(data_set)) as pbar:
        for i in range(len(data_set)):

            img_id,full_img,ann_img,pathology,rad_time = data_set[i]
            # if img_id == "01106_20990909_L_MLO_2.dcm":
            #     t = img_id
            # else:
            #     pbar.update(1)
            #     continue
            if full_img is None:
                print('{} : mask shape not matched'.format(img_id))
                pbar.update(1)
                continue

            if type=='test' and np.sum(ann_img) < 5:
                images.append(dict(
                    id=i,
                    label=rad_time,
                    file_name=img_id+'.png',
                    height=full_img.shape[0],
                    width=full_img.shape[1]))
                if verbose:
                    verbose_img = (full_img.copy()/4095*255).astype(np.uint8)
                    verbose_img = cv2.cvtColor(verbose_img,cv2.COLOR_GRAY2RGB)
                for idx in np.unique(ann_img):
                    if idx != 0:
                        mask_pixels = np.argwhere(ann_img > 0)
                        y, x = mask_pixels[0]
                        data_anno = dict(
                            image_id=i,
                            id=obj_count,
                            category_id=idx,
                            bbox=(x,y,1,1),
                            area=1,
                            segmentation=mask_pixels,
                            iscrowd=0)

                        annotations_lonly.append(data_anno)
                        # annotations_ptype[-1]['category_id'] = 1
                        obj_count += 1
                        if verbose:
                            cv2.rectangle(verbose_img,(x-10,y-10),(x+10,y+10),(255,0,0),3)
                            # cv2.putText(img=verbose_img,
                            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            #             text='category_id: {}'.format(data_anno['category_id'], get_category_name(data_anno['category_id'],categories_5cls)), 
                            #             org=(x, y-10),  
                            #             fontScale=0.5, color=(0,255,0), thickness=1)
                img_path = os.path.join(dataset_name,type,'{}.png').format(img_id)
                # img = np.expand_dims(full_img, axis=-1)
                # img = img.astype(np.float32)
                # img = (img-img.min())/img.max()*255
                # mmcv.dump(full_img, img_path)
                cv2.imwrite(img_path,full_img,)
                if verbose:
                    verbose_img_path = os.path.join(dataset_name,type+'_verbose','{}.jpeg').format(img_id)
                    verbose_img = cv2.resize(verbose_img,(600,800))
                    cv2.imwrite(verbose_img_path,verbose_img)                
                # meta_file.write('{},{}\n'.format(img_id,pathology))
                pbar.update(1)
            if type!='test' and np.sum(ann_img) > 5:
                images.append(dict(
                    id=i,
                    label=rad_time,
                    file_name=img_id+'.png',
                    height=full_img.shape[0],
                    width=full_img.shape[1]))
                if verbose:
                    verbose_img = (full_img.copy()/4095*255).astype(np.uint8)
                    verbose_img = cv2.cvtColor(verbose_img,cv2.COLOR_GRAY2RGB)
                for idx in np.unique(ann_img):
                    if idx != 0:
                        ann_img = np.where(ann_img==idx,255,0).astype(np.uint8)
                        # cv2.imshow('graycsale image',ann_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()     
                        contours,_ = cv2.findContours(
                            ann_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cont_areas = [ cv2.contourArea(cont) for cont in contours ]

                        for c_idx, area in enumerate(cont_areas):

                            if area < 40:
                                continue                        
                            x,y,w,h = cv2.boundingRect(contours[c_idx])
                            mask_pixels = np.argwhere(ann_img > 0)

                            data_anno = dict(
                                image_id=i,
                                id=obj_count,
                                category_id=idx,
                                bbox=[x,y,w,h],
                                area=area,
                                segmentation=[contours[c_idx].flatten()],
                                iscrowd=0)

                            annotations_lonly.append(data_anno)
                            # annotations_ptype[-1]['category_id'] = 1
                            obj_count += 1
                            if verbose:
                                cv2.rectangle(verbose_img,(x,y),(x+w,y+h),(0,255,0),3)
                                # cv2.putText(img=verbose_img,
                                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                #             text='category_id: {}'.format(data_anno['category_id'], get_category_name(data_anno['category_id'],categories_5cls)), 
                                #             org=(x, y-10),  
                                #             fontScale=0.5, color=(0,255,0), thickness=1)
                img_path = os.path.join(dataset_name,type,'{}.png').format(img_id)
                # img = np.expand_dims(full_img, axis=-1)
                # img = img.astype(np.float32)
                # img = (img-img.min())/img.max()*255
                # mmcv.dump(full_img, img_path)
                cv2.imwrite(img_path,full_img,)
                if verbose:
                    verbose_img_path = os.path.join(dataset_name,type+'_verbose','{}.jpeg').format(img_id)
                    verbose_img = cv2.resize(verbose_img,(600,800))
                    cv2.imwrite(verbose_img_path,verbose_img)                
                # meta_file.write('{},{}\n'.format(img_id,pathology))
            pbar.update(1)

    coco_format_json_lonly = dict(
        images=images,
        annotations=annotations_lonly,
        categories=categories_lonly)

    mmcv.dump(coco_format_json_lonly, os.path.join(args.data_root,dataset_name,type,'annotation_coco.json'))

def ddsm_generate_img_mask_pairs(data_set,dataset_name,train = True, verbose=True):

    if train:
        type = 'train'
    else:
        type = 'test'

    # roi_file = pd.read_csv(os.path.join(args.data_root,'csv/'+type+'_roi.csv'))

    # data_set = DDSMSet(os.path.join(args.data_root,'cbis-ddsm-png/'),roi_file,target_size=(2000,3000))
    # data_set_module = load_data_module(dataset_name)
    # data_set = data_set_module(args.data_root)
    annotations_5cls = []
    annotations_ltype = []
    annotations_ptype = []
    annotations_lonly = []
    images = []
    categories_5cls=[{'id':1, 'name': 'be_calc'},{'id':2, 'name': 'be_mass'},{'id':2, 'name': 'ma_mass'},{'id':2, 'name': 'ma_mass'}]
    categories_ltype=[{'id':1, 'name': 'calc'},{'id':2, 'name': 'mass'}]
    categories_ptype=[{'id':1, 'name': 'benign'},{'id':2, 'name': 'malignant'},]
    categories_lonly=[{'id':1, 'name': 'lesion'}]
    obj_count = 0
    if not os.path.exists(os.path.join(dataset_name,type)):
        os.makedirs(os.path.join(dataset_name,type))
    if verbose:
        if not os.path.exists(os.path.join(dataset_name,type+'_verbose')):
            os.makedirs(os.path.join(dataset_name,type+'_verbose'))  

    with tqdm(total=len(data_set)) as pbar:
        for i in range(len(data_set)):
         
            img_id,full_img,ann_img,pathology,rad_time = data_set[i]

            images.append(dict(
                id=i,
                label=pathology,
                file_name=img_id+'.png',
                height=full_img.shape[0],
                width=full_img.shape[1]))
            if verbose:
                verbose_img = (full_img.copy()/65535*255).astype(np.uint8)
                verbose_img = cv2.cvtColor(verbose_img,cv2.COLOR_GRAY2RGB)
            for idx in np.unique(ann_img):
                if idx != 0:
                    ann_img = np.where(ann_img==idx,255,0).astype(np.uint8)

                    # cv2.imshow('graycsale image',ann_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()     
                    contours,_ = cv2.findContours(
                        ann_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cont_areas = [ cv2.contourArea(cont) for cont in contours ]

                    for c_idx, area in enumerate(cont_areas):

                        if area < 40:
                            continue                        
                        x,y,w,h = cv2.boundingRect(contours[c_idx])
                        mask_pixels = np.argwhere(ann_img > 0)

                        data_anno = dict(
                            image_id=i,
                            id=obj_count,
                            category_id=idx,
                            bbox=[x,y,w,h],
                            area=area,
                            segmentation=[contours[c_idx].flatten()],
                            iscrowd=0)
                            
                        annotations_5cls.append(data_anno)
                        annotations_ltype.append(data_anno)
                        annotations_ltype[-1]['category_id'] = annotations_ltype[-1]['category_id']//2 if annotations_ltype[-1]['category_id']>2 else annotations_ltype[-1]['category_id']
                        annotations_ptype.append(data_anno)
                        annotations_ptype[-1]['category_id'] = 1 if annotations_ptype[-1]['category_id']<2 else 2
                        annotations_lonly.append(data_anno)
                        annotations_ptype[-1]['category_id'] = 1
                        obj_count += 1
                        if verbose:
                            cv2.rectangle(verbose_img,(x,y),(x+w,y+h),(0,255,0),3)
                            cv2.putText(img=verbose_img,
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                        text='category_id: {} type: {}'.format(data_anno['category_id'], get_category_name(data_anno['category_id'],categories_5cls)), 
                                        org=(x, y-10),  
                                        fontScale=0.5, color=(0,255,0), thickness=1)
            img_path = os.path.join(dataset_name,type,'{}.png').format(img_id)
            # img = np.expand_dims(full_img, axis=-1)
            # img = img.astype(np.float32)
            # img = (img-img.min())/img.max()*255
            # mmcv.dump(full_img, img_path)
            cv2.imwrite(img_path,full_img,)
            if verbose:
                verbose_img_path = os.path.join(dataset_name,type+'_verbose','{}.jpeg').format(img_id)
                verbose_img = cv2.resize(verbose_img,(600,800))
                cv2.imwrite(verbose_img_path,verbose_img)                
            # meta_file.write('{},{}\n'.format(img_id,pathology))
            pbar.update(1)
    coco_format_json_5cls = dict(
        images=images,
        annotations=annotations_5cls,
        categories=categories_5cls)
    coco_format_json_ltype = dict(
        images=images,
        annotations=annotations_ltype,
        categories=categories_ltype)
    coco_format_json_ptype = dict(
        images=images,
        annotations=annotations_ptype,
        categories=categories_ptype)
    coco_format_json_lonly = dict(
        images=images,
        annotations=annotations_lonly,
        categories=categories_lonly)
    mmcv.dump(coco_format_json_5cls, os.path.join(args.data_root,dataset_name,type,'annotation_coco_5cls.json'))
    mmcv.dump(coco_format_json_ltype, os.path.join(args.data_root,dataset_name,type,'annotation_coco_ltype.json'))
    mmcv.dump(coco_format_json_ptype, os.path.join(args.data_root,dataset_name,type,'annotation_coco_ptype.json'))
    mmcv.dump(coco_format_json_lonly, os.path.join(args.data_root,dataset_name,type,'annotation_coco_lonly.json'))



if __name__ == '__main__':
   
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--data_name',default='ddsm',type=str)
    parser.add_argument('--output_dir',default='/home/xumingjie/dataset/ddsm_coco',type=str)
    parser.add_argument('--data_root',default='/mnt/hdd/datasets/',type=str)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.environ['data_root']
    print('DATA_ROOT:',args.data_root)
    
    # args.data_root = os.path.expanduser(args.data_root)
    main()
