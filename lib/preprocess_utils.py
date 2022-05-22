import os
from os import path
from pathlib import Path

import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset



## put you dir here 

def generate_new_meta():
    calc_train = pd.read_csv('csv/calc_case_description_train_set.csv')
    mass_train = pd.read_csv('csv/mass_case_description_train_set.csv')

    calc_test = pd.read_csv('csv/calc_case_description_test_set.csv')
    mass_test = pd.read_csv('csv/mass_case_description_test_set.csv')

    meta_data = pd.read_csv('cbis-ddsm-png/reconstruct_meta.csv')

    meta_data['Subject ID'] =  meta_data['Subject ID'].str.strip()
    meta_data['Series Description'] =  meta_data['Series Description'].str.strip()

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

def show_img(img):
  plt.figure(figsize=(10, 10))
  plt.imshow(img, cmap=plt.cm.bone)
  plt.show()

def show_img_cv(img, title = 'img' ):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_to_8bit(img):
    if np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    return (img * 255).astype(np.int32)

def convert_to_16bit(img):
    if np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    return (img * 65535).astype(np.float32)

def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False, rescale_factor=None):
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    if target_size is None and target_height is None:
        raise Exception('One of [target_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = pydicom.dcmread(fname).pixel_array
        if gs_255:
            img = convert_to_8bit(img)
        else:
            img = convert_to_16bit(img)
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_height is not None:
        target_width = int(float(target_height)/img.shape[0]*img.shape[1])
    else:
        target_height, target_width = target_size
    if (target_height, target_width) != img.shape:
        img = cv2.resize(
            img, dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img *= target_scale/img_max
    if rescale_factor is not None:
        img *= rescale_factor
    return img

def get_max_connected_area(img, erosion = False):
    img_8u = img.copy().astype('uint8')

    if erosion:
        kernel = np.ones((20, 20), np.uint8)
        # Using cv2.erode() method 
        img_8u = cv2.erode(img_8u, kernel, cv2.BORDER_REFLECT)
        img_8u = cv2.dilate(img_8u,kernel,iterations = 1)
    
    # img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
    contours,_ = cv2.findContours(
        img_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    if len(cont_areas):
        idx = np.argmax(cont_areas)  # find the largest contour.
    else:
        idx = -1
    return idx,cont_areas,contours

def crop_img(img, bbox):
    '''Crop an image using bounding box
    '''
    x,y,w,h = bbox
    return img[y:y+h, x:x+w]

def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    margin_size = int(margin_size)
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

def crop_val(v, minv, maxv):
    v = v if v >= minv else minv
    v = v if v <= maxv else maxv
    return v

def segment_breast(img, low_int_threshold=0.05, crop=True,erosion= False):
    '''Perform breast segmentation
    Args:
        low_int_threshold([float or int]): Low intensity threshold to 
                filter out background. It can be a fraction of the max 
                intensity value or an integer intensity value.
        crop ([bool]): Whether or not to crop the image.
    Returns:
        An image of the segmented breast.
    NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
        which has a max value of 255.
    '''
    # Create img for thresholding and contours.
    img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
    
    if low_int_threshold < 1.:
        low_th = int(img_8u.max()*low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
    idx,cont_areas,contours = get_max_connected_area(img_bin, erosion = erosion)
    # contours,_ = cv2.findContours(
    #     img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    # idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
    breast_mask = cv2.drawContours(
        np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.
    # segment the breast.
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
    x,y,w,h = cv2.boundingRect(contours[idx])
    if crop:
        img_breast_only = img_breast_only[y:y+h, x:x+w]
    return img_breast_only, (x,y,w,h)

def overlap_patch_roi(patch_center, patch_size, roi_mask, 
                     cutoff=.5):
    black_img = np.zeros(roi_mask.shape)
    black_img[int(patch_center[1] - patch_size/2):patch_center[1] + int(patch_size/2), 
                int(patch_center[0]- patch_size/2):patch_center[0] + int(patch_size/2)] = 1
    roi_area = (roi_mask>0).sum()
    patch_area = (black_img>0).sum()

    inter_img = roi_mask*black_img
    inter_area = (inter_img >0).sum()

    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)

def draw_rect(img, sel_centers, patchsize):
    # print(img.shape)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for idx,[cx, cy] in enumerate(sel_centers):
        x1 = cx - patchsize//2
        x2 = cx + patchsize//2
        y1 = cy - patchsize//2
        y2 = cy + patchsize//2
        # print(x1,y1, x2,y2)
        if (x1 * y1<0) or (x2>img.shape[1]) or (y2>img.shape[0]):
            print('Rect bigger than image')
        cv2.rectangle(img,(x1,y1),(x2, y2), (0, 255, 0), 1)
    return img

def draw_rect(img, rect):
    # print(img.shape)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), (0, 255, 0), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_candidate_patch(shape,rw, rh,patch_size, step_divisor=8, 
                        rx=0, ry =0, JITTER_HEIGHT= 10,JITTER_WIDTH=10):
    def get_patch_(shape,x,y, patch_size):
        black_img = np.zeros(shape)
        black_img[y - patch_size//2:y + patch_size//2, x- patch_size//2:x + patch_size//2] = 1
        return black_img
    
    if rw <=0 or rh<=0:
    #if True:
        print('rw/rh <0',rx,ry)
        rxx = np.array([rx,rx,rx,rx])
        ryy = np.array([ry,ry,ry,ry])
    else:
        # JITTER_HEIGHT = rh//JITTER_HEIGHT if rh >= JITTER_HEIGHT else 1
        # JITTER_WIDTH = rw//JITTER_WIDTH if rw >= JITTER_WIDTH else 1
        rxx =  np.arange(rx, rx+rw, max(rw//step_divisor,1))
        ryy = np.arange(ry, ry+rh, max(rh//step_divisor,1))
    xx, yy = np.meshgrid(rxx, ryy)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    tl = np.concatenate((xx, yy), axis=1)
    tl = tl.astype(np.int32)
    jitter = np.random.randint([-JITTER_HEIGHT, -JITTER_WIDTH], [JITTER_HEIGHT, JITTER_WIDTH], tl.shape)
    tl += jitter
    patches = []
    for [x,y] in tl:
        patches.append(get_patch_(shape,x,y,patch_size))
    patches = np.array(patches,dtype=np.int32)
    # tl = torch.from_numpy(tl)
    return patches, tl

def sample_patches_(records, img, roi_mask, out_dir,txn, img_id, pos, abn_type, patch_size,
                   pos_cutoff, neg_cutoff,
                   nb_bkg, nb_abn,
                   bkg_dir='background', pos_dir='malignant', neg_dir='benign', rect_dir='rect',
                   JITTER_HEIGHT= 10,JITTER_WIDTH=10,
                   verbose=False):
    
    #region
    if pos:
        roi_out = os.path.join(out_dir, pos_dir)
    else:
        roi_out = os.path.join(out_dir, neg_dir)
    bkg_out = os.path.join(out_dir, bkg_dir)
    rect_out = os.path.join(out_dir, rect_dir)

    img = add_img_margins(img, patch_size/2+JITTER_HEIGHT)
    roi_mask = add_img_margins(roi_mask, patch_size/2+JITTER_WIDTH)
    
    # Get ROI bounding box.
    idx,cont_areas,contours = get_max_connected_area(roi_mask)

    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    roi_area_with_margin = img[ry-patch_size//2-JITTER_HEIGHT:ry+rh+patch_size//2+JITTER_HEIGHT, rx-patch_size//2-JITTER_WIDTH:rx+rw+patch_size//2+JITTER_WIDTH]
    roi_mask_with_margin = roi_mask[ry-patch_size//2-JITTER_HEIGHT:ry+rh+patch_size//2+JITTER_HEIGHT, rx-patch_size//2-JITTER_WIDTH:rx+rw+patch_size//2+JITTER_WIDTH]
    # show_img(roi_area_with_margin)
    # show_img(roi_mask_with_margin)
    # print(roi_area_with_margin.shape)

    rng = np.random.RandomState(12345)
    # Sample abnormality first.
    def save_patch(patch_img, out_dir, count):
        # patch_img = Image.fromarray(patch_img.astype(np.uint8))
        filename = img_id + "_%04d" % (count)
        fullname = os.path.join(out_dir, filename)

        print(fullname)
        # txn.put(key = fullname.encode('ascii'), value = patch_img)

        # hf.create_dataset(fullname,data=patch_img.astype(np.uint8))
        # patch_img.save(fullname)

    #endregion
    # sampled_abn = 0
    patches, tl = get_candidate_patch(roi_mask_with_margin.shape, 
                                        rw, rh, patch_size, 
                                        rx=patch_size//2, ry = patch_size//2,
                                        JITTER_HEIGHT=JITTER_HEIGHT, JITTER_WIDTH=JITTER_WIDTH
                                        )
    
    sel_centers = np.array([], dtype=np.int32).reshape(0,2)
    while True:
        patches_1D = patches.reshape((tl.shape[0], roi_mask_with_margin.shape[0]*roi_mask_with_margin.shape[1]))
        mask_1D = roi_mask_with_margin.reshape((roi_mask_with_margin.shape[0]*roi_mask_with_margin.shape[1]))
        
        def overlap_patch_roi_(patch):
            roi_area = (mask_1D>0).sum()
            patch_area = (patch>0).sum()
            inter_area = (mask_1D*patch>0).sum()

            return (inter_area/roi_area > pos_cutoff or inter_area/patch_area > pos_cutoff)


        overlap_idx = np.apply_along_axis(overlap_patch_roi_, 1, patches_1D)
        overlap_idx = np.argwhere(overlap_idx).squeeze(-1)
        if overlap_idx.shape[0] <= 1:
            pos_cutoff -= 0.05
            print('img:{}  new cutoff:{}'.format(img_id,pos_cutoff))
            if pos_cutoff <= neg_cutoff:
                print('img:{}  less than:{}'.format(img_id,pos_cutoff))
                # show_img(roi_area_with_margin)
                # show_img(roi_mask_with_margin)
                img_rect = draw_rect(cv2.equalizeHist(roi_area_with_margin.astype(np.uint8)), tl, patch_size)
                cv2.imwrite('patches/{}_mask_margine.jpg'.format(img_id), roi_mask_with_margin)
                cv2.imwrite('patches/{}_raw_margine.jpg'.format(img_id), roi_area_with_margin.astype(np.uint8))
                cv2.imwrite('patches/{}.jpg'.format(img_id), img_rect) 
                return
            continue
        sel_centers = np.concatenate((sel_centers , tl[overlap_idx]))
        if sel_centers.shape[0] < nb_abn:
            sel_centers_t = np.transpose(sel_centers)
            
            img_rect = draw_rect(cv2.equalizeHist(roi_area_with_margin.astype(np.uint8)), tl[overlap_idx], patch_size)
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


    img_rect = draw_rect(cv2.equalizeHist(roi_area_with_margin.astype(np.uint8)), sel_centers, patch_size)

    # cv2.imwrite('patches/{}.jpg'.format(img_id), img_rect) 
    for idx,[x, y] in enumerate(sel_centers):
        patch_img =roi_area_with_margin[y - patch_size//2:y + patch_size//2, x - patch_size//2:x + patch_size//2]     
        save_patch(patch_img, roi_out, idx)
    if verbose:
        patch_img = img[ry:ry+rh, rx:rx+rw]
        save_patch(patch_img, roi_out, 9999)



    sampled_bkg = 0
    while sampled_bkg < nb_bkg:
        x = rng.randint(patch_size//2, img.shape[1] - patch_size//2)
        y = rng.randint(patch_size//2, img.shape[0] - patch_size//2)
        patch = img[int(y - patch_size//2):y + int(patch_size//2), 
                    int(x - patch_size//2):x + int(patch_size//2)]
        if (patch>0).sum() > 0:     
            if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
                save_patch(patch,bkg_out,sampled_bkg)
                sampled_bkg += 1
                if verbose:
                    print("sampled a bkg patch at (x,y) center=", (x,y))

def sample_patches(img, roi_mask, out_dir, out_df, img_id, pos, abn_type, patch_size,
                   pos_cutoff, neg_cutoff,
                   nb_bkg, nb_abn,
                   bkg_dir='background', pos_dir='malignant', neg_dir='benign', 
                   verbose=False):
    if pos:
        roi_out = os.path.join(out_dir, pos_dir)
    else:
        roi_out = os.path.join(out_dir, neg_dir)
    bkg_out = os.path.join(out_dir, bkg_dir)
    # basename = '_'.join([img_id])
    # output_pd = pd.DataFrame(columns = ['Subject ID', 'pathology', 'path'])

    # Check directory
    if not os.path.exists(roi_out):
      os.makedirs(roi_out)
    if not os.path.exists(bkg_out):
      os.makedirs(bkg_out)

    # Expand margine, pathes should be not out of image
    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    
    # Get ROI bounding box.
    idx,cont_areas,contours = get_max_connected_area(roi_mask)

    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print("ROI centroid=", (cx,cy))

    rng = np.random.RandomState(12345)
    # Sample abnormality first.
    sampled_abn = 0
    nb_try = 0

    while pos_cutoff > 0:
        sample_size = max(nb_abn, 50)
        x_list = np.random.randint(rx,rx + rw, size=sample_size)
        y_list = np.random.randint(ry,ry + rh, size=sample_size)

        for x, y in zip(x_list, y_list):
            if overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=pos_cutoff):
                patch = img[int(y - patch_size/2):y + int(patch_size/2), 
                            int(x - patch_size/2):x + int(patch_size/2)]
                patch_img = Image.fromarray(patch.astype(np.uint8))
                # patch = patch.reshape((patch.shape[0], patch.shape[1], 1))
                filename = img_id + "_%04d" % (sampled_abn) + ".png"
                fullname = os.path.join(roi_out, filename)
                patch_img.save(fullname)
                out_df.append([img_id,abn_type,pos,fullname])
                sampled_abn += 1
                if sampled_abn==nb_abn:
                    break
                if verbose:
                    print("sampled an abn patch at (x,y) center=", (x,y))
        if sampled_abn==nb_abn:
            break
        if pos_cutoff <= neg_cutoff:
            raise Exception("overlap cutoff becomes non-positive, "
                            "check roi mask input.")
        else:
            # m = torch.nn.Sigmoid()
            # input = torch.tensor(sampled_abn/nb_abn*10)
            # pos_cutoff = m(input).item() * pos_cutoff
            pos_cutoff = pos_cutoff-0.5

    sampled_bkg = 0
    while sampled_bkg < nb_bkg:
        x = rng.randint(patch_size/2, img.shape[1] - patch_size/2)
        y = rng.randint(patch_size/2, img.shape[0] - patch_size/2)
        patch = img[int(y - patch_size/2):y + int(patch_size/2), 
                    int(x - patch_size/2):x + int(patch_size/2)]
        if (patch>0).sum() > 0:     
            if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
                # patch = img[int(y - patch_size/2):y + int(patch_size/2), 
                #             int(x - patch_size/2):x + int(patch_size/2)]
                patch_img = Image.fromarray(patch.astype(np.uint8))
                filename = img_id + "_%04d" % (sampled_bkg) + ".png"
                fullname = os.path.join(bkg_out, filename)
                # patch_img.save(fullname)
                out_df.append([img_id,'background',False,fullname])
                sampled_bkg += 1
                if verbose:
                    print("sampled a bkg patch at (x,y) center=", (x,y))

def generate_patch(
    row,
    txn,
    records,
    img_dir = 'cbis-ddsm-png/',
    out_dir = 'patches/',
    target_height = 1024,
    patch_size=224, 
    positiv_overlap = 0.75, 
    negative_overlap = 0.35, 
    number_positive = 10,
    number_negative = 5,    
    ):
    full_img_path = path.join(img_dir,row['image file path']) 
    mask_img_path = path.join(img_dir,row['ROI mask file path'])
    img_id = Path(row['ROI mask file path']).stem
    abn_type = row['abnormality type']
    positiv = True if row['pathology'] == 'MALIGNANT' else False

    full_img = read_resize_img(full_img_path, target_height=target_height,gs_255=True)
    mask_img = read_resize_img(mask_img_path, target_height=target_height,gs_255=True)

    full_img,bbox = segment_breast(full_img)
    mask_img = crop_img(mask_img,bbox)
    try:
        sample_patches_(full_img, mask_img, out_dir,txn, img_id, positiv, abn_type,
                patch_size, positiv_overlap, negative_overlap, 
                number_negative, number_positive,
                records,
                verbose=False)
    except Exception as e:
        print(img_id)
        print(e.args)