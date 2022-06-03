import os
from os import path
from pathlib import Path

import cv2
import torch
import pydicom
import mmcv
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset



## put you dir here 
def process_pipeline(img_file_path,resize_shape,crop_borders_size):
    target_height, target_width = resize_shape
    img = read_resize_img(img_file_path, crop_borders_size=(0,0,0.04,0.04))
    
    img_segment,crop_rect,breast_mask = segment_breast(img)
    breast_mask = crop_img(breast_mask, crop_rect)
    img_segment,_ = remove_pectoral(img=img_segment,breast_mask=breast_mask,high_int_threshold=0.9,sm_kn_size=5)
    img_filped = horizontal_flip(img_segment,breast_mask)
    img_filped = clahe(img_filped.astype(np.uint16), 65535)
    img_resized = cv2.resize(img_filped,dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img_resized = convert_to_16bit(img_resized)/65535*255
    img_resized = mmcv.image.imnormalize(img_resized,mean=np.array([123.675]),std=np.array([58.395]),to_rgb=False)
    return img_resized

def max_pix_val(dtype):
    if dtype == np.dtype('uint8'):
        maxval = 2**8 - 1
    elif dtype == np.dtype('uint16'):
        maxval = 2**16 - 1
    else:
        raise Exception('Unknown dtype found in input image array')
    return maxval

def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                        smooth_boundary=False, kernel_size=15):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin (2D array): 2D numpy array of binary image.
        lab_val ([int]): integer value used for the label of the largest 
                object. Default is 255.
        fill_holes ([boolean]): whether fill the holes inside the largest 
                object or not. Default is false.
        smooth_boundary ([boolean]): whether smooth the boundary of the 
                largest object using morphological opening or not. Default 
                is false.
        kernel_size ([int]): the size of the kernel used for morphological 
                operation. Default is 15.
    Returns:
        a binary image as a mask for the largest object.
    '''
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
                                            ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    # import pdb; pdb.set_trace()
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
                        newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, 
                                        kernel_)
        
    return largest_mask

def remove_pectoral(img, high_int_threshold=.9, 
                    morph_kn_size=10, n_morph_op=7, sm_kn_size=25):
    '''Remove the pectoral muscle region from an input image

    Args:
        img (2D array): input image as a numpy 2D array.
        breast_mask (2D array):
        high_int_threshold ([int]): a global threshold for high intensity 
                regions such as the pectoral muscle. Default is 200.
        morph_kn_size ([int]): kernel size for morphological operations 
                such as erosions and dilations. Default is 3.
        n_morph_op ([int]): number of morphological operations. Default is 7.
        sm_kn_size ([int]): kernel size for final smoothing (i.e. opening). 
                Default is 25.
    Returns:
        an output image with pectoral muscle region removed as a numpy 
        2D array.
    Notes: this has not been tested on .dcm files yet. It may not work!!!
    '''
    # Enhance contrast and then thresholding.
    img_8u = convert_to_8bit(np.clip(img,0.4*65535, 0.5*65535)).astype(np.uint8)
    # t1 = np.mean(img_8u)
    # t2 = np.min(img_8u)
    # img = convert_to_16bit(img)
    # img_equ = cv2.equalizeHist(img_8u)
    img_equ = img_8u
    if high_int_threshold < 1.:
        high_th = int(img_8u.max()*high_int_threshold)
    else:
        high_th = int(high_int_threshold)
    maxval = max_pix_val(img_8u.dtype)
    _, img_bin = cv2.threshold(img_equ, high_th, 
                                maxval=maxval, type=cv2.THRESH_BINARY)
    
    
    # pect_marker_img = np.zeros(img_bin.shape, dtype=np.int32)
    # # Sure foreground (shall be pectoral).
    # pect_mask_init = select_largest_obj(img_bin, lab_val=maxval, 
    #                                             fill_holes=True, 
    #                                             smooth_boundary=False)
    
    kernel_ = np.ones((morph_kn_size, morph_kn_size), dtype=np.uint8)
    pect_mask_eroded = cv2.erode(img_bin, kernel_, 
                                    iterations=n_morph_op)
    # show_img(pect_mask_eroded)
    idx,cont_areas,contours = get_max_connected_area(pect_mask_eroded)                                
    pectoral = cv2.drawContours(
        np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.
    
    pectoral = 255-pectoral
    # pect_marker_img[pect_mask_eroded > 0] = 255
    # # Sure background - breast.
    # pect_mask_dilated = cv2.dilate(img_bin, kernel_, 
    #                                 iterations=n_morph_op)
    # show_img(pect_mask_dilated)
    # pect_marker_img[pect_mask_dilated == 0] = 128
    # # Sure background - pure background.
    # pect_marker_img[breast_mask == 0] = 64
    # # Watershed segmentation.
    # img_equ_3c = cv2.cvtColor(img_equ, cv2.COLOR_GRAY2BGR)
    # cv2.watershed(img_equ_3c, pect_marker_img)
    # img_equ_3c[pect_marker_img == -1] = (0, 0, 255)
    # # Extract only the breast and smooth.
    # breast_only_mask = pect_marker_img.copy()
    # breast_only_mask[breast_only_mask == -1] = 0
    # breast_only_mask = breast_only_mask.astype(np.uint8)
    # breast_only_mask[breast_only_mask != 128] = 0
    # breast_only_mask[breast_only_mask == 128] = 255
    # kernel_ = np.ones((sm_kn_size, sm_kn_size), dtype=np.uint8)
    # breast_only_mask = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, 
    #                                     kernel_)
    img_breast_only = cv2.bitwise_and(img,img,mask = pectoral)

    return img_breast_only

def clahe(img,max_pixel_val = 255,clip=2.0, tile=(8, 8)):
    
    '''
    This function applies the Contrast-Limited Adaptive
    Histogram Equalisation filter to a given image.
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to edit.
    clip : {int or floa}
        Threshold for contrast limiting.
    tile : {tuple (int, int)}
        Size of grid for histogram equalization. Input
        image will be divided into equally sized
        rectangular tiles. `tile` defines the number of
        tiles in row and column.
    
    Returns
    -------
    clahe_img : {numpy.ndarray}
        The edited image.
    '''
    
    # Convert to uint8.
    # img = skimage.img_as_ubyte(img)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=max_pixel_val,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    if max_pixel_val == 255:
        img = img.astype("uint8")
    if max_pixel_val == 65535:
        img = img.astype("uint16")
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img)

    return clahe_img

def crop_borders(img,border_size=(0.01,0.04,0.01,0.04)):
    
    '''
    This function crops 1% from all four sides of the given
    image.
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.
        
    Returns
    -------
    cropped_img: {numpy.ndarray}
        The cropped image.
    '''
    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * border_size[0])
    r_crop = int(ncols * (1 - border_size[1]))
    u_crop = int(nrows * border_size[2])
    d_crop = int(nrows * (1 - border_size[3]))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    
    return cropped_img

def horizontal_flip(img, mask):
    
    '''
    This function figures out how to flip (also entails whether
    or not to flip) a given mammogram and its mask. The correct
    orientation is the breast being on the left (i.e. facing
    right) and it being the right side up. i.e. When the
    mammogram is oriented correctly, the breast is expected to
    be found in the bottom left quadrant of the frame.
    
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The corresponding mask of the CC image to flip.

    Returns
    -------
    horizontal_flip : {boolean}
        True means need to flip horizontally,
        False means otherwise.
    '''
    
    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2
    
    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)
    
    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])
    top_sum = sum(row_sum[0:y_center])
    bottom_sum = sum(row_sum[y_center:-1])
    
    if left_sum < right_sum:
        img = cv2.flip(img, 1)

    return img

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
    return (img * 255).astype(np.float32)

def convert_to_16bit(img):
    if np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    return (img * 65535).astype(np.float32)

def linear_nor(img):
    if np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    return img

def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False, rescale_factor=None,
                    ) -> np.float32:
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    # if target_size is None and target_height is None:
    #     raise Exception('One of [target_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = pydicom.dcmread(fname).pixel_array
    else:
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    # if gs_255:
    #     img = convert_to_8bit(img)
    # else:
    #     img = convert_to_16bit(img)

    if target_height is not None:
        target_width = int(float(target_height)/img.shape[0]*img.shape[1])
    elif target_size is not None:
        target_height, target_width = target_size
    else:
        target_height, target_width = img.shape
    if (target_height, target_width) != img.shape:
        img = cv2.resize(
            img, dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)            
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

def segment_breast(img, low_int_threshold=0.05, crop=True, erosion= False):
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
    return img_breast_only, (x,y,w,h), breast_mask

def overlap_patch_roi(patch_center, patch_size, roi_mask, 
                     cutoff=.5):
    black_img = np.zeros(roi_mask.shape)
    black_img[int(patch_center[1] - patch_size/2):patch_center[1] + int(patch_size/2), 
                int(patch_center[0]- patch_size/2):patch_center[0] + int(patch_size/2)] = 255


    # cv2.imshow("grayscale image", black_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("grayscale image", roi_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    roi_area = (roi_mask>0).sum()
    patch_area = (black_img>0).sum()

    inter_area = (roi_mask*black_img>0).sum()
    # print(inter_area, roi_area, patch_area, str(inter_area/roi_area > cutoff or inter_area/patch_area > cutoff))
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