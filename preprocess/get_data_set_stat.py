import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from lib.preprocess_utils import convert_to_16bit

img_dir = '/media/xumingjie/study/datasets/ddsm_set/img_dir/'
img_list = glob(img_dir+'/*/*.png')

psum = 0
pcount = 0

with tqdm(total=len(img_list)) as pbar:
    for img_file in img_list:
        img = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH)
        img = img - np.min(img)
        img = img / np.max(img)
        psum += img.sum()
        pcount += img.shape[0]*img.shape[1]
        pbar.update(1)

pmean = psum/pcount
# pmean = 15179.378704950348
print('data set mean value',pmean)

p_var_sum = 0

with tqdm(total=len(img_list)) as pbar:
    for img_file in img_list:
        img = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH)

        img = img - np.min(img)
        img = img / np.max(img)

        p_var_sum += ((img-pmean)**2).sum()
        pcount += img.shape[0]*img.shape[1]
        pbar.update(1)

pvar = p_var_sum/pcount


print('data set var',pvar)
print('data set std',pvar**0.5)

