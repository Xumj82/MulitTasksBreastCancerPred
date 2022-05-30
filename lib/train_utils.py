
import os
import cv2
import random
from pathlib2 import Path
from sklearn import metrics
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split

import os
import PIL
import torch
import torch.nn as nn


def resize_keep_aspectratio(image_src,dst_size):
    src_h,src_w = image_src.shape[:2]
    # print(src_h,src_w)
    dst_h,dst_w = dst_size 
    
    #判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h)/src_w)#按照ｗ做等比缩放
    w = dst_h * (float(src_w)/src_h)#按照h做等比缩放
    
    h = int(h)
    w = int(w)
    
    if h <= dst_h:
        image_dst = cv2.resize(image_src,(dst_w,int(h)))
    else:
        image_dst = cv2.resize(image_src,(int(w),dst_h))
    
    h_,w_ = image_dst.shape[:2]
    # print(h_,w_)
    
    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_+1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_+1) / 2)
    
    value = [0,0,0]
    borderType = cv2.BORDER_CONSTANT
    # print(top, down, left, right)
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
 
    return image_dst

def show_grid(images, row=3, col=3):
    fig = plt.figure(figsize=(col*5, row*5))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(col, row),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
def show_img(path, resize=False):
    if resize :
        img = resize_keep_aspectratio(cv2.imread(path),dst_size=(512,512))
    else:
        img = cv2.imread(path,0)
    # img = cv2.resize(cv2.imread(path), dsize=(512,512))
    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap='bone')

def masked_img(img, kernal_size = (50,50), iter = 2, ):
    while(iter):
        (_,thr) = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
        kernel  = np.ones(kernal_size,np.uint8)
        morphimg = cv2.morphologyEx(np.uint8(thr),cv2.MORPH_OPEN, kernel)
        img = cv2.bitwise_and(img, img, mask=morphimg)
        iter -= 1
    return img

def show_random_masked(patients,dataset,index= -1,row=2,col=2):
    images = []
    while True:
        patient=np.random.choice(patients,1)[0]
        side = "RIGHT"
        row_CC = dataset[(dataset.patient_id == patient) & (dataset['image view'] == "CC") &  (dataset['left or right breast'] == "RIGHT")]
        row_MLO = dataset[(dataset.patient_id == patient) & (dataset['image view'] == "MLO")&  (dataset['left or right breast'] == "RIGHT")]
        if(row_CC.shape[0] <= 0 or row_MLO.shape[0]<= 0):
            side = "LEFT"
            row_CC = dataset[(dataset.patient_id == patient) & (dataset['image view'] == "CC") &  (dataset['left or right breast'] == "LEFT")]
            row_MLO = dataset[(dataset.patient_id == patient) & (dataset['image view'] == "MLO")&  (dataset['left or right breast'] == "LEFT")]
        if(row_CC.shape[0] > 0 and row_MLO.shape[0] > 0):
            break
    
    print(patient, side)
    
    
    if row_CC.shape[0] > 0:
        print(row_CC.image_path.iloc[0])
        img_CC = cv2.imread(row_CC.image_path.iloc[0], 0)
        img_CC = resize_keep_aspectratio(img_CC,(512,512))
        masked_CC = masked_img(img_CC)
        images.append(img_CC)
        images.append(masked_CC)
    else:
        print("No CC image for {}".format(patient))
        return
        
    if row_MLO.shape[0] > 0:
        print(row_MLO.image_path.iloc[0])
        img_MLO = cv2.imread(row_MLO.image_path.iloc[0], 0)
        img_MLO = resize_keep_aspectratio(img_MLO,(512,512))
        masked_MLO = masked_img(img_MLO)
        images.append(img_MLO)
        images.append(masked_MLO)
    else:
        print("No MLO image for {}".format(patient))
        return
    print("Y-CC:{} Y-MLO:{}".format(row_CC.Y.iloc[0], row_MLO.Y.iloc[0]))

    fig = plt.figure(figsize=(col*5, row*5))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(col, row),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def train_test_split_on_patient(data_list ,test_size=0.1, random_state = 45):
    patients = []
    for index, row in data_list.iterrows():
        patientid = row['img_id'].split('_')[1] + row['img_id'].split('_')[2]
        patients.append(patientid)
    data_list['PatientID'] = patients

    # data_train, data_test = train_test_split(data_list, test_size=test_size, random_state=45)
    patientdict = list( dict.fromkeys(patients) )
    traindict, testdict = train_test_split(patientdict, test_size=test_size, random_state=random_state)
    traindict = pd.DataFrame(data=traindict, columns=["PatientID"])
    testdict = pd.DataFrame(data=testdict, columns=["PatientID"])
    data_train = pd.merge(traindict, data_list,  on ='PatientID', how ='left')
    data_test = pd.merge(testdict, data_list,  on ='PatientID', how ='left')
    return data_train,data_test

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(root=args['TRAIN']['load_dir'], version=args['TRAIN']['load_ver'], v_num=args['TRAIN']['load_v_num'], best=args['TRAIN']['load_best'])

def plot_roc( model_name, fprs, tprs,thresholds, classes):
    # classes = ['backgound','mass','calcification']
    plt.figure()
    for i in range(len(classes)):
        fpr = fprs[i].cpu()
        tpr = tprs[i].cpu()
        thr = thresholds[i].cpu()
        auc = metrics.auc(fpr, tpr)
        print(max(thr))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_sensitive = tpr[optimal_idx]
        optimal_specificity = 1-fpr[optimal_idx]
        optimal_threshold = thr[optimal_idx]
        plt.plot(
            fpr,
            tpr,
            color=(random.random(),random.random(),random.random()),
            # color=(0,0,0),
            lw=2,
            label="{0} (area = {1:0.2f}) ".format(classes[i],auc),
        )
        print("type:{0}  sensitive:{1:0.2f} specificity:{2:0.2f} threshold:{3:0.2f}".format(classes[i],optimal_sensitive,optimal_specificity,optimal_threshold))

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig('asset/{}_roc_curve.png'.format(model_name))

def dir_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
            center = factor - 1
    else:
            center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b

def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c)>0:
        for l in c:
            apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def load_pretrained(pretrained, model):
    print(f"==============> Loading weight {pretrained} for fine-tuning......")
    checkpoint = torch.load(pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    print(f"=> loaded successfully '{pretrained}'")

    del checkpoint
    torch.cuda.empty_cache()
