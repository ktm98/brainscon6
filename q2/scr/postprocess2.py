import os
import math
import sys
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

from tqdm.auto import tqdm
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import glob
import re
from skimage.util import view_as_blocks
from sklearn.metrics import f1_score
import torch

from scr.preprocess import preprocess
from scr.utils import get_score


import warnings 
warnings.filterwarnings('ignore')

# ref: https://note.nkmk.me/python-opencv-hconcat-vconcat-np-tile/
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def get_preds_and_image(df, base_image):
    _df = df.loc[df['base_image']==base_image]
    
    _df = _df.sort_values('location').reset_index()
    images = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(252)]
    # masks = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(252)]
    idxs = np.array([None for _ in range(252)])
    preds = np.zeros(252)
    probs = np.zeros(252)
    for idx in _df.index:
        row = _df.iloc[idx]
        
#         mask_path = row['seg_mask_pred']
        path = row['image']
        
        loc = row['location']
        if '_' in path[-8:-4]:
            continue
            
        # masks_ = []
        # for exp in exps:
        #     mask_path = row[f'{exp}_mask']
        #     msk = cv2.imread(mask_path)[:, :, ::-1]
        #     masks_.append(msk)
#         if None in masks_:
#             continue
        # mask = np.mean(masks_, axis=0)
#         mask = cv2.imread(mask_path)[:, :, ::-1]
        img = cv2.imread(path)[:, :, ::-1]
        # if row['train'] == 0:
        #     img[:, :, 2] = 255
        # elif row['label'] == 1:
        #     img[:, :, 1] = 255
#         if row['preds'] == 1 :
#             img[:, :, 0] = 255
        images[loc - 1] = img
        # masks[loc - 1] = mask
        idxs[loc - 1] = row['id']
        preds[loc - 1] = row['preds']
        probs[loc - 1] = row['probs']
    images = np.array(images).reshape(14, 18, 256, 256, 3)
    # masks = np.array(masks).reshape(14, 18, 256, 256, 3)
    images = concat_tile(images)
    # masks = concat_tile(masks)
#     fig = plt.figure(figsize=(9, 4))
#     axes = fig.subplots(1, 2)
#     axes[0].imshow(images)
#     axes[1].imshow(masks)
# #     plt.imshow(images)
#     plt.show()
    preds = preds.reshape(14, 18)
    probs = probs.reshape(14, 18)
    return idxs, preds, images, probs


def post_process(preds, images, probs, thresh=50, iterations=5, count_thresh=200):
    # masks = np.mean(masks, axis=2)
    
    # masks = (masks > thresh).astype(np.uint8)
    preds = preds.astype(np.uint8)
    assert preds.shape == (14, 18)

    images = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    assert images.dtype == np.uint8
    masks = np.zeros_like(images)

    
    probs_ = torch.nn.ReplicationPad2d(1)(torch.tensor(probs.reshape(1, 1, 14, 18))).float()
    # print(probs)
    probs2 = torch.nn.functional.conv2d(probs_**2,
                                weight=torch.tensor( [[[[0.0,  1.0,  0.0],
                                                        [1.0,  0.0,  1.0],
                                                        [0.0,  1.0,  0.0]]]]).float(),
                                                        stride=1)
    # print(probs2.numpy().reshape(14, 18))
    probs += probs2.numpy().reshape(14, 18) / 20
    



    return preds, masks, probs

def main(exp):

    train_df = pd.read_csv(f'./exp/{exp}/oof_df.csv')
    train_df = train_df.loc[train_df['fold']>=0].reset_index()
    test_df = pd.read_csv(f'./exp/{exp}/test_df.csv')
    data_df = pd.concat([train_df, test_df])

    idxs = []
    preds = []
    probs = []
    for i, base_img in enumerate(data_df['base_image'].unique()):
        
        # fig = plt.figure(figsize=(14, 7))
        # axes = fig.subplots(1, 3)
        
        idx, preds_, image, prob = get_preds_and_image(data_df, base_img)
        
#         axes[1].imshow()
        
        preds_, masks, prob = post_process(preds_, image, prob)
        idxs.append(idx)
        preds.append(preds_.reshape(-1))
        probs.append(prob.reshape(-1))
        
        # axes[0].imshow(image)
        # axes[1].imshow(preds_)
        # axes[2].imshow(masks)
        # plt.show()
        
#         if i > 10: break
    
    idxs = np.concatenate(idxs)
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    
    new_df = pd.DataFrame()
    new_df['id'] = idxs
    # new_df['pred'] = preds
    new_df['probs_'] = probs

    # print(new_df['id'].values.shape)
    new_df = new_df.dropna(subset=['id'])
    

    _df = pd.merge(data_df, new_df, on='id', how='left')


    # _df['pred'] = _df['pred'].fillna(0)
    _df['probs_'] = _df['probs_'].fillna(0)
    _df['probs_'] = _df['probs_'] / _df['probs_'].max()
    target = _df.loc[_df['train']==1, ['label']]
    pred_ = _df.loc[_df['train']==1, ['probs_']]
    threshs = []
    for fold in range(3):
        score, thresh = get_score(target.loc[_df['fold']==fold, 'label'].values, pred_.loc[_df['fold']==fold, 'probs_'].values)
        threshs.append(thresh)
        print(f'fold {fold}: score: {score} thresh: {thresh}')
    threshs = np.mean(threshs) - 0.15
    # score = f1_score(target, pred_)
    # print(f'score: {score}')
    _df['preds_'] = 1 - (_df['probs_'].values.reshape(-1) < threshs)
 

    OUTPUT_DIR = f'./exp/{exp}/'
    _df.loc[_df['train']==1].to_csv(OUTPUT_DIR + 'oof_df_pp2.csv', index=False)

    _df.loc[_df['train']==0, ['id', 'preds_']].to_csv(OUTPUT_DIR + 'submission_pp2.csv', index=False, header=None)

    _df.loc[_df['train']==0].to_csv(OUTPUT_DIR + 'test_df_pp2.csv', index=False)

    
   
if __name__ == '__main__':
    exp = sys.argv[1]
    main(exp)