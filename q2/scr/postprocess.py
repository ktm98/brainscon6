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

from scr.preprocess import preprocess
import optuna

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
    return idxs, preds, images


def post_process(preds, images, thresh=50, iterations=5, count_thresh=200):
    # masks = np.mean(masks, axis=2)
    
    # masks = (masks > thresh).astype(np.uint8)
    preds = preds.astype(np.uint8)
    assert preds.shape == (14, 18)

    images = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    assert images.dtype == np.uint8
    masks = np.zeros_like(images)
    # 
#     neighborhood = np.array([[0, 1, 0],
#                              [1, 1, 1],
#                              [0, 1, 0]], dtype=np.uint8)
#     masks = cv2.erode(masks, neighborhood, iterations=iterations)
#     masks = cv2.dilate(masks, neighborhood, iterations=iterations)
    
    # masks = morphology.remove_small_objects(masks, min_size=20)
    # masks = morphology.remove_small_holes(masks, area_threshold=20)
        
    # masks = masks.astype(np.uint8)
    
    contours, hierarchy = cv2.findContours(preds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in contours:
        # area = cv2.contourArea(cont)

        # rect = cv2.minAreaRect(cont)
        
        # (cx, cy), (width, height), angle = rect
        cx, cy, width, height = cv2.boundingRect(cont)
        # box = np.int0(cv2.boxPoints(rect))
        # rate = np.sqrt(np.sum((box[1] - box[0])**2) / (np.sum((box[3] - box[0])**2)) )

        if width*height >4:
            preds[cy:cy+height, cx:cx+width] = 0
        elif width*height == 4:
            if preds[cy:cy+height, cx:cx+width].sum() == 2:
                preds[cy:cy+height, cx:cx+width] = 0
                # print('here')
            elif preds[cy:cy+height, cx:cx+width].sum() == 3:
                # 円を見つける
                img = images[256*cy:256*cy+256*height, 256*cx:256*cx+256*width]
                msk = np.zeros_like(img)
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.4, minDist=1000, param1=100, param2=20, minRadius=50, maxRadius=1000)
                if circles is None:
                    print('circles are not found! change the params')
                    continue
                    # preds[cy:cy+height, cx:cx+width] = 0
                else:
                   
                    radius_max_idx = np.argmax(circles[0, :, 2])
                    # for circle in circles[0, :]:
                        # 円周を描画する
                    cv2.circle(msk, (int(circles[0, radius_max_idx, 0]), int(circles[0, radius_max_idx, 1])), int(circles[0, radius_max_idx, 2]), (1, 1, 1), -1)
                        # 中心点を描画する
                        # cv2.circle(circle_mask, (int(circle[0]), int(circle[1])), 2, (255, 255, 255), 3)
                    prd0 = msk[:256, :256].sum()
                    prd1 = msk[:256, 256:].sum()
                    prd2 = msk[256:, :256].sum()
                    prd3 = msk[256:, 256:].sum()
                    prd  = np.array([[prd0, prd1],
                                     [prd2, prd3]])
                    prd = prd > 20
                    preds[cy:cy+height, cx:cx+width] = prd
                    masks[256*cy:256*cy+256*height, 256*cx:256*cx+256*width] = msk 
        elif width*height <=2 :
            img = images[256*cy:256*cy+256*height, 256*cx:256*cx+256*width]

            msk = np.zeros_like(img)
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.45, minDist=1000, param1=100, param2=20, minRadius=20, maxRadius=1000)
            if circles is None:
                preds[cy:cy+height, cx:cx+width] = 0
            else:
                for circle in circles[0, :]:
                    # 円周を描画する
                    cv2.circle(msk, (int(circle[0]), int(circle[1])), int(circle[2]), (1, 1, 1), -1)
                    # 中心点を描画する
                    # cv2.circle(circle_mask, (int(circle[0]), int(circle[1])), 2, (255, 255, 255), 3)

                prd = msk.sum() > 1000
                preds[cy:cy+height, cx:cx+width] = prd
                masks[256*cy:256*cy+256*height, 256*cx:256*cx+256*width] = msk 


#         if area < 40:
#             masks = cv2.drawContours(masks, [box], -1 ,(0, 0, 0), -1)
        # if area > 256*256*4:
        #     masks = cv2.drawContours(masks, [box], -1 ,(0,0,0), -1)
        # if rate < 1/2 or rate > 2:
        #     masks = cv2.drawContours(masks, [box], -1 ,(0,0, 0), -1)
            
    
    
#     mask = mask.reshape(mask.shape[0], mask.shape[1])
    # height, width = masks.shape[0], masks.shape[1]
    #     patch = mask.reshape(height//14, 14, width//18, 18).swapaxes(1, 2).reshape(-1, height//14, width//18)
    # patch = view_as_blocks(masks, (height//14, width//18)).reshape(-1, height//14, width//18)
    
    
    # count_masks = patch.sum(axis=(1, 2))
    
    # pred = np.zeros(252)
    
    # pred[count_masks>count_thresh] = 1
    # pred = pred.reshape(14, 18)
    
    # post process
#     contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for cont in contours:
# #             box = np.int0(cv2.boxPoints(rect))
#         x, y, w, h = cv2.boundingRect(cont)
#         area = w*h
# #             (cx, cy), (width, height), angle = rect

#         if area > 4:
#             pred[y:y+h, x:x+w] = 0
#         elif w/h >2 or w/h < 1/2:
#             pred[y:y+h, x:x+w] = 0
    return preds, masks

def main(exp):
#     train_df = pd.read_csv('../input/Q2/train_label.csv', names=['id', 'label'])
#     test_df = pd.read_csv('../input/Q2/sample_submission.csv', names=['id', 'label'])

#     IMAGE_DIR = '../input/SDNET2018/'
#     train_df['image'] = train_df['id'].apply(lambda x: IMAGE_DIR + x)
#     test_df['image'] = test_df['id'].apply(lambda x: IMAGE_DIR + x)

# #     train_df['seg_image'] = train_df['id'].apply(lambda x: '../input/train_patch_256/train_patch_256/W/' + x.split('/')[-1][:-3] + 'png')
# #     train_df['seg_mask'] = train_df['id'].apply(lambda x: '../input/train_patch_256/train_patch_256/W_masks/' + x.split('/')[-1][:-3] + 'png')

#     train_df, test_df = preprocess(train_df, test_df)

# #     train_df['seg_mask_pred'] = train_df['id'].apply(lambda x: EXP_DIR + 'mask_pred/' + x.split('/')[-1][:-3] + '.png')
# #     test_df['seg_mask_pred'] = test_df['id'].apply(lambda x: EXP_DIR + 'mask_pred/' + x.split('/')[-1][:-3] + '.png')

#     data_df = pd.concat([train_df, test_df])
#     for exp in exps:
#         data_df[f'{exp}_mask'] = data_df['id'].apply(lambda x: '../q2/exp/' + exp + '/mask_pred/' + x.split('/')[-1][:-3] + '.png')
    train_df = pd.read_csv(f'./exp/{exp}/oof_df.csv')
    test_df = pd.read_csv(f'./exp/{exp}/test_df.csv')
    data_df = pd.concat([train_df, test_df])

    idxs = []
    preds = []
    for i, base_img in enumerate(data_df['base_image'].unique()):
        
        # fig = plt.figure(figsize=(14, 7))
        # axes = fig.subplots(1, 3)
        
        idx, preds_, image = get_preds_and_image(data_df, base_img)
        
#         axes[1].imshow()
        # print(base_img)
        preds_, masks = post_process(preds_, image)
        idxs.append(idx)
        preds.append(preds_.reshape(-1))
        
        # axes[0].imshow(image)
        # axes[1].imshow(preds_)
        # axes[2].imshow(masks)
        # plt.show()
        
#         if i > 10: break
    
    idxs = np.concatenate(idxs)
    preds = np.concatenate(preds)
    
    new_df = pd.DataFrame()
    new_df['id'] = idxs
    new_df['pred'] = preds
    
    _df = data_df.merge(new_df, on='id', how='left')
    _df['pred'] = _df['pred'].fillna(0)
    target = _df.loc[_df['train']==1, 'label']
    pred_ = _df.loc[_df['train']==1, 'pred']
    score = f1_score(target, pred_)
    print(f'score: {score}')
 

    OUTPUT_DIR = f'./exp/{exp}/'
    _df.loc[_df['train']==1].to_csv(OUTPUT_DIR + 'oof_df_pp.csv', index=False)

    _df.loc[_df['train']==0, ['id', 'pred']].to_csv(OUTPUT_DIR + 'submission_pp.csv', index=False, header=None)

    _df.loc[_df['train']==0].to_csv(OUTPUT_DIR + 'test_df_pp.csv', index=False)

    
   
if __name__ == '__main__':
    exp = sys.argv[1]
    main(exp)