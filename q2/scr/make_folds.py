import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob
import re

from sklearn.model_selection import StratifiedKFold

from .preprocess import preprocess

import warnings 
warnings.filterwarnings('ignore')

def make_folds(df, shuffle=True, random_state=None):
    '''
    この関数はcross validationをするときのfoldを分割する関数です。
    分割は
    1. 各foldでのlabelの割合がなるべく均等になる
    2. 分割前の画像(base_image)の左上、左下、右上、右下の4つの領域のうちテスト領域を除いた3領域が別のfoldに入るように調整
    するように分割します。
    そうするために、各base_imageで3領域のlabelが1の数を求め、累積label数が最小のfoldに、そのbase_imageのlabel数が最大だった領域を割り当てます。
    同様に、累積label数が最大だった場合は、label数が最小の領域を割り当てます。

    '''
    base_images = df['base_image'].unique()
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(base_images)
    
    fold_counts = np.array([0, 0, 0])
    fold_index = [[], [], []]
    for i, b_img in enumerate(base_images):
        _df = df[df['base_image']==b_img]
        num_labels = np.array([0, 0, 0])
        for j in range(3):
            num_label = _df.loc[_df['location_id']==j, 'label'].sum()
            num_labels[j] = num_label
        fold_order = np.argsort(fold_counts)  # ラベルが一番少ないfoldにnum_labelsが一番大きい値を入れる
        num_label_index = np.argsort(-num_labels)
        
        num_labels_in_folds = np.array([0, 0, 0])
        for j in range(3):
            fold = fold_order[j]
            location_id = num_label_index[j]
            num_labels_in_folds[fold] = num_labels[location_id]
            
            fold_index[fold].append(_df.loc[_df['location_id']==location_id].index.tolist())
            
        fold_counts += num_labels_in_folds
#         print('num: ', fold_counts)
            

    df['fold'] = -1
    
    
    for fold, fi in enumerate(fold_index):
        fold_idx = np.concatenate(fi)
        df.loc[fold_idx, 'fold'] = fold

    
    return df
        
if __name__ == '__main__':
    train_df = pd.read_csv('../input/Q2/train_label.csv', names=['id', 'label'])
    test_df = pd.read_csv('../input/Q2/sample_submission.csv', names=['id', 'label'])

    IMAGE_DIR = '../input/SDNET2018/'
    train_df['image'] = train_df['id'].apply(lambda x: IMAGE_DIR + x)
    test_df['image'] = test_df['id'].apply(lambda x: IMAGE_DIR + x)


    train_df, test_df = preprocess(train_df, test_df)
    train_df = make_folds(train_df, shuffle=True, random_state=0)

    for fold in range(3):
        count_labels = train_df.loc[train_df['fold']==fold, 'label'].sum()
        print(f'fold {fold}: {count_labels}')

    train_df.to_csv('../input/Q2/folds.csv', index=False)