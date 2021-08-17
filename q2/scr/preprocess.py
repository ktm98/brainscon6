import os
import math
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

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

import warnings 
warnings.filterwarnings('ignore')


def map_location(location):
    # 0 1 
    # 2 3
    
    # 上
    if location <= 126:
        # 左上
        if (location - 1) % 18 < 9:
            return 0
        # 右上
        else:
            return 1
    else:
        # 左下
        if (location - 1) % 18 < 9:
            return 2
        # 右下
        else:
            return 3

def preprocess_df(df):
    df['base_image'] = df['id'].apply(lambda x: int(x.split('/')[2][:4]))
    df['location'] = df['id'].apply(lambda x: int(re.search(r'\d{4}-\d{1,3}', x).group()[5:]))
    df['location_id'] = df['location'].apply(map_location)
    df['row_id'] = df['location'].apply(lambda x: (x-1) // 18)
    df['col_id'] = df['location'].apply(lambda x: (x-1) % 18)
    df['patch_id'] = df.apply(lambda row: row['row_id'] % 7 * 9 + row['col_id'] % 9, axis=1)
    return df

def preprocess(train_df, test_df):
    train_df = preprocess_df(train_df)
    test_df = preprocess_df(test_df)

    train_df['train'] = 1
    test_df['train'] = 0

    for i, b_img in enumerate(train_df['base_image'].unique()):
        # _df_tr = train_df.loc[train_df['base_image']==b_img]
        _df_te = test_df.loc[test_df['base_image']==b_img]
        _test_location_id = _df_te['location_id'].values[0]

        
        train_df.loc[(train_df['base_image']==b_img) & (train_df['location_id']==3), 'location_id'] = _test_location_id
        train_df.loc[train_df['base_image']==b_img, 'location_id'] = (train_df.loc[train_df['base_image']==b_img, 'location_id'] + i) % 3

    return train_df, test_df
    

