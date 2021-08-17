import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import argparse
import yaml

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import pickle

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW, Optimizer
import torchvision.models as models
from torchvision import transforms as T
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

from torch.cuda.amp import autocast, GradScaler

# import lightly

from scr.preprocess import preprocess
from scr.make_folds import make_folds
from scr.optimizer import Ranger, SAM
from scr.scheduler import GradualWarmupSchedulerV2
from scr.utils import seed_everything, AverageMeter, get_score, asMinutes, timeSince, fix_model_state_dict
from scr.mixup import mixup_data, mixup_criterion
from scr.dataset import ImageDataset
from scr.model import CustomModel
from scr.transform import get_transforms


import warnings 
warnings.filterwarnings('ignore')


def get_args():
    # 引数の導入
    parser = argparse.ArgumentParser(description='YAMLありの例')
    parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
    args = parser.parse_args()
    return args


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            y_preds = torch.sigmoid(y_preds)
            avg_preds.append(y_preds.to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


if __name__ == '__main__':

    def get_result(result_df):
        preds = result_df['probs'].values
        labels = result_df[config['target']].values
        score, thresh = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}, thresh: {thresh}')
        return thresh

    args = get_args()

    exp = args.config_path[-10:-4]
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(config['seed'])
    
    OUTPUT_DIR = f'./exp/{exp}/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    EXP_DIR = OUTPUT_DIR
    
    LOGGER = init_logger(OUTPUT_DIR+'train.log')
    LOGGER.info('================ inference =========================')
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    train_df = pd.read_csv('../input/Q2/train_label.csv', names=['id', 'label'])
    test_df = pd.read_csv('../input/Q2/sample_submission.csv', names=['id', 'label'])

    IMAGE_DIR = '../input/SDNET2018/'
    train_df['image'] = train_df['id'].apply(lambda x: IMAGE_DIR + x)
    test_df['image'] = test_df['id'].apply(lambda x: IMAGE_DIR + x)


    train_df, test_df = preprocess(train_df, test_df)
    train_df = make_folds(train_df, shuffle=True, random_state=0)

    folds = pd.read_csv(EXP_DIR + 'oof_df.csv')
    
    LOGGER.info('============= CV =================-')
    thresh = get_result(folds)


    model = CustomModel(config, pretrained=False, target_size=1)
    MODEL_DIR = EXP_DIR
    states = [torch.load(MODEL_DIR+f'{config["model_name"]}_fold{fold}_best_loss.pth') for fold in config['trn_fold']]
    test_dataset = ImageDataset(test_df['image'].values, transform=get_transforms(data='valid', cfg=config))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=config['num_workers'], pin_memory=True)
    proba = inference(model, states, test_loader, DEVICE)
    predictions =  1 - (proba.reshape(-1) < thresh)

    test_df['preds'] = predictions
    test_df['probs'] = proba.reshape(-1)
    test_df.to_csv(OUTPUT_DIR + 'test_df.csv')
    test_df[['id', 'preds']].to_csv(OUTPUT_DIR + 'submission.csv', index=False, header=False)
