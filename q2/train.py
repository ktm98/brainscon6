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
from scr.utils import seed_everything, AverageMeter, get_score, asMinutes, timeSince, fix_model_state_dict, multi_getattr
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

def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):
    if cfg['apex']:
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0

    input_list, output_list, lams = [], [], []

    for step, (images, labels) in enumerate(train_loader):
        batch_size = labels.size(0)
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)
        if not isinstance(criterion, nn.CrossEntropyLoss):
            labels = labels.reshape(batch_size, -1)


        

        if cfg['augmentation']['mixup']['use']:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=cfg['augmentation']['mixup']['alpha'])


        if cfg['apex']:
            with autocast():
                y_preds = model(images)
                if cfg['augmentation']['mixup']['use']:
                    loss_fn = mixup_criterion(labels_a, labels_b, lam)
                    loss = loss_fn(criterion, y_preds)  # view
                else:
                    loss = criterion(y_preds, labels)  # view
        else:
            y_preds = model(images)
            if cfg['augmentation']['mixup']['use']:
                loss_fn = mixup_criterion(labels_a, labels_b, lam)
                loss = loss_fn(criterion, y_preds)  # view
            else:
                loss = criterion(y_preds, labels)  # view
        # record loss
        if (cfg['optimizer']['use_sam']) and (cfg['gradient_accumulation_steps']>1):
            input_list.append(images)
            
            if cfg['augmentation']['mixup']['use']:
                output_list.append([labels_a, labels_b])
                lams.append(lam)
            else:
                output_list.append(labels)

        losses.update(loss.item(), batch_size)
        if cfg['gradient_accumulation_steps'] > 1:
            loss = loss / cfg['gradient_accumulation_steps']
        if cfg['apex'] and (not cfg['optimizer']['use_sam']):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
        if (step + 1) % cfg['gradient_accumulation_steps'] == 0:
            if cfg['apex']:
                if cfg['optimizer']['use_sam']:
                    optimizer.first_step(zero_grad=True)
                    # scaler.step(optimizer.first_step)
                    # scaler.update()
                    for i in range(len(input_list)):
                        with autocast():
                            pred =  model(input_list[i])
                        if cfg['augmentation']['mixup']['use']:
                            loss_fn = mixup_criterion(output_list[i][0], output_list[i][1], lams[i])
                            loss = loss_fn(criterion, y_preds)  # view
                        else:
                            loss = criterion(pred, output_list[i])
                        loss = loss / cfg['gradient_accumulation_steps']
                        # scaler.scale(loss).backward()
                        loss.backward()
                    optimizer.second_step(zero_grad=True)
                    # scaler.step(optimizer.second_step)
                    # scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if cfg['optimizer']['use_sam']:
                    optimizer.first_step(zero_grad=True)
                    y_preds = model(images)
                    if cfg['augmentation']['mixup']['use']:
                        loss_fn = mixup_criterion(labels_a, labels_b, lam)
                        loss = loss_fn(criterion, y_preds)  # view
                    else:
                        loss = criterion(y_preds, labels)  # view
                    loss.backward()
                    # loss.backward()
                    optimizer.second_step(zero_grad=True)  
                else:
                    optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg['print_freq'] == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        batch_size = labels.size(0)
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        if not isinstance(criterion, nn.CrossEntropyLoss):
            labels = labels.reshape(batch_size, -1)

        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if cfg['gradient_accumulation_steps'] > 1:
            loss = loss / cfg['gradient_accumulation_steps']
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg['print_freq'] == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def train_loop(folds, fold, cfg):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[cfg['target']].values 



    train_target = train_folds[cfg['target']].values.astype(np.float32)
    valid_target = valid_folds[cfg['target']].values.astype(np.float32)






    # target_type = np.int32 if CFG.criterion == 'CrossEntropyLoss' else np.float32



    train_dataset = ImageDataset(train_folds['image'].values, train_target, 
                                 transform=get_transforms(data='train', cfg=cfg))
    valid_dataset = ImageDataset(valid_folds['image'].values, valid_target, 
                                 transform=get_transforms(data='valid', cfg=cfg))

    train_loader = DataLoader(train_dataset, 
                            batch_size=cfg['batch_size'], 
                            shuffle=True, 
                            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=cfg['batch_size'] * 2, 
                            shuffle=False, 
                            num_workers=cfg['num_workers'], pin_memory=True, drop_last=False)


    def get_optimizer(model):
        if cfg['optimizer']['name'] == 'Adam':
            optimizer =  Adam
        elif cfg['optimizer']['name'] == 'SGD':
            optimizer = SGD
        elif cfg['optimizer']['name'] == 'AdamW':
            optimizer = AdamW
        elif cfg['optimizer']['name'] == 'Ranger':
            optimizer = Ranger
        else:
            LOGGER.info(f'Optimizer {cfg["optimizer"]["name"]} is not implementated')
        
        if cfg['optimizer']['use_sam']:
            return SAM(param_group, optimizer, **cfg['optimizer']['params'])
        else:
            return optimizer(param_group, **cfg['optimizer']['params'])


    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if cfg['scheduler']['name']=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', **cfg['scheduler']['params'])
        elif cfg['scheduler']['name']=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, last_epoch=-1, **cfg['scheduler']['params'])
        elif cfg['scheduler']['name']=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_mult=1, last_epoch=-1, **cfg['scheduler']['params'])
        # elif cfg['scheduler']['name'] == 'GradualWarmupSchedulerV2':
        #     scheduler_cosine = CosineAnnealingLR(optimizer, T_max=CFG.cosine_epochs - CFG.warmup_epochs, eta_min=CFG.min_lr, last_epoch=-1)
        #     scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=CFG.multiplier, total_epoch=CFG.warmup_epochs, after_scheduler=scheduler_cosine)
        elif cfg['scheduler']['name'] == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, max_lr=cfg['scheduler']['params']['max_lr'],
                                    pct_start=cfg['scheduler']['params']['pct_start'],
                                    div_factor=cfg['scheduler']['params']['div_factor'],
                                    epochs=cfg['epochs'],
                                    steps_per_epoch=math.ceil(len(train_loader)/cfg['gradient_accumulation_steps']))
        else:
            LOGGER.info(f'Scheduler {cfg["scheduler"]["name"]} is not implementated')
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(cfg, pretrained=False, target_size=1)
    model.to(DEVICE)

    if 'pram_group' in cfg['optimizer'].keys():
        param_group = {
            [dict({'params': multi_getattr(model, k).parameters(), **v}) for k, v in cfg['optimizer']['param_group'].items()]
        }
    else:
        param_group = [
            {'params': model.model.parameters(), 'lr': cfg['optimizer']['params']['lr']/50},
            {'params': model.fc.parameters()}
        ]

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================

    criterion = getattr(nn, cfg['loss_fn'])()

    best_score = 0
    best_loss = np.inf
    
    for epoch in range(cfg['epochs']):
        
        start_time = time.time()
        
        # train
        if (epoch + 1) / cfg['epochs'] > 0.90 and 'train_weak' in cfg['augmentation']['transform'].keys():

            train_dataset = ImageDataset(train_folds['image'].values, train_target, 
                                transform=get_transforms(data='train_weak', cfg=cfg))
            train_loader = DataLoader(train_dataset, 
                            batch_size=cfg['batch_size'], 
                            shuffle=True, 
                            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)


        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, DEVICE, cfg)

        # eval

        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, DEVICE, cfg)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, GradualWarmupSchedulerV2):
            scheduler.step()

        # scoring

        score, thresh = get_score(valid_labels, preds)

        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} f1_score: {score:.4f} thresh: {thresh:.4f}  time: {elapsed:.0f}s')

        if score > best_score:
            best_score = score

            torch.save({'model': model.state_dict(), 
                    'preds': preds},
                    OUTPUT_DIR+f'{ cfg["model_name"]}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
                
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{cfg["model_name"]}_fold{fold}_best_loss.pth')


    valid_preds = torch.load(OUTPUT_DIR+f'{cfg["model_name"]}_fold{fold}_best_loss.pth', 
                                        map_location=torch.device('cpu'))['preds']
    valid_folds['probs'] = valid_preds


    return valid_folds




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
    
    LOGGER = init_logger(OUTPUT_DIR+'train.log')
    LOGGER.info(config)
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    train_df = pd.read_csv('../input/Q2/train_label.csv', names=['id', 'label'])
    test_df = pd.read_csv('../input/Q2/sample_submission.csv', names=['id', 'label'])

    IMAGE_DIR = '../input/SDNET2018/'
    train_df['image'] = train_df['id'].apply(lambda x: IMAGE_DIR + x)
    test_df['image'] = test_df['id'].apply(lambda x: IMAGE_DIR + x)


    train_df, test_df = preprocess(train_df, test_df)
    train_df = make_folds(train_df, shuffle=True, random_state=0)

    if 'pseudo_label' in config.keys():
        if config['pseudo_label']:
            pseudo_df = pd.read_csv(f'./exp/{config["pseudo_exp"]}/test_df_pp.csv')
            pseudo_labels = pseudo_df['pred'].values

            _test_df = test_df.copy()
            _test_df['label'] = pseudo_labels
            _test_df['fold'] = -1
            train_df = pd.concat([train_df, _test_df])

    if config['debug']:
        train_df = train_df.head(1000)
        config['epochs'] = 1

    oof_df = pd.DataFrame()
    for fold in range(config['n_fold']):
        if fold in config['trn_fold']:
            _oof_df = train_loop(train_df, fold, config)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f'============ fold {fold} result ==========')
            get_result(_oof_df)
    LOGGER.info(f'============== CV =============')
    thresh = get_result(oof_df)
    oof_df['preds'] = 1 - (oof_df['probs'] < thresh)

    oof_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)
