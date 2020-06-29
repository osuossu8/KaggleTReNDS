import numpy as np
import pandas as pd
import collections
import cv2
import datetime
import gc
import glob
import h5py
import logging
import math
import operator
import os 
import pickle
import random
import re
import sklearn
import scipy.signal
import scipy.stats as stats
import seaborn as sns
import string
import sys
import time
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from contextlib import contextmanager
from collections import Counter, defaultdict, OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from skimage import measure
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from torch.utils import model_zoo
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

sys.path.append("/usr/src/app/kaggle/trends-assessment-prediction")

EXP_ID = 'exp14'
import configs.config14 as config
import src.engine14 as engine
from src.model7 import resnet50, resnet50_medicalnet
from src.machine_learning_util import seed_everything, prepare_labels, timer, to_pickle, unpickle


SEED = 718
seed_everything(SEED)


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)


class TReNDSDataset:
    def __init__(self, df, target_cols, indices, 
                 loading_features,
                 fnc_features,
                 map_path,
                 is_train):
        self.df = df.iloc[indices]
        self.target = df.iloc[indices][target_cols+['Id']]
        self.loading_features = loading_features
        self.fnc_features = fnc_features
        self.map_path = map_path
        self.target_cols = target_cols
        self.is_train = is_train

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):

        IDX = self.df.iloc[item].Id        
        path = self.map_path + str(IDX)
        
        all_maps = h5py.File(path + '.mat', 'r')['SM_feature'][()]

        all_maps = all_maps[5:-5, 5:-5, 5:-5, 5:-5]

        x_max = all_maps.max(axis=0)
        x_std = all_maps.std(axis=0)
        x_sum = all_maps.sum(axis=0)    

        all_maps = np.stack([x_max, x_std, x_sum], 0)

        all_maps = normalize_minmax(all_maps)

        flip_aug = True
        if self.is_train:
            if flip_aug and np.random.rand() >= 0.5:
                all_maps = all_maps[:, :, ::-1, :].copy()


        targets = self.target[self.target.Id==IDX][self.target_cols].values

        return {
            'idx':torch.tensor(IDX, dtype=torch.float32),
            'features': torch.tensor(all_maps, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
        }


def run_one_fold(fold_id):

    fnc_df = pd.read_csv(config.FNC_PATH)
    loading_df = pd.read_csv(config.LOADING_PATH)
    labels_df = pd.read_csv(config.TRAIN_SCORES_PATH)


    fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
    df = fnc_df.merge(loading_df, on="Id")
    labels_df["is_train"] = True

    df = df.merge(labels_df, on="Id", how="left")

    df_test = df[df["is_train"] != True].copy()
    df_train = df[df["is_train"] == True].copy()

    target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

    df_train = df_train.dropna().reset_index(drop=True)

    num_folds = config.NUM_FOLDS
    kf = KFold(n_splits = num_folds, random_state = SEED)
    splits = list(kf.split(X=df_train))

    train_idx = splits[fold_id][0]
    val_idx = splits[fold_id][1]

    print(len(train_idx), len(val_idx))


    train_dataset = TReNDSDataset(df=df_train, target_cols=target_cols, indices=train_idx, 
                                  loading_features=loading_features,
                                  fnc_features=fnc_features,
                                  map_path=config.TRAIN_MAP_PATH,
                                  is_train=True)

    train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, 
                batch_size=config.TRAIN_BATCH_SIZE,
                num_workers=0, pin_memory=True)

    val_dataset = TReNDSDataset(df=df_train, target_cols=target_cols, indices=val_idx, 
                                loading_features=loading_features,
                                fnc_features=fnc_features,
                                map_path=config.TRAIN_MAP_PATH,
                                is_train=False)

    val_loader = torch.utils.data.DataLoader(
                val_dataset, shuffle=False, 
                batch_size=config.VALID_BATCH_SIZE,
                num_workers=0, pin_memory=True)

    del train_dataset, val_dataset
    gc.collect()


    device = config.DEVICE
    # params = {}
    # params['shortcut_type'] = 'A'
    model = resnet50()

    # https://github.com/Tencent/MedicalNet/blob/35ecd5be96ae4edfc1be29816f9847c11d067db0/model.py#L89
    net_dict = model.state_dict() 
    # $B0eNE%I%a%$%s(B
    pretrain = torch.load("inputs/pretrain/resnet_50.pth")
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys() and 'conv1' not in k}
    net_dict.update(pretrain_dict)

    model.load_state_dict(net_dict)
    print("pretrained model loaded !")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    patience = config.PATIENCE
    p = 0
    min_loss = 999
    best_score = -999

    for epoch in range(1, config.EPOCHS + 1):

        LOGGER.info("Starting {} epoch...".format(epoch))

        engine.train_fn(train_loader, model, optimizer, device, scheduler)
        score, val_loss, val_idices, val_preds = engine.eval_fn(val_loader, model, device)
        scheduler.step()

        if val_loss < min_loss:
            min_loss = val_loss
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config.OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
            LOGGER.info("val loss is {}".format(val_loss))
            LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))
            p = 0 

        if p > 0: 
            LOGGER.info(f'val loss is not updated while {p} epochs of training')
        p += 1
        if p > patience:
            LOGGER.info(f'Early Stopping')
            break

    to_pickle(os.path.join(config.OUT_DIR, '{}_fold{}.pkl'.format(EXP_ID, fold_id)), [val_idices, val_preds])
    LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))


if __name__ == '__main__':

    fold0_only = config.FOLD0_ONLY

    for fold_id in range(5):

        LOGGER.info("Starting fold {} ...".format(fold_id))

        run_one_fold(fold_id)

        if fold0_only:
            LOGGER.info("This is fold0 only experiment.")
            break