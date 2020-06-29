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

EXP_ID = 'exp13'
import configs.config13 as config
from src.model6 import resnet50_medicalnet
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
    def __init__(self, df, map_path):
        self.df = df
        self.map_path = map_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):

        IDX = self.df.iloc[item].Id        
        path = self.map_path + str(IDX)
        
        all_maps = h5py.File(path + '.mat', 'r')['SM_feature'][()]
        all_maps = normalize_minmax(all_maps)

        return {
            'features': torch.tensor(all_maps, dtype=torch.float32),
        }


def feature_extraction_fn(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))

    y_pred = []
    with torch.no_grad():
        for bi, d in enumerate(tk0):
            features = d["features"].to(device, dtype=torch.float32)
            outputs = model(features)
            y_pred.append(outputs.cpu().detach().numpy())

    y_pred = np.concatenate(y_pred, 0)
    return y_pred


def run_one_fold():

    fnc_df = pd.read_csv(config.FNC_PATH)
    loading_df = pd.read_csv(config.LOADING_PATH)
    labels_df = pd.read_csv(config.TRAIN_SCORES_PATH)

    fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
    df = fnc_df.merge(loading_df, on="Id")
    labels_df["is_train"] = True

    df = df.merge(labels_df, on="Id", how="left")

    df_test = df[df["is_train"] != True].copy()
    df_train = df[df["is_train"] == True].copy()
    print(df_train.shape)

    # df_train = df_train.dropna().reset_index(drop=True)
    # df_train = df_train.head(100)

    train_dataset = TReNDSDataset(df=df_train, map_path=config.TRAIN_MAP_PATH)
_loader, total=len(data_loader))

    y_pred = []
    with torch.no_grad():
        for bi, d in enumerate(tk0):
            features = d["features"].to(device, dtype=torch.float32)
            outputs = model(features)
            y_pred.append(outputs.cpu().detach().numpy())

    y_pred = np.concatenate(y_pred, 0)
    return y_pred
train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=False, 
                batch_size=32,
                num_workers=0, pin_memory=True)

    del train_dataset
    gc.collect()

    device = 'cuda'
    model = resnet50_medicalnet()

    # https://github.com/Tencent/MedicalNet/blob/35ecd5be96ae4edfc1be29816f9847c11d067db0/model.py#L89
    net_dict = model.state_dict() 
    # 医療ドメイン
    pretrain = torch.load("inputs/pretrain/resnet_50.pth")
    # 動画データセット
    # pretrain = torch.load("inputs/3dResNetsPyTorchWeight/r3d50_KMS_200ep.pth")
    # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys() and 'conv1' not in k}
    net_dict.update(pretrain_dict)

    model.load_state_dict(net_dict)
    print("pretrained model loaded !")

    model = model.to(device)

    y_preds = feature_extraction_fn(train_loader, model, device)

    print(y_preds.shape)
    print(df_train.shape)
    # to_pickle(os.path.join(config.OUT_DIR, '3dResNet50_extracted_features.pkl'), y_preds)
    to_pickle(os.path.join(config.OUT_DIR, '3dMedNet50_extracted_features.pkl'), y_preds)


if __name__ == '__main__':
    run_one_fold()

