#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: h12345jack
@file: logistic_function.py
@time: 2018/12/16
"""

import os
import sys
import re
import time
import json
import pickle
import logging
import math
import random as rd
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.utils.testing import ignore_warnings
# from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from collections import defaultdict

import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn import linear_model
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import Normalizer, StandardScaler

from common import DATASET_NUM_DIC

EMBEDDING_SIZE = 64

def read_train_test_data_sign(dataset, k, PER):
    train_X = []
    train_y = []
    with open('./experiment-data/{}/{}_u{}_{}.train'.format(dataset,dataset, k, PER)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            train_X.append((i, j))
            train_y.append(flag)
    test_X = []
    test_y = []
    with open('./experiment-data/{}/{}_u{}_{}.test'.format(dataset,dataset, k, PER)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            test_X.append((i, j))
            test_y.append(flag)
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


@ignore_warnings(category=ConvergenceWarning)
def sign_prediction(dataset, k, embeddings, per):
    train_X, train_y, test_X, test_y  = read_train_test_data_sign(dataset, k, per)

    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))


    logistic_function = linear_model.LogisticRegression()
    logistic_function.fit(train_X1, train_y)
    pred = logistic_function.predict(test_X1)
    pred_p = logistic_function.predict_proba(test_X1)

    cnt = test_y + pred
   
    recall = metrics.recall_score(test_y, pred)
    pos_ratio =  np.sum(test_y) / test_y.shape[0]
    cnt_correct_1 = recall * len(pred) * pos_ratio

    accuracy =  metrics.accuracy_score(test_y, pred)
    f1_score0 =  metrics.f1_score(test_y, pred)
    f1_ma =  metrics.f1_score(test_y, pred, average='macro')
    f1_mi =  metrics.f1_score(test_y, pred, average='micro')

    auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    print("pos_ratio:", pos_ratio)
    print('accuracy:', accuracy)
    print("f1_score:", f1_score0)
    print("micro f1_score:", f1_mi)
    print("macro f1_score:", f1_ma)
    print("auc score:", auc_score)

    return pos_ratio, accuracy, f1_score0, f1_ma, f1_mi,  auc_score

def logistic_embedding_sign(k, dataset, epoch, dirname, per):

    print(epoch, dataset, per)
    fpath = os.path.join(dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    embeddings = np.load(fpath)
    pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score = sign_prediction(dataset, k, embeddings, per)
    return pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score

