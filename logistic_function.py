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

SINE_MODEL_PATH_DIC = {
    'epinions': './embeddings/sine_epinions_models',
    'slashdot': './embeddings/sine_slashdot_models',
    'bitcoin_alpha': './embeddings/sine_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/sine_bitcoin_otc_models'
}

SIDE_MODEL_PATH_DIC = {
    'epinions': './embeddings/side_epinions_models',
    'slashdot': './embeddings/side_slashdot_models',
    'bitcoin_alpha': './embeddings/side_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/side_bitcoin_otc_models'
}


def read_train_test_data_sign(dataset, k, PER):
    train_X = []
    train_y = []
    with open('./experiment-data/{}/{}_u{}_{}.train'.format(dataset,dataset, k, PER)) as f:
    # with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
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
    # with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
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
    # print(sum(pred))
    pred_p = logistic_function.predict_proba(test_X1)

    cnt = test_y + pred
    # print("correct_pos :", len(cnt[cnt==2]), "||", "correct_neg :", len(cnt[cnt==0]))
    

    recall = metrics.recall_score(test_y, pred)
    pos_ratio =  np.sum(test_y) / test_y.shape[0]
    cnt_correct_1 = recall * len(pred) * pos_ratio
    # print("check: ", cnt_correct_1)
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

@ignore_warnings(category=ConvergenceWarning)
def read_train_test_data_link(dataset, k, per):
    train_edges = []
    test_edges = []
    
    train_edges_flag = []
    test_edges_flag = []
    
    train_edge_num = 0
    test_edge_num = 0
    with open('./experiment-data/{}/{}_u{}_{}.train'.format(dataset,dataset, k, per)) as f:
    # with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            train_edges.append((i, j))
            train_edges_flag.append((i, j, flag))
            train_edge_num += 1


    with open('./experiment-data/{}/{}_u{}_{}.test'.format(dataset,dataset, k, per)) as f:
    # with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            test_edges.append((i, j))
            test_edges_flag.append((i, j, flag))
            test_edge_num += 1

 
    return train_edges, test_edges, train_edges_flag, test_edges_flag


@ignore_warnings(category=ConvergenceWarning)
def link_prediction(dataset, k, embeddings, per):
    node_num = DATASET_NUM_DIC[dataset]
    train_edges, test_edges, train_edges_flag, test_edges_flag  = read_train_test_data_link(dataset, k, per)
    train_edges_null, test_edges_null = [],[]

    for _ in range(3 * len(test_edges)):
        u = rd.choice(range(node_num))
        v = rd.choice(range(node_num))
        while (u,v) in test_edges:
            v = rd.choice(range(node_num))
        test_edges_null.append([u, v, 'n'])

    for _ in range(3 * len(train_edges)):
        u = rd.choice(range(node_num))
        v = rd.choice(range(node_num))
        while (u,v)  in train_edges:
            v = rd.choice(range(node_num))
        train_edges_null.append([u, v, 'n'])



    train_x = np.zeros((len(train_edges) + len(train_edges_null), (EMBEDDING_SIZE) * 2))
    train_y = np.zeros((len(train_edges) + len(train_edges_null), 1))
    for i, edge in enumerate(train_edges_flag):
        u = int(edge[0])
        v = int(edge[1])
        train_x[i, : EMBEDDING_SIZE] = embeddings[u]
        train_x[i, EMBEDDING_SIZE: ] = embeddings[v]

        if int(edge[2]) > 0:
            train_y[i] = 1
        else:
            train_y[i] = -1

    for i, edge in enumerate(train_edges_null):
        i += len(train_edges)
        u = int(edge[0])
        v = int(edge[1])
        train_x[i, : EMBEDDING_SIZE] = embeddings[u]
        train_x[i, EMBEDDING_SIZE: ] = embeddings[v]
        train_y[i] = 0


    test_x = np.zeros((len(test_edges) + len(test_edges_null), (EMBEDDING_SIZE) * 2))
    test_y = np.zeros((len(test_edges) + len(test_edges_null), 1))
    for i, edge in enumerate(test_edges_flag):
        u = int(edge[0])
        v = int(edge[1])
        test_x[i, : EMBEDDING_SIZE] = embeddings[u]
        test_x[i, EMBEDDING_SIZE: ] = embeddings[v]

        if int(edge[2]) > 0:
            test_y[i] = 1
        else:
            test_y[i] = -1

    for i, edge in enumerate(test_edges_null):
        i += len(test_edges)
        u = int(edge[0])
        v = int(edge[1])
        test_x[i, : EMBEDDING_SIZE] = embeddings[u]
        test_x[i, EMBEDDING_SIZE: ] = embeddings[v]
        test_y[i] = 0

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = linear_model.LogisticRegressionCV(fit_intercept=True, max_iter=100, multi_class='multinomial', Cs=np.logspace(-2, 2, 20),
                              cv=2, penalty="l2", solver="lbfgs", tol=0.01)

    # logistic_function = linear_model.LogisticRegression()
    lr.fit(train_x, train_y.ravel())
    pred_prob = lr.predict_proba(test_x)

    labels = test_y.copy()
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = 1
        else:
            labels[i] = 0
    auc_score_pos = metrics.roc_auc_score(labels, pred_prob[:, 2])

    labels = test_y.copy()
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = 1
        else:
            labels[i] = 0
    auc_score_neg = metrics.roc_auc_score(labels, pred_prob[:, 0])
    
    labels = test_y.copy()
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = 1
        else:
            labels[i] = 0
    auc_score_null = metrics.roc_auc_score(labels, pred_prob[:, 1])

    print("AUC_pos:", auc_score_pos)
    print('AUC_neg:', auc_score_neg)
    print("AUC_null:", auc_score_null)


    return auc_score_pos, auc_score_neg, auc_score_null

@ignore_warnings(category=ConvergenceWarning)
def logistic_embedding_link(k, dataset, epoch, dirname, per):

    print(epoch, dataset, per)
    fpath = os.path.join(dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    embeddings = np.load(fpath)
    auc_p, auc_n, auc_null = link_prediction(dataset, k, embeddings, per)
    return auc_p, auc_n, auc_null


def logistic_embedding9(k=1, dataset='epinions', epoch=10, dirname='sigat'):
    """use sigat embedding to train logistic function
    Returns:
        pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score
    """

    filename = os.path.join('embeddings', dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    embeddings = np.load(filename)
    pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score = sign_prediction(dataset, k, embeddings, 'sigat')
    return pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score

def main():
    dataset = 'bitcoin_alpha'
    
    for i in range(6):
        epo = i * 20
        pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score = logistic_embedding9(k=1, dataset=dataset, epoch=epo, dirname='sigat')

    
    # print("pos_ratio:", pos_ratio)
    # print('accuracy:', accuracy)
    # print("f1_score:", f1_score0)
    # print("macro f1_score:", f1_ma)
    # print("micro f1_score:", f1_mi)
    # print("auc score:",auc_score)
        


if __name__ == "__main__":
    main()

