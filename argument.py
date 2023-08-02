import os
import random
import argparse
import numpy as np
import torch
from common import DATASET_NUM_DIC

# pre-processing settings
parser = argparse.ArgumentParser()
parser.add_argument('--devices_cpu', type=str, default='cpu', help='Devices')
parser.add_argument('--devices_gpu', type=str, default='cuda:0', help='Devices')
parser.add_argument('--seed', type=int, default=111, help='Random seed.')
parser.add_argument('--dataset', default='bitcoin_otc', help='Dataset(slashdot_aminer, epinions_aminer, bitcoin_otc, bitcoin_alpha)')
parser.add_argument('--k', default=1, help='Folder k (1,2,3,4,5)') 
parser.add_argument('--hop', default='2', help='hop')
parser.add_argument("--percent", default=80,  help="Sparsity setting(80, 60, 40, 20)")   
parser.add_argument("--p_thres", type=float, default=0.98, help="Positive threshold (beta_+). Default is 0.98") # we set p_thres == n_thres
parser.add_argument("--n_thres", type=float, default=0.98, help="Negative threshold (beta_-). Default is 0.98")
parser.add_argument("--func", default="setproMTX",  help="select a function of (countTRI, extract, setsubgraph, predict, setproMTX). Default is \"setproMTX\"")
               
args = parser.parse_args()
RANDOM_SEED=args.seed
NUM_NODE = DATASET_NUM_DIC[args.dataset]
DEVICES_CPU = torch.device(args.devices_cpu)
DEVICES_GPU = torch.device(args.devices_gpu)
K = args.k
DATASET = args.dataset
PER = args.percent
HOP= int(args.hop)
FUNCTION=args.func
P_THRESN=args.p_thres
N_THRESN=args.n_thres

