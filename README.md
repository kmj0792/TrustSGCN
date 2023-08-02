# TrustSGCN: Learning Trustworthiness on Edge Signs for Effective Signed Graph Convolutional Networks

This repository provides a reference implementation of TrustSGCN as described in the following paper:
TrustSGCN: Learning Trustworthiness on Edge Signs for Effective Signed Graph Convolutional Networks
46th International ACM SIGIR Conference on Research and Development in Information Retrieval (ACM SIGIR 2023)

## Authors
- Min-Jeong Kim (kmj0792@hanyang.ac.kr)
- Yeon-Chang Lee (yeonchang@gatech.edu)
- Sang-Wook Kim (wook@hanyang.ac.kr)

## Input

## Output

## Arguments

## Procedure
1. Get ratios of balanced/unbalanced triads (```pre_analysis``` percentage).
2. Extract 23 topological features of train dataset.
3. Get target node's extended EgoNets.
4. Predict edge sign and confidence scores between two nodes using 23 topological features.
5. Measure trustworthiness of edge signs in the Egonet using two conditions.
6. Performs different embedding propagation (trustworthy or untrustworthy).

## Basic Usage

```
# 1. Get ratios of balanced/unbalanced triads (```pre_analysis``` percentage).
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=countTRI

<!-- 2. Extract 23 topological features of train dataset. -->
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=extract

<!-- 3. Get target node's extended EgoNets. -->
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=setsubgraph

<!-- 4. Predict edge sign and confidence scores between two nodes using 23 topological features. -->
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=predict

<!-- 5. Measure trustworthiness of edge signs in the Egonet using two conditions. -->
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=setproMTX

<!-- 6. Performs different embedding propagation (trustworthy or untrustworthy). -->
python trustsgcn.py --dataset=bitcoin_alpha --batch_size=300 --percent=80 --k=1 --hop=2 --p_thres=0.98 --n_thres=0.98 --sample_num=30 --lamda1=1 --get_dgl=True --task=sign
```
