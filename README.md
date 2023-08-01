# TrustSGCN: Learning Trustworthiness on Edge Signs for Effective Signed Graph Convolutional Networks

This repository provides a reference implementation of TrustSGCN as described in the following paper:
TrustSGCN: Learning Trustworthiness on Edge Signs for Effective Signed Graph Convolutional Networks
46th International ACM SIGIR Conference on Research and Development in Information Retrieval (ACM SIGIR 2023)

## Authors
- Min-Jeong Kim (kmj0792@hanyang.ac.kr)
- Yeon-Chang Lee (yeonchang@gatech.edu)
- Sang-Wook Kim (wook@hanyang.ac.kr)

## Preprocessing
### To get pre_analysis percentage
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=countTRI 

### To get 23 topological feature for FExtra
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=extract

### To get target node's EgoNets
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=setsubgraph

### FExtra (output: predict  sign, confidence score)
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=predict

### Two condition으로 trustworthy or untrustworthy 인지 구분
python preprocessing.py --dataset=bitcoin_alpha --hop=2 --percent=80 --p_thres= --n_thres --func=setproMTX
