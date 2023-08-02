import sys, os, argparse
import numpy as np
from tqdm import tqdm
import util
import matrix
import FExtra
import feature
from argument import DATASET, NUM_NODE, DEVICES_GPU, K, HOP, PER
from collections import defaultdict
import pickle
import time

def extract_features(flag, fea_path, n_user):
    input_path = './experiment-data/{}/{}_u{}_80.train'.format(DATASET, DATASET, K)
    output_path = fea_path + "/{}_features.txt".format(DATASET)

    df_train = util.txt_to_df(input_path)
    df_train_pos, df_train_neg = util.split_sign(df_train)
    mtx = matrix.create_mtx(df_train, df_train_pos, df_train_neg, n_user)

    if flag:
        print("Start extracting features.........")
        time1 = time.time()
        features_train = feature.extract_features(output_path, df_train, mtx)
        running_time = time.time() - time1
        print("time: ", running_time)
    else:
        if os.path.isfile(output_path):
            print("Load features.........")
            features_train = util.read_features_from_file(output_path)
        else:
            sys.exit("A file of features is not existed. Please extract them first.")
    return df_train, features_train, mtx 


# no save time
def predict_FExtra_scores(flag,  fea_path,  features_train, mtx, n_user):
 
    output_path = fea_path + "/{}_FExtra.txt".format(DATASET)
    fextra = FExtra.FExtra(features_train, 23)
    if flag:
        print("Start predicting FExtra scores........")
        pbar=tqdm(total=n_user)  
        with open(output_path, 'w') as f: 
            for snode in range(n_user):
      
                util.save_predict(snode, mtx, fextra, f, n_user)
                pbar.update(1)
        pbar.close()
    else:
        print("Load FExtra scores........")
        if not os.path.isfile(output_path):
            sys.exit("A file of FExtra scores is not existed. Please predict them first.")

    del mtx[2]
    del mtx[1]
    del mtx[0][3]
    del mtx[0][2]
    return  mtx

def predict_FExtra_scores_save_time(flag,  fea_path, features_train, mtx, n_user, subgraph):
 
    output_path = fea_path + "/{}_{}_{}_FExtra.pkl".format(DATASET, HOP, PER)

    time1 = time.time()
    fextra = FExtra.FExtra(features_train, 23)# train feature로 학습된 모델
    running_time = time.time() - time1
    print("feture로 학습하는 time: ", running_time)
   
    if flag:
        print("Start predicting FExtra scores........")
        time2= time.time()
        pbar=tqdm(total=len(subgraph.keys()))
        with open(output_path, 'w') as f: 
            for snode in subgraph.keys():
                adj_set = set()
                for p2, sign, hop in subgraph.get(snode):
                    adj_set.add(p2)

                util.save_predict_save_time(snode, mtx, fextra, f, n_user, adj_set)
                pbar.update(1)
        pbar.close()
        
        running_time2 = time.time() - time2
        print("fextra 예측 time: ", running_time2)
        

    else:
        print("Load FExtra scores........")
        if not os.path.isfile(output_path):
            sys.exit("A file of FExtra scores is not existed. Please predict them first.")

    del mtx[2]
    del mtx[1]
    del mtx[0][3]
    del mtx[0][2]
    return  mtx

def process(fea_path, df_train,  n_user, sign_thres, adj_dic_all):
    print("Run Propagation MTX")
    time1 = time.time()

    input_path = fea_path + "/{}_{}_{}_FExtra.pkl".format(DATASET, HOP, PER)
    num_snode=0.0

    all_pred_score = np.zeros(shape=(n_user, n_user))
    all_pred_sign = np.zeros(shape=(n_user, n_user))
    with open(input_path, 'r') as f:
        pbar = tqdm(total=len(adj_dic_all.keys()))
        for snode in adj_dic_all.keys():
            num_snode += 1.0
            scores_snode, signs_snode = util.read_predict(snode, n_user, f)
            
            #self score = 1
            scores_snode[snode] = 1
            signs_snode[snode] = 1

            scores_snode = np.array(scores_snode)
            signs_snode = np.array(signs_snode)

            real_edge = df_train[df_train.sid == snode].tid.values
            real_sign = df_train[df_train.sid == snode].feedback.values

            scores_snode[real_edge] = 1 
            signs_snode[real_edge] = real_sign

            all_pred_score[snode]=scores_snode
            all_pred_sign[snode]= signs_snode
           
            pbar.update(1)
        pbar.close()
   
    lists_T1, lists_T2, lists_U1, lists_U2= matrix.set_TrustSGCN_mtx(all_pred_score, all_pred_sign, n_user, sign_thres, adj_dic_all)

    running_time = time.time() - time1
    print("time: ", running_time)
    print("Finish Make Propagation MTX")

    return lists_T1, lists_T2, lists_U1, lists_U2





