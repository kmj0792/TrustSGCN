import pandas as pd
import numpy as np
import feature

def txt_to_df(lt_input):
    df = pd.read_csv(lt_input, sep='\t', names=['sid', 'tid', 'feedback'])
    return df


def split_sign(df):
    df_pos = df[df.feedback == 1].reset_index(drop=True)
    df_neg = df[df.feedback == -1].reset_index(drop=True)
    df_neg.feedback = 1
    return df_pos, df_neg


def group_by_test(df):
    return df.groupby('sid')


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_features_from_file(filename):
    with open(filename, "r") as f:
        features = []
        lines = f.readlines()
        for line in lines:
            int_list = str_list_to_int(line.split())
            features.append(feature.Feature(int_list[0], int_list[1], int_list[2:]))
    return features


# original
def save_predict(snode, mtx, fextra, f, n_user):
   
    features_snode = feature.extract_features_for_seed(snode, mtx, n_user) 
    scores_snode, signs_snode = fextra.compute_scores(features_snode) 
    scores_snode[snode] = 1
    signs_snode[snode] = 1
    f.write(str(snode))
    f.write("\t")
    f.write("\t".join(str(item) for item in scores_snode))
    f.write("\t")
    f.write("\t".join(str(item) for item in signs_snode))
    f.write("\n")
    return

# to save time
def save_predict_save_time(snode, mtx, fextra, f, n_user, adj_set):
   
    features_snode = feature.extract_features_for_seed_save_time(snode, mtx, n_user, adj_set) 
    scores_snode, signs_snode = fextra.compute_scores(features_snode) 
    scores_snode[snode] = 1
    signs_snode[snode] = 1
    f.write(str(snode))
    f.write("\t")
    f.write("\t".join(str(item) for item in scores_snode))
    f.write("\t")
    f.write("\t".join(str(item) for item in signs_snode))
    f.write("\n")
    return

def read_predict(snode, n_user, f):
    line = f.readline() 
    line_lt = line.split()
    while int(line_lt[0]) != snode:
        line = f.readline()
        line_lt = line.split()
    return list(map(float, line_lt[1:n_user+1])), list(map(int, line_lt[n_user+1:]))
