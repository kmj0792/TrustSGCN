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


#원래
def save_predict(snode, mtx, fextra, f, n_user):
   
    features_snode = feature.extract_features_for_seed(snode, mtx, n_user) # test데이터셋의 피처 추출
    scores_snode, signs_snode = fextra.compute_scores(features_snode) # train 피처들로 fextra 계산
    scores_snode[snode] = 1
    signs_snode[snode] = 1
    f.write(str(snode))
    f.write("\t")
    f.write("\t".join(str(item) for item in scores_snode))
    f.write("\t")
    f.write("\t".join(str(item) for item in signs_snode))
    f.write("\n")
    return

def save_predict_save_time(snode, mtx, fextra, f, n_user, adj_set):
   
    features_snode = feature.extract_features_for_seed_save_time(snode, mtx, n_user, adj_set) # test데이터셋의 피처 추출
    scores_snode, signs_snode = fextra.compute_scores(features_snode) # train 피처들로 fextra 계산
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
    line = f.readline() # FEXFTRA 파일을 한줄씩 읽어
    line_lt = line.split()
    while int(line_lt[0]) != snode:
        line = f.readline()
        line_lt = line.split()
    return list(map(float, line_lt[1:n_user+1])), list(map(int, line_lt[n_user+1:]))


def evaluation_ranking(seed, dic_ans, vec_scores, snode_pos_nodes, top_n, type):
    # initialize metrics
    hits = {}
    precision = {}
    recall = {}
    dcg = {}
    idcg = {}
    ndcg = {}

    dic_scores = {x: y for x, y in zip(range(len(vec_scores)), vec_scores)}
    del dic_scores[seed]
    for del_node in snode_pos_nodes:
        del dic_scores[del_node]

    if type == "top":
        dic_scores_n = sorted(dic_scores.items(), key=lambda x: x[1], reverse=True)[
                       0:min(len(dic_scores), top_n[len(top_n) - 1])]
        ans = [name for name, value in dic_ans.items() if value == 1]
    else:  # bottom
        dic_scores_n = sorted(dic_scores.items(), key=lambda x: x[1], reverse=False)[
                       0:min(len(dic_scores), top_n[len(top_n) - 1])]
        ans = [name for name, value in dic_ans.items() if value == -1]

    if len(ans) == 0:
        precision = None
        recall = None
        ndcg = None
        return precision, recall, ndcg

    # bottom
    # dic_scores_bottomn = sorted(dic_scores.items(), key=lambda x: x[1], reverse=False)[0:min(len(dic_scores), top_n[len(top_n)-1])]
    # ans_neg = [name for name, value in dic_ans.items() if value == -1]

    for n in top_n:
        hits[n] = 0
        precision[n] = 0
        recall[n] = 0
        dcg[n] = 0.0
        idcg[n] = 0.0
        ndcg[n] = 0.0
        # set idcg
        if len(ans) > n:
            for rank in range(n):
                rank += 1
                idcg[n] += 1.0 / np.log2(rank + 1.0)
        else:
            for rank in range(len(ans)):
                rank += 1
                idcg[n] += 1.0 / np.log2(rank + 1.0)

    # compute accuracies
    rank = 0
    for (node, score) in dic_scores_n:
        rank += 1
        if node in ans:  # hits!
            for n in top_n:
                if rank <= n:
                    hits[n] += 1
                    dcg[n] += 1.0 / np.log2(rank + 1.0)
    for n in top_n:
        precision[n] = hits[n] / (1.0 * n)
        if len(ans) > n:
            recall[n] = hits[n] / (1.0 * n)
        else:
            recall[n] = hits[n] / (1.0 * len(ans))
        ndcg[n] = dcg[n] / idcg[n]

    return precision, recall, ndcg
