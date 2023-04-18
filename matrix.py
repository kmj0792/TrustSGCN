import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, csc_matrix


def df_to_csr_matrix(df, n_user):
    mtx_csr = csr_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    return mtx_csr


def df_to_csc_matrix(df, n_user):
    mtx_csc = csc_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    return mtx_csc


def set_mtx_triad(mtx1, mtx2):
    mtx = [mtx1 * mtx2,
           mtx1 * csr_matrix.transpose(mtx2),
           (csr_matrix.transpose(mtx1) * mtx2).tocsr(),
           (csr_matrix.transpose(mtx1) * csr_matrix.transpose(mtx2)).tocsr()]
    return mtx


# create matrix for feature extraction
def create_mtx(df, df_pos, df_neg, n_user):
    temp1 = csr_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    temp2 = csr_matrix((df.feedback, (df.tid, df.sid)), shape=(n_user, n_user))
    mtx_train_csr = temp1 + temp2
    mtx_train_csr.data = np.ones_like(mtx_train_csr.data)
    mtx_common = mtx_train_csr * mtx_train_csr

    mtx_cs = [df_to_csr_matrix(df_pos, n_user), df_to_csr_matrix(df_neg, n_user),
              df_to_csc_matrix(df_pos, n_user), df_to_csc_matrix(df_neg, n_user)]

    mtx_triad = []
    mtx_triad.extend(set_mtx_triad(mtx_cs[0], mtx_cs[0]))
    mtx_triad.extend(set_mtx_triad(mtx_cs[0], mtx_cs[1]))
    mtx_triad.extend(set_mtx_triad(mtx_cs[1], mtx_cs[0]))
    mtx_triad.extend(set_mtx_triad(mtx_cs[1], mtx_cs[1]))

    mtx = [mtx_cs, mtx_common, mtx_triad]
    return mtx


# set matrix for OBOE
def set_OBOE_mtx( scores, signs, n_user, sign_thres, adj_dic_all):
    # 0 == -1 edge
    cnt = 0
    '''for i in range(n_user):
        for j in range(n_user):
            if(signs[i][j]==0):
                signs[i][j]=-1'''
    
    lists_T1 = list()
    lists_T2 = list()
    lists_U1 = list()
    lists_U2 = list()


    pbar=tqdm(total=len(adj_dic_all.keys()))
    for i in adj_dic_all.keys():
        for j, sign, hop in adj_dic_all.get(i):
            cnt +=  1
            fifi = [i, j, sign, hop]
            #1hop은 다 믿어
            if hop == 1 and sign == 1:
                lists_T1.append(fifi)

            elif hop == 1 and sign == -1:
                lists_T2.append(fifi)
      
            #2hop은 balance theory 기반으로 예측한 것이기 때문에 검증 필요
            elif(sign == signs[i][j]): # 조건1: balance theory ==  fextra
                if sign==1: # pos edge
                    if(scores[i][j]>sign_thres[0]): # 조건2: over thre
                        lists_T1.append(fifi)
           
                    else:
                        #ppp, ppn, npp, npn
                        lists_U1.append(fifi)
                        
                        
                if sign==-1:
                    if(abs(scores[i][j])>sign_thres[1]): # over thre
                        lists_T2.append(fifi)
                        # temp = ['T2']
                        # temp = [0.0, 1.0, 1.0, 0.0]       
                    else: 
                        #pnp, pnn, nnp, nnn
                        lists_U2.append(fifi)
                        # temp = ['U2']# _n_
                        # temp = [pre_analysis[1][0], pre_analysis[1][1], pre_analysis[3][0], pre_analysis[3][1]]
                    
            elif((hop != 1) and (sign != signs[i][j])):
                if sign == 1:
                    #ppp, ppn, npp, npn
                    lists_U1.append(fifi)
                    
        
                if sign==-1:
                    #pnp, pnn, nnp, nnn
                    lists_U2.append(fifi)

        pbar.update(1)
    pbar.close()  

    return lists_T1, lists_T2, lists_U1, lists_U2


def set_Balance_mtx( scores, signs, n_user, sign_thres, adj_dic_all):
    # 0 == -1 edge
    for i in range(n_user):
        for j in range(n_user):
            if(signs[i][j]==0):
                signs[i][j]=-1
    
    lists_T1 = list()
    lists_T2 = list()
    lists_U1 = list()
    lists_U2 = list()

    pbar=tqdm(total=len(adj_dic_all.keys()))
    for i in adj_dic_all.keys():
        # final_adj_lists_all = adj_dic_all.get(i)[:] # copy
        for j, sign, hop in adj_dic_all.get(i):
            fifi = [i, j, sign, hop]
            #1hop은 다 믿어
            if sign == 1:
                lists_T1.append(fifi)
                # temp = ['T1']
                # temp = [1.0, 0.0, 0.0, 1.0]
            elif sign == -1:
                lists_T2.append(fifi)
                # temp = ['T2']
                # temp = [0.0, 1.0, 1.0, 0.0]

            # fifi.extend(temp)
            # lists_all.append(fifi)

        pbar.update(1)
    pbar.close()  

    return lists_T1, lists_T2, lists_U1, lists_U2 #lists_all

def set_FExtra_mtx( scores, signs, n_user, sign_thres, adj_dic_all):
 
    lists_T1 = list()
    lists_T2 = list()
    lists_U1 = list()
    lists_U2 = list()

    pbar=tqdm(total=len(adj_dic_all.keys()))
    for i in adj_dic_all.keys():
        # final_adj_lists_all = adj_dic_all.get(i)[:] # copy
        for j, sign, hop in adj_dic_all.get(i):
            fifi = [i, j, sign, hop]
            #1hop은 다 믿어
            if signs[i][j] == 1 :
                lists_T1.append(fifi)
                # temp = ['T1']
                # temp = [1.0, 0.0, 0.0, 1.0]
            elif signs[i][j] == -1:
                lists_T2.append(fifi)
                # temp = ['T2']
                # temp = [0.0, 1.0, 1.0, 0.0]
        pbar.update(1)
    pbar.close()  

    return lists_T1, lists_T2, lists_U1, lists_U2
# normalize matrix for OBOE
def normalize_OBOE_mtx(mtx_FS, n_user):
    nor_mtx_FS = []
    # [0]: positive nodes which don't have neighbors
    # [1]: negative nodes which don't have neighbors
    no_neighs = []
    for i in [0, 2]:
        no_neighs_snode = []
        sid_p, sid_n, tid_p, tid_n, feedback_p, feedback_n = [], [], [], [], [], []
        for snode in range(n_user):
            neighs_p, neighs_feedback_p = get_neighbors_and_feedback(mtx_FS[0+i], snode)
            neighs_n, neighs_feedback_n = get_neighbors_and_feedback(mtx_FS[1+i], snode)
            neighs_len = float(sum(neighs_feedback_p) + sum(neighs_feedback_n))
            if neighs_len == 0:
                no_neighs_snode.append(snode)
                continue
            sid_p.extend([snode] * len(neighs_p))
            sid_n.extend([snode] * len(neighs_n))
            tid_p.extend(neighs_p)
            tid_n.extend(neighs_n)
            feedback_p.extend(neighs_feedback_p / neighs_len)
            feedback_n.extend(neighs_feedback_n / neighs_len)
        nor_mtx_FS.append(csr_matrix((feedback_p, (sid_p, tid_p)), shape=(n_user, n_user)))
        nor_mtx_FS.append(csr_matrix((feedback_n, (sid_n, tid_n)), shape=(n_user, n_user)))
        no_neighs.append(no_neighs_snode)
    return nor_mtx_FS, no_neighs




def get_neighbors(mtx, idx):
    start = mtx.indptr[idx].astype(int)
    end = mtx.indptr[idx + 1].astype(int)
    return mtx.indices[start:end]


def get_neighbors_and_feedback(mtx, idx):
    start1 = mtx.indptr[idx].astype(int)
    end1 = mtx.indptr[idx + 1].astype(int)
    return mtx.indices[start1:end1], mtx.data[start1:end1]


def get_degree(mtx, idx):
    return mtx.indptr[idx + 1] - mtx.indptr[idx]
