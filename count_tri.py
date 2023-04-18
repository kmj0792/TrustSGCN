version = 'undirec' # direc or undirec


# full_graph_filename:  "./ours/experiment-data/{}/{}.txt".format(dataset[i], dataset[i])
def call_check_triangle_sign(full_graph_filename, dataset, split_chr=' ', ):
    ppp, ppn, pnp, pnn, npp, npn, nnp, nnn, cnt_tri_all = check_triangle_sign(full_graph_filename, split_chr)
    ppp_ppn= ppp+ppn
    pnp_pnn= pnp+pnn
    npp_npn= npp+npn
    nnp_nnn= nnp+nnn


    with open("./count_triangle/" + dataset + "_" + version +"_cnt_tri_sign.txt", "a") as fw:
        fw.writelines("\n")
        fw.writelines("number of triangles checked: " + str(cnt_tri_all) + "\n")
        '''for key in cnt_tri_neg:'''
        fw.writelines("number of triangles with ppp: " +  str(ppp) + "\t" + str(ppp / (ppp+ppn)) + "\n")
        fw.writelines("number of triangles with ppn: " +  str(ppn) + "\t" + str(ppn / (ppp+ppn)) + "\n")
        fw.writelines("number of triangles with pnp: " +  str(pnp) + "\t" + str(pnp / (pnp+pnn)) + "\n")
        fw.writelines("number of triangles with pnn: " +  str(pnn) + "\t" + str(pnn / (pnp+pnn)) + "\n")
        fw.writelines("number of triangles with npp: " +  str(npp) + "\t" + str(npp / (npp+npn)) + "\n")
        fw.writelines("number of triangles with npn: " +  str(npn) + "\t" + str(npn / (npp+npn)) + "\n")
        fw.writelines("number of triangles with nnp: " +  str(nnp) + "\t" + str(nnp / (nnp+nnn)) + "\n")
        fw.writelines("number of triangles with nnn: " +  str(nnn) + "\t" + str(nnn / (nnp+nnn)) + "\n")


def check_triangle_sign(full_graph_filename, split_chr=" "):
    pos = 1; neg = -1

    adj_dict_drct, edge_sign_dict_drct = get_adj_dict_from_file(full_graph_filename, split_chr)

    cnt_tri_neg = {0: 0, 1: 0, 2: 0, 3: 0}
    cnt_tri_pos = {0: 0, 1: 0, 2: 0, 3: 0}
    cnt_tri_all = 0
    ppp=0
    ppn=0
    pnp=0
    pnn=0
    npp=0
    npn=0
    nnp=0
    nnn = 0 #; 
    # get_triangles( ) <- iterator (expecially generator) for triangles. (uses 'yield' keyword)

    tri_dict=get_triangles(adj_dict_drct)
    for triangle in get_triangles(adj_dict_drct): # count negative edges in a single triangle
        prior1sign = edge_sign_dict_drct[triangle[0]][triangle[1]]
        prior2sign = edge_sign_dict_drct[triangle[1]][triangle[2]]
        posteriorsign = edge_sign_dict_drct[triangle[0]][triangle[2]]
        if version == "direc":
            if (prior1sign==pos) and (prior2sign==pos):
                if posteriorsign==pos:
                    ppp+=1
                elif posteriorsign==neg:
                    ppn+=1
            elif (prior1sign==pos) and (prior2sign==neg):
                if posteriorsign==pos:
                    pnp+=1
                elif posteriorsign==neg:
                    pnn+=1
            elif (prior1sign==neg) and (prior2sign==pos):
                if posteriorsign==pos:
                    npp+=1
                elif posteriorsign==neg:
                    npn+=1
            elif (prior1sign==neg) and (prior2sign==neg):
                if posteriorsign==pos:
                    nnp+=1
                elif posteriorsign==neg:
                    nnn+=1
        if version == "undirec":
            if prior1sign+prior2sign+posteriorsign == 3:
                ppp+=1
            elif prior1sign+prior2sign+posteriorsign == 1:
                ppn+=1
                pnp+=1
                npp+=1
            elif prior1sign+prior2sign+posteriorsign == -1:
                pnn+=1
                npn+=1
                nnp+=1               
            elif prior1sign+prior2sign+posteriorsign == -3:
                nnn+=1
            
        '''cnt_tri_pos[cnt_pos] += 1
        cnt_tri_neg[cnt_neg] += 1'''
        cnt_tri_all += 1

        if cnt_tri_all % 1000 == 0:
            print("\t\tuntill...", cnt_tri_all)

    return ppp, ppn, pnp, pnn, npp, npn, nnp, nnn, cnt_tri_all

""" rewrite with yc."""
def get_triangles(adj_dict_undrct):
    visited_centers = set() # mark visited node

    for center in adj_dict_undrct:
        visited_center_nhds = set()

        for nhd_1 in adj_dict_undrct[center]:
            if nhd_1 == center:
                continue # raise ValueError # to prevent self-loops
            
            if nhd_1 in visited_centers:
                continue

            if nhd_1 in adj_dict_undrct:
                for nhd_2 in adj_dict_undrct[nhd_1]:
                    if nhd_2 in [center, nhd_1]: # if center or nhd_1 exists
                        continue
                    if nhd_2 in visited_centers or nhd_2 in visited_center_nhds: # 2 terminate conditions;
                        continue
                    
                    if center in adj_dict_undrct:
                        if nhd_2 in adj_dict_undrct[center]:
                            yield (center, nhd_1, nhd_2)
                    # else: continue
            visited_center_nhds.add(nhd_1)

        visited_centers.add(center)
    return visited_centers


def get_adj_dict_from_file(filename, split_chr=' '):
    adj_dict_drct = {} # {center: [nhd, nhd, ...], ..}
    edge_sign_dict_drct = {} # {center: {nhd: sign, nhd: sign}, ..}
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line != '':
            edge = [int(x) for x in line.split(split_chr)]
            if len(edge) < 3:
                line = f.readline().strip()
                continue

            if version == 'undirec':
                add_to_dict_undirect(adj_dict_drct, edge[0], edge[1])
                add_sign_to_dict_undirect(edge_sign_dict_drct, edge[0], edge[1], edge[2])
            if version == 'direc':
                add_to_dict_direct(adj_dict_drct, edge[0], edge[1])
                add_sign_to_dict_direct(edge_sign_dict_drct, edge[0], edge[1], edge[2])

            line = f.readline().strip()

    for key in adj_dict_drct:
        temp = set(adj_dict_drct[key])
        adj_dict_drct[key] = temp # 중복 제거

    print("Finish read graph... ")
    return adj_dict_drct, edge_sign_dict_drct


def add_sign_to_dict_undirect(dictionary, key1, key2, sign):
    if key1 not in dictionary:
        dictionary[key1] = {}
    if key2 not in dictionary:
        dictionary[key2] = {}

    dictionary[key1][key2] = sign
    dictionary[key2][key1] = sign

def add_sign_to_dict_direct(dictionary, key1, key2, sign):
    if key1 not in dictionary:
        dictionary[key1] = {}

    dictionary[key1][key2] = sign



def add_to_dict_undirect(dictionary, key1, key2):
    # undirect !!
    if key1 not in dictionary:
        dictionary[key1] = []
    if key2 not in dictionary:
        dictionary[key2] = []
    dictionary[key1].append(key2)
    dictionary[key2].append(key1)

def add_to_dict_direct(dictionary, key1, key2):
    # direct !!
    if key1 not in dictionary:
        dictionary[key1] = []
    dictionary[key1].append(key2)


def cntTriangle(data):
    dataset = [data]
    split_chr = ["\t"] # bitcoin_alpha는 " ", 나머지는 "\t"
    # s1 = time.time()
    for i in range(len(dataset)):
        # s_temp = time.time()
        print(dataset[i] + ".....")
        full_graph_filename = "./experiment-data/{}/{}.txt".format(dataset[i], dataset[i])
        call_check_triangle_sign(full_graph_filename, dataset[i], split_chr[i])
