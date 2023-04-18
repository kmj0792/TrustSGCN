import os
import logging
import time
import sys

DATASET_NUM_DIC = {
    'epinions_aminer': 25148,
    'slashdot_aminer': 13182,
    'bitcoin_alpha': 3780,
    'bitcoin_otc': 5878,
    'wikiRfA_pos_neg': 11258,
    'wikiRfA_pos_neg_ASiNE': 11258,
    'epinions' : 178096,
    'slashdot':82140,
    'Alpha':3783,
    'OTC':5881
}

SAMPLER_NUM_DIC = {
    'epinions_aminer': 20,
    'slashdot_aminer': 50,
    'bitcoin_alpha': 30,
    'bitcoin_otc': 30
}

EDGE_NUM_DIC = {
    'epinions_aminer': 105061 ,
    'slashdot_aminer': 36338 ,
    'bitcoin_alpha': 14081 ,
    'bitcoin_otc': 21434 ,
    'wikiRfA_pos_neg': 178096 ,

}

