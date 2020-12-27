import pickle
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import dagshub

sys.path.append('./model_scripts/common/')
from tools import *

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=int)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
parser.add_argument("MODEL", help="model to use", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH
MODEL = args.MODEL # MODEL = 'svm'

TASK = 'model validation' # 'model evaluation'
MODEL_DIR = './models/{}_regex_graph_v{}.sav'.format(MODEL, GRAPH_VER)
TFIDF_DIR = './models/tfidf_{}_graph_v{}.pickle'.format(MODEL, GRAPH_VER)
CODE_COLUMN = 'code_block'
TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)
SCRIPT_DIR = './predict_tag.ipynb'

if __name__ == '__main__':
    df = load_data(DATASET_PATH)
    code_blocks = df[CODE_COLUMN]
    nrows = df.shape[0]
    print("loaded")
    tfidf_params = {'min_df': 5
                    , 'max_df': 0.3
                    , 'smooth_idf': True}
    meta = {'DATASET_PATH': DATASET_PATH
            ,'TFIDF_DIR': TFIDF_DIR
            ,'MODEL_DIR': MODEL_DIR
            ,'nrows': nrows
            ,'label': TAGS_TO_PREDICT
            ,'model': MODEL
            ,'graph_ver': GRAPH_VER
            ,'script_dir': SCRIPT_DIR
            ,'task': TASK}
    code_blocks_tfidf = tfidf_transform(code_blocks, tfidf_params, TFIDF_DIR)
    with dagshub.dagshub_logger() as logger:
        _, y, y_pred, metrics = get_metrics(code_blocks_tfidf, df[TAGS_TO_PREDICT], TAGS_TO_PREDICT, MODEL_DIR)
        logger.log_hyperparams(meta)
        logger.log_metrics(metrics)
    print("finished")