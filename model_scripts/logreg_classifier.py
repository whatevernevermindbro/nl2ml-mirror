import pickle
import argparse
import json
import sys, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import dagshub

sys.path.append('./model_scripts/common/')
from tools import *

# def logreg_evaluate(df, code_blocks, TAG_TO_PREDICT):
#     code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
#     X_train, X_test, y_train, y_test = train_test_split(code_blocks_tfidf, df[TAG_TO_PREDICT], test_size=0.25)
#     clf = LogisticRegression(random_state=421).fit(X_train, y_train)
#     print("inited the model")
#     pickle.dump(clf, open(MODEL_DIR, 'wb'))
#     print("saved the model")
#     y_pred = clf.predict(X_test)
#     accuracy = clf.score(X_test, y_test)
#     f1 = f1_score(y_pred, y_test, average='weighted')
#     print(f'Mean Accuracy {round(accuracy*100, 2)}%')
#     print(f'F1-score {round(f1*100, 2)}%')
#     errors = y_test - y_pred
#     plt.hist(errors)
#     # plot_precision_recall_curve(clf, X_test, y_test)
#     # plot_confusion_matrix(clf, X_test, y_test, values_format='d')
#     def mean_confidence_interval(data, confidence=0.95):
#         a = 1.0 * np.array(data)
#         n = len(a)
#         m, se = np.mean(a), scipy.stats.sem(a)
#         h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
#         return m, m-h, m+h
#     conf_interval = mean_confidence_interval(errors, 0.95)
#     print(conf_interval)
#     metrics = {'test_accuracy': accuracy
#                , 'test_f1_score': f1}
#     return metrics

# def get_predictions(X, y, TAGS_TO_PREDICT, MODEL_DIR):
#     clf = pickle.load(open(MODEL_DIR, 'rb'))
#     # result = loaded_model.score(X, y)
#     y_pred = clf.predict(X)
#     accuracy = accuracy_score(y_pred, y)
#     f1 = f1_score(y_pred, y, average='weighted')
#     print(f'Mean Accuracy {round(accuracy*100, 2)}%')
#     print(f'F1-score {round(f1*100, 2)}%')
#     errors = y - y_pred
#     plt.hist(errors)
#     plot_precision_recall_curve(clf, X, y)
#     plot_confusion_matrix(clf, X, y, values_format='d')
#     def mean_confidence_interval(data, confidence=0.95):
#         a = 1.0 * np.array(data)
#         n = len(a)
#         m, se = np.mean(a), scipy.stats.sem(a)
#         h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
#         return m, m-h, m+h
#     conf_interval = mean_confidence_interval(errors, 0.95)
#     print(conf_interval)
#     metrics = {'test_accuracy': accuracy
#                , 'test_f1_score': f1}
#     return metrics

def logreg_multioutput_evaluate(df, code_blocks, TAGS_TO_PREDICT):
    code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
    print("tfifd-ed")
    X_train, X_test, Y_train, Y_test = train_test_split(code_blocks_tfidf, df[TAGS_TO_PREDICT], test_size=0.25)
    print("splitted to train and test")
    clf = MultiOutputRegressor(LogisticRegression(random_state=421)).fit(X_train, Y_train)
    print("trained the model")
    pickle.dump(clf, open(MODEL_DIR, 'wb'))
    print("saved the model")
    Y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, Y_test)
    f1 = f1_score(Y_pred, Y_test, average='weighted')
    print(f'Mean Accuracy {round(accuracy*100, 2)}%')
    print(f'F1-score {round(f1*100, 2)}%')
    # errors = Y_test - Y_pred
    # plt.hist(errors)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    # plot_confusion_matrix(clf, X_test, Y_test, values_format='d')
    metrics = {'test_accuracy': accuracy
               , 'test_f1_score': f1}
    return metrics

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=int)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

# REPO_PATH = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/'
MODEL_DIR = './models/logreg_regex_graph_v{}.sav'.format(GRAPH_VER)
TFIDF_DIR = './models/tfidf_logreg_graph_v{}.pickle'.format(GRAPH_VER)
CODE_COLUMN = 'code_block'
TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)
PREDICT_COL = 'pred_{}'.format(TAGS_TO_PREDICT)
SCRIPT_DIR = 'logreg_classifier.ipynb'
TASK = 'training LogReg'

if __name__ == '__main__':
    df = load_data(DATASET_PATH)
    code_blocks = df[CODE_COLUMN]
    nrows = df.shape[0]
    print("loaded the data")
    tfidf_params = {'min_df': 5
                    , 'max_df': 0.3
                    , 'smooth_idf': True}
    data_meta = {'DATASET_PATH': DATASET_PATH
                ,'TFIDF_DIR': TFIDF_DIR
                ,'MODEL_DIR': MODEL_DIR
                ,'nrows': nrows
                ,'label': TAGS_TO_PREDICT
                ,'graph_ver': GRAPH_VER
                ,'script_dir': SCRIPT_DIR
                ,'task': TASK}
    with dagshub.dagshub_logger() as logger:
        metrics = logreg_multioutput_evaluate(df, code_blocks, TAGS_TO_PREDICT)
        logger.log_hyperparams(data_meta)
        logger.log_hyperparams(tfidf_params)
        logger.log_metrics(metrics)
        print("saved the dicts")
    print("finished")