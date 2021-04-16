import pickle
from datetime import datetime
import argparse
import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
import dagshub

from common.tools import *


EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"


def SVM_multioutput_evaluate(df, tfidf_params, TFIDF_DIR, svm_params):
    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], tfidf_params, TFIDF_DIR)
    print(df.columns)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values

    clf = LinearSVC(**svm_params)
    print("Model params:", clf.get_params())

    n_folds = 20
    kf = KFold(n_splits=n_folds)

    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("{:2}/{:2} training".format(i, n_folds))
        clf.fit(X_train, y_train)

        print("{:2}/{:2} predicting on test".format(i, n_folds))
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)

    metrics = {'test_accuracy': accuracies.mean(), 'test_f1_score': f1s.mean()}
    print(metrics)

    print("saving model")

    clf.fit(X, y)
    pickle.dump(clf, open(MODEL_DIR, "wb"))

    return metrics


parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

MODEL_DIR = "../models/linear_svm_regex_graph_v{}.sav".format(GRAPH_VER)
TFIDF_DIR = "../models/tfidf_svm_graph_v{}.pickle".format(GRAPH_VER)

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    tfidf_params = {
        'min_df': 5,
        'max_df': 0.3,
        'smooth_idf': True,
    }
    kfold_params = {
        "random_state": 42,
        "shuffle": True,
    }
    svm_params = {
        'C': 5,
        'random_state': 241,
    }
    data_meta = {
        'DATASET_PATH': DATASET_PATH,
        'nrows': nrows,
        'label': TAGS_TO_PREDICT,
        'model': MODEL_DIR,
        'script_dir': __file__,
    }

    metrics_path = os.path.join(EXPERIMENT_DATA_PATH, "metrics.csv")
    params_path = os.path.join(EXPERIMENT_DATA_PATH, "params.yml")
    with dagshub.dagshub_logger(metrics_path=metrics_path, hparams_path=params_path) as logger:
        print("evaluating..")
        metrics = SVM_multioutput_evaluate(df, tfidf_params, TFIDF_DIR, svm_params)
        print("saving the results..")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"tfidf": tfidf_params})
        logger.log_hyperparams({"model": svm_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_metrics(metrics)
    print("finished")
