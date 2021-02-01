import pickle
from datetime import datetime
import argparse
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
import dagshub

sys.path.append('./models_scripts/common/')
from tools import *

# def SVM_evaluate(df, code_blocks, tfidf_params, TFIDF_DIR, SVM_params):
#     code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
#     X_train, X_test, y_train, y_test = train_test_split(code_blocks_tfidf, df[TAGS_TO_PREDICT], test_size=0.3)
#     # grid = {"C": [100]}
#     # cv = KFold(n_splits=2, shuffle=True, random_state=241)
#     model = SVC(kernel="linear", random_state=241)
#     # gs = GridSearchCV(model, grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)
#     # gs.fit(X_train[:25000], y_train.ravel()[:25000])
#     # C = gs.best_params_.get('C')
#     # model = SVC(**SVM_params)
#     print("Train SVM params:", model.get_params())
#     n_estimators = 10
#     clf = BaggingClassifier(model, max_samples=1.0 / n_estimators, n_estimators=n_estimators)
#     # clf = model
#     print("starting training..")
#     clf.fit(X_train, y_train)
#     print("saving the model")
#     pickle.dump(clf, open(MODEL_DIR, 'wb'))
#     print("predicting on the test..")
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     # confus_matrix = confusion_matrix(model, X_test, y_test)
#     metrics = {'test_accuracy': accuracy
#             , 'test_f1_score': f1}
#     print(metrics)
#     return metrics

def SVM_multioutput_evaluate(df, tfidf_params, TFIDF_DIR, SVM_params):
    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], tfidf_params, TFIDF_DIR)
    print(df.columns)
    X_train, X_test, y_train, y_test = train_test_split(code_blocks_tfidf, df[TAGS_TO_PREDICT], test_size=0.3)
    # grid = {"C": [100]}
    # cv = KFold(n_splits=2, shuffle=True, random_state=241)
    model = SVC(kernel="linear", random_state=241)
    # gs = GridSearchCV(model, grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)
    # gs.fit(X_train[:25000], y_train.ravel()[:25000])
    # C = gs.best_params_.get('C')
    # model = SVC(**SVM_params)
    print("Train SVM params:", model.get_params())
    n_estimators = 10
    clf = MultiOutputRegressor(BaggingClassifier(model, max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    # clf = model
    print("starting training..")
    clf.fit(X_train, y_train)
    print("saving the model")
    pickle.dump(clf, open(MODEL_DIR, 'wb'))
    print("predicting on the test..")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # confus_matrix = confusion_matrix(model, X_test, y_test)
    metrics = {'test_accuracy': accuracy
            , 'test_f1_score': f1}
    print(metrics)
    return metrics

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=int)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH
MODEL_DIR = './models/svm_regex_graph_v{}.sav'.format(GRAPH_VER)
TFIDF_DIR = './models/tfidf_svm_graph_v{}.pickle'.format(GRAPH_VER)
CODE_COLUMN = 'code_block'
TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)
print(TAGS_TO_PREDICT)
PREDICT_COL = 'pred_{}'.format(TAGS_TO_PREDICT)
SCRIPT_DIR = __file__
TASK = 'training SVM'

if __name__ == '__main__':
    df = load_data(DATASET_PATH)
    print(df.columns)
    nrows = df.shape[0]
    print("loaded")
    tfidf_params = {'min_df': 5
            , 'max_df': 0.3
            , 'smooth_idf': True}
    SVM_params = {'C':100
            , 'kernel':"linear"
            , 'random_state':241}
    data_meta = {'DATASET_PATH': DATASET_PATH
                ,'nrows': nrows
                ,'label': TAGS_TO_PREDICT
                ,'model': MODEL_DIR
                ,'script_dir': SCRIPT_DIR}

    with dagshub.dagshub_logger() as logger:
        print("evaluating..")
        metrics = SVM_multioutput_evaluate(df, tfidf_params, TFIDF_DIR, SVM_params)
        print("saving the results..")
        logger.log_hyperparams(data_meta)
        logger.log_hyperparams(tfidf_params)
        logger.log_hyperparams(SVM_params)
        logger.log_metrics(metrics)
    print("finished")