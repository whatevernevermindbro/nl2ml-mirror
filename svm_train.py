import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor

import dagshub

def load_code_blocks(DATASET_PATH, CODE_COLUMN):
    df = pd.read_csv(DATASET_PATH, encoding='utf-8', comment='#', sep=',')#, quoting=csv.QUOTE_NONE, error_bad_lines=False)#, sep=','
    df.dropna(axis=0, inplace=True)
    code_blocks = df[CODE_COLUMN]
    # test_size = 0.1
    # test_rows = round(df.shape[0]*test_size)
    # train_rows = df.shape[0] - test_rows
    # train_code_blocks = df[CODE_COLUMN][0:test_rows]
    # test_code_blocks = df[CODE_COLUMN][train_rows:]
    return df, code_blocks

def tfidf_fit_transform(code_blocks, params, TFIDF_DIR):
    vectorizer = TfidfVectorizer(**params)
    tfidf = vectorizer.fit(code_blocks)
    pickle.dump(tfidf, open(TFIDF_DIR, "wb"))
    print('TF-IDF model has been saved')
    code_blocks_tfidf = tfidf.transform(code_blocks)
    return code_blocks_tfidf

def SVM_evaluate(df, code_blocks, tfidf_params, TFIDF_DIR, SVM_params):
    code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
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
    clf = BaggingClassifier(model, max_samples=1.0 / n_estimators, n_estimators=n_estimators)
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

def SVM_multioutput_evaluate(df, code_blocks, tfidf_params, TFIDF_DIR, SVM_params):
    code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
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

if __name__ == '__main__':
    GRAPH_VERSION = 5
    DATASET_PATH = './data/code_blocks_regex_graph_v{}.csv'.format(GRAPH_VERSION)
    MODEL_DIR = './models/svm_regex_graph_v{}.sav'.format(GRAPH_VERSION)
    TFIDF_DIR = './models/tfidf_svm_graph_v{}.pickle'.format(GRAPH_VERSION)
    CODE_COLUMN = 'code_block'
    TAGS_TO_PREDICT = ['import', 'data_import', 'data_export', 'preprocessing',
                        'visualization', 'model', 'deep_learning_model', 'train', 'predict']
    SCRIPT_DIR = __file__
    
    df, code_blocks = load_code_blocks(DATASET_PATH, CODE_COLUMN)
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
        metrics = SVM_multioutput_evaluate(df, code_blocks, tfidf_params, TFIDF_DIR, SVM_params)
        print("saving the results..")
        logger.log_hyperparams(data_meta)
        logger.log_hyperparams(tfidf_params)
        logger.log_hyperparams(SVM_params)
        logger.log_metrics(metrics)
    print("finished")