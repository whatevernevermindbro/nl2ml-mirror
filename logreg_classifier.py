##TODO:
# Arguments from the input
# Excessive functions -> parameters
# Common module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import dagshub
import pickle

def load_corpus(DATASET_PATH, CODE_COLUMN):
    df = pd.read_csv(DATASET_PATH, encoding='utf-8', comment='#', sep='\t')#, quoting=csv.QUOTE_NONE, error_bad_lines=False)#, sep=','
    df.dropna(axis=0, inplace=True)
    corpus = df[CODE_COLUMN]
    test_size = 0.1
    test_rows = round(df.shape[0]*test_size)
    train_rows = df.shape[0] - test_rows
    train_corpus = df[CODE_COLUMN][0:test_rows]
    test_corpus = df[CODE_COLUMN][train_rows:]
    return df, corpus

def tfidf_transform(corpus, tfidf_params, TFIDF_DIR):
    tfidf = pickle.load(open(TFIDF_DIR, 'rb'))
    features = tfidf.transform(corpus)
    return features

def tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR):
    tfidf = TfidfVectorizer(tfidf_params)
    print(code_blocks.head())
    tfidf = tfidf.fit(code_blocks)
    pickle.dump(tfidf, open(TFIDF_DIR, "wb"))
    code_blocks_tfidf = tfidf.transform(code_blocks)
    return code_blocks_tfidf

def logreg_evaluate(df, code_blocks, TAG_TO_PREDICT):
    code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
    X_train, X_test, y_train, y_test = train_test_split(code_blocks_tfidf, df[TAG_TO_PREDICT], test_size=0.25)
    clf = LogisticRegression(random_state=421).fit(X_train, y_train)
    print("saving the model")
    pickle.dump(clf, open(MODEL_DIR, 'wb'))
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    f1 = f1_score(y_pred, y_test, average='weighted')
    print(f'Mean Accuracy {round(accuracy*100, 2)}%')
    print(f'F1-score {round(f1*100, 2)}%')
    errors = y_test - y_pred
    plt.hist(errors)
    plot_precision_recall_curve(clf, X_test, y_test)
    plot_confusion_matrix(clf, X_test, y_test, values_format='d')
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h
    conf_interval = mean_confidence_interval(errors, 0.95)
    print(conf_interval)
    metrics = {'test_accuracy': accuracy
               , 'test_f1_score': f1}
    return metrics

def logreg_multioutput_evaluate(df, code_blocks, TAGS_TO_PREDICT):
    code_blocks_tfidf = tfidf_fit_transform(code_blocks, tfidf_params, TFIDF_DIR)
    print("splitting")
    X_train, X_test, Y_train, Y_test = train_test_split(code_blocks_tfidf, df[TAGS_TO_PREDICT], test_size=0.25)
    print("training the model")
    clf = MultiOutputRegressor(LogisticRegression(random_state=421)).fit(X_train, Y_train)
    print("saving the model")
    pickle.dump(clf, open(MODEL_DIR, 'wb'))
    Y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, Y_test)
    f1 = f1_score(Y_pred, Y_test, average='weighted')
    print(f'Mean Accuracy {round(accuracy*100, 2)}%')
    print(f'F1-score {round(f1*100, 2)}%')
    # errors = Y_test - Y_pred
    # plt.hist(errors)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    # plot_confusion_matrix(clf, X_test, Y_test, values_format='d')
    # def mean_confidence_interval(data, confidence=0.95):
    #     a = 1.0 * np.array(data)
    #     n = len(a)
    #     m, se = np.mean(a), scipy.stats.sem(a)
    #     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    #     return m, m-h, m+h
    # conf_interval = mean_confidence_interval(errors, 0.95)
    # print(conf_interval)
    metrics = {'test_accuracy': accuracy
               , 'test_f1_score': f1}
    return metrics

def get_predictions(X, y, TAGS_TO_PREDICT, MODEL_DIR):
    clf = pickle.load(open(MODEL_DIR, 'rb'))
    # result = loaded_model.score(X, y)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_pred, y)
    f1 = f1_score(y_pred, y, average='weighted')
    print(f'Mean Accuracy {round(accuracy*100, 2)}%')
    print(f'F1-score {round(f1*100, 2)}%')
    errors = y - y_pred
    plt.hist(errors)
    plot_precision_recall_curve(clf, X, y)
    plot_confusion_matrix(clf, X, y, values_format='d')
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h
    conf_interval = mean_confidence_interval(errors, 0.95)
    print(conf_interval)
    metrics = {'test_accuracy': accuracy
               , 'test_f1_score': f1}
    return metrics

GRAPH_VER = 5
DATASET_PATH = './data/kaggle_10_regex_v{}.csv'.format(GRAPH_VER)
MODEL_DIR = './models/logreg_regex_graph_v{}.sav'.format(GRAPH_VER)
TFIDF_DIR = './models/tfidf_logreg_graph_v{}.pickle'.format(GRAPH_VER)
CODE_COLUMN = 'code_block'
TAGS_TO_PREDICT = ['import', 'data_import', 'data_export', 'preprocessing',
                    'visualization', 'model', 'train', 'predict']
PREDICT_COL = 'pred_{}'.format(TAGS_TO_PREDICT)
SCRIPT_DIR = 'logreg_classifier.ipynb'

VAL_CHUNK_SIZE = 10
VAL_CODE_COLUMN = 'code'
VAL_TAGS_TO_PREDICT = 'tag'
VAL_DATASET_PATH = './data/chunks_{}_validate.csv'.format(VAL_CHUNK_SIZE)

if __name__ == '__main__':
    df, code_blocks = load_corpus(DATASET_PATH, CODE_COLUMN)
    nrows = df.shape[0]
    print("loaded")
    tfidf_params = {'min_df': 5
                    , 'max_df': 0.3
                    , 'smooth_idf': True}
    data_meta = {'DATASET_PATH': DATASET_PATH
                ,'nrows': nrows
                ,'label': TAGS_TO_PREDICT
                ,'model': MODEL_DIR
                ,'script_dir': SCRIPT_DIR
                ,'task': 'training and evaluation'}
    print("tfidf-ed")
    with dagshub.dagshub_logger() as logger:
        metrics = logreg_multioutput_evaluate(df, code_blocks, TAGS_TO_PREDICT)
        # metrics = get_predictions(features, df[TAGS_TO_PREDICT], TAGS_TO_PREDICT, MODEL_DIR)
        logger.log_hyperparams(data_meta)
        logger.log_hyperparams(tfidf_params)
        logger.log_metrics(metrics)
    print("finished")