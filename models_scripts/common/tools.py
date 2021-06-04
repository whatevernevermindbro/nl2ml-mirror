import pickle
import json

import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


GRAPH_PATH = "../data/actual_graph.csv"


def load_data(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, encoding='utf-8', comment='#', sep=',')
    df.dropna(axis=0, inplace=True)
    return df


def get_graph_vertices(GRAPH_VER):
    GRAPH_DIR = '../graph/graph_v{}.txt'.format(GRAPH_VER)
    with open(GRAPH_DIR, "r") as graph_file:
        graph = json.load(graph_file)
        vertices = list(graph.keys())
    print('vertices parsed: {}'.format(vertices))
    return vertices


def tfidf_transform(corpus, tfidf_params, TFIDF_DIR):
    tfidf = pickle.load(open(TFIDF_DIR, 'rb'))
    features = tfidf.transform(corpus)
    return features


def tfidf_fit_transform(code_blocks: pd.DataFrame, tfidf_params: dict, tfidf_path=None):
    tfidf = TfidfVectorizer(**tfidf_params).fit(code_blocks)
    if tfidf_path is not None:
        pickle.dump(tfidf, open(tfidf_path, "wb"))
    code_blocks_tfidf = tfidf.transform(code_blocks)
    code_blocks_tfidf.sort_indices()
    return code_blocks_tfidf


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_metrics(X, y, TAGS_TO_PREDICT, MODEL_DIR):
    clf = pickle.load(open(MODEL_DIR, 'rb'))
    print("the model has been loaded")
    y_pred = clf.predict(X)
    print("predictions were calculated")
    accuracy = clf.score(X, y)
    f1 = f1_score(y_pred, y, average='weighted')
    print(f'Mean Accuracy {round(accuracy*100, 2)}%')
    print(f'F1-score {round(f1*100, 2)}%')
    metrics = {'test_accuracy': accuracy, 'test_f1_score': f1}
    return X, y, y_pred, metrics
