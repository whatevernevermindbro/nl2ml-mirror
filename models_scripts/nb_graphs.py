import argparse
import os
import pickle
import sys

import dagshub
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from tokenizers import Tokenizer

from common.tools import *


plt.rcParams["axes.labelsize"] = 12


parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

TOKENIZER_PATH = "../models/bpe_tokenizer.json"

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42

TFIDF_STEPS = 25
TFIDF_PARAM_SPACE = {
    "min_df": (1, 50),
    "max_df": (-2500, -0),
}

NB_STEPS = 10
NB_TYPE = "Bernoulli"
NB_PARAM_SPACE = {
    "alpha": (1e-3, 1),
}

KFOLD_PARAMS = {
    "n_splits": 3,
    "random_state": RANDOM_STATE,
    "shuffle": True,
}


def cross_val_scores(kf, clf, X, y):
    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    return f1s.mean(), accuracies.mean()


def prepare_metrics(df):
    tokenizer = make_tokenizer(Tokenizer.from_file(TOKENIZER_PATH))
    kf = StratifiedKFold(**KFOLD_PARAMS)

    accuracies = []
    f1_scores = []

    train_docs = df.shape[0]

    min_dfs = np.linspace(*TFIDF_PARAM_SPACE["min_df"], TFIDF_STEPS).astype(np.int)
    max_dfs = np.linspace(
        train_docs + TFIDF_PARAM_SPACE["max_df"][0],
        train_docs + TFIDF_PARAM_SPACE["max_df"][1],
        TFIDF_STEPS
    ).astype(np.int)
    for min_df in min_dfs:
        for max_df in max_dfs:
            tfidf_params = {
                "min_df": min_df,
                "max_df": max_df,
                "smooth_idf": True,
                "tokenizer": tokenizer,
                "token_pattern": None,
            }
            code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], tfidf_params)
            X, y = code_blocks_tfidf, df[TARGET_COLUMN].values

            best_metrics = None
            for alpha in np.linspace(*NB_PARAM_SPACE["alpha"], NB_STEPS):
                if NB_TYPE == "Multinomial":
                    clf = MultinomialNB(alpha=alpha)
                else:
                    clf = BernoulliNB(alpha=alpha)

                metrics = cross_val_scores(kf, clf, X, y)

                if best_metrics is None or metrics[0] > best_metrics[0]:
                    best_metrics = metrics

            f1_scores.append(best_metrics[0])
            accuracies.append(best_metrics[1])

    f1_scores = np.array(f1_scores).reshape((TFIDF_STEPS, -1))
    accuracies = np.array(accuracies).reshape((TFIDF_STEPS, -1))

    f1_scores = pd.DataFrame(f1_scores.T, columns=min_dfs, index=max_dfs)
    accuracies = pd.DataFrame(accuracies.T, columns=min_dfs, index=max_dfs)
    return f1_scores, accuracies


def save_heatmap(data, title, xlabel, ylabel, path):
    fig = plt.figure()
    ax = sns.heatmap(data)
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    print("loaded")

    f1_scores, accuracies = prepare_metrics(df)

    print("finished metrics. drawing heatmaps")

    save_heatmap(
        f1_scores,
        "F1-мера",
        "Минимальное число документов,\n содержащих токен (min_df)",
        "Максимальное число документов,\n содержащих токен (max_df)",
        f"./heatmap_f1score_{NB_TYPE}.pdf"
    )
    save_heatmap(
        accuracies,
        "Доля верных ответов",
        "Минимальное число документов,\n содержащих токен (min_df)",
        "Максимальное число документов,\n содержащих токен (max_df)",
        f"./heatmap_accuracy_{NB_TYPE}.pdf"
    )

    print("done")
