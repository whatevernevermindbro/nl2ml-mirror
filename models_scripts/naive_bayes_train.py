import argparse
import os
import pickle
import sys

import dagshub
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB

from common.tools import *


parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

MODEL_DIR = "../models/nb_graph_v{}.sav".format(GRAPH_VER)
COUNTVEC_DIR = "../models/countvec_nb_graph_v{}.pickle".format(GRAPH_VER)

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
N_TRIALS = 100

HYPERPARAM_SPACE = {
    "cntvec_min_df": (1, 50),
    "cntvec_max_df": (0.2, 1.0),
    "nb_type": ("ComplementNB", "MultinomialNB"),
    "nb_alpha": (1e-6, 10),
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


class Objective:
    def __init__(self, df, kfold_params, cntvec_min_df, cntvec_max_df, nb_type, nb_alpha):
        self.kf = StratifiedKFold(**kfold_params)
        self.min_df_range = cntvec_min_df
        self.max_df_range = cntvec_max_df
        self.nbs = nb_type
        self.alpha_range = nb_alpha
        self.df = df

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("cntvec__min_df", *self.min_df_range),
            "max_df": trial.suggest_loguniform("cntvec__max_df", *self.max_df_range),
        }
        code_blocks_tfidf = count_fit_transform(self.df[CODE_COLUMN], tfidf_params)
        X, y = code_blocks_tfidf, self.df[TARGET_COLUMN].values

        nb_params = {
            "alpha": trial.suggest_loguniform("nb__alpha", *self.alpha_range)
        }
        nb_type = trial.suggest_categorical("nb__nb_type", self.nbs)

        if nb_type == "ComplementNB":
            clf = ComplementNB(**nb_params)
        else:
            clf = MultinomialNB(**nb_params)

        f1_mean, _ = cross_val_scores(self.kf, clf, X, y)
        return f1_mean


def select_hyperparams(df, kfold_params, cntvec_path, model_path):
    """
    Uses optuna to find hyperparams that maximize F1 score
    :param df: labelled dataset
    :param kfold_params: parameters for sklearn's KFold
    :param tfidf_dir: where to save trained tf-idf
    :return: dict with parameters and metrics
    """
    study = optuna.create_study(direction="maximize", study_name="svm with kernels")
    objective = Objective(df, kfold_params, **HYPERPARAM_SPACE)

    study.optimize(objective, n_trials=N_TRIALS)

    best_cntvec_params = dict()
    best_nb_params = dict()
    for key, value in study.best_params.items():
        model_name, param_name = key.split("__")
        if model_name == "cntvec":
            best_cntvec_params[param_name] = value
        elif model_name == "nb":
            best_nb_params[param_name] = value

    code_blocks_tfidf = count_fit_transform(df[CODE_COLUMN], best_cntvec_params, cntvec_path)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values

    nb_type = best_nb_params["nb_type"]
    best_nb_params.pop("nb_type")
    if nb_type == "ComplementNB":
        clf = ComplementNB(**best_nb_params)
    else:
        clf = MultinomialNB(**best_nb_params)
    best_nb_params["nb_type"] = nb_type

    f1_mean, accuracy_mean = cross_val_scores(objective.kf, clf, X, y)

    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(test_f1_score=f1_mean, test_accuracy=accuracy_mean)

    return best_cntvec_params, best_nb_params, metrics


if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 6,
        "random_state": RANDOM_STATE,
        "shuffle": True,
    }
    data_meta = {
        "DATASET_PATH": DATASET_PATH,
        "nrows": nrows,
        "label": TAGS_TO_PREDICT,
        "model": MODEL_DIR,
        "script_dir": __file__,
    }

    metrics_path = os.path.join(EXPERIMENT_DATA_PATH, "metrics.csv")
    params_path = os.path.join(EXPERIMENT_DATA_PATH, "params.yml")

    with dagshub.dagshub_logger(metrics_path=metrics_path, hparams_path=params_path) as logger:
        print("selecting hyperparameters")
        cntvec_params, nb_params, metrics = select_hyperparams(df, kfold_params, COUNTVEC_DIR, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"countvec": cntvec_params})
        logger.log_hyperparams({"model": nb_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_metrics(metrics)
    print("finished")
