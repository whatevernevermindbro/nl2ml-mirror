import argparse
import os
import pickle
import sys

import dagshub
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import itertools, random
import scipy as sp
from sklearn.decomposition import LatentDirichletAllocation

from common.tools import *


parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

MODEL_DIR = "../models/linear_svm_regex_graph_v{}.sav".format(GRAPH_VER)
TFIDF_DIR = "../models/tfidf_svm_graph_v{}.pickle".format(GRAPH_VER)

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
N_TRIALS = 70
MAX_ITER = 10000

HYPERPARAM_SPACE = {
    "svm_c": (1e-1, 1e3),
    "tfidf_min_df": (1, 50),
    "tfidf_max_df": (0.2, 1.0),
    "lda_n_components": (5, 1000),
    "lda_learning_decay": (0.5, 1.0),
    "svm_kernel": ["linear", "poly", "rbf"],
    "svm_degree": (2, 6),  # in case of poly kernel
}


def cross_val_scores(kf, clf, X, y, lda):
    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = lda.fit_transform(X_train)
        X_test = lda.transform(X_test)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    return f1s.mean(), accuracies.mean()


class Objective:
    def __init__(self, df, kfold_params, svm_c, tfidf_min_df, tfidf_max_df, lda_n_components, lda_learning_decay, svm_kernel, svm_degree):
        self.kf = KFold(**kfold_params)
        self.c_range = svm_c
        self.min_df_range = tfidf_min_df
        self.max_df_range = tfidf_max_df
        self.lda_n_components = lda_n_components
        self.lda_learning_decay = lda_learning_decay
        self.kernels = svm_kernel
        self.poly_degrees = svm_degree
        self.df = df

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *self.min_df_range),
            "max_df": trial.suggest_loguniform("tfidf__max_df", *self.max_df_range),
            "smooth_idf": True,
        }

        code_blocks_tfidf = tfidf_fit_transform(self.df[CODE_COLUMN], tfidf_params)
        X, y = code_blocks_tfidf, self.df[TARGET_COLUMN].values

        lda_params = {
            "n_components": trial.suggest_int("lda__n_components", *self.lda_n_components),
            "learning_decay": trial.suggest_uniform("lda__learning_decay", *self.lda_learning_decay),
            "random_state": RANDOM_STATE,
        }
        lda = LatentDirichletAllocation(**lda_params)
        X = lda.fit_transform(X)

        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *self.c_range),
            "kernel": trial.suggest_categorical("svm__kernel", self.kernels),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.poly_degrees)
        clf = SVC(**svm_params)

        f1_mean, _ = cross_val_scores(self.kf, clf, X, y, lda)
        return f1_mean


def select_hyperparams(df, kfold_params, tfidf_path, model_path):
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

    best_tfidf_params = {
        "smooth_idf": True,
    }
    best_svm_params = {
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
    }
    best_lda_params = {
        "random_state": RANDOM_STATE,
    }
    for key, value in study.best_params.items():
        model_name, param_name = key.split("__")
        if model_name == "tfidf":
            best_tfidf_params[param_name] = value
        elif model_name == "svm":
            best_svm_params[param_name] = value
        elif model_name == "lda":
            best_lda_params[param_name] = value

    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], best_tfidf_params, tfidf_path)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values
    lda = LatentDirichletAllocation(**best_lda_params)

    clf = SVC(**best_svm_params)

    f1_mean, accuracy_mean = cross_val_scores(objective.kf, clf, X, y, lda)

    X = lda.fit_transform(X)
    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(test_f1_score=f1_mean, test_accuracy=accuracy_mean)

    return best_tfidf_params, best_svm_params, best_lda_params, metrics


if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 10,
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
        tfidf_params, svm_params, lda_params, metrics = select_hyperparams(df, kfold_params, TFIDF_DIR, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"tfidf": tfidf_params})
        logger.log_hyperparams({"model": svm_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_hyperparams({"lda": lda_params})
        logger.log_metrics(metrics)
    print("finished")
