import argparse
import logging
import os
import pickle
import sys

import dagshub
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from common.tools import *


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

MODEL_DIR = "../models/hyper_svm_regex_graph_v{}.sav".format(GRAPH_VER)
TFIDF_DIR = "../models/tfidf_hyper_svm_graph_v{}.pickle".format(GRAPH_VER)

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
N_TRIALS = 500
MAX_ITER = 10000

HYPERPARAM_SPACE = {
    "svm_c": (1e-2, 1e3),
    "tfidf_min_df": (1, 10),
    "tfidf_max_df": (0.2, 0.7),
    "svm_kernel": ["linear", "poly", "rbf"],
    "svm_degree": (2, 3),  # in case of poly kernel
    "context_size": (1, 30),
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
    return f1s.mean(), f1s.std(), accuracies.mean(), accuracies.std()


class Objective:
    def __init__(self, df, kfold_params, search_space):
        self.kf = StratifiedKFold(**kfold_params)
        self.hparam_space = search_space
        self.df = df

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *self.hparam_space["tfidf_min_df"]),
            "max_df": trial.suggest_loguniform("tfidf__max_df", *self.hparam_space["tfidf_max_df"]),
            "smooth_idf": True,
        }
        code_blocks_tfidf = tfidf_fit_transform(self.df[CODE_COLUMN], tfidf_params)
        X, y = code_blocks_tfidf.toarray(), self.df[TARGET_COLUMN].values
        
        context_size = trial.suggest_int("ctx__context_size", *self.hparam_space["context_size"])
        X_ctx = ohe_context(self.df["context"], context_size, y.max())
        X = sp.hstack([X, X_ctx], format="csr")

        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *self.hparam_space["svm_c"]),
            "kernel": trial.suggest_categorical("svm__kernel", self.hparam_space["svm_kernel"]),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.hparam_space["svm_degree"])

        clf = SVC(**svm_params)

        f1_mean, _, _, _ = cross_val_scores(self.kf, clf, X, y)
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
    objective = Objective(df, kfold_params, HYPERPARAM_SPACE)

    study.optimize(objective, n_trials=N_TRIALS)

    best_tfidf_params = {
        "smooth_idf": True,
    }
    best_svm_params = {
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
    }
    best_ctx_params = dict()
    for key, value in study.best_params.items():
        model_name, param_name = key.split("__")
        if model_name == "tfidf":
            best_tfidf_params[param_name] = value
        elif model_name == "svm":
            best_svm_params[param_name] = value
        elif model_name == "ctx":
            best_ctx_params[param_name] = value

    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], best_tfidf_params, tfidf_path)
    X, y = code_blocks_tfidf.toarray(), df[TARGET_COLUMN].values
    X_ctx = ohe_context(df["context"], best_ctx_params["context_size"], y.max())
    X = sp.hstack([X, X_ctx], format="csr")

    clf = SVC(**best_svm_params)

    f1_mean, f1_std, accuracy_mean, accuracy_std = cross_val_scores(objective.kf, clf, X, y)

    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(
        test_f1_score=f1_mean,
        test_accuracy=accuracy_mean,
        test_f1_std=f1_std,
        test_accuracy_std=accuracy_std,
    )

    return best_tfidf_params, best_svm_params, best_ctx_params, metrics


if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 3,
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
        tfidf_params, svm_params, ctx_params, metrics = select_hyperparams(df, kfold_params, TFIDF_DIR, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"tfidf": tfidf_params})
        logger.log_hyperparams({"model": svm_params})
        logger.log_hyperparams({"context": ctx_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_metrics(metrics)
    print("finished")
