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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import itertools, random
import scipy as sp

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
    "rff_n_features": (10, 1000),
    "rff_pca_dim": (5, 100),
    "svm_kernel": ["linear", "poly", "rbf"],
    "svm_degree": (2, 6),  # in case of poly kernel
}


class RFF:
    def __init__(self, n_features, pca_dim, random_state):
        self.n_features = n_features
        self.pca_dim = pca_dim
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        if not isinstance(X, np.ndarray):
            X = np.array(X.todense())
        if self.pca_dim > min(X.shape[0], X.shape[1]):
            self.pca_dim = min(X.shape[0], X.shape[1])

        self.pca = PCA(n_components=self.pca_dim)

        X_trans = self.pca.fit_transform(X)
        pairs = rng.choice(range(X_trans.shape[0]), 10000)
        pairs2 = rng.choice(range(1, X_trans.shape[0] - 2), 10000)
        pairs2 = (pairs + pairs2) % X_trans.shape[0]

        if not isinstance(X_trans, np.ndarray):
            X_new = np.array((X_trans[pairs] - X_trans[pairs2]).todense())
        else:
            X_new = X_trans[pairs] - X_trans[pairs2]
        square_sums = []
        for row in X_new:
            s = 0
            for elem in row.flatten():
                # print(elem)
                s += elem ** 2
            square_sums.append(s)
        sigma2 = np.median(square_sums)

        self.w = rng.normal(0, 1.0/np.sqrt(sigma2), size=(self.n_features, X_trans.shape[1]))
        self.b = rng.uniform(-np.pi, np.pi, size=(self.n_features, 1))

        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X.todense())
        X_trans = self.pca.transform(X)
        return (np.cos(self.w @ X_trans.T + self.b)).T

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



def cross_val_scores(kf, clf, X, y, rff):
    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = rff.fit_transform(X_train, y_train)
        X_test = rff.transform(X_test)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    return f1s.mean(), accuracies.mean()


class Objective:
    def __init__(self, df, kfold_params, svm_c, tfidf_min_df, tfidf_max_df, rff_n_features, rff_pca_dim, svm_kernel, svm_degree):
        self.kf = KFold(**kfold_params)
        self.c_range = svm_c
        self.min_df_range = tfidf_min_df
        self.max_df_range = tfidf_max_df
        self.rff_n_features = rff_n_features
        self.rff_pca_dim = rff_pca_dim
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

        rff_params = {
            "n_features": trial.suggest_int("rff__n_features", *self.rff_n_features),
            "pca_dim": trial.suggest_int("rff__pca_dim", *self.rff_pca_dim)
        }
        rff = RFF(rff_params["n_features"], rff_params["pca_dim"], RANDOM_STATE)
        X = rff.fit_transform(X, y)


        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *self.c_range),
            "kernel": trial.suggest_categorical("svm__kernel", self.kernels),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.poly_degrees)
        clf = SVC(**svm_params)

        f1_mean, _ = cross_val_scores(self.kf, clf, X, y, rff)
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
    best_rff_params = {
        "random_state": RANDOM_STATE,
    }
    for key, value in study.best_params.items():
        model_name, param_name = key.split("__")
        if model_name == "tfidf":
            best_tfidf_params[param_name] = value
        elif model_name == "svm":
            best_svm_params[param_name] = value
        elif model_name == "rff":
            best_rff_params[param_name] = value

    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], best_tfidf_params, tfidf_path)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values
    rff = RFF(best_rff_params["n_features"], best_rff_params["pca_dim"], RANDOM_STATE)

    clf = SVC(**best_svm_params)

    f1_mean, accuracy_mean = cross_val_scores(objective.kf, clf, X, y, rff)

    X = rff.fit_transform(X, y)
    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(test_f1_score=f1_mean, test_accuracy=accuracy_mean)

    return best_tfidf_params, best_svm_params, best_rff_params, metrics


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
        tfidf_params, svm_params, rff_params, metrics = select_hyperparams(df, kfold_params, TFIDF_DIR, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"tfidf": tfidf_params})
        logger.log_hyperparams({"model": svm_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_hyperparams({"rff": rff_params})
        logger.log_metrics(metrics)
    print("finished")
