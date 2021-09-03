import argparse
import logging
import os
import pickle
import sys

import dagshub
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from tokenizers import Tokenizer

from common.tools import *


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

TOKENIZER_PATH = "../models/bpe_tokenizer.json"
MODEL_DIR = "../models/hyper_svm_regex_graph_v{}.sav".format(GRAPH_VER)
TFIDF_DIR = "../models/tfidf_hyper_svm_graph_v{}.pickle".format(GRAPH_VER)

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
N_TRIALS = 1
MAX_ITER = 10000

HYPERPARAM_SPACE = {
    "svm_c": (1e-1, 1e3),
    "tfidf_min_df": (1, 10),
    "tfidf_max_df": (0.2, 0.7),
    "svm_kernel": ["linear", "poly", "rbf"],
    "svm_degree": (2, 6),  # in case of poly kernel
    "masking_rate": (0.5, 1.0)
}


def cross_val_scores(kf, clf, X, y, tfidf_params, masking_rate):
    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = augment_mask(X_train, CODE_COLUMN, masking_rate)

        X_train = tfidf_fit_transform(X_train[CODE_COLUMN], tfidf_params, TFIDF_DIR).toarray()
        X_test = tfidf_transform(X_test[CODE_COLUMN], tfidf_params, TFIDF_DIR).toarray()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    return f1s.mean(), f1s.std(), accuracies.mean(), accuracies.std()


class Objective:
    def __init__(self, df, kfold_params, svm_c, tfidf_min_df, tfidf_max_df, svm_kernel, svm_degree, masking_rate):
        self.kf = KFold(**kfold_params)
        self.c_range = svm_c
        self.min_df_range = tfidf_min_df
        self.max_df_range = tfidf_max_df
        self.kernels = svm_kernel
        self.poly_degrees = svm_degree
        self.masking_rate = masking_rate
        self.df = df
        self.tokenizer = make_tokenizer(Tokenizer.from_file(TOKENIZER_PATH))

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *self.min_df_range),
            "max_df": trial.suggest_loguniform("tfidf__max_df", *self.max_df_range),
            "smooth_idf": True,
            "ngram_range": (1, 2),
            "tokenizer": self.tokenizer,
        }
        X, y = self.df, self.df[TARGET_COLUMN].values

        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *self.c_range),
            "kernel": trial.suggest_categorical("svm__kernel", self.kernels),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }

        masking_rate = trial.suggest_loguniform("augmentation__masking_rate", *self.masking_rate)
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.poly_degrees)

        clf = SVC(**svm_params)

        f1_mean, _, _, _ = cross_val_scores(self.kf, clf, X, y, tfidf_params, masking_rate)
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
    best_bagging_params = {
        "random_state": RANDOM_STATE
    }
    for key, value in study.best_params.items():
        model_name, param_name = key.split("__")
        if model_name == "tfidf":
            best_tfidf_params[param_name] = value
        elif model_name == "svm":
            best_svm_params[param_name] = value
        elif model_name == "bagging":
            best_bagging_params[param_name] = value
        elif model_name == "augmentation":
            best_masking_rate = value

    X, y = df, df[TARGET_COLUMN].values
    clf = SVC(**best_svm_params)

    f1_mean, f1_std, accuracy_mean, accuracy_std = cross_val_scores(objective.kf, clf, X, y, best_tfidf_params, best_masking_rate)

    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(
        test_f1_score=f1_mean,
        test_accuracy=accuracy_mean,
        test_f1_std=f1_std,
        test_accuracy_std=accuracy_std,
    )

    return best_tfidf_params, best_svm_params, best_masking_rate, metrics


if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 15,
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
        tfidf_params, svm_params, masking_rate, metrics = select_hyperparams(df, kfold_params, TFIDF_DIR, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"tfidf": tfidf_params})
        logger.log_hyperparams({"masking_rate": masking_rate})
        logger.log_hyperparams({"model": svm_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_metrics(metrics)
    print("finished")
