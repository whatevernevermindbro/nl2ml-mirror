# Baselines and Bigrams: Simple, Good Sentiment and Topic Classification
# Sida Wang and Christopher Manning
# https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
import argparse
import os
import warnings

import dagshub
import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import safe_sparse_dot
from tokenizers import Tokenizer

from common.tools import *


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

TOKENIZER_PATH = "../models/bpe_tokenizer.json"
MODEL_DIR = "../models/nbsvm_regex_graph_v{}.sav".format(GRAPH_VER)
TFIDF_DIR = "../models/tfidf_nbsvm_graph_v{}.pickle".format(GRAPH_VER)

TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
N_TRIALS = 50
MAX_ITER = 10000

SEARCH_SPACE = {
    "tfidf_min_df": (1, 5),
    "tfidf_max_df": (0.8, 1.0),
    "nbsvc_alpha": (1e-3, 10),
    "nbsvc_binarize": (True,),
    "svm_c": (1e-1, 1e3),
}


class NBSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha, svm_params, binarize=False):
        """
        :param alpha: smoothing parameter for Naive Bayes
        """
        super(NBSVC, self).__init__()

        self.alpha = alpha
        self.svm_params = svm_params
        self.binarize = binarize

    def _compute_ratios(self, X, y):
        """
        Computes ratios as in Multinomial Naive Bayes
        """
        n_classes = self.label_encoder.classes_.shape[0]

        self.ratios_ = np.full((n_classes, X.shape[1]), self.alpha)
        self.ratios_ += safe_sparse_dot(y.T, X)
        normalize(self.ratios_, norm="l1", copy=False)

        self.ratios_ = np.log(self.ratios_) - np.log(1 - self.ratios_)

        self.ratios_ = sparse.csr_matrix(self.ratios_)

    def fit(self, X, y=None):
        if self.binarize:
            X = (X > 0).astype(np.float)

        self.label_encoder = LabelBinarizer()
        y = self.label_encoder.fit_transform(y)

        self._compute_ratios(X, y)

        self.clfs_ = []
        for i in range(len(self.label_encoder.classes_)):
            X_i = X.multiply(self.ratios_[i])
            clf = LinearSVC(**self.svm_params).fit(X_i, y[:, i])
            self.clfs_.append(clf)

        return self

    def predict(self, X):
        if self.binarize:
            X = (X > 0).astype(np.float)

        decisions = np.zeros((X.shape[0], self.label_encoder.classes_.shape[0]))

        for i in range(len(self.label_encoder.classes_)):
            X_i = X.multiply(self.ratios_[i])
            decisions[:, i] = self.clfs_[i].decision_function(X_i)

        return self.label_encoder.inverse_transform(decisions, threshold=0)


class Objective:
    def __init__(self, df, kfold_params):
        self.kf = KFold(**kfold_params)
        self.df = df
        self.tokenizer = make_tokenizer(Tokenizer.from_file(TOKENIZER_PATH))

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *SEARCH_SPACE["tfidf_min_df"]),
            "max_df": trial.suggest_loguniform("tfidf__max_df", *SEARCH_SPACE["tfidf_max_df"]),
            "smooth_idf": True,
            "ngram_range": (1, 2),
            "tokenizer": self.tokenizer,
        }
        code_blocks_tfidf = tfidf_fit_transform(self.df[CODE_COLUMN], tfidf_params)
        X, y = code_blocks_tfidf, self.df[TARGET_COLUMN].values

        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *SEARCH_SPACE["svm_c"]),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }

        alpha = trial.suggest_loguniform("nbsvc__alpha", *SEARCH_SPACE["nbsvc_alpha"])
        is_binarized = trial.suggest_categorical("nbsvc__binarize", SEARCH_SPACE["nbsvc_binarize"])
        clf = NBSVC(alpha, svm_params, is_binarized)

        f1_mean, _ = cross_val_scores(self.kf, clf, X, y)
        return f1_mean


def select_hyperparams(df, kfold_params, tfidf_path, model_path):
    """
    Uses optuna to find hyperparams that maximize F1 score
    :param df: labelled dataset
    :param kfold_params: parameters for sklearn's KFold
    :param tfidf_dir: where to save trained tf-idf
    :return: dict with parameters and metrics
    """
    study = optuna.create_study(direction="maximize", study_name="nb-svm", sampler=optuna.samplers.TPESampler())
    objective = Objective(df, kfold_params)

    study.optimize(objective, n_trials=N_TRIALS)

    best_tfidf_params = {
        "smooth_idf": True,
        "ngram_range": (1, 2),
        "tokenizer": make_tokenizer(Tokenizer.from_file(TOKENIZER_PATH))
    }
    best_svm_params = {
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
    }
    best_nb_params = dict()
    for key, value in study.best_params.items():
        model_name, param_name = key.split("__")
        if model_name == "tfidf":
            best_tfidf_params[param_name] = value
        elif model_name == "svm":
            best_svm_params[param_name] = value
        elif model_name == "nbsvc":
            best_nb_params[param_name] = value

    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], best_tfidf_params)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values
    clf = NBSVC(best_nb_params["alpha"], best_svm_params)

    f1_mean, accuracy_mean = cross_val_scores(objective.kf, clf, X, y)

    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(test_f1_score=f1_mean, test_accuracy=accuracy_mean)

    best_tfidf_params["tokenizer"] = "BPE"
    best_svm_params["kernel"] = "linear"
    best_model_params = {
        "svc": best_svm_params,
        "nb": best_nb_params,
    }
    return best_tfidf_params, best_model_params, metrics


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
        "model": MODEL_DIR,
        "script_dir": __file__,
    }

    metrics_path = os.path.join(EXPERIMENT_DATA_PATH, "metrics.csv")
    params_path = os.path.join(EXPERIMENT_DATA_PATH, "params.yml")
    with dagshub.dagshub_logger(metrics_path=metrics_path, hparams_path=params_path) as logger:
        print("selecting hyperparameters")
        tfidf_params, model_params, metrics = select_hyperparams(df, kfold_params, TFIDF_DIR, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"tfidf": tfidf_params})
        logger.log_hyperparams({"model": model_params})
        logger.log_hyperparams({"kfold": kfold_params})
        logger.log_metrics(metrics)
    print("finished")
