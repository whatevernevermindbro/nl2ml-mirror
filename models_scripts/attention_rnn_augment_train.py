import argparse
from collections import defaultdict
import os
import pickle
import shutil

import dagshub
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from common.nn_tools import *


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

MODEL_DIR = "../models/rnn_codebert_graph_v{}.pt".format(GRAPH_VER)
CHECKPOINT_PATH_TEMPLATE = "../checkpoints/rnn_codebert_trial{}.pt"
LEARNING_HISTORY_PATH_TEMPLATE = "../checkpoints/rnn_codebert_history_trial{}.pickle"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 768
MAX_SEQUENCE_LENGTH = 512  # this is required by transformers for some reason
RANDOM_SEED = 42

N_EPOCHS = 50
N_TRIALS = 30

torch.manual_seed(RANDOM_SEED)

SPECIAL_TOKENS = []
for i in range(50):
    SPECIAL_TOKENS.append("[VAR" + str(i) + "]")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", additional_special_tokens=SPECIAL_TOKENS)
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)

EXPERIMENT_DATA_PATH = "../attention_rnn"
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

SEARCH_SPACE = {
    "rnn_size": (32, 256),
    "rnn_layers": (1, 4),
    "lin_size": (16, 256),
    "masking_rate": (0.5, 1.0),
}


class CodeblocksDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(CodeblocksDataset, self).__init__()

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            "code": self.data.iloc[idx][CODE_COLUMN],
            "label": self.data.iloc[idx][TARGET_COLUMN]
        }


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Classifier(nn.Module):
    def __init__(self, rnn_size, rnn_layers, lin_size, n_classes):
        super(Classifier, self).__init__()

        self.rnn = nn.LSTM(
            EMBEDDING_SIZE, rnn_size, num_layers=rnn_layers,
            batch_first=True, dropout=0.25, bidirectional=True
        )

        self.attention = Attention(2 * rnn_size, MAX_SEQUENCE_LENGTH)

        self.decoder = nn.Sequential(
            nn.Linear(2 * rnn_size, lin_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(lin_size, n_classes)
        )

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print("Total param size: {}".format(size))

    def forward(self, data):
        x, lengths = data
        # initial shape (batch_size, time, features)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.to("cpu"), batch_first=True)
        out, _ = self.rnn(x)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=MAX_SEQUENCE_LENGTH)
        mask = (
            torch.arange(MAX_SEQUENCE_LENGTH).expand(lengths.size(0), MAX_SEQUENCE_LENGTH).to(DEVICE) < lengths.unsqueeze(1)
        )
        out = self.attention(out, mask)

        return self.decoder(out)


def prep_data():
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df.drop_duplicates(inplace=True)
    codes, uniques = pd.factorize(df[TARGET_COLUMN])
    df[TARGET_COLUMN] = codes
    df.dropna(inplace=True)
    return df, len(uniques)


class DataProcessor(object):
    def __init__(self, masking_rate):
        self.masking_rate = masking_rate

    def __call__(self, batch):
        batch = augment_mask_list(batch, self.masking_rate)
        labels = torch.LongTensor([obj["label"] for obj in batch])

        tokens = []
        lengths = []
        for obj in batch:
            code_tokens = tokenizer.tokenize(obj["code"], truncation=True, max_length=MAX_SEQUENCE_LENGTH)
            token_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            lengths.append(len(token_ids))
            tokens.append(torch.tensor(token_ids))

        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        lengths = torch.LongTensor(lengths)
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        tokens = tokens[indices]
        with torch.no_grad():
            tokens = codebert_model(tokens.to(DEVICE))[0]
        # pack = nn.utils.rnn.pack_padded_sequence(tokens, sorted_lengths, batch_first=True)
        return (tokens, sorted_lengths), labels[indices]



def process_data(batch):
    labels = torch.LongTensor([obj["label"] for obj in batch])

    tokens = []
    lengths = []
    for obj in batch:
        code_tokens = tokenizer.tokenize(obj["code"], truncation=True, max_length=MAX_SEQUENCE_LENGTH)
        token_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        lengths.append(len(token_ids))
        tokens.append(torch.tensor(token_ids))

    tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    lengths = torch.LongTensor(lengths)
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tokens = tokens[indices]
    with torch.no_grad():
        tokens = codebert_model(tokens.to(DEVICE))[0]
    # pack = nn.utils.rnn.pack_padded_sequence(tokens, sorted_lengths, batch_first=True)
    return (tokens, sorted_lengths), labels[indices]


def train_new_model(df_train, df_test, n_epochs, params, masking_rate, lr=3e-1):
    model = Classifier(**params)
    model = model.to(DEVICE)

    data_processor = DataProcessor(masking_rate)

    train_dataloader = torch.utils.data.DataLoader(
        CodeblocksDataset(df_train), batch_size=16, collate_fn=data_processor, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        CodeblocksDataset(df_test), batch_size=16, collate_fn=data_processor
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=n_epochs
    )
    criterion = FocalLoss()

    history = defaultdict(list)
    for epoch in range(n_epochs):
        train_loss, train_acc, train_f1 = train_with_augment(model, DEVICE, train_dataloader, epoch, criterion,
                                                             optimizer, CODE_COLUMN, masking_rate)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        print("evaluating")
        test_loss, test_acc, test_f1 = test(model, DEVICE, test_dataloader, epoch, criterion)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)
        scheduler.step()

    return model, history


class Objective:
    def __init__(self, n_classes, df_train, df_test):
        self.n_classes = n_classes
        self.df_train = df_train
        self.df_test = df_test

    def __call__(self, trial):
        params = {
            "rnn_size": trial.suggest_int("rnn_size", *SEARCH_SPACE["rnn_size"]),
            "rnn_layers": trial.suggest_int("rnn_layers", *SEARCH_SPACE["rnn_layers"]),
            "lin_size": trial.suggest_int("lin_size", *SEARCH_SPACE["lin_size"]),
            "n_classes": self.n_classes,
        }
        masking_rate = trial.suggest_uniform("masking_rate", *SEARCH_SPACE["masking_rate"])
        model, history = train_new_model(self.df_train, self.df_test, N_EPOCHS, params, masking_rate)

        checkpoint_path = CHECKPOINT_PATH_TEMPLATE.format(trial.number)
        history_path = LEARNING_HISTORY_PATH_TEMPLATE.format(trial.number)
        torch.save(model.state_dict(), checkpoint_path)
        pickle.dump(history, open(history_path, "wb"))

        best_f1 = np.array(history["test_f1"]).max()
        return best_f1


def select_hyperparams(df, n_classes, model_path):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    study = optuna.create_study(direction="maximize", study_name="rnn with codebert", sampler=optuna.samplers.TPESampler())
    objective = Objective(n_classes, df_train, df_test)

    study.optimize(objective, n_trials=N_TRIALS)

    model_params = study.best_params
    model_params["n_classes"] = n_classes

    best_checkpoint_path = CHECKPOINT_PATH_TEMPLATE.format(study.best_trial.number)
    best_history_path = LEARNING_HISTORY_PATH_TEMPLATE.format(study.best_trial.number)

    history = pickle.load(open(best_history_path, "rb"))

    shutil.copy(best_checkpoint_path, model_path)
    return model_params, history


if __name__ == "__main__":
    df, n_classes = prep_data()

    data_meta = {
        "DATASET_PATH": DATASET_PATH,
        "model": MODEL_DIR,
        "script_dir": __file__,
        "random_seed": RANDOM_SEED,
    }

    metrics_path = os.path.join(EXPERIMENT_DATA_PATH, "cnn_metrics.csv")
    params_path = os.path.join(EXPERIMENT_DATA_PATH, "cnn_params.yml")
    with dagshub.dagshub_logger(metrics_path=metrics_path, hparams_path=params_path) as logger:
        print("selecting hyperparameters")
        model_params, history = select_hyperparams(df, n_classes, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"model": model_params})
        for i in range(N_EPOCHS):
            metrics = dict()
            metrics["train_loss"] = history["train_loss"][i]
            metrics["test_loss"] = history["test_loss"][i]
            metrics["train_f1_score"] = history["train_f1"][i]
            metrics["test_f1_score"] = history["test_f1"][i]
            metrics["train_accuracy"] = history["train_acc"][i]
            metrics["test_accuracy"] = history["test_acc"][i]
            logger.log_metrics(metrics, step_num=i)

    print("finished")
