import argparse
from collections import defaultdict
import os
import pickle
import shutil

import dagshub
import optuna
import pandas as pd
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

N_EPOCHS = 25
N_TRIALS = 15

torch.manual_seed(RANDOM_SEED)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

SEARCH_SPACE = {
    "rnn_size": (64, 512),
    "n_rnn_layers": (1, 5),
    "lin_size": (32, 512),
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


class Classifier(nn.Module):
    def __init__(self, rnn_size, n_rnn_layers, lin_size, n_classes):
        super(Classifier, self).__init__()

        # TODO: add convolutions?

        self.birnn = nn.LSTM(
            EMBEDDING_SIZE,
            rnn_size,
            num_layers=n_rnn_layers,
            bidirectional=True,
            dropout=0.1,
            batch_first=True,
        )

        decoder_layers = []
        for i in range(3):
            if i == 0:
                decoder_layers.append(nn.Linear(2 * rnn_size, lin_size))
            else:
                decoder_layers.append(nn.Linear(lin_size, lin_size))
            decoder_layers.extend([nn.Dropout(0.1, inplace=True), nn.GELU()])
        decoder_layers.append(nn.Linear(lin_size, n_classes))
        self.decoder = nn.Sequential(*decoder_layers)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print("Total param size: {}".format(size))

    def forward(self, x):
        out, _ = self.birnn(x)
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        indices = torch.LongTensor((unpacked_len - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1))
        indices = indices.to(DEVICE)
        out = unpacked.gather(dim=1, index=indices).squeeze(dim=1)
        return self.decoder(out)


def prep_data():
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df.drop_duplicates(inplace=True)
    codes, uniques = pd.factorize(df[TARGET_COLUMN])
    df[TARGET_COLUMN] = codes
    df.dropna(inplace=True)
    return df, len(uniques)


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
    pack = nn.utils.rnn.pack_padded_sequence(tokens, sorted_lengths, batch_first=True)
    return pack, labels[indices]


def train_new_model(df_train, df_test, n_epochs, params, lr=3e-3):
    model = Classifier(**params)
    model = model.to(DEVICE)

    train_dataloader = torch.utils.data.DataLoader(
        CodeblocksDataset(df_train), batch_size=16, collate_fn=process_data, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        CodeblocksDataset(df_test), batch_size=16, collate_fn=process_data
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=n_epochs
    # )
    criterion = F1_Loss()

    history = defaultdict(list)
    for epoch in range(n_epochs):
        train_loss, train_acc, train_f1 = train(model, DEVICE, train_dataloader, epoch, criterion, optimizer)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        print("evaluating")
        test_loss, test_acc, test_f1 = test(model, DEVICE, test_dataloader, epoch, criterion)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)
        # scheduler.step()

    return model, history


class Objective:
    def __init__(self, n_classes, df_train, df_test):
        self.n_classes = n_classes
        self.df_train = df_train
        self.df_test = df_test

    def __call__(self, trial):
        params = {
            "rnn_size": trial.suggest_int("rnn_size", *SEARCH_SPACE["rnn_size"]),
            "n_rnn_layers": trial.suggest_int("n_rnn_layers", *SEARCH_SPACE["n_rnn_layers"]),
            "lin_size": trial.suggest_int("lin_size", *SEARCH_SPACE["lin_size"]),
            "n_classes": self.n_classes,
        }
        model, history = train_new_model(self.df_train, self.df_test, N_EPOCHS, params)

        checkpoint_path = CHECKPOINT_PATH_TEMPLATE.format(trial.number)
        history_path = LEARNING_HISTORY_PATH_TEMPLATE.format(trial.number)
        torch.save(model.state_dict(), checkpoint_path)
        pickle.dump(history, open(history_path, "wb"))

        last_f1 = history["test_f1"][-1]
        return last_f1


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

    metrics = {
        "test_f1_score": history["test_f1"][-1],
        "test_accuracy": history["test_acc"][-1],
        "history": history,
    }

    shutil.copy(best_checkpoint_path, model_path)
    return model_params, metrics


if __name__ == "__main__":
    df, n_classes = prep_data()

    data_meta = {
        "DATASET_PATH": DATASET_PATH,
        "model": MODEL_DIR,
        "script_dir": __file__,
        "random_seed": RANDOM_SEED,
    }

    metrics_path = os.path.join(EXPERIMENT_DATA_PATH, "nn_metrics.csv")
    params_path = os.path.join(EXPERIMENT_DATA_PATH, "nn_params.yml")
    with dagshub.dagshub_logger(metrics_path=metrics_path, hparams_path=params_path) as logger:
        print("selecting hyperparameters")
        model_params, metrics = select_hyperparams(df, n_classes, MODEL_DIR)
        print("logging the results")
        logger.log_hyperparams({"data": data_meta})
        logger.log_hyperparams({"model": model_params})
        logger.log_metrics(metrics)
    print("finished")
