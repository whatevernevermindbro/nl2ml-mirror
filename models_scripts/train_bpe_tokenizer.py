import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from common.tools import *


parser = argparse.ArgumentParser()
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

DATASET_PATH = args.DATASET_PATH

MODEL_DIR = "../models/bpe_tokenizer.json"

CORPUS_TMP_DIR = "./corpus_files"
KERNEL_TEMPLATE_NAME = "kernel{}.txt"

CODE_COLUMN = "code_block"
NOTEBOOK_ID_COLUMN = "kaggle_id"

VOCAB_SIZE = 50000
MIN_FREQ = 3
DROPOUT = 0.15

SPECIAL_TOKENS = []

def make_corpus(df):
    print("Creating corpus")
    os.mkdir(CORPUS_TMP_DIR)
    files = set()
    for _, row in df.iterrows():
        kernel_path = os.path.join(CORPUS_TMP_DIR, KERNEL_TEMPLATE_NAME.format(row[NOTEBOOK_ID_COLUMN]))
        with open(kernel_path, "a") as f:
            f.write(row[CODE_COLUMN])
            f.write("\n")
        files.add(kernel_path)
    print("Done")
    return list(files)


if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    corpus_files = make_corpus(df)

    tokenizer = Tokenizer(BPE(dropout=DROPOUT))

    for i in range(50):
        SPECIAL_TOKENS.append("[VAR" + str(i) + "]")

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQ, special_tokens=SPECIAL_TOKENS)

    tokenizer.pre_tokenizer = Whitespace()

    print("Start BPE training")
    tokenizer.train(corpus_files, trainer)
    print("Done")

    tokenizer.save(MODEL_DIR)
