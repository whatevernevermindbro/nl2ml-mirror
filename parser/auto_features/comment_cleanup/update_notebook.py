import os
import argparse

import warnings
warnings.simplefilter('ignore')

from scipy.special import softmax
from simpletransformers.classification import ClassificationModel

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import utils.preprocessing as preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("DATA", help="path to input code blocks data", type=str)
parser.add_argument("UPDATED_DATA", help="path to output code blocks data", type=str)
args = parser.parse_args()
DATA = args.DATA
UPDATED_DATA = args.UPDATED_DATA

# DATA = "../data/code_blocks_clean.csv"
# UPDATED_DATA = "./data_parts/code_blocks_updated.csv"
SAVE_UPDATED_DATA = False
MODEL_FOLDER = '../../../models/comment_cleanup_model/outputs/'
DELETED_COMMETNS_FILE = './deleted_comments.txt'
NUM_OF_COMMENTS_TO_WRITE_TO_FILE = 20

f = open(DELETED_COMMETNS_FILE, "w")

def get_all_comments():
    all_blocks = pd.read_csv(DATA)
    all_blocks = all_blocks["code_block"].to_frame()

    comment_blocks_idx = (
        all_blocks["code_block"].str.contains("#") | 
        (all_blocks["code_block"].str.contains("'''") & 
         (all_blocks["code_block"].str.count("'''") % 2 == 0)) |
        (all_blocks["code_block"].str.contains('"""') & 
         (all_blocks["code_block"].str.count('"""') % 2 == 0))
    )

    block = all_blocks[comment_blocks_idx]
    
    prep_pipeline = [
        preprocessing.trim_symbols,
        preprocessing.single_lines,
        preprocessing.multiple_lines,
        preprocessing.extract_comments,
    ]
    
    for prep_func in prep_pipeline:
        block = block.apply(prep_func, axis=1)
    
    comments = []
    idx = []
    for index, row in block.iterrows():
        count = len(row['comments'])
        comments.extend(row['comments'])
        idx.extend(index for i in range(count))
    return comments, idx


data, data_idx = get_all_comments()

model = ClassificationModel("roberta", MODEL_FOLDER, use_cuda=False)

updated_data = pd.read_csv(DATA)
updated_data = updated_data["code_block"].to_frame()
sub = 0
for idx in range(len(data)):
    comment = data[idx][1]
    prediction = model.predict([str(comment)])[1]
    vals = softmax(prediction,axis=1)
    if idx != 0 and data_idx[idx] != data_idx[idx - 1]:
        sub = 0
    
    if not (vals[0][0] < vals[0][1]):
        if NUM_OF_COMMENTS_TO_WRITE_TO_FILE > 0:
            f.write(comment + "\n")
            f.write('----------------------\n')
            NUM_OF_COMMENTS_TO_WRITE_TO_FILE -= 1
        new_comment = updated_data.loc[data_idx[idx], 'code_block']
        new_comment = new_comment[0 : data[idx][0] - sub] + new_comment[data[idx][0] + len(comment) + 6 - sub:]
        updated_data.loc[data_idx[idx], 'code_block'] = new_comment
        if idx != 0 and data_idx[idx] == data_idx[idx - 1]:
            sub +=  len(comment) + 6
        if idx != 0 and data_idx[idx] != data_idx[idx - 1]:
            sub =  len(comment) + 6
if SAVE_UPDATED_DATA:
    updated_data.to_csv(UPDATED_DATA)
f.close()
