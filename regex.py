import re
import csv
import json
import argparse

import pandas as pd
import numpy as np

def tokens_search(df, tokens, new_column_name):
    df[new_column_name] = 0
    for i in range(len(df)):
        percents = str(round(100*i/len(df),1))
        print(percents + '%\r', end='')
        row = df[CODE_COLUMN][i]
        for token in tokens:
            result = re.search(token.replace('(','\('), row)
            if result!=None:
                df[new_column_name][i] = 1
                break
    return df

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=int)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH

OUTPUT_DATASET_PATH = '{}_graph_v{}.csv'.format(DATASET_PATH[:-4], GRAPH_VER)
CODE_COLUMN = 'code_block'
GRAPH_DIR = './graph/graph_v{}.txt'.format(GRAPH_VER)

if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH, encoding='utf-8', sep=',')
    print('opened input data')
    if df[CODE_COLUMN].isna().sum() > 0:
        print('Empty chunks found: {}'.format(df[CODE_COLUMN].isna().sum()))
        df = df.dropna(subset=[CODE_COLUMN]).reset_index()

    with open(GRAPH_DIR, "r") as graph_file:
        graph = json.load(graph_file)
    print('opened graph')

    for i in range(0, len(graph)):
        vertice = list(graph.keys())[i]
        print('\n' + vertice)
        tokens = graph[vertice]
        df = tokens_search(df, tokens, vertice)
    print('labelled')

    df.to_csv(OUTPUT_DATASET_PATH, index=False)
    print('saved and finished')