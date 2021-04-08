import re
import csv
import json
import argparse
import yaml

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import dagshub

def tokens_search(df, tokens, new_column_name):
    df[new_column_name] = 0
    for i in range(len(df)):
        percents = str(round(100*i/len(df),1))
        print(percents + '%\r', end='')
        row = df[CODE_COLUMN][i]
        for token in tokens:
            result = re.search(token, row)
            if result!=None:
                df[new_column_name][i] = 1
                break
    return df
evaluation = False
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=int)
    parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
    parser.add_argument("-eval", "--evaluation", help="evalute regex after creating", type=bool)
    args = parser.parse_args()

    GRAPH_VER = args.GRAPH_VER
    DATASET_PATH = args.DATASET_PATH
    if args.evaluation is not None:
        evaluation = args.evaluation
except:
    print('Got no arguments, taking default arguments from params.yaml')
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    GRAPH_VER = params['GRAPH_VER']
    DATASET_PATH = params['regex']['DATASET_PATH']
    if params['regex']['evaluation'] is not None:
        evaluation = params['evaluation']

OUTPUT_DATASET_PATH = '{}_regex_graph_v{}.csv'.format(DATASET_PATH[:-4], GRAPH_VER)
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

    vertices = ["Other.something_strange"]
    for i in range(0, len(graph)):
        vertex = list(graph.keys())[i]
        vertices.append(vertex)
        print('\n creating labels for {}'.format(vertex))
        tokens = graph[vertex]
        df = tokens_search(df, tokens, vertex)
    print('labelled')

    df.to_csv(OUTPUT_DATASET_PATH, index=False)
    print('saved and finished')
    if evaluation:
        print('evaluating regex v{}'.format(GRAPH_VER))
        VALIDATION_DATA_PATH = "./data/golden_884_set.csv"
        TAGS = vertices
        REGEX_TAGS = [el + '_regex_v{}'.format(GRAPH_VER) for el in TAGS]
        regexed_data = pd.read_csv(VALIDATION_DATA_PATH)
        print('loaded validation data')
        for i, tag in enumerate(TAGS):
            print('\n creating labels for {}'.format(tag))
            tokens = graph[tag]
            regexed_data = tokens_search(regexed_data, tokens, REGEX_TAGS[i])
        print('val data labelled')
        Y_test, Y_pred = regexed_data[TAGS], regexed_data[REGEX_TAGS]
        base_f1 = f1_score(Y_test, Y_pred, average='weighted')
        base_precision = precision_score(Y_test, Y_pred, average='weighted')
        base_recall = recall_score(Y_test, Y_pred, average='weighted')
        regex_results = {'test_f1_score': base_f1
                        , 'test_precision': base_precision
                        , 'test_recall': base_recall}
        for i, tag in enumerate(TAGS):
            tag_results = (round(f1_score(Y_test.iloc[:, i], Y_pred.iloc[:, i], average='weighted'),4),\
                            round(precision_score(Y_test.iloc[:, i], Y_pred.iloc[:, i], average='weighted'),4),\
                            round(recall_score(Y_test.iloc[:, i], Y_pred.iloc[:, i], average='weighted'),4))
            print(tag)
            print(tag_results)
            regex_results.update({'test_f1_score_{}'.format(tag): tag_results[0]
                                , 'test_precision_{}'.format(tag): tag_results[1]
                                , 'test_recall_{}'.format(tag): tag_results[2]})
            print('------')
        data_meta = {'DATASET_PATH': VALIDATION_DATA_PATH
                    ,'nrows': regexed_data.shape[0]
                    ,'graph_ver': GRAPH_VER
                    ,'label': TAGS
                    ,'model': 'regex_v{}'.format(GRAPH_VER)
                    ,'script_dir': './regex.ipynb'
                    ,'task': 'regex evaluation'}
        with dagshub.dagshub_logger() as logger:
            logger.log_hyperparams(data_meta)
            logger.log_metrics(regex_results)