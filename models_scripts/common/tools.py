import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, encoding='utf-8', comment='#', sep=',')#, quoting=csv.QUOTE_NONE, error_bad_lines=False)#, sep=','
    df.dropna(axis=0, inplace=True)
    return df

def tfidf_transform(corpus, tfidf_params, TFIDF_DIR):
    tfidf = pickle.load(open(TFIDF_DIR, 'rb'))
    features = tfidf.transform(corpus)
    return features

def tfidf_fit_transform(code_blocks:pd.DataFrame, tfidf_params:dict, TFIDF_DIR:str):
    tfidf = TfidfVectorizer(tfidf_params)
    print(code_blocks.head())
    tfidf = tfidf.fit(code_blocks)
    pickle.dump(tfidf, open(TFIDF_DIR, "wb"))
    code_blocks_tfidf = tfidf.transform(code_blocks)
    return code_blocks_tfidf
