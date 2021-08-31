import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
parser.add_argument("GRAPH_PATH", help="path to graph data", type=str)
args = parser.parse_args()

DATASET_PATH = args.DATASET_PATH
CONTEXT_DATASET_PATH = os.path.join("../data/", "context_" + os.path.basename(DATASET_PATH))
GRAPH_PATH = args.GRAPH_PATH

data = pd.read_csv(DATASET_PATH, index_col=0)
data.drop_duplicates(inplace=True)

kernel_ids = data.kaggle_id.unique()
context_series = pd.Series(index=data.index, dtype=np.object_, name="context")

for id in kernel_ids:
    kernel_blocks = data[data.kaggle_id == id]
    kernel_blocks.sort_values(by=["code_block_id"])
    cur_context = []
    for block in kernel_blocks.itertuples():
        context_series[block.Index] = cur_context.copy()
        cur_context.append(block.graph_vertex_id)

data["context"] = context_series

data.to_csv(CONTEXT_DATASET_PATH)
