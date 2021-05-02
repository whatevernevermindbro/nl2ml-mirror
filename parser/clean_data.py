import argparse
import os
import subprocess

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
parser.add_argument("GRAPH_PATH", help="path to graph data", type=str)
args = parser.parse_args()

DATASET_PATH = args.DATASET_PATH
CLEANED_DATASET_PATH = os.path.join("../data/", "clean_" + os.path.basename(DATASET_PATH))
GRAPH_PATH = args.GRAPH_PATH

SUPPORT_VERTICES = [
    "Other.something_strange",
    "Other.not_enough_vertices",
    "Other.commented",
]

data = pd.read_csv(DATASET_PATH)

for i in data.index:
    code = data.loc[i, "code_block"]
    if code.startswith("`"):
        code = code[1:]
    if code.endswith("`"):
        code = code[:-1]

    with open("tmp.py", "w", encoding="utf-8") as f:
        f.write(code)
    subprocess.run(["autopep8", "--in-place", "--aggressive", "tmp.py"])

    with open("tmp.py", "r", encoding="utf-8") as f:
        code_lines = filter(lambda s: not s.lstrip().startswith("#"), f.readlines())
    code = "".join(code_lines)

    if code.find("'''") != -1:
        print(code)
        print()
    if code.find('"""') != -1:
        print(code)
        print()

    while code.find("'''") != -1:
        comment_start = code.index("'''")
        comment_end = code.find("'''", comment_start + 3)
        if comment_end == -1:
            code = code[:comment_start]
        else:
            code = code[:comment_start] + code[comment_end + 3:]

    while code.find('"""') != -1:
        comment_start = code.index('"""')
        comment_end = code.find('"""', comment_start + 3)
        if comment_end == -1:
            code = code[:comment_start]
        else:
            code = code[:comment_start] + code[comment_end + 3:]

    data.loc[i, "code_block"] = code.strip()

os.remove("tmp.py")

# we should also remove empty
data.dropna(axis=0, inplace=True)

graph_data = pd.read_csv(GRAPH_PATH, index_col=0)
graph_data["full_name"] = graph_data.graph_vertex + "." + graph_data.graph_vertex_subclass
support_ids = graph_data[graph_data.full_name.isin(SUPPORT_VERTICES)].index.values
mask = data.graph_vertex_id.isin(support_ids)
data = data[~mask]

data.to_csv(CLEANED_DATASET_PATH)
