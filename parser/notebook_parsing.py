import copy
import json
import requests

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import trange


KAGGLE_LINK = "https://www.kaggle.com/"

KERNELS_PATH = "../data/kaggle_kernels.csv"
CODE_BLOCKS_PATH = "../data/code_blocks_new.csv"

METADATA_FIELDS = {
    "kaggle_score": "kernel__bestPublicScore",
    "kaggle_comments": "menuLinks__-1__count",
    "kaggle_upvotes": "kernel__upvoteCount",
    "kaggle_link": "baseUrl",
    "kaggle_id": "kernel__id",
}
CODE_BLOCK_COLUMN = "code_blocks"
CODE_BLOCK_ID_COLUMN = "code_block_id"
ALL_COLUMNS = (
    list(METADATA_FIELDS.keys()) +
    [CODE_BLOCK_COLUMN, CODE_BLOCK_ID_COLUMN]
)


def is_notebook_view(tag):
    return (tag.name == "script" and tag.has_attr("class") and
            tag["class"][0] == "kaggle-component")


def collect_metadata(kernel_json):
    '''
    Collects data for fields in METADATA_FIELDS
    '''
    def nested_lookup(json_obj, complex_key):
        keys = complex_key.split("__")
        current_level = json_obj
        for key in keys:
            if isinstance(current_level, list):
                next_level = current_level[int(key)]
            else:
                next_level = current_level.get(key)

            if next_level is None:
                return None
            current_level = next_level

        return current_level

    result = dict()
    for field_name, field_json_key in METADATA_FIELDS.items():
        result[field_name] = nested_lookup(kernel_json, field_json_key)
    return result


def code_blocks(ipynb_source):
    '''
    Code block generator
    '''
    ipynb_source = json.loads(ipynb_source)

    for cell in ipynb_source["cells"]:
        if cell["cell_type"] == "code":
            yield cell["source"]


def process_notebook(notebook_ref):
    '''
    Loads notebook from kaggle and parses its codeblocks and metadata
    '''
    response = requests.get(KAGGLE_LINK + notebook_ref)

    soup = BeautifulSoup(response.text, "html.parser")
    potential_notebook_views = soup.find_all(is_notebook_view)

    notebook_view = potential_notebook_views[1]

    notebook_data_raw = notebook_view.string

    data_begin_marker = "Kaggle.State.push("
    data_end_marker = ");performance"

    data_begin = notebook_data_raw.index(data_begin_marker) + len(data_begin_marker)
    data_end = notebook_data_raw.index(data_end_marker)

    notebook_data = json.loads(notebook_data_raw[data_begin: data_end])

    metadata = collect_metadata(notebook_data)
    data = []
    for idx, block in enumerate(code_blocks(notebook_data["kernelBlob"]["source"])):
        code_block_data = copy.copy(metadata)
        code_block_data[CODE_BLOCK_COLUMN] = block
        code_block_data[CODE_BLOCK_ID_COLUMN] = idx
        data.append(code_block_data)

    return pd.DataFrame(data, columns=ALL_COLUMNS)


kernels_df = pd.read_csv(KERNELS_PATH)
code_blocks_df = pd.DataFrame(columns=ALL_COLUMNS)
for i in trange(kernels_df.shape[0]):
    new_blocks = process_notebook(kernels_df.loc[i, "ref"])
    code_blocks_df = code_blocks_df.append(new_blocks, ignore_index=True)

code_blocks_df.to_csv(CODE_BLOCKS_PATH)
