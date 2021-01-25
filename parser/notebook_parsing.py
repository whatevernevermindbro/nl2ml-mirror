import copy
import json
import requests

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import trange

from utils.source_scraping import SourceScraper


KAGGLE_LINK = "https://www.kaggle.com/"

METADATA_FIELDS = {
    "kaggle_score": "kernel__bestPublicScore",
    "kaggle_comments": "menuLinks__-1__count",
    "kaggle_upvotes": "kernel__upvoteCount",
    "kaggle_link": "baseUrl",
    "kaggle_id": "kernel__id",
}
CODE_BLOCK_COLUMN = "code_blocks"
CODE_BLOCK_ID_COLUMN = "code_block_id"
DATA_SOURCE_COLUMN = "data_source"
ALL_COLUMNS = (
    list(METADATA_FIELDS.keys()) +
    [CODE_BLOCK_COLUMN, CODE_BLOCK_ID_COLUMN, DATA_SOURCE_COLUMN]
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


def collect_data_source(kernel_json, kernel_ref, source_scraper):
    '''
    Finds data source slug and figures out the link
    '''
    user_dataset_count = 0
    sources = []
    for data_source in kernel_json["dataSources"]:
        source_slug = data_source["mountSlug"]
        source_type = data_source["sourceType"]
        # Data source is either a competition or a user-made dataset
        if source_type == "Competition":
            sources.append(f"c/{source_slug}")
        else:
            user_dataset_count += 1

    if user_dataset_count == 0:
        return sources
    try:
        raw_links = source_scraper.get_source_links(KAGGLE_LINK + kernel_ref)
    except Exception as e:
        return sources

    sources = []
    for raw_link in raw_links:
        sources.append(raw_link[len(KAGGLE_LINK):])
    return sources


def code_blocks(ipynb_source):
    '''
    Code block generator
    '''
    ipynb_source = json.loads(ipynb_source)

    for cell in ipynb_source["cells"]:
        if cell["cell_type"] == "code":
            yield cell["source"]


def process_notebook(notebook_ref, source_scraper):
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
    metadata[DATA_SOURCE_COLUMN] = collect_data_source(
        notebook_data,
        notebook_ref,
        source_scraper
    )

    data = []
    for idx, block in enumerate(code_blocks(notebook_data["kernelBlob"]["source"])):
        code_block_data = copy.copy(metadata)
        code_block_data[CODE_BLOCK_COLUMN] = block
        code_block_data[CODE_BLOCK_ID_COLUMN] = idx
        data.append(code_block_data)

    return pd.DataFrame(data, columns=ALL_COLUMNS)


def extract_code_blocks(kernels_df, filters=None):
    code_blocks_df = pd.DataFrame(columns=ALL_COLUMNS)
    with trange(kernels_df.shape[0]) as kernel_indices_iterator, SourceScraper() as source_scraper:
        for i in kernel_indices_iterator:
            kernel_ref = kernels_df.loc[i, "ref"]
            kernel_indices_iterator.set_description(f"Notebook {i: >7}")

            new_blocks = process_notebook(kernel_ref, source_scraper)

            kernel_indices_iterator.set_postfix({"code blocks": new_blocks.shape[0]})
            code_blocks_df = code_blocks_df.append(new_blocks, ignore_index=True)

    code_blocks_df['kaggle_score'] = pd.to_numeric(code_blocks_df['kaggle_score'], errors='ignore')
    code_blocks_df['kaggle_comments'] = pd.to_numeric(code_blocks_df['kaggle_comments'], errors='ignore')
    code_blocks_df['kaggle_upvotes'] = pd.to_numeric(code_blocks_df['kaggle_upvotes'], errors='ignore')

    if (filters['kaggle_score'] and filters['--competition']):
        if (filters['minimize_score']):
            code_blocks_df = code_blocks_df[code_blocks_df['kaggle_score'] <= float(filters['kaggle_score'])]
        else:
            code_blocks_df = code_blocks_df[code_blocks_df['kaggle_score'] >= float(filters['kaggle_score'])]

    if(filters['upvotes']):
        code_blocks_df = code_blocks_df[code_blocks_df['kaggle_upvotes'] >= int(filters['upvotes'])]
    if(filters['comments']):
        code_blocks_df = code_blocks_df[code_blocks_df['kaggle_comments'] >= int(filters['comments'])]

    return code_blocks_df
