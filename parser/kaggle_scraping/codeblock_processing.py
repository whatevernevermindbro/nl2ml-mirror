import copy
from csv import writer
from io import StringIO
import json

from bs4 import BeautifulSoup
import pandas as pd
import requests

from .notebook_scraping import get_source_links


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
    [DATA_SOURCE_COLUMN, CODE_BLOCK_COLUMN, CODE_BLOCK_ID_COLUMN]
)


def is_kernel_view(tag):
    return (tag.name == "script" and tag.has_attr("class") and
            tag["class"][0] == "kaggle-component")


def collect_metadata(kernel_json):
    """
    Collects data for fields in METADATA_FIELDS
    """
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

    result = []
    for field_name, field_json_key in METADATA_FIELDS.items():
        result.append(nested_lookup(kernel_json, field_json_key))
    return result


def collect_data_sources(webdriver, kernel_ref):
    """
    Finds data source slug and figures out the link
    """
    raw_links = get_source_links(webdriver, KAGGLE_LINK + kernel_ref)

    sources = []
    for raw_link in raw_links:
        if raw_link is not None:
            sources.append(raw_link[len(KAGGLE_LINK):])
        else:
            sources.append(raw_link)
    return sources


def code_blocks(kernel_source):
    """
    Code block generator
    """
    ipynb_source = json.loads(kernel_source)

    for cell in ipynb_source["cells"]:
        if cell["cell_type"] == "code":
            yield cell["source"]


def process_kernel(buffer_writer, webdriver, kernel_ref):
    """
    Loads notebook from kaggle and parses its codeblocks and metadata
    """
    response = requests.get(KAGGLE_LINK + kernel_ref, timeout=webdriver.max_load_wait)

    soup = BeautifulSoup(response.text, "html.parser")
    potential_notebook_views = soup.find_all(is_kernel_view)

    kernel_view = potential_notebook_views[1]

    kernel_raw = kernel_view.string

    data_begin_marker = "Kaggle.State.push("
    data_end_marker = ");performance"

    data_begin = kernel_raw.index(data_begin_marker) + len(data_begin_marker)
    data_end = kernel_raw.index(data_end_marker)

    kernel_json = json.loads(kernel_raw[data_begin: data_end])
    if kernel_json.get("kernelBlob") is None or kernel_json["kernelBlob"].get("source") is None:
        return 0

    metadata = collect_metadata(kernel_json)
    metadata.append(collect_data_sources(
        webdriver,
        kernel_ref
    ))

    new_blocks = 0
    for idx, block in enumerate(code_blocks(kernel_json["kernelBlob"]["source"])):
        code_block_data = copy.copy(metadata)
        code_block_data.append(block)
        code_block_data.append(idx)
        buffer_writer.writerow(code_block_data)
        new_blocks += 1

    return new_blocks


def apply_filters(code_blocks_df, filters):
    code_blocks_df['kaggle_score'] = pd.to_numeric(code_blocks_df['kaggle_score'], errors='ignore')
    code_blocks_df['kaggle_comments'] = pd.to_numeric(code_blocks_df['kaggle_comments'], errors='ignore')
    code_blocks_df['kaggle_upvotes'] = pd.to_numeric(code_blocks_df['kaggle_upvotes'], errors='ignore')

    if (filters['kaggle_score'] and filters['--competition']):
        if (filters['minimize_score']):
            code_blocks_df = code_blocks_df[code_blocks_df['kaggle_score'] <= float(filters['kaggle_score'])]
        else:
            code_blocks_df = code_blocks_df[code_blocks_df['kaggle_score'] >= float(filters['kaggle_score'])]

    if filters['upvotes']:
        code_blocks_df = code_blocks_df[code_blocks_df['kaggle_upvotes'] >= int(filters['upvotes'])]
    if filters['comments']:
        code_blocks_df = code_blocks_df[code_blocks_df['kaggle_comments'] >= int(filters['comments'])]
    return code_blocks_df


def extract_code_blocks(webdriver, kernel_ref):
    buffer = StringIO()
    buffer_writer = writer(buffer)
    new_blocks = process_kernel(buffer_writer, webdriver, kernel_ref)
    if new_blocks == 0:
        raise RuntimeError("No new blocks =(")
    buffer.seek(0)
    return buffer
