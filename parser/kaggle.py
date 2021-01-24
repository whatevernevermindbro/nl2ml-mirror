import argparse
from datetime import datetime
from io import StringIO
import subprocess

import pandas as pd

from notebook_parsing import extract_code_blocks


PAGE_COUNT = 10
CODEBLOCKS_FILENAME_TEMPLATE = "../data/code_blocks_new_{}.csv"


def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList


parser = argparse.ArgumentParser(description='Process kaggle notebooks.')
parser.add_argument('--page-size', dest='--page-size', default='1001')

parser.add_argument('--language', dest='--language', default='python')

parser.add_argument('--kernel-type', dest='--kernel-type', default='notebook')

parser.add_argument('--sort-by', dest='--sort-by', nargs='+', default='dateCreated')

args = parser.parse_args()

d = vars(args)

# filters_parser = argparse.ArgumentParser(description='Filter kaggle notebooks.')
# filters_parser.add_argument('--public_score', dest = '--public_score', nargs='?')
# filters_parser.add_argument('--upvotes', dest = '--upvotes', nargs='?')
# filters_parser.add_argument('--comments', dest = '--comments', nargs='?')
#
args = flatten(list(map(list, d.items())))
#
# filters = vars(filters_parser.parse_args())

command = ["kaggle", "k", "list", "--csv", flatten(args), "-p"]
command = flatten(command)


frames = []
for n in range(1, PAGE_COUNT + 1):
    print("Loading kernels from page", n)
    result = subprocess.run(
        args=flatten([command, str(n)]),
        capture_output=True,
        encoding="utf-8"
    )
    if result.returncode != 0:
        print("Page", n, "failed")
        continue

    frames.append(pd.read_csv(StringIO(result.stdout), header=0))

kernel_df = pd.concat(frames, ignore_index=True)
codeblocks_df = extract_code_blocks(kernel_df)

codeblocks_filename = CODEBLOCKS_FILENAME_TEMPLATE.format(datetime.date(datetime.now()))
codeblocks_df.to_csv(codeblocks_filename)
