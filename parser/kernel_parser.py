import argparse
from io import StringIO

import pandas as pd
from tqdm import tqdm

from kaggle_scraping import KaggleWebDriver, extract_code_blocks


KERNEL_FILENAME = "./kernels_list21.csv"
refs = pd.read_csv(KERNEL_FILENAME)

parser = argparse.ArgumentParser(description="Parse kaggle notebooks")
parser.add_argument("--process_id", dest="--process_id", default="0")

args = vars(parser.parse_args())

CODEBLOCK_FILENAME = "code_blocks.csv"
ERRORS_FILENAME = "errors_blocks.csv"

webdriver = KaggleWebDriver()
webdriver.load()

pbar = tqdm(total=((refs.shape[0] + 2)// 3))

with open(CODEBLOCK_FILENAME, mode='w') as f:
    with open(ERRORS_FILENAME, mode='w') as e:
        for i in range(int(args["--process_id"]), refs.shape[0], 3):
            try:
                buf = extract_code_blocks(webdriver, refs.ref[i])
                print(buf.getvalue(), file=f)
            except Exception as expt:
                e.write(refs.ref[i] + "\n")
            pbar.update(1)

webdriver.close()
pbar.close()
