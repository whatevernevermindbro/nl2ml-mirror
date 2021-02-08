import argparse
from time import sleep

import pandas as pd
from tqdm import tqdm

from kaggle_scraping import KaggleWebDriver, extract_code_blocks


KERNEL_FILENAME = "./kernels_list21.csv"
COOLDOWN_LIMIT = 50
COOLDOWN_TIME = 120

refs = pd.read_csv(KERNEL_FILENAME)

parser = argparse.ArgumentParser(description="Parse kaggle notebooks")
parser.add_argument("--process_id", dest="--process_id", default="0")

args = vars(parser.parse_args())

CODEBLOCK_FILENAME = "code_blocks.csv"
ERRORS_FILENAME = "errors_blocks.csv"

webdriver = KaggleWebDriver()
webdriver.load()

pbar = tqdm(total=((refs.shape[0] + 2) // 3))

with open(CODEBLOCK_FILENAME, mode='w') as f:
    with open(ERRORS_FILENAME, mode='w') as e:
        processed_count = 0
        for i in range(int(args["--process_id"]), refs.shape[0], 3):
            processed_count += 1
            try:
                buf = extract_code_blocks(webdriver, refs.ref[i])
                print(buf.getvalue(), file=f)
            except Exception as expt:
                e.write(refs.ref[i] + ",")
                e.write(str(expt))
                e.write("\n")
            pbar.update(1)
            if processed_count >= COOLDOWN_LIMIT:
                processed_count = 0
                sleep(COOLDOWN_TIME)

webdriver.close()
pbar.close()
