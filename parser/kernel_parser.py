from kaggle_scraping import KaggleWebDriver, extract_code_blocks
import pandas as pd
from io import StringIO

KERNEL_FILENAME = "./kernels_part.csv"
refs = pd.read_csv(KERNEL_FILENAME)

parser = argparse.ArgumentParser(description="Parse kaggle notebooks")
parser.add_argument("--process_id", dest="--process_id", default="0")

args = vars(parser.parse_args())

CODEBLOCK_FILENAME = "code_blocks.csv"
ERRORS_FILENAME = "errors_blocks.csv"

webdriver = KaggleWebDriver()

with open(CODEBLOCK_FILENAME, mode='w') as f:
    with open(ERRORS_FILENAME, mode='w') as e:
        for i in range(int(args["--process_id"], refs.shape[0], 3):
            try:
                buf = extract_code_blocks(webdriver, refs.iloc[[i]])
                print(buf.getvalue(), file=f)
            except (Exception):
                e.write(refs.iloc[[i]] + "\n")
