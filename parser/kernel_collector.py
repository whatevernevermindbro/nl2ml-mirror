import argparse

import tqdm

from kaggle_scraping import KaggleWebDriver, search_and_write_to_file


KERNEL_FILENAME = "./kernels_part.csv"

filters = dict(
    kernelLanguage="Python",
)

parser = argparse.ArgumentParser(description="Load kaggle notebooks")
parser.add_argument("--kernel_count", dest="count", type=int, default=100)
parser.add_argument("--process_id", dest="process_id", type=int, default=0)

args = vars(parser.parse_args())

pbar = tqdm.tqdm(total=int(args["--kernel_count"]))

driver = KaggleWebDriver()
driver.load()
fd = open(KERNEL_FILENAME, "w", newline="")

search_and_write_to_file(fd, driver, "notebooks", args["count"], pbar, args["process_id"], 3, filters)

driver.close()
fd.close()

