import argparse

import tqdm

from kaggle_scraping import KaggleWebDriver, search_and_write_to_file


KERNEL_FILENAME = "../data/competitions_refs.csv"

parser = argparse.ArgumentParser(description="Find references to kaggle competitions")
parser.add_argument("--competition_count", dest="count", type=int, default=100)
parser.add_argument("--process_id", dest="process_id", type=int, default=0)
parser.add_argument("--total_processes", dest="total_processes", type=int, default=1)

args = vars(parser.parse_args())

pbar = tqdm.tqdm(total=args["count"])

driver = KaggleWebDriver()
driver.load()
fd = open(KERNEL_FILENAME, "w", newline="")

search_and_write_to_file(fd, driver, "competitions", args["count"], pbar, args["process_id"], args["total_processes"])

driver.close()
fd.close()

