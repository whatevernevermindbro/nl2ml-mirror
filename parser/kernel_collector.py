import argparse

import tqdm

from kaggle_scraping import KaggleWebDriver, write_kernels_to_file


KERNEL_FILENAME = "./kernels_part.csv"

parser = argparse.ArgumentParser(description="Load kaggle notebooks")
parser.add_argument("--kernel_count", dest="--kernel_count", default="100")
parser.add_argument("--process_id", dest="--process_id", default="0")

args = vars(parser.parse_args())

pbar = tqdm.tqdm(total=int(args["--kernel_count"]))

driver = KaggleWebDriver()
driver.load()
fd = open(KERNEL_FILENAME, "w", newline="")

write_kernels_to_file(fd, driver, "Python", int(args["--kernel_count"]), int(args["--process_id"]), pbar)

driver.close()
fd.close()

