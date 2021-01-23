import argparse
import subprocess
from subprocess import check_output
import pandas as pd

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

from datetime import datetime
date = datetime.date(datetime.now())
filename = "../data/kaggle_kernels_" + str(date) + ".csv"

parser = argparse.ArgumentParser(description='Process kaggle notebooks.')
parser.add_argument('--page-size', dest='--page-size', default='1001')

parser.add_argument('--language', dest='--language', default='python')

parser.add_argument('--kernel-type', dest='--kernel-type', default='notebook')

parser.add_argument('--sort-by', dest='--sort-by', nargs='+', default='dateCreated')

args = parser.parse_args()

d = vars(args)

filters_parser = argparse.ArgumentParser(description='Filter kaggle notebooks.')
filters_parser.add_argument('--public_score', dest = '--public_score', nargs='?')
filters_parser.add_argument('--upvotes', dest = '--upvotes', nargs='?')
filters_parser.add_argument('--comments', dest = '--comments', nargs='?')

args = flatten(list(map(list, d.items())))

filters = vars(filters_parser.parse_args())
# print(filters)
command = ["kaggle", "k", "list", "--csv", flatten(args), "-p"]
command = flatten(command)

# print(command)

with open (filename, "w") as file:
    for n in range(1, 12):
        print("\n", n, "\n")
        subprocess.call(args=flatten([command, str(n)]), stdout=file)

# df = pd.read_csv('NL2ML/data/kaggle_kernels/21/kk_01_21_2021(orig).csv')#filename)
# print(df.dtypes)
# print(df.head())

# получение нужных колонок
# фильтрация
