#!/bin/sh

date_=`date +"%m_%d_%Y"`

for n in 1 2 3 4 5 6 7 8 9 10 11; do
	kaggle k list --page-size 1001 --csv --language python --kernel-type notebook --sort-by dateCreated -p $n >> "NL2ML/data/kaggle_kernels/kk_${date_}.csv"
done