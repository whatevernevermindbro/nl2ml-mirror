#!/bin/sh

date_=`date +"%T"`

for n in 1 2 3 4 5 6 7 8 9 10 11; do
  filename="kernel_lists/kk_${date_}_${n}.csv"
	kaggle k list --page-size 100 --csv --language python --kernel-type notebook --sort-by hotness --competition $1 -p $n >> $filename
	if [[ $(head -n1 $filename) == No* ]] ;
	then
	  break
	fi
done
