#!/bin/sh

date_=`date +"%T"`
comp_slug_=`echo ${1} | cut -d'/' -f2`

for n in 1 2 3 4 5 6 7 8 9 10 11; do
  filename="kernel_lists/kk_${comp_slug_}_${n}.csv"
	kaggle k list --page-size 100 --csv --language python --kernel-type notebook --sort-by hotness --competition $comp_slug_ -p $n >> $filename
	if [[ $(head -n1 $filename) == No* || $(wc -l $filename | awk '{ print $1 }') != 101 ]] ;
	then
	  break
	fi
done
