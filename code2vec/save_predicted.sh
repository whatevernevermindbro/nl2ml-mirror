#!/usr/bin/env bash
####################################################################
source my_predict.sh
grep "predicted" prediction.txt > new_pred.txt
sed -n '1~2p' new_pred.txt > last_pred.txt
#rm prediction.txt new_pred.txt
