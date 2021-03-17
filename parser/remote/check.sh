for n in 1 2 3
do
    echo "Selectel $n"
    server="selectel_playground$n"
    ssh $server "
      cat /home/kek/parser.log
    "
    echo
    ssh $server "
      wc -l /home/kek/errors_blocks.csv
    "
    echo "===================="
done
