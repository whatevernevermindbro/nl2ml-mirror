for n in 3
do
    server="selectel_playground$n"
    scp -q kernel_parser.py "$server:/home/kek"
    scp -q ../data/kernels_list21.csv "$server:/home/kek"
    ssh $server "mkdir -p /home/kek/kaggle_scraping"
    scp -q kaggle_scraping/*.py "$server:/home/kek/kaggle_scraping"
    echo "===== SUCCESSFUL PUSH TO $server ====="
done

