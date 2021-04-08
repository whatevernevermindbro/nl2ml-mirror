for n in 1 2 3
do
    server="ocean$n"
    scp -q kernel_parser.py "$server:/home/kek"
    scp -q ../data/additional_kernels2.csv "$server:/home/kek/additional_kernels.csv"
    ssh $server "mkdir -p /home/kek/kaggle_scraping"
    scp -q kaggle_scraping/*.py "$server:/home/kek/kaggle_scraping"
    echo "===== SUCCESSFUL PUSH TO $server ====="
done

