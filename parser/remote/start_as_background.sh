for n in 1 2 3
do
    server="ocean$n"
    process_id=$(( n - 1 ))
    ssh $server "
        cd /home/kek;
        sudo -u kek nohup python3 kernel_parser.py --process_id $process_id 1>./parser.err 2>./parser.log &
    "
    echo "===== SUCCESSFUL START AT $server ====="
done

