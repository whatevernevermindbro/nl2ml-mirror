for n in 1 2 3
do
    server="ocean$n"
    proc_info=`ssh $server "ps -fA | grep 'nohup' | head -n1"`
    proc=`echo $proc_info | awk '{ print $2 }'`
    ssh $server "kill $proc"
    ssh $server "
        killall chrome; killall chromedriver
    "
done
