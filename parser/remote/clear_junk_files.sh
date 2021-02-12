for n in 1 2 3
do
    server="selectel_playground$n"
    ssh $server "rm /home/kek/*.pdf"
done
