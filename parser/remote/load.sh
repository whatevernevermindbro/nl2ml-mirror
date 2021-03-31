for n in 1 2 3
do
  server="ocean$n"
  filename="./cb$n.csv"
  scp "$server:/home/kek/code_blocks.csv" $filename
done

