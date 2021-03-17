for n in 1 2 3
do
  server="selectel_playground$n"
  filename="./cb$n.sh"
  scp "$server:/home/kek/code_blocks.csv" $filename
done

