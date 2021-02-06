echo "Selectel 1"
ssh selectel_playground "
  cat /home/kek/parser.log
"

echo

echo "Selectel 2"
ssh selectel_playground2 "
  cat /home/kek/parser.log
"

echo

echo "Selectel 3"
ssh selectel_playground3 "
  cat /home/kek/parser.log
"

echo
