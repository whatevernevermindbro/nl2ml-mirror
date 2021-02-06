ssh selectel_playground "
  cd /home/kek;
  sudo -u kek nohup python3 kernel_parser.py --process_id 0 1>./parser.err 2>./parser.log &
"

ssh selectel_playground2 "
  cd /home/kek;
  sudo -u kek nohup python3 kernel_parser.py --process_id 1 1>./parser.err 2>./parser.log &
"

ssh selectel_playground3 "
  cd /home/kek;
  sudo -u kek nohup python3 kernel_parser.py --process_id 2 1>./parser.err 2>./parser.log &
"
