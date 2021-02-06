ssh selectel_playground "
  cd /home/kek;
  sudo -u kek nohup python3 kernel_collector.py --kernel_count 3300 --process_id 0 1>./collector.err 2>./collector.log &
"

ssh selectel_playground2 "
  cd /home/kek;
  sudo -u kek nohup python3 kernel_collector.py --kernel_count 3300 --process_id 1 1>./collector.err 2>./collector.log &
"

ssh selectel_playground3 "
  cd /home/kek;
  sudo -u kek nohup python3 kernel_collector.py --kernel_count 3300 --process_id 2 1>./collector.err 2>./collector.log &
"
