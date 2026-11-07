#!/bin/bash
# usage: queue.sh LOG cmd...: wait (under a lock, one job at a time) until fewer than 3 e00613 GPU processes run
# and 20 GB is free, start cmd, hold the lock 120 s so the new process is counted, then wait for cmd to end
LOG=$1; shift
exec 9>/tmp/e00613_queue.lock
flock 9
while true; do
  n=$(pgrep -fc 'python .*e00613_(arm|c1|free|free_o2|nbarm|absorb)')
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$n" -lt 3 ] && [ "$avail" -gt 20 ]; then break; fi
  sleep 30
done
"$@" > "$LOG" 2>&1 &
pid=$!
sleep 120
flock -u 9
wait $pid
