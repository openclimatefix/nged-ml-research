#!/bin/bash
# usage: queue.sh LOG cmd...: wait until fewer than 3 e00613 GPU processes run and 20 GB is free, then run cmd
LOG=$1; shift
while true; do
  n=$(pgrep -fc 'python .*e00613_(arm|c1|free|free_o2|nbarm|absorb)')
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$n" -lt 3 ] && [ "$avail" -gt 20 ]; then break; fi
  sleep 30
done
exec "$@" > "$LOG" 2>&1
