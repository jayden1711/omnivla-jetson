#!/bin/bash
# wait_job.sh SESSION LOGFILE DONE_PATTERN TIMEOUT_MIN - bounded wait for a tmux job on the Jetson (ssh host $JETSON_HOST).
# Exits 0 when DONE_PATTERN appears in LOGFILE; 2 on timeout; 3 if the tmux session is gone without DONE_PATTERN;
# 4 if no file matching LOGFILE (a path or glob on the Jetson; the newest match is watched) changed for 20 minutes.
# DONE_PATTERN is searched in all matches. Polls every 30 s.
H=${JETSON_HOST:-jetson}; S=$1; L=$2; P=$3; T=$4; start=$(date +%s); last_sum=""; last_change=$(date +%s)
while true; do
  out=$(ssh -o ConnectTimeout=10 "$H" "grep -qs \"$P\" $L && echo DONE; tmux has-session -t $S 2>/dev/null && echo ALIVE; stat -c %n:%Y:%s \$(ls -t $L 2>/dev/null | head -1) 2>/dev/null" 2>/dev/null)
  now=$(date +%s)
  echo "$out" | grep -q DONE && { echo "[WAIT] done after $(( (now - start) / 60 )) min"; exit 0; }
  sum=$(echo "$out" | tail -1)
  [ "$sum" != "$last_sum" ] && { last_sum=$sum; last_change=$now; }
  if [ -n "$out" ] && ! echo "$out" | grep -q ALIVE; then echo "[WAIT] tmux session $S gone without '$P' after $(( (now - start) / 60 )) min"; exit 3; fi
  [ $(( now - last_change )) -ge 1200 ] && { echo "[WAIT] $L unchanged for 20 min: stalled"; exit 4; }
  [ $(( now - start )) -ge $(( T * 60 )) ] && { echo "[WAIT] timeout after $T min"; exit 2; }
  sleep 30
done
