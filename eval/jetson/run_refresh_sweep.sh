#!/bin/bash
# Goal-cache refresh sweep (deploy_refresh_sweep.py), one process per config; resumes from refresh_sweep.done.
ROOT=${OMNIVLA_ROOT:-/mnt/nvme/omnivla}
cd $ROOT/OmniVLA
for a in "5 1" "5 999" "10 1" "10 999" "21 1" "21 2" "21 3" "21 4" "21 999"; do
  set -- $a
  grep -q "^DONE $a\$" refresh_sweep.done 2>/dev/null && continue
  echo "=== $a (MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB) $(date +%T)" >> refresh_sweep.log
  VALIDATE_TAG=_g $ROOT/deploy/launch.sh deploy_refresh_sweep.py $1 $2 >> refresh_sweep.log 2>&1 && echo "DONE $a" >> refresh_sweep.done
done
echo "SWEEP FINISHED $(date +%T)" >> refresh_sweep.log
