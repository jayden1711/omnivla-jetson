#!/bin/bash
# 30-minute image-goal soaks with goal_refresh 1 (default) and 3.
ROOT=${OMNIVLA_ROOT:-/mnt/nvme/omnivla}
cd $ROOT/OmniVLA
rm -f results_soak6_r1.csv results_soak6_r3.csv soak6_r1.log soak6_r3.log
for N in 1 3; do
  SOAK_MODE=6 GOAL_REFRESH=$N VALIDATE_TAG=_r$N timeout -k 60 2400 $ROOT/deploy/launch.sh final_validate.py --soak 30 > soak6_r$N.log 2>&1
done
echo "SOAK6 DONE" >> soak6_r3.log
