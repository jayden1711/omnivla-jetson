#!/bin/bash
# ROS offline replay tests on the Jetson (no rover): pose mode vs mf_results/e2e_deploy_gptq@4 (with RERUN4=1), and
# image goal with goal_refresh 1 and 3, each followed by a bit-exact reproduction of every chunk (verify_replay6.py).
ROOT=${OMNIVLA_ROOT:-/mnt/nvme/omnivla}
cd $ROOT/deploy/ros2
R=$ROOT/OmniVLA/results
[ -f $ROOT/OmniVLA/seq.npz ] || { echo 'seq.npz missing'; exit 2; }
rm -f $R/ros_replay6_r*.log $R/verify_replay6_r*.log      # a stale done marker would end a wait early
# pose-mode replay runs only with RERUN4=1
[ "$RERUN4" = 1 ] && REPLAY_REF=e2e_deploy_gptq@4 timeout -k 60 900 ./run_ros_test.sh > $R/ros_replay.log 2>&1
for N in 1 3; do
  cd $ROOT/deploy/ros2
  REPLAY_MODE=6 GOAL_REFRESH=$N timeout -k 60 900 ./run_ros_test.sh > $R/ros_replay6_r$N.log 2>&1
  cd $ROOT/OmniVLA
  timeout -k 30 1800 $ROOT/deploy/launch.sh verify_replay6.py > results/verify_replay6_r$N.log 2>&1
  for f in ros_replay6.json ros_replay6_chunks.json ros_replay6_servo.csv; do mv $R/$f $R/${f/replay6/replay6_r$N}; done
done
echo "REPLAYS DONE" >> $R/ros_replay6_r3.log
