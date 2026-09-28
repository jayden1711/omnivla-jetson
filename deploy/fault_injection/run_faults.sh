#!/bin/bash
# run_faults.sh - checks that the ROS replay test catches injected faults (no rover). Runs the replay test against the
# fault-injection copy of the node for each fault and mode; mode 6 also runs verify_replay6.py on the chunks.
# Needs eval/jetson/verify_replay6.py copied into $OMNIVLA_ROOT/OmniVLA. Results: $OMNIVLA_ROOT/OmniVLA/results/ros_replay*_f<fault>.{log,json}.
ROOT=${OMNIVLA_ROOT:-/mnt/nvme/omnivla}
FAULTS=${FAULTS:-"none steer_flip stale_goal delay2s estop_ignored wrong_mode"}
MODES=${MODES:-"4 6"}
OMNI=$ROOT/OmniVLA; R=$OMNI/results; ROS2=$ROOT/deploy/ros2
export NODE_SCRIPT=$ROOT/deploy/fault_injection/omnivla_nav_node_fault.py
export NODE_PAT="[f]ault_injection/omnivla_nav_node_fault.py"   # [f]: pkill -f must not match itself
[ -f $OMNI/seq.npz ] || { echo "seq.npz missing"; exit 2; }
for M in $MODES; do
  for F in $FAULTS; do
    T=$([ $M = 4 ] && echo "" || echo 6)_f$F
    rm -f $R/ros_replay${T}.log $R/ros_replay${T}.json $R/ros_replay${T}_chunks.json $R/verify_replay6_f$F.log
    echo "=== mode $M fault $F $(date +%T)" >> $R/run_faults.log
    cd $ROS2
    OMNI_FAULT=$F REPLAY_MODE=$M GOAL_REFRESH=1 REPLAY_REF=e2e_deploy_gptq@4 REPLAY_OUT_TAG=_f$F \
      timeout -k 60 900 ./run_ros_test.sh > $R/ros_replay${T}.log 2>&1
    echo "rc $?" >> $R/ros_replay${T}.log
    if [ $M = 6 ]; then
      cd $OMNI
      timeout -k 30 1800 $ROOT/deploy/launch.sh verify_replay6.py 1000000000 results/ros_replay${T}_chunks.json > $R/verify_replay6_f$F.log 2>&1
      echo "rc $?" >> $R/verify_replay6_f$F.log
    fi
  done
done
echo "FAULTS DONE $(date +%T)" >> $R/run_faults.log
