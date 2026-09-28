#!/bin/bash
# run_ros_test.sh - offline ROS 2 replay test (no rover): localhost-only DDS, bridge sends to a UDP sink on 127.0.0.1.
#   ./run_ros_test.sh      -> $OMNIVLA_ROOT/OmniVLA/results/ros_replay.json, [ROSTEST] lines
DEPLOY="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
MARK=/tmp/rostest_$$; rm -f $MARK.*
cd "$DEPLOY/ros2"
/usr/bin/python3 servo_udp_bridge.py --ros-args -p beaglebone_host:=127.0.0.1 -p beaglebone_port:=15005 \
  -r /servo_udp_bridge/servo_cmd:=/omnivla_nav/servo_cmd > /tmp/rostest_bridge.log 2>&1 &
BR=$!
export REPLAY_MODE=${REPLAY_MODE:-4} GOAL_REFRESH=${GOAL_REFRESH:-1}     # REPLAY_MODE=6: image goal + goal cache
NODE_PAT=${NODE_PAT:-"[r]os2/omnivla_nav_node.py"}          # [r]: pkill -f must not match its own command line
if [ -n "$NODE_SCRIPT" ]; then
  "$DEPLOY/launch.sh" "$NODE_SCRIPT" --ros-args -p mode:=$REPLAY_MODE -p goal_refresh:=$GOAL_REFRESH -p goal_spacing_m:=0.25 > /tmp/rostest_node.log 2>&1 &
else
  "$DEPLOY/launch.sh" --ros --ros-args -p mode:=$REPLAY_MODE -p goal_refresh:=$GOAL_REFRESH -p goal_spacing_m:=0.25 > /tmp/rostest_node.log 2>&1 &
fi
NODE=$!
# a node started with & ignores SIGINT until rclpy installs its handlers: escalate SIGINT -> SIGTERM -> SIGKILL
stop_pid() {  # PID PATTERN SECONDS_INT  (PID = the launcher; PATTERN = the python process, which may not exist yet)
  pkill -INT -f "$2"
  for i in $(seq 1 $3); do kill -0 $1 2>/dev/null || return 0; sleep 1; done
  echo "[ROSTEST] $2 still running after SIGINT + $3 s -> SIGTERM (process, children, launcher)"
  pkill -TERM -f "$2"; pkill -TERM -P $1; kill -TERM $1 2>/dev/null
  for i in $(seq 1 15); do kill -0 $1 2>/dev/null || return 0; sleep 1; done
  echo "[ROSTEST] -> SIGKILL"; pkill -KILL -f "$2"; pkill -KILL -P $1; kill -KILL $1 2>/dev/null
}
trap 'stop_pid $NODE "$NODE_PAT" 20; stop_pid $BR "[s]ervo_udp_bridge.py" 5' TERM INT
/usr/bin/python3 replay_test.py $MARK &
TEST=$!
while [ ! -f $MARK.scenario_done ] && kill -0 $TEST 2>/dev/null; do sleep 1; done
[ -f $MARK.scenario_done ] || echo "[ROSTEST] FAIL tester exited before the scenario finished"
touch $MARK.sigint; stop_pid $NODE "$NODE_PAT" 60   # node shutdown: sends neutral, then the bridge watchdog
wait $NODE; touch $MARK.node_stopped
wait $TEST; RC=$?
stop_pid $BR "[s]ervo_udp_bridge.py" 10; wait $BR
echo "[ROSTEST] exit $RC (node log /tmp/rostest_node.log, bridge log /tmp/rostest_bridge.log)"
exit $RC
