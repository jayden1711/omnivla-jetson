#!/bin/bash
# launch.sh - run the OmniVLA runtime (or any script) with the settings the 8 GB Orin needs.
#   ./launch.sh                         smoke test (loads the model, 3 pose-mode predictions, prints a command)
#   ./launch.sh some_script.py args     run a script (omnivla_deploy importable)
#   ./launch.sh --ros [ros2 args]       run the ROS 2 node (ros2/omnivla_nav_node.py), e.g. --ros --ros-args -p mode:=4
# Env: OMNIVLA_VENV (default $DEPLOY/venv), OMNIVLA_REPO (default /mnt/nvme/omnivla/OmniVLA), ROS_SETUP.
set -e
DEPLOY="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV="${OMNIVLA_VENV:-$DEPLOY/venv}"
export OMNIVLA_REPO="${OMNIVLA_REPO:-/mnt/nvme/omnivla/OmniVLA}"
source "$VENV/bin/activate"
python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.8.0') and torch.cuda.is_available() else 1)" \
  || { echo "[launch] torch 2.8.0 with CUDA not found in $VENV - see docs/deployment.md step 4"; exit 1; }
[ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -ge 5700 ] \
  || echo "[launch] WARNING: < 5.7 GB available; stop desktop/docker/snapd (docs/deployment.md step 1) or the 7B model may not fit"
sudo -n /usr/bin/jetson_clocks                     # max clocks (sudoers entry, docs/deployment.md step 3)
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
sudo -n /usr/local/bin/omni-dropcache &            # drop page cache every second: unified memory is shared with the GPU
trap 'sudo -n /usr/bin/pkill -f omni-dropcache' EXIT
export PYTHONPATH="$DEPLOY:$DEPLOY/ros2:$OMNIVLA_REPO${PYTHONPATH:+:$PYTHONPATH}"
if [ "$1" = "--ros" ]; then
  shift
  set +e; source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"; set -e
  python "$DEPLOY/ros2/omnivla_nav_node.py" "$@"
elif [ $# -eq 0 ]; then
  python "$DEPLOY/omnivla_deploy.py" "$DEPLOY"
else
  python "$@"
fi
