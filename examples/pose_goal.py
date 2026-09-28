# Pose goal (mode 4): drive toward a point given in the robot frame.
#   ./deploy/launch.sh examples/pose_goal.py CURRENT.jpg [X_M Y_M]      (x forward, y left, meters)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivla_jetson import OmniVLAJetson

image = sys.argv[1]
x, y = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (3.0, 0.0)
model = OmniVLAJetson(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "weights"))
model.warmup(modes=(4,))
p = model.predict(image, goal_pose=OmniVLAJetson.goal_pose_from_xy_yaw(x, y, 0.0))
print(f"waypoints (model units):\n{p.waypoints.round(3)}")
print(f"command: v {p.linear:.3f} m/s, w {p.angular:.3f} rad/s ({p.latency_s * 1000:.0f} ms)")
