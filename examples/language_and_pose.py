# Language + pose goal (mode 8): a text instruction plus a target point. Not validated on the Jetson yet.
#   ./deploy/launch.sh examples/language_and_pose.py CURRENT.jpg "stay on the sidewalk" X_M Y_M
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivla_jetson import OmniVLAJetson

image, instruction, x, y = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
model = OmniVLAJetson(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "weights"))
model.warmup(modes=(8,))
p = model.predict(image, instruction=instruction, goal_pose=OmniVLAJetson.goal_pose_from_xy_yaw(x, y, 0.0))
print(f"waypoints (model units):\n{p.waypoints.round(3)}")
print(f"command: v {p.linear:.3f} m/s, w {p.angular:.3f} rad/s ({p.latency_s * 1000:.0f} ms)")
