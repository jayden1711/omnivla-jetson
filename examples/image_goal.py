# Image goal (mode 6): drive toward the place where GOAL.jpg was taken. The goal's vision features are computed once
# and reused while the goal image stays the same.
#   ./deploy/launch.sh examples/image_goal.py CURRENT.jpg GOAL.jpg
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivla_jetson import OmniVLAJetson

image, goal = sys.argv[1], sys.argv[2]
model = OmniVLAJetson(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "weights"))
model.warmup(modes=(6,))
p = model.predict(image, goal_image=goal)
print(f"waypoints (model units):\n{p.waypoints.round(3)}")
print(f"command: v {p.linear:.3f} m/s, w {p.angular:.3f} rad/s ({p.latency_s * 1000:.0f} ms)")
