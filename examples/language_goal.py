# Language goal (mode 7): follow a text instruction. Not validated on the Jetson yet; see results/lang_summary.md for
# the off-device evaluation.
#   ./deploy/launch.sh examples/language_goal.py CURRENT.jpg "move toward the blue trash bin"
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivla_jetson import OmniVLAJetson

image, instruction = sys.argv[1], sys.argv[2]
model = OmniVLAJetson(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "weights"))
model.warmup(modes=(7,))
p = model.predict(image, instruction=instruction)
print(f"waypoints (model units):\n{p.waypoints.round(3)}")
print(f"command: v {p.linear:.3f} m/s, w {p.angular:.3f} rad/s ({p.latency_s * 1000:.0f} ms)")
