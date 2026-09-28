"""OmniVLAJetson: a small wrapper around deploy/omnivla_deploy.py (OmniVLADeploy). It picks the mode from the goals you
pass, accepts images as PIL images, numpy arrays or file paths, and returns the waypoints together with the velocity
command from OmniVLA's own controller. Every prediction is OmniVLADeploy.predict's output, unchanged.

    from omnivla_jetson import OmniVLAJetson
    m = OmniVLAJetson("/path/to/omnivla-jetson/deploy/weights")
    p = m.predict(image, goal_pose=OmniVLAJetson.goal_pose_from_xy_yaw(3.0, 0.0, 0.0))
    p.waypoints          # (8, 4) float32: x, y, cos, sin per chunk step (model units)
    p.linear, p.angular  # m/s, rad/s

Run scripts that use it through deploy/launch.sh (clocks, allocator settings, page-cache dropping).
"""
import math, os, sys
from dataclasses import dataclass, field

import numpy as np

MODES = {("pose",): 4, ("image",): 6, ("instruction",): 7, ("instruction", "pose"): 8}
MODE_NAMES = {4: "pose goal", 6: "image goal", 7: "language", 8: "language + pose"}
VALIDATED_MODES = (4, 6)                     # checked on the Jetson; 7 and 8 were only evaluated off-device


@dataclass
class Prediction:
    waypoints: np.ndarray                    # (8, 4): x, y, cos(yaw), sin(yaw) per chunk step, model units
    linear: float                            # m/s, OmniVLA's controller (waypoint 4, limits 0.3 m/s, 0.3 rad/s)
    angular: float                           # rad/s
    mode: int                                # 4 pose, 6 image, 7 language, 8 language + pose
    latency_s: float                         # model forward time
    raw: dict = field(repr=False, default_factory=dict)   # OmniVLADeploy.predict's full output (cache info, ...)

    @property
    def command(self):
        return self.linear, self.angular


def _find_runtime(weights_path, deploy_dir):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [deploy_dir, os.environ.get("OMNIVLA_DEPLOY"), os.path.dirname(os.path.abspath(weights_path)),
             os.path.join(here, "deploy")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "omnivla_deploy.py")):
            return d
    raise FileNotFoundError("omnivla_deploy.py not found; pass deploy_dir= or set OMNIVLA_DEPLOY")


def _load_runtime_module(d):
    if d not in sys.path:
        sys.path.insert(0, d)
    import omnivla_deploy
    return omnivla_deploy


def _as_pil(img, what):
    from PIL import Image
    if img is None:
        return None
    if isinstance(img, (str, os.PathLike)):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{what}: numpy images must be HxWx3 uint8 RGB")
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img if img.mode == "RGB" else img.convert("RGB")
    raise TypeError(f"{what}: expected a PIL image, an HxWx3 uint8 array or a file path, got {type(img).__name__}")


class OmniVLAJetson:
    """weights_path: the weights folder (deploy/weights after setup_jetson.sh). deploy_dir: the folder with
    omnivla_deploy.py (default: the weights folder's parent, then OMNIVLA_DEPLOY, then this checkout's deploy/).
    runtime_kwargs go to OmniVLADeploy (repo, prune_frac, goal_refresh, verbose, ...). runtime: an already built
    OmniVLADeploy-like object (used by the tests); nothing is loaded then. Language modes run without image-token
    pruning by default (runtime_kwargs lang_prune_frac=0.0); pose and image goals keep the deployed 75%."""

    def __init__(self, weights_path, deploy_dir=None, controller=None, runtime=None, **runtime_kwargs):
        self.controller = dict(controller or {})             # actions_to_cmd kwargs: waypoint_select, spacing, maxv, maxw, dt
        if runtime is not None:
            self.runtime, self._cmd = runtime, runtime.actions_to_cmd
            return
        d = _find_runtime(weights_path, deploy_dir)
        mod = _load_runtime_module(d)
        W = os.path.abspath(weights_path)
        prev = os.environ.get("OMNIVLA_WEIGHTS")
        os.environ["OMNIVLA_WEIGHTS"] = W                    # OmniVLADeploy loads os.path.join(deploy_dir, OMNIVLA_WEIGHTS)
        try:
            self.runtime = mod.OmniVLADeploy(os.path.dirname(W), **runtime_kwargs)
        finally:
            if prev is None:
                os.environ.pop("OMNIVLA_WEIGHTS", None)
            else:
                os.environ["OMNIVLA_WEIGHTS"] = prev
        self._cmd = mod.OmniVLADeploy.actions_to_cmd

    @staticmethod
    def mode_for(goal_image=None, goal_pose=None, instruction=None):
        key = tuple(k for k, v in (("image", goal_image), ("instruction", instruction), ("pose", goal_pose)) if v is not None)
        key = tuple(sorted(key))
        if key not in MODES:
            raise ValueError("give exactly one of goal_image, goal_pose, instruction, or instruction + goal_pose "
                             f"(got: {', '.join(key) or 'no goal'})")
        return MODES[key]

    def predict(self, image, goal_image=None, goal_pose=None, instruction=None, goal_id=None):
        """image: the current camera frame. goal_image: image-goal mode (its vision features are cached while it does
        not change; goal_id= skips hashing the image). goal_pose: [x, y, cos, sin] in model units
        (goal_pose_from_xy_yaw / goal_pose_from_gps). instruction: text, e.g. "move toward the blue trash bin"."""
        if instruction is not None and not str(instruction).strip():
            raise ValueError("instruction is empty")
        mode = self.mode_for(goal_image, goal_pose, instruction)
        kw = dict(mode=mode)
        if goal_pose is not None:
            gp = np.asarray(goal_pose, dtype=np.float64)
            if gp.shape != (4,):
                raise ValueError("goal_pose must be [x, y, cos, sin]; use goal_pose_from_xy_yaw(x_m, y_m, yaw_rad)")
            kw["goal_pose"] = gp
        if goal_image is not None:
            kw["goal_image"] = _as_pil(goal_image, "goal_image")
            if goal_id is not None:
                kw["goal_id"] = goal_id
        if instruction is not None:
            kw["lang"] = str(instruction)
        out = self.runtime.predict(_as_pil(image, "image"), **kw)
        v, w = self._cmd(out["actions"], **self.controller)
        return Prediction(waypoints=out["actions"], linear=float(v), angular=float(w), mode=mode,
                          latency_s=float(out.get("t_fwd", float("nan"))), raw=out)

    def warmup(self, modes=(4, 6)):
        """compile the GPU kernels before serving (~14 s on the Orin)"""
        self.runtime.warmup(modes=modes)

    @staticmethod
    def goal_pose_from_xy_yaw(x_m, y_m, yaw_rad=0.0, spacing_m=0.1):
        """goal in the robot frame (x forward, y left, meters; yaw in radians) -> model units, as the ROS node's
        goal_xy_yaw parameter (deploy/ros2/omnivla_nav_node.py)"""
        return np.array([x_m / spacing_m, y_m / spacing_m, math.cos(yaw_rad), math.sin(yaw_rad)])

    def goal_pose_from_gps(self, cur_lat, cur_lon, cur_compass_deg, goal_lat, goal_lon, goal_compass_rad=0.0):
        """OmniVLA's GPS goal normalization (OmniVLADeploy.goal_pose_from_gps; needs the utm package)"""
        return self.runtime.goal_pose_from_gps(cur_lat, cur_lon, cur_compass_deg, goal_lat, goal_lon, goal_compass_rad)
