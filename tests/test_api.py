# Tests of the omnivla_jetson wrapper without a Jetson or a GPU (numpy + Pillow only; torch is stubbed if missing).
# A stand-in runtime replays the validated deployment's recorded outputs (deploy/tests/reference/reference.npz) and
# records every call, so the tests check that the wrapper passes the goals through unchanged, returns the runtime's
# actions bit for bit, and computes the command with the runtime's own controller (OmniVLADeploy.actions_to_cmd).
# The same check against the real model runs on the Jetson: ./deploy/launch.sh deploy/tools/api_check.py
#   python -m unittest discover -s tests -v            (from the repo root)
import importlib.util, os, sys, tempfile, types, unittest
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "deploy"))
if importlib.util.find_spec("torch") is None:                    # omnivla_deploy imports torch at the top
    _t = types.ModuleType("torch"); _t.no_grad = lambda: (lambda f: f); sys.modules["torch"] = _t
from omnivla_deploy import OmniVLADeploy                         # noqa: E402
from omnivla_jetson import OmniVLAJetson, Prediction              # noqa: E402

REF = np.load(os.path.join(ROOT, "deploy", "tests", "reference", "reference.npz"))
FRAMES = sorted(k[6:] for k in REF.files if k.startswith("goal__"))


class ReplayRuntime:
    """OmniVLADeploy stand-in: returns the recorded output for (mode, frame); the frame is found from the goal"""
    actions_to_cmd = staticmethod(OmniVLADeploy.actions_to_cmd)
    goal_pose_from_gps = staticmethod(OmniVLADeploy.goal_pose_from_gps)

    def __init__(self):
        self.calls, self.frame = [], None

    def predict(self, image, mode=4, goal_pose=None, goal_image=None, lang=None, goal_id=None):
        self.calls.append(dict(image=image, mode=mode, goal_pose=goal_pose, goal_image=goal_image, lang=lang, goal_id=goal_id))
        f = self.frame
        if mode == 4:
            f = next(x for x in FRAMES if np.array_equal(REF[f"goal__{x}"], goal_pose))
        act = REF[f"act_m{mode}__{f}"]
        return dict(actions=act, t_fwd=0.123)

    def warmup(self, modes=(4,), n=2):
        self.calls.append(dict(warmup=modes))


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.rt = ReplayRuntime(); self.m = OmniVLAJetson("unused", runtime=self.rt)
        self.img = Image.new("RGB", (224, 224), (90, 120, 150))

    def test_pose_goal_matches_recorded_outputs(self):
        for f in FRAMES:
            p = self.m.predict(self.img, goal_pose=REF[f"goal__{f}"])
            c = self.rt.calls[-1]
            self.assertEqual(c["mode"], 4)
            self.assertTrue(np.array_equal(c["goal_pose"], REF[f"goal__{f}"]))
            self.assertEqual(c["goal_pose"].dtype, np.float64)
            self.assertIs(c["image"], self.img)
            self.assertTrue(np.array_equal(p.waypoints, REF[f"act_m4__{f}"]))
            self.assertEqual(p.waypoints.dtype, np.float32)
            self.assertEqual((p.linear, p.angular), tuple(float(x) for x in OmniVLADeploy.actions_to_cmd(REF[f"act_m4__{f}"])))
            self.assertEqual(p.latency_s, 0.123)

    def test_image_goal_matches_recorded_outputs(self):
        for f in FRAMES:
            self.rt.frame = f
            goal = np.full((224, 224, 3), 200, np.uint8)
            p = self.m.predict(self.img, goal_image=goal, goal_id="g1")
            c = self.rt.calls[-1]
            self.assertEqual(c["mode"], 6)
            self.assertTrue(np.array_equal(np.asarray(c["goal_image"]), goal))
            self.assertEqual(c["goal_id"], "g1")
            self.assertIsNone(c["goal_pose"]); self.assertIsNone(c["lang"])
            self.assertTrue(np.array_equal(p.waypoints, REF[f"act_m6__{f}"]))
            self.assertEqual(p.command, tuple(float(x) for x in OmniVLADeploy.actions_to_cmd(REF[f"act_m6__{f}"])))

    def test_language_modes(self):
        self.rt.frame = FRAMES[0]
        p = self.m.predict(self.img, instruction="move toward the blue trash bin")
        self.assertEqual((p.mode, self.rt.calls[-1]["lang"]), (7, "move toward the blue trash bin"))
        self.assertTrue(np.array_equal(p.waypoints, REF[f"act_m7__{FRAMES[0]}"]))
        g = REF[f"goal__{FRAMES[1]}"]
        p = self.m.predict(self.img, instruction="follow the path", goal_pose=g)
        c = self.rt.calls[-1]
        self.assertEqual((p.mode, c["mode"], c["lang"]), (8, 8, "follow the path"))
        self.assertTrue(np.array_equal(c["goal_pose"], g))

    def test_image_inputs(self):
        self.rt.frame = FRAMES[0]
        arr = np.asarray(self.img)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png"); self.img.save(path)
            for src in (arr, path, self.img.convert("L").convert("RGB"), self.img.convert("RGBA")):
                self.m.predict(src, goal_pose=REF[f"goal__{FRAMES[0]}"])
                got = self.rt.calls[-1]["image"]
                self.assertEqual(got.mode, "RGB")
        self.assertTrue(np.array_equal(np.asarray(self.rt.calls[0]["image"]), arr))
        with self.assertRaises(ValueError):
            self.m.predict(arr.astype(np.float32), goal_pose=REF[f"goal__{FRAMES[0]}"])

    def test_bad_goal_combinations(self):
        g = REF[f"goal__{FRAMES[0]}"]
        for kw in ({}, dict(goal_image=self.img, goal_pose=g), dict(goal_image=self.img, instruction="x"),
                   dict(instruction="   "), dict(goal_pose=[1.0, 2.0])):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.m.predict(self.img, **kw)
        self.assertEqual(self.rt.calls, [])

    def test_controller_kwargs(self):
        m = OmniVLAJetson("unused", runtime=self.rt, controller=dict(maxv=0.5, maxw=0.5, waypoint_select=2))
        f = FRAMES[0]
        p = m.predict(self.img, goal_pose=REF[f"goal__{f}"])
        self.assertEqual(p.command, tuple(float(x) for x in OmniVLADeploy.actions_to_cmd(REF[f"act_m4__{f}"], waypoint_select=2, maxv=0.5, maxw=0.5)))

    def test_goal_helpers(self):
        self.assertTrue(np.allclose(OmniVLAJetson.goal_pose_from_xy_yaw(3.0, -1.0, np.pi / 2), [30.0, -10.0, 0.0, 1.0]))
        if importlib.util.find_spec("utm"):
            a = self.m.goal_pose_from_gps(37.87371258374039, -122.26729417226024, 270.0, 37.8738930785863, -122.26746181032362)
            self.assertTrue(np.array_equal(a, OmniVLADeploy.goal_pose_from_gps(37.87371258374039, -122.26729417226024, 270.0,
                                                                                  37.8738930785863, -122.26746181032362)))

    def test_prediction_type(self):
        self.rt.frame = FRAMES[0]
        p = self.m.predict(self.img, instruction="go")
        self.assertIsInstance(p, Prediction); self.assertIn("actions", p.raw)
        self.m.warmup(); self.assertEqual(self.rt.calls[-1], dict(warmup=(4, 6)))


class TestLoading(unittest.TestCase):
    """weights_path handling with a fake omnivla_deploy.py (no model is loaded)"""
    FAKE = ("import os\nclass OmniVLADeploy:\n    seen = []\n    def __init__(self, deploy_dir, **kw):\n"
            "        OmniVLADeploy.seen.append((deploy_dir, os.environ.get('OMNIVLA_WEIGHTS'), kw))\n"
            "    @staticmethod\n    def actions_to_cmd(a, **k):\n        return 0.0, 0.0\n")

    def test_weights_path_and_env_restored(self):
        with tempfile.TemporaryDirectory() as d:
            dep = os.path.join(d, "deploy"); os.makedirs(os.path.join(dep, "my_weights"))
            with open(os.path.join(dep, "omnivla_deploy.py"), "w") as fh:
                fh.write(self.FAKE)
            saved = sys.modules.pop("omnivla_deploy"); sys.path.insert(0, dep)
            os.environ["OMNIVLA_WEIGHTS"] = "previous"
            try:
                OmniVLAJetson(os.path.join(dep, "my_weights"), deploy_dir=dep, goal_refresh=3)
                fake = sys.modules["omnivla_deploy"].OmniVLADeploy
                deploy_dir, w, kw = fake.seen[-1]
                self.assertEqual(os.path.join(deploy_dir, w), os.path.abspath(os.path.join(dep, "my_weights")))
                self.assertEqual(kw, dict(goal_refresh=3))
                self.assertEqual(os.environ["OMNIVLA_WEIGHTS"], "previous")
            finally:
                os.environ.pop("OMNIVLA_WEIGHTS"); sys.path.remove(dep); sys.modules["omnivla_deploy"] = saved


if __name__ == "__main__":
    unittest.main()
