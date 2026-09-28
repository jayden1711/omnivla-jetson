"""Offline replay test of omnivla_nav_node + servo_udp_bridge (no rover), run by run_ros_test.sh.

FrodoBots frames are published as the camera (each held 1 s at 15 Hz) with their ground-truth goals. Scenario: camera
pause at 40 s (STALE), estop at 60-65 s, manual override at 75-85 s, stop at 100 s. Checks chunks against the reference
outputs, rates, caps, servo mapping, shutdown behavior and the UDP packet format. Writes results/ros_replay*.json.
"""
import csv, json, math, os, socket, struct, sys, threading, time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float32MultiArray, String

ROOT = os.environ.get("OMNIVLA_ROOT", "/mnt/nvme/omnivla")
OMNI = f"{ROOT}/OmniVLA"
FRAMES = f"{ROOT}/frames"
MODE = int(os.environ.get("REPLAY_MODE", "4"))       # 6: image goal (sequential clips, one goal image per clip, goal cache)
REF = f"{OMNI}/mf_results/" + os.environ.get("REPLAY_REF", "e2e_deploy@4")
TAG = ("" if MODE == 4 else f"{MODE}") + os.environ.get("REPLAY_OUT_TAG", "")   # e.g. _fault_x: keep the real results
MARK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rostest"
SINK_PORT, HZ, HOLD, T_END = 15005, 15.0, 1.0, 100.0
PAUSE, ESTOP, OVERRIDE = (40.0, 42.0), (60.0, 65.0), (75.0, 85.0)

rows = [r for r in csv.DictReader(open(f"{OMNI}/gt_tests.csv")) if r["reliable_path"] == "True" and r["img_ok"] == "True"]
names = sorted(r["frame"] for r in rows)
gz = np.load(f"{OMNI}/gt_frodobots.npz")
if MODE == 6:                                       # clips of 101 frames (20 fps), played at their native timing
    seqr = list(csv.DictReader(open(f"{OMNI}/seq_frames.csv")))
    CLIPS = sorted({r["clip"] for r in seqr})
    CLIPF = {c: [r["frame"] for r in sorted((r for r in seqr if r["clip"] == c), key=lambda r: int(r["idx"]))] for c in CLIPS}
    CLIP_S = 5.0
    gz = np.load(f"{OMNI}/seq.npz")


class Tester(Node):
    def __init__(self):
        super().__init__("replay_tester")
        self.pub_img = self.create_publisher(CompressedImage, "/troupe/camera/color/compressed", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "/omnivla_nav/goal", 10)
        self.pub_gimg = self.create_publisher(CompressedImage, "/omnivla_nav/goal_image", 1)
        self.pub_estop = self.create_publisher(Bool, "/omnivla_nav/estop", 10)
        self.pub_ovr = self.create_publisher(Bool, "/omnivla_nav/manual_override", 10)
        self.t0 = None; self.ev = {}; self.log = {"servo": [], "twist": [], "status": [], "chunk": [], "udp": []}
        rel = lambda: (time.monotonic() - self.t0) if self.t0 else -1.0
        self.rel = rel
        self.create_subscription(Float32MultiArray, "/omnivla_nav/servo_cmd", lambda m: self.log["servo"].append((rel(), list(m.data))), 100)
        self.create_subscription(Twist, "/omnivla_nav/cmd_vel", lambda m: self.log["twist"].append((rel(), m.linear.x, m.angular.z)), 100)
        self.create_subscription(String, "/omnivla_nav/status", lambda m: self.log["status"].append((rel(), json.loads(m.data))), 100)
        self.create_subscription(String, "/omnivla_nav/chunk", lambda m: self.log["chunk"].append((rel(), json.loads(m.data))), 100)

    def udp_sink(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", SINK_PORT)); s.settimeout(0.2)
        while not getattr(self, "stop_sink", False):
            try:
                pkt, _ = s.recvfrom(64)
            except socket.timeout:
                continue
            self.log["udp"].append((self.rel(), len(pkt), *(struct.unpack("<Bf", pkt) if len(pkt) == 5 else (None, None))))


def goal_msg(f):
    u = gz[f"goal__{f}"].astype(np.float64)                   # action units (0.25 m) -> meters; node uses goal_spacing_m 0.25
    m = PoseStamped(); m.header.frame_id = "base_link"
    m.pose.position.x, m.pose.position.y = float(u[0] * 0.25), float(u[1] * 0.25)
    yaw = math.atan2(u[3], u[2]); m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return m


def main():
    rclpy.init()
    n = Tester()
    th = threading.Thread(target=rclpy.spin, args=(n,), daemon=True); th.start()
    sk = threading.Thread(target=n.udp_sink, daemon=True); sk.start()
    t_wait = time.monotonic()
    while not n.log["status"]:                               # wait for the node (model load ~50 s)
        time.sleep(0.5)
        if time.monotonic() - t_wait > 300:
            print("[ROSTEST] FAIL node did not come up"); sys.exit(2)
    if MODE == 6:
        jpeg = {f: open(f"{OMNI}/seq/{f}.jpg", "rb").read() for c in CLIPS for f in CLIPF[c]}
        gjpg = {c: open(f"{OMNI}/seq/goals/{c}.jpg", "rb").read() for c in CLIPS}
    else:
        jpeg = {f: open(f"{FRAMES}/{f}.jpg", "rb").read() for f in names}
    n.t0 = time.monotonic(); cur = None; est = ovr = False
    while (t := n.rel()) < T_END:
        if MODE == 6:
            c = CLIPS[min(int(t / CLIP_S), len(CLIPS) - 1)]
            f = CLIPF[c][min(int((t % CLIP_S) * 20), 100)]
            if c != cur:
                g = CompressedImage(); g.format = "jpeg"; g.data = gjpg[c]; n.pub_gimg.publish(g); cur = c; time.sleep(0.05)
        else:
            f = names[min(int(t / HOLD), len(names) - 1)]
            if f != cur:
                n.pub_goal.publish(goal_msg(f)); cur = f; time.sleep(0.05)
        e = ESTOP[0] <= t < ESTOP[1]; o = OVERRIDE[0] <= t < OVERRIDE[1]
        if e != est:
            b = Bool(); b.data = e; n.pub_estop.publish(b); est = e; n.ev["estop_on" if e else "estop_off"] = n.rel()
        if o != ovr:
            b = Bool(); b.data = o; n.pub_ovr.publish(b); ovr = o; n.ev["override_on" if o else "override_off"] = n.rel()
        if not (PAUSE[0] <= t < PAUSE[1]):
            m = CompressedImage(); m.format = "jpeg"; m.header.frame_id = f; m.data = jpeg[f]
            m.header.stamp = n.get_clock().now().to_msg(); n.pub_img.publish(m)
        time.sleep(1.0 / HZ)
    open(f"{MARK}.scenario_done", "w").close()
    t_stop_wait = time.monotonic()
    while not os.path.exists(f"{MARK}.sigint") and time.monotonic() - t_stop_wait < 30:
        time.sleep(0.01)
    t_sigint = n.rel()
    while not os.path.exists(f"{MARK}.node_stopped") and time.monotonic() - t_stop_wait < 60:
        time.sleep(0.1)
    t_node_stop = n.rel()
    time.sleep(3.0); n.stop_sink = True; time.sleep(0.3)
    try:
        ok = report(n.log, t_node_stop, t_sigint, n.ev)
    except Exception as e:                                   # a crashing report must be a FAIL, never a silent pass
        import traceback; traceback.print_exc()
        print(f"[ROSTEST] FAIL report_crashed: {e!r}", flush=True); ok = False
    n.destroy_node(); rclpy.shutdown()
    os._exit(0 if ok else 1)


def report(L, t_node_stop, t_sigint, ev=None):
    ev = ev or {}
    R, ok = {}, True
    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond); R[name] = dict(pass_=bool(cond), detail=detail)
        print(f"[ROSTEST] {'PASS' if cond else 'FAIL'} {name}: {detail}", flush=True)
    # 1) chunks equal the validated deploy@4 outputs for the same frame (mode 4); cache bookkeeping (mode 6)
    d, lat, tf, race = [], [], [], 0
    if MODE == 6:
        tf = [c["t_fwd"] for _, c in L["chunk"]]; lat = [c["latency"] for _, c in L["chunk"]]
        miss = sum(1 for _, c in L["chunk"] if not all(k in c for k in ("goal_id", "kv_reused", "cache_age", "kv_src")))
        check("image_goal_chunk_fields", L["chunk"] and miss == 0, f"{miss}/{len(L['chunk'])} chunks without goal-cache fields (a mode-6 runtime call always returns them)")
        if miss:
            L["chunk"] = [(t, c) for t, c in L["chunk"] if "goal_id" in c]
        prev, bad, n_new, n_reuse = None, 0, 0, 0
        for _, c in L["chunk"]:
            if c["goal_id"] != prev:                      # first chunk of a new goal must be a full pass
                n_new += 1; bad += int(c["kv_reused"]); prev = c["goal_id"]
            n_reuse += int(c["kv_reused"])
            bad += int(c["kv_reused"] and c["cache_age"] >= int(os.environ.get("GOAL_REFRESH", "1")))
        n_played = min(len(CLIPS), int(T_END / CLIP_S))              # clips shown in the scenario (20 of 24 in 100 s)
        check("goal_cache_refresh", bad == 0 and n_new >= n_played - 1,
              f"{n_new} goal changes for {n_played} clips played, {'all refreshed' if bad == 0 else str(bad) + ' violations'}; {n_reuse}/{len(L['chunk'])} chunks reused the goal K/V, max age < {os.environ.get('GOAL_REFRESH', '1')}")
        json.dump([c for _, c in L["chunk"]], open(f"{OMNI}/results/ros_replay{TAG}_chunks.json", "w"))
    for t, c in (L["chunk"] if MODE == 4 else []):
        lat.append(c["latency"]); tf.append(c["t_fwd"])
        p = f"{REF}/{c['frame_id']}.npz"
        want = gz[f"goal__{c['frame_id']}"].astype(np.float64)
        if not np.allclose(np.array(c["goal"]), want, atol=1e-9):   # image and goal travel on separate topics:
            race += 1; continue                                        # the first inference of a frame may use the old goal
        if os.path.exists(p):
            d.append(float(np.abs(np.array(c["actions"]) - np.load(p)["act"][0]).max()))
    if MODE == 4:
      n_c = len(L["chunk"])
      check("goal_fresh", n_c > 0 and race <= 0.2 * n_c,   # topic races are ~5-8% of chunks; a node that ignores goal updates ~all
            f"{race}/{n_c} chunks used a goal other than the one published for their frame (<= 20%)")
      check("chunks_match_validated", d and max(d) <= 1e-3, f"{len(d)} chunks with this frame's goal, max|d| {max(d) if d else None:.2e}, exact {sum(x == 0 for x in d)}; {race} chunks used the previous frame's goal (topic race, excluded)")
    check("inference_latency", bool(tf) and np.median(tf) < 1.0, f"fwd median {np.median(tf)*1000 if tf else float('nan'):.0f} ms")
    # frame arrival -> chunk published: pose ~0.5 s, image goal ~1.1 s (p95)
    check("chunk_latency", bool(lat) and np.percentile(lat, 95) <= 1.5,
          f"frame->chunk median {np.median(lat)*1000 if lat else float('nan'):.0f} ms, p95 {np.percentile(lat, 95)*1000 if lat else float('nan'):.0f} ms (<= 1500)")
    S = [(t, s) for t, s in L["servo"] if t >= 0]
    win = lambda a, b: [(t, s) for t, s in S if a <= t < b]
    neutral = lambda s: abs(s[1] - 7.5) < 1e-6 and abs(s[3] - 7.5) < 1e-6
    rate = len(win(5, 39)) / 34.0
    check("servo_rate_20hz", 17 <= rate <= 21, f"{rate:.1f} msg/s in [5,39) s")
    st = [(t, s["state"]) for t, s in L["status"] if t >= 0]
    hist = {k: sum(1 for t, s in st if 5 <= t < 39 and s == k) for k in sorted({s for _, s in st})}
    run_frac = np.mean([s == "RUN" for t, s in st if 5 <= t < 39])
    t_first = next((t for t, s in st if s == "RUN"), None)
    check("drives_when_healthy", run_frac > 0.9 and t_first is not None and t_first < 1.5, f"RUN {run_frac*100:.0f}% of ticks in [5,39) s {hist}; first RUN at {t_first if t_first is not None else float('nan'):.2f} s after the first frame")
    t_stale = next((t for t, s in st if t >= PAUSE[0] and s == "STALE"), None)
    check("stale_frame_stop", t_stale is not None and t_stale - PAUSE[0] <= 0.5 + 1.0 / HZ + 0.1 and all(neutral(s) for _, s in win(t_stale + 0.06, PAUSE[1])),
          f"STALE {t_stale - PAUSE[0] if t_stale else float('nan'):.2f} s after the camera stopped (limit 0.5 s + one frame period); neutral during the pause")
    t_resume = next((t for t, s in st if t >= PAUSE[1] and s == "RUN"), None)
    check("resumes_after_stale", t_resume is not None and t_resume - PAUSE[1] < 2.0, f"RUN again {t_resume - PAUSE[1] if t_resume else float('nan'):.2f} s after frames resumed")
    t_e = ev.get("estop_on", ESTOP[0])                   # windows start when the tester actually published the event
    e = win(t_e + 0.1, ESTOP[1])
    check("estop_neutral", e and all(neutral(s) for _, s in e) and all(s == "ESTOP" for t, s in st if t_e + 0.1 <= t < ESTOP[1]),
          f"{len(e)} commands in [estop published + 0.1 s, end), {sum(not neutral(s) for _, s in e)} non-neutral; estop published at +{t_e - ESTOP[0]:.3f} s")
    t_o = ev.get("override_on", OVERRIDE[0])
    hand = win(t_o, t_o + 0.1); o = win(t_o + 0.1, OVERRIDE[1])  # the node sends ONE neutral when override turns on (handover)
    # regular 20 Hz ticks may still go out until the override MESSAGE reaches the node; after its handover neutral: nothing
    i_n = next((i for i, (_, s_) in enumerate(hand) if neutral(s_)), None)
    hand_ok = i_n is not None and all(neutral(s_) for _, s_ in hand[i_n:])
    check("override_silent", len(o) == 0 and hand_ok,
          f"{len(o)} node commands in [override published + 0.1 s, end) (expect 0); in the first 0.1 s: {len(hand)} command(s), "
          f"handover neutral sent and nothing but neutral after it: {hand_ok}; override published at +{t_o - OVERRIDE[0]:.3f} s")
    tw = [(t, v, w) for t, v, w in L["twist"] if t >= 0]
    check("speed_caps", all(v <= 0.3 + 1e-6 and abs(w) <= 0.3 + 1e-6 and v >= 0 for _, v, w in tw),
          f"max v {max(v for _, v, _ in tw):.3f} m/s, max |w| {max(abs(w) for *_, w in tw):.3f} rad/s (caps 0.3 / 0.3)")
    pc = [x for _, s in S for x in (s[1], s[3])]; thr = [s[1] for _, s in S]
    check("pulse_ranges", min(pc) >= 5.0 and max(pc) <= 10.0 and min(thr) >= 7.5 and max(thr) <= 8.1 + 1e-6,
          f"pulse {min(pc):.2f}-{max(pc):.2f} %, throttle {min(thr):.2f}-{max(thr):.2f} % (forward-only, cap 8.1)")
    U = L["udp"]
    fmt_ok = all(ln == 5 and ch in (1, 2) and 5.0 <= pct <= 10.0 for _, ln, ch, pct in U)
    n_fwd = sum(1 for t, *_ in U if t < t_node_stop - 0.2); n_sv = sum(1 for t, _ in S if t < t_node_stop - 0.3)
    check("udp_packets_servo_cmd_t", U and fmt_ok and abs(n_fwd - 2 * n_sv) <= 0.05 * 2 * n_sv + 4,
          f"{len(U)} packets, all 5 bytes '<Bf', channels 1/2; {n_fwd} forwarded vs 2x{n_sv} commands")
    after = [(t, ch, pct) for t, ln, ch, pct in U if t >= t_sigint]
    drive = [t - t_sigint for t, ch, p in after if abs(p - 7.5) > 1e-6]
    last_t = max((t for t, *_ in after), default=None)
    # a control tick already running when SIGINT arrives can still send one drive command: the Python handler only runs
    # between bytecodes. Require every drive packet within 0.15 s of SIGINT (2 control periods + DDS/UDP), then neutral,
    # then silence within 2 s (bridge watchdog).
    check("stop_then_watchdog_silence", after and (not drive or max(drive) <= 0.15) and abs(after[-1][2] - 7.5) < 1e-6 and last_t - t_sigint < 2.0,
          f"{len(after)} packets after SIGINT; {len(drive)} non-neutral" + (f" (latest +{max(drive):.3f} s)" if drive else "") +
          f"; last packet neutral at +{(last_t - t_sigint) if last_t else 0:.2f} s, then silent (bridge watchdog)")
    # servo output must be the node's own mapping of the command it publishes in the same tick (catches a flipped or
    # mis-scaled steering/throttle stage between controller and servo); mapping = rover_protocol.ServoMap defaults = node defaults
    from rover_protocol import ServoMap, ChunkExecutor
    sm = ServoMap(); SV = [(t, s_) for t, s_ in L["servo"] if 0 <= t < t_sigint]; TW = [(t, v, w) for t, v, w in L["twist"] if 0 <= t < t_sigint]
    si, bad_sv, n_sv2 = 0, 0, 0
    for t, v, w in TW:
        while si + 1 < len(SV) and SV[si + 1][0] <= t + 0.005:
            si += 1
        if not SV or abs(SV[si][0] - t) > 0.03:
            continue
        et, es = sm.to_pct(v, w); n_sv2 += 1
        bad_sv += abs(SV[si][1][1] - et) > 1e-3 or abs(SV[si][1][3] - es) > 1e-3
    check("servo_matches_command", n_sv2 > 0.8 * len(TW) and bad_sv <= 0.01 * n_sv2,
          f"{n_sv2} servo/cmd_vel pairs, {bad_sv} where the servo pulses differ from ServoMap(cmd_vel) (<= 1%)")
    # commanded turn direction must be what the ChunkExecutor derives from the latest chunk (catches a flipped controller)
    CH = [(t, c) for t, c in L["chunk"] if t >= 0]; ci, agree_n, tot_n = 0, 0, 0
    for t, v, w in TW:
        while ci + 1 < len(CH) and CH[ci + 1][0] <= t:
            ci += 1
        if not CH or CH[ci][0] > t or abs(w) < 0.01:
            continue
        ex = ChunkExecutor(); ex.set_chunk(CH[ci][1]["actions"], CH[ci][0] - CH[ci][1]["latency"])
        cmd = ex.command(t)
        if cmd is None or abs(cmd[1]) < 0.05:
            continue
        tot_n += 1; agree_n += np.sign(cmd[1]) == np.sign(w)
    check("steer_follows_chunk", tot_n >= 50 and agree_n >= 0.85 * tot_n,
          f"turn direction of cmd_vel vs the executor's command from the latest chunk: {agree_n}/{tot_n} agree (>= 85%)")
    # direction sanity vs the human's path (GT heading change at step 8), first chunk per frame: sign agreement on frames
    # where the human turned (|yaw| > 0.1 rad) and Spearman correlation (Pearson is outlier-driven here).
    yaw_pred, yaw_gt, seen = [], [], set()
    for _, c in L["chunk"]:
        if c["frame_id"] in seen:
            continue
        seen.add(c["frame_id"])
        if f"act__{c['frame_id']}" not in gz.files:
            continue
        a = np.array(c["actions"]); g = gz[f"act__{c['frame_id']}"]
        yaw_pred.append(math.atan2(a[7, 3], a[7, 2])); yaw_gt.append(math.atan2(g[7, 3], g[7, 2]))
    from scipy.stats import spearmanr
    yp, yg = np.array(yaw_pred), np.array(yaw_gt); t = np.abs(yg) > 0.1
    agree = float(np.mean(np.sign(yp[t]) == np.sign(yg[t]))) if t.any() else float("nan")
    rho = float(spearmanr(yp, yg).correlation)
    # Catches wiring/sign errors only. Floors sit below each mode's offline reference (pose: sign 0.89, Spearman 0.73;
    # image goal: 0.71 / 0.31) and well above chance.
    f_sign, f_rho = (0.7, 0.4) if MODE == 4 else (0.55, 0.2)
    check("turns_follow_actual_path", agree >= f_sign and rho >= f_rho,
          f"{len(yp)} frames: turn-sign agreement on {int(t.sum())} turning frames {agree:.2f} (>= {f_sign}), Spearman {rho:.2f} (>= {f_rho})")
    os.makedirs(f"{OMNI}/results", exist_ok=True)
    json.dump(dict(ok=ok, checks=R, n_chunks=len(L["chunk"]), n_servo=len(S), n_udp=len(U)), open(f"{OMNI}/results/ros_replay{TAG}.json", "w"), indent=1)
    with open(f"{OMNI}/results/ros_replay{TAG}_servo.csv", "w") as fh:
        w = csv.writer(fh); w.writerow(["t_s", "throttle_ch", "throttle_pct", "steer_ch", "steer_pct"])
        for t, s in S:
            w.writerow([round(t, 3)] + [round(x, 4) for x in s])
    print(f"[ROSTEST] {'ALL PASS' if ok else 'SOME CHECKS FAILED'}", flush=True)
    return ok


if __name__ == "__main__":
    main()
