"""omnivla_nav_node.py - ROS 2 (Humble/Jazzy) navigation node for the Troupe rover, running on the Jetson.

Camera -> OmniVLA (omnivla_deploy.py) -> 8-step action chunk -> servo commands. An inference thread works on the newest
frame; a 20 Hz control timer executes the current chunk in time (rover_protocol.ChunkExecutor).
Subscribes: <camera_topic> (CompressedImage), ~/goal (PoseStamped, robot frame, mode 4), ~/goal_image (mode 6),
  ~/estop (Bool, latching), ~/manual_override (Bool: node goes silent so a teleop can drive).
Publishes: ~/servo_cmd ([throttle_ch, throttle_pct, steer_ch, steer_pct], forwarded by servo_udp_bridge.py), ~/cmd_vel,
  ~/status (JSON), ~/chunk (JSON per inference).
Stops (neutral) on stale frames, used-up chunk, estop, startup, shutdown, and low memory (the process heap creeps
~60 MB / 10 min; restart the node between runs). Set udp_host to send packets directly from the Jetson.
"""
import io, json, socket, threading, time

import numpy as np
import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float32MultiArray, String
from PIL import Image

from rover_protocol import ChunkExecutor, ServoMap, NEUTRAL_PCT


def _yaw(q):
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class OmniVLANavNode(Node):
    def __init__(self, model=None):
        super().__init__("omnivla_nav")
        P = self.declare_parameter
        self.camera_topic = P("camera_topic", "/troupe/camera/color/compressed").value
        self.mode = int(P("mode", 4).value)
        self.stale_s = float(P("stale_frame_s", 0.5).value)
        self.min_mem = int(P("min_mem_available_mb", 300).value)
        self.mem_avail, self.t_mem = None, 0.0
        self.hz = float(P("control_hz", 20.0).value)
        goal_xyyaw = list(P("goal_xy_yaw", [3.0, 0.0, 0.0]).value)            # initial pose goal (m, m, rad)
        self.goal_spacing = float(P("goal_spacing_m", 0.1).value)             # meters per model unit for the goal
        self.exec = ChunkExecutor(spacing_m=float(P("waypoint_spacing_m", 0.1).value),
                                  max_v=float(P("max_v", 0.3).value), max_w=float(P("max_w", 0.3).value))
        self.smap = ServoMap(throttle_ch=int(P("throttle_ch", 1).value), steer_ch=int(P("steer_ch", 2).value),
                             throttle_invert=bool(P("throttle_invert", True).value), steer_invert=bool(P("steer_invert", True).value),
                             throttle_max_pct_offset=float(P("throttle_max_pct_offset", 0.6).value), v_full=float(P("v_full", 0.3).value),
                             steer_range_pct=float(P("steer_range_pct", 2.5).value), max_steer_rad=float(P("max_steer_rad", 0.45).value),
                             wheelbase_m=float(P("wheelbase_m", 0.32).value))
        self.udp_host, self.udp_port = str(P("udp_host", "").value), int(P("udp_port", 5005).value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_host else None

        self.lock = threading.Lock()
        self.frame = None            # (jpeg bytes, receive time monotonic, seq)
        self.seq, self.done_seq = 0, -1
        self.goal = self._goal_norm(*goal_xyyaw); self.goal_image = None; self.goal_id = 0; self.kv_src = None
        self.goal_refresh = int(P("goal_refresh", 0).value)                   # image goal: full pass every N (0 = runtime default)
        self.estop, self.override = False, False
        self.last_lat, self.n_inf, self.state = float("nan"), 0, "WAIT"

        if model is None:
            from omnivla_deploy import OmniVLADeploy
            import os
            kw = dict(goal_refresh=self.goal_refresh) if self.goal_refresh > 0 else {}
            model = OmniVLADeploy(os.environ.get("OMNIVLA_DEPLOY", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), **kw)
            model.warmup(modes=(self.mode,))              # compile kernels BEFORE taking frames (first call ~14 s)
        self.model = model
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)  # matches RELIABLE pubs
        self.create_subscription(CompressedImage, self.camera_topic, self._on_frame, qos)
        self.create_subscription(PoseStamped, "~/goal", self._on_goal, 10)
        self.create_subscription(CompressedImage, "~/goal_image", self._on_goal_image, 1)
        self.create_subscription(Bool, "~/estop", self._on_estop, 10)
        self.create_subscription(Bool, "~/manual_override", self._on_override, 10)
        self.pub_servo = self.create_publisher(Float32MultiArray, "~/servo_cmd", 10)
        self.pub_twist = self.create_publisher(Twist, "~/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.pub_chunk = self.create_publisher(String, "~/chunk", 10)

        self.running = True
        self.worker = threading.Thread(target=self._inference_loop, daemon=True); self.worker.start()
        self.create_timer(1.0 / self.hz, self._control_tick)
        self.get_logger().info(f"ready: mode {self.mode}, camera {self.camera_topic}, stale {self.stale_s}s, "
                               f"caps v {self.exec.max_v} w {self.exec.max_w}, udp {'off' if not self.sock else self.udp_host}")

    def _goal_norm(self, x, y, yaw):
        return np.array([x / self.goal_spacing, y / self.goal_spacing, np.cos(yaw), np.sin(yaw)])

    # ---- callbacks ----
    def _on_frame(self, msg):
        with self.lock:
            self.seq += 1
            self.frame = (bytes(msg.data), time.monotonic(), self.seq, msg.header.frame_id)

    def _on_goal(self, msg):
        p = msg.pose
        with self.lock:
            self.goal = self._goal_norm(p.position.x, p.position.y, _yaw(p.orientation))

    def _on_goal_image(self, msg):
        img = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        with self.lock:
            self.goal_image = img; self.goal_id += 1                    # every goal message = new goal -> cache refresh

    def _on_estop(self, msg):
        self.estop = bool(msg.data)
        self.get_logger().warn(f"estop {'ENGAGED' if self.estop else 'released'}")

    def _on_override(self, msg):
        if bool(msg.data) and not self.override:
            self._send_neutral()                              # hand over from a stopped state
        self.override = bool(msg.data)
        self.get_logger().warn(f"manual override {'ON (node silent)' if self.override else 'OFF'}")

    # ---- inference thread: newest frame only; drops frames it cannot keep up with ----
    def _inference_loop(self):
        while self.running:
            with self.lock:
                fr, goal, gimg, gid = self.frame, self.goal, self.goal_image, self.goal_id
            if fr is None or fr[2] == self.done_seq or time.monotonic() - fr[1] > self.stale_s:
                time.sleep(0.005); continue
            if self.mode == 6 and gimg is None:
                time.sleep(0.05); continue
            img = Image.open(io.BytesIO(fr[0])).convert("RGB")
            kw = dict(goal_image=gimg, goal_id=gid) if self.mode == 6 else dict(goal_pose=goal)
            try:
                out = self.model.predict(img, mode=self.mode, **kw)
            except Exception as e:                           # never leave the rover moving on an inference error
                self.get_logger().error(f"inference failed: {e!r}"); self.exec.chunk = None; time.sleep(0.1); continue
            with self.lock:
                self.exec.set_chunk(out["actions"], fr[1])   # chunk time = frame arrival (camera latency not included)
            self.done_seq, self.last_lat, self.n_inf = fr[2], time.monotonic() - fr[1], self.n_inf + 1
            info = dict(frame_id=fr[3], actions=np.asarray(out["actions"]).tolist(), goal=np.asarray(goal).tolist(),
                        latency=round(self.last_lat, 4), t_fwd=round(out["t_fwd"], 4))
            if "cache" in out:                                # image goal: which full pass the reused goal K/V came from
                if not out["cache"]["kv_reused"]:
                    self.kv_src = fr[3]
                info.update(goal_id=gid, kv_reused=out["cache"]["kv_reused"], cache_age=out["cache"]["age"], kv_src=self.kv_src)
            c = String(); c.data = json.dumps(info)
            self.pub_chunk.publish(c)

    # ---- control timer ----
    def _control_tick(self):
        now = time.monotonic()
        if now - self.t_mem > 1.0:
            self.t_mem = now
            self.mem_avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
        with self.lock:
            age = now - self.frame[1] if self.frame else float("inf")
            cmd = self.exec.command(now)
        if self.override:
            state, v, w = "OVERRIDE", 0.0, 0.0
        elif self.estop:
            state, v, w = "ESTOP", 0.0, 0.0
        elif self.mem_avail is not None and self.mem_avail < self.min_mem:
            state, v, w = "LOWMEM", 0.0, 0.0
        elif age > self.stale_s:
            state, v, w = "STALE", 0.0, 0.0
        elif cmd is None:
            state, v, w = ("WAIT" if self.exec.chunk is None else "EXPIRED"), 0.0, 0.0
        else:
            state, (v, w, _) = "RUN", cmd
        if state != self.state:
            self.get_logger().info(f"state {self.state} -> {state}")
        self.state = state
        if state != "OVERRIDE":
            thr, st = self.smap.to_pct(v, w) if state == "RUN" else (NEUTRAL_PCT, NEUTRAL_PCT)
            self._send(thr, st)
            t = Twist(); t.linear.x, t.angular.z = float(v), float(w); self.pub_twist.publish(t)
        s = String(); s.data = json.dumps(dict(state=state, frame_age=round(age, 3) if age != float("inf") else None,
                                               step=cmd[2] if cmd else None, latency=round(self.last_lat, 3), n_inf=self.n_inf,
                                               mem_available_mb=self.mem_avail))
        self.pub_status.publish(s)

    def _send(self, thr, st):
        m = Float32MultiArray(); m.data = [float(self.smap.throttle_ch), float(thr), float(self.smap.steer_ch), float(st)]
        self.pub_servo.publish(m)
        if self.sock:
            for p in self.smap.packets(thr, st):
                self.sock.sendto(p, (self.udp_host, self.udp_port))

    def _send_neutral(self):
        self._send(NEUTRAL_PCT, NEUTRAL_PCT)

    def shutdown(self):
        self.running = False
        for _ in range(3):
            try:
                self._send_neutral()
            except Exception as e:                           # context already gone: UDP path (if on) still sent
                self.get_logger().error(f"final neutral not published: {e!r}")


def main(args=None):
    # rclpy's default SIGINT handler invalidates the context before the final neutral can be sent; background processes
    # of non-interactive shells also inherit SIGINT=IGNORED. Both signals must reach the finally-block.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
    node = OmniVLANavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown(); node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
