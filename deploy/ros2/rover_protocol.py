"""Troupe rover servo protocol and OmniVLA chunk execution (pure Python, no ROS).

Servo daemon on the BeagleBone Blue (default 192.168.7.2:5005): one 5-byte UDP packet per channel, struct "<Bf"
(uint8 channel 1-8, float32 pulse_pct: 5.0 = 1000 us, 7.5 = neutral, 10.0 = 2000 us). The daemon has no watchdog.
Channel map and directions differ between Troupe versions: verify on the bench with the wheels off the ground.
"""
import math
import struct
from dataclasses import dataclass, field

SERVO_CMD_FMT = "<Bf"          # uint8 channel, float32 pulse_pct, little-endian, packed (5 bytes)
NEUTRAL_PCT = 7.5
STEP_S = 0.3                   # OmniVLA action chunk: 8 waypoints, 0.3 s apart
N_STEPS = 8


def pack_servo(channel, pulse_pct):
    if not 1 <= int(channel) <= 8:
        raise ValueError("channel must be 1-8")
    return struct.pack(SERVO_CMD_FMT, int(channel), float(pulse_pct))


def unpack_servo(pkt):
    ch, pct = struct.unpack(SERVO_CMD_FMT, pkt)
    return ch, pct


@dataclass
class ServoMap:
    throttle_ch: int = 1
    steer_ch: int = 2
    throttle_invert: bool = True        # True: forward = pct above neutral (newest Troupe drive_controller.py)
    steer_invert: bool = True           # True: left (positive yaw rate) = pct above neutral
    throttle_max_pct_offset: float = 0.6  # |pct - 7.5| at v = v_full; conservative (full range is 2.5)
    v_full: float = 0.3                 # m/s commanded at throttle_max_pct_offset (calibrate on the bench)
    steer_range_pct: float = 2.5        # |pct - 7.5| at full steering lock
    max_steer_rad: float = 0.45         # full-lock wheel angle (calibrate)
    wheelbase_m: float = 0.32           # placeholder: measure the chassis
    min_v_for_steer: float = 0.05       # below this, steer from yaw rate as if at this speed
    pct_min: float = 5.0
    pct_max: float = 10.0

    def to_pct(self, v, w):
        """(v m/s >= 0, w rad/s, + = left) -> (throttle_pct, steer_pct). Forward only, like the Troupe controllers."""
        v = max(0.0, float(v))
        t_off = min(v / self.v_full, 1.0) * self.throttle_max_pct_offset if self.v_full > 0 else 0.0
        thr = NEUTRAL_PCT + (t_off if self.throttle_invert else -t_off)
        delta = math.atan(self.wheelbase_m * float(w) / max(v, self.min_v_for_steer))   # Ackermann: tan(d) = L w / v
        s_off = max(-1.0, min(1.0, delta / self.max_steer_rad)) * self.steer_range_pct
        st = NEUTRAL_PCT + (s_off if self.steer_invert else -s_off)
        clamp = lambda p: max(self.pct_min, min(self.pct_max, p))
        return clamp(thr), clamp(st)

    def packets(self, throttle_pct, steer_pct):
        return [pack_servo(self.throttle_ch, throttle_pct), pack_servo(self.steer_ch, steer_pct)]

    def neutral_packets(self):
        return self.packets(NEUTRAL_PCT, NEUTRAL_PCT)


@dataclass
class ChunkExecutor:
    """Executes the latest chunk open-loop in time. Waypoint i is the target at t_img + (i+1)*0.3 s (robot frame at
    image time); a new chunk is entered at its own age (latency compensation). spacing_m = meters per action unit
    (OmniVLA's controller: 0.1 m; FrodoBots training data: 0.25 m)."""
    spacing_m: float = 0.1
    max_v: float = 0.3
    max_w: float = 0.3
    chunk: list = field(default=None)
    t_img: float = None

    def set_chunk(self, actions, t_img):
        self.chunk, self.t_img = [list(map(float, a)) for a in actions], float(t_img)

    def step_index(self, now):
        if self.chunk is None:
            return None
        k = int((now - self.t_img) // STEP_S)
        return k if 0 <= k < N_STEPS else None

    def command(self, now):
        """-> (v, w, k) or None when there is no chunk or it is used up (the caller must stop)."""
        k = self.step_index(now)
        if k is None:
            return None
        x0, y0, c0, s0 = (0.0, 0.0, 1.0, 0.0) if k == 0 else self.chunk[k - 1][:4]
        x1, y1, c1, s1 = self.chunk[k][:4]
        dx, dy = (x1 - x0) * self.spacing_m, (y1 - y0) * self.spacing_m
        h0 = math.atan2(s0, c0)
        fwd = dx * math.cos(h0) + dy * math.sin(h0)                     # progress along the current heading
        v = max(0.0, fwd) / STEP_S                                       # forward only (no reversing)
        dth = math.atan2(s1, c1) - h0
        dth = (dth + math.pi) % (2 * math.pi) - math.pi
        w = dth / STEP_S
        if v > self.max_v:                                               # keep the curvature when capping speed
            w *= self.max_v / v; v = self.max_v
        if abs(w) > self.max_w:
            v *= self.max_w / abs(w); w = math.copysign(self.max_w, w)
        return v, w, k
