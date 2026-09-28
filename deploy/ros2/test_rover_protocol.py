"""Unit tests for rover_protocol (no ROS, no GPU):  python -m pytest test_rover_protocol.py  (or python test_rover_protocol.py)"""
import math, struct

from rover_protocol import (ChunkExecutor, ServoMap, NEUTRAL_PCT, STEP_S, pack_servo, unpack_servo)


def test_packet_matches_servo_cmd_t():
    p = pack_servo(1, 7.5)
    assert len(p) == 5 and p[0] == 1 and struct.unpack("<f", p[1:])[0] == 7.5      # packed uint8 + little-endian float
    assert unpack_servo(pack_servo(2, 8.25)) == (2, 8.25)
    for bad in (0, 9):
        try:
            pack_servo(bad, 7.5); assert False
        except ValueError:
            pass


def test_servo_map_neutral_forward_left():
    m = ServoMap()
    assert m.to_pct(0.0, 0.0) == (NEUTRAL_PCT, NEUTRAL_PCT)
    thr, st = m.to_pct(0.3, 0.0)
    assert thr == NEUTRAL_PCT + 0.6 and st == NEUTRAL_PCT                     # forward = above neutral (invert=True)
    thr, st = m.to_pct(0.2, 0.3)
    assert st > NEUTRAL_PCT                                                  # left turn (w>0) = above neutral
    thr2, _ = ServoMap(throttle_invert=False).to_pct(0.3, 0.0)
    assert thr2 == NEUTRAL_PCT - 0.6
    assert m.to_pct(-1.0, 0.0)[0] == NEUTRAL_PCT                            # never reverses
    thr, st = m.to_pct(99.0, 99.0)
    assert m.pct_min <= thr <= NEUTRAL_PCT + 0.6 and m.pct_min <= st <= m.pct_max
    pk = m.packets(*m.to_pct(0.3, 0.0))
    assert [unpack_servo(x)[0] for x in pk] == [1, 2]


def test_chunk_executor_timing_and_caps():
    ex = ChunkExecutor(spacing_m=0.1, max_v=0.3, max_w=0.3)
    assert ex.command(0.0) is None                                           # no chunk -> caller stops
    chunk = [[1.0 * (i + 1), 0.0, 1.0, 0.0] for i in range(8)]              # straight, 1 unit per step = 0.333 m/s
    ex.set_chunk(chunk, t_img=10.0)
    v, w, k = ex.command(10.0 + 0.01)
    assert k == 0 and abs(v - 0.3) < 1e-9 and w == 0.0                       # capped at max_v
    assert ex.command(10.0 + 2.5 * STEP_S)[2] == 2                           # entered at its own age
    assert ex.command(10.0 + 8 * STEP_S + 0.01) is None                      # used up after 2.4 s -> stop
    assert ex.command(9.9) is None                                           # not yet valid
    yaw = [0.1 * (i + 1) for i in range(8)]                                  # turning left 0.1 rad per step
    ex.set_chunk([[0.5 * (i + 1), 0.0, math.cos(a), math.sin(a)] for i, a in enumerate(yaw)], t_img=0.0)
    v, w, _ = ex.command(0.01)
    assert w > 0 and abs(w) <= 0.3 + 1e-9 and 0 <= v <= 0.3


if __name__ == "__main__":
    for f in [test_packet_matches_servo_cmd_t, test_servo_map_neutral_forward_left, test_chunk_executor_timing_and_caps]:
        f(); print("PASS", f.__name__)
