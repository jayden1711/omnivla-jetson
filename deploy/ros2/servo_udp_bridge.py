"""Servo UDP bridge for the rover's Raspberry Pi: forwards ~/servo_cmd ([throttle_ch, throttle_pct, steer_ch, steer_pct])
as "<Bf" UDP packets to the servo daemon. Watchdog: no command for watchdog_s -> neutral for 1 s, then silent (so a
manual teleop can drive). Remap: --ros-args -r ~/servo_cmd:=/omnivla_nav/servo_cmd
"""
import socket, time

import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32MultiArray

from rover_protocol import pack_servo, NEUTRAL_PCT


class ServoUdpBridge(Node):
    def __init__(self):
        super().__init__("servo_udp_bridge")
        P = self.declare_parameter
        self.host, self.port = str(P("beaglebone_host", "192.168.7.2").value), int(P("beaglebone_port", 5005).value)
        self.wd = float(P("watchdog_s", 0.3).value)
        self.pct_min, self.pct_max = float(P("pct_min", 5.0).value), float(P("pct_max", 10.0).value)
        self.max_thr = float(P("max_throttle_offset", 1.0).value)
        self.throttle_ch = int(P("throttle_ch", 1).value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last, self.neutral_until, self.n = None, 0.0, 0
        self.create_subscription(Float32MultiArray, "~/servo_cmd", self._on_cmd, 10)
        self.create_timer(0.05, self._watchdog)
        self.get_logger().info(f"forwarding to {self.host}:{self.port}, watchdog {self.wd}s")

    def _send(self, ch, pct):
        pct = max(self.pct_min, min(self.pct_max, pct))
        if ch == self.throttle_ch:
            pct = max(NEUTRAL_PCT - self.max_thr, min(NEUTRAL_PCT + self.max_thr, pct))
        self.sock.sendto(pack_servo(ch, pct), (self.host, self.port)); self.n += 1

    def _on_cmd(self, msg):
        d = list(msg.data)
        if len(d) != 4:
            self.get_logger().error(f"bad servo_cmd length {len(d)}"); return
        self._send(int(d[0]), d[1]); self._send(int(d[2]), d[3])
        self.last, self.channels = time.monotonic(), (int(d[0]), int(d[2]))
        self.neutral_until = 0.0

    def _watchdog(self):
        now = time.monotonic()
        if self.last is None:
            return
        if now - self.last > self.wd:
            if self.neutral_until == 0.0:
                self.neutral_until = now + 1.0
                self.get_logger().warn("no servo_cmd: watchdog -> neutral")
            if now < self.neutral_until:
                for ch in self.channels:
                    self._send(ch, NEUTRAL_PCT)

    def shutdown(self):
        if self.last is not None:
            for ch in self.channels:
                self._send(ch, NEUTRAL_PCT)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    # background processes of non-interactive shells inherit SIGINT=IGNORED; both signals must reach the neutral stop
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
    n = ServoUdpBridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.shutdown(); n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
