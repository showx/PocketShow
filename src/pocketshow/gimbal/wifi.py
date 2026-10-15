from __future__ import annotations

import logging
import struct
import threading
import time

from pocketshow.gimbal.base import GimbalPort
from pocketshow.native import core
from pocketshow.pocket3.udp import DEV_GIMBAL, DjiUdpClient

_core = core()

logger = logging.getLogger(__name__)

CMD_GIMBAL_PWM = 0x01
CMD_GIMBAL_SPEED = 0x0C
CMD_GIMBAL_ABS = 0x14
CMD_GIMBAL_ATTITUDE = 0x05

SPEED_CENTER = 1024


class WifiGimbal(GimbalPort):
    """经 WiFi UDP 发送 DUML 云台速度（CmdSet 0x04）。

    真机客户端的 30Hz 发送在 Rust 里，不跟 YOLO 抢 GIL。测试用的假客户端仍走 Python 线程。
    """

    def __init__(self, client: DjiUdpClient, send_hz: float = 30.0) -> None:
        self.client = client
        self.send_hz = max(5.0, send_hz)
        self.yaw = 0.0
        self.pitch = 0.0
        self.attitude = (0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._running = False
        self._rust = False
        self._thread: threading.Thread | None = None
        self.client.register_duml_callback(4, CMD_GIMBAL_ATTITUDE, self._on_attitude)

    def start(self) -> None:
        if self._running:
            return
        start_gimbal = getattr(self.client, "start_gimbal", None)
        if start_gimbal is not None:
            start_gimbal(self.send_hz)
            self._rust = True
            self._running = True
            self.client.set_gimbal_velocity(self.yaw, self.pitch)
            logger.info("WifiGimbal Rust 控制环 %.0f Hz", self.send_hz)
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="wifi-gimbal")
        self._thread.start()
        logger.info("WifiGimbal 控制环 %.0f Hz", self.send_hz)

    def set_velocity(self, yaw: float, pitch: float) -> None:
        yaw = max(-1.0, min(1.0, yaw))
        pitch = max(-1.0, min(1.0, pitch))
        with self._lock:
            self.yaw = yaw
            self.pitch = pitch
        if self._rust:
            self.client.set_gimbal_velocity(yaw, pitch)

    def recenter(self) -> None:
        self.set_velocity(0.0, 0.0)
        try:
            self.client.send_duml_req(DEV_GIMBAL, 0, 4, CMD_GIMBAL_ABS, _core.gimbal_recenter())
        except OSError:
            logger.warning("recenter 发送失败")

    def close(self) -> None:
        self.set_velocity(0.0, 0.0)
        time.sleep(min(0.1, 2.0 / self.send_hz))
        if self._rust:
            self.client.stop_gimbal()
            self._running = False
            self._rust = False
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _on_attitude(self, pkt: dict) -> None:
        payload = pkt.get("payload", b"")
        if len(payload) < 6:
            return
        yaw, roll, pitch = struct.unpack_from("<hhh", payload, 0)
        self.attitude = (yaw / 10.0, pitch / 10.0, roll / 10.0)

    def _loop(self) -> None:
        interval = 1.0 / self.send_hz
        while self._running:
            with self._lock:
                yaw, pitch = self.yaw, self.pitch
            self._send(yaw, pitch)
            time.sleep(interval)

    def _send(self, yaw: float, pitch: float) -> None:
        pwm, speed = _core.gimbal_packets(yaw, pitch)
        self.client.send_duml_req(DEV_GIMBAL, 0, 4, CMD_GIMBAL_PWM, pwm)
        self.client.send_duml_req(DEV_GIMBAL, 0, 4, CMD_GIMBAL_SPEED, speed)
