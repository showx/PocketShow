from __future__ import annotations

import logging
import struct
import threading
import time

from pocketshow.gimbal.base import GimbalPort
from pocketshow.pocket3.udp import DEV_GIMBAL, DjiUdpClient

logger = logging.getLogger(__name__)

CMD_GIMBAL_PWM = 0x01
CMD_GIMBAL_SPEED = 0x0C
CMD_GIMBAL_ABS = 0x14
CMD_GIMBAL_ATTITUDE = 0x05

SPEED_CENTER = 1024
SPEED_SPAN = 1024
CONTROL_FLAGS = 0x8000
CONTROL_EXTRA = 0x0042


class WifiGimbal(GimbalPort):
    """经 WiFi UDP 发送 DUML 云台速度（CmdSet 0x04）。"""

    def __init__(self, client: DjiUdpClient, send_hz: float = 30.0) -> None:
        self.client = client
        self.send_hz = max(5.0, send_hz)
        self.yaw = 0.0
        self.pitch = 0.0
        self.attitude = (0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.client.register_duml_callback(4, CMD_GIMBAL_ATTITUDE, self._on_attitude)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="wifi-gimbal")
        self._thread.start()
        logger.info("WifiGimbal 控制环 %.0f Hz", self.send_hz)

    def set_velocity(self, yaw: float, pitch: float) -> None:
        with self._lock:
            self.yaw = max(-1.0, min(1.0, yaw))
            self.pitch = max(-1.0, min(1.0, pitch))

    def recenter(self) -> None:
        self.set_velocity(0.0, 0.0)
        payload = struct.pack("<hhhBB", 0, 0, 0, 0x07, 30)
        try:
            self.client.send_duml_req(DEV_GIMBAL, 0, 4, CMD_GIMBAL_ABS, payload)
        except OSError:
            logger.warning("recenter 发送失败")

    def close(self) -> None:
        self.set_velocity(0.0, 0.0)
        time.sleep(min(0.1, 2.0 / self.send_hz))
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
        yaw_pwm = int(SPEED_CENTER + yaw * SPEED_SPAN)
        pitch_pwm = int(SPEED_CENTER + pitch * SPEED_SPAN)
        yaw_pwm = max(0, min(2048, yaw_pwm))
        pitch_pwm = max(0, min(2048, pitch_pwm))
        # Pocket 3 WiFi 实测可用：CmdId 0x01 PWM 速度包
        pwm = struct.pack("<HHHHH", yaw_pwm, SPEED_CENTER, pitch_pwm, CONTROL_FLAGS, CONTROL_EXTRA)
        self.client.send_duml_req(DEV_GIMBAL, 0, 4, CMD_GIMBAL_PWM, pwm)
        # 同时发 0x0C 角速度（度/秒 * 10），部分固件认这个
        # 0x0C：角速度，单位 0.1°/s，映射 ±30°/s
        speed = struct.pack(
            "<hhhB",
            max(-1800, min(1800, int(pitch * 30 * 10))),
            0,
            max(-1800, min(1800, int(yaw * 30 * 10))),
            0x01,
        )
        self.client.send_duml_req(DEV_GIMBAL, 0, 4, CMD_GIMBAL_SPEED, speed)
