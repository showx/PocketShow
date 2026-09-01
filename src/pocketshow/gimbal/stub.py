from __future__ import annotations

import logging
import threading

from pocketshow.gimbal.base import GimbalPort

logger = logging.getLogger(__name__)


class StubGimbal(GimbalPort):
    """不驱动硬件，把指令留给预览 HUD 和日志。"""

    def __init__(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.0
        self._lock = threading.Lock()

    def set_velocity(self, yaw: float, pitch: float) -> None:
        yaw = max(-1.0, min(1.0, yaw))
        pitch = max(-1.0, min(1.0, pitch))
        with self._lock:
            self.yaw = yaw
            self.pitch = pitch
        if abs(yaw) > 0.02 or abs(pitch) > 0.02:
            logger.debug("stub gimbal yaw=%.3f pitch=%.3f", yaw, pitch)

    def recenter(self) -> None:
        with self._lock:
            self.yaw = 0.0
            self.pitch = 0.0
        logger.info("stub gimbal recenter")

    def close(self) -> None:
        self.recenter()
