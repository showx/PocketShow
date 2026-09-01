from __future__ import annotations

from abc import ABC, abstractmethod

from pocketshow.types import FollowCommand


class GimbalPort(ABC):
    """云台控制口：决策层只依赖速度指令，不关心 USB/WiFi。"""

    @abstractmethod
    def set_velocity(self, yaw: float, pitch: float) -> None:
        """yaw/pitch 归一化速度，范围 [-1, 1]。yaw>0 右转，pitch>0 上仰。"""

    @abstractmethod
    def recenter(self) -> None:
        """回中并停转。"""

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""

    def apply(self, command: FollowCommand) -> None:
        self.set_velocity(command.yaw_rate, command.pitch_rate)
