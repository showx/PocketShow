"""Pocket 3 WiFi UDP。会话、ACK 和云台节拍在 Rust 里跑，避免和检测抢 GIL。"""

from __future__ import annotations

import logging

from pocketshow.native import core

logger = logging.getLogger(__name__)

_core = core()

TYPE_HANDSHAKE = 0x00
TYPE_TELEMETRY = 0x01
TYPE_VIDEO = 0x02
TYPE_ACK_TELEMETRY = 0x03
TYPE_ACK = 0x04
TYPE_COMMAND = 0x05

DEV_APP = 2
DEV_GIMBAL = 4
DEV_DM368 = 8


class DjiUdpClient:
    """Pocket 3 WiFi UDP（端口 9004）：握手、ACK、DUML 命令、H.264 收流。"""

    def __init__(self, camera_ip: str = "192.168.2.1", camera_port: int = 9004) -> None:
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self._rs = _core.UdpClient(camera_ip, camera_port)

    def set_video_callback(self, callback) -> None:
        self._rs.set_video_callback(callback)

    def register_duml_callback(self, cmd_set: int, cmd_id: int, callback) -> None:
        self._rs.register_duml_callback(int(cmd_set), int(cmd_id), callback)

    def connect(self, timeout: float = 10.0) -> bool:
        logger.info("UDP 连接 %s:%s", self.camera_ip, self.camera_port)
        ok = bool(self._rs.connect(float(timeout)))
        if ok:
            logger.info(
                "握手成功 camera_session=0x%04x seed=0x%04x",
                self._rs.session_id(),
                self._rs.seed(),
            )
        else:
            logger.error("UDP 握手失败：%s", self._rs.last_error() or "未知原因")
        return ok

    def start(self) -> None:
        self._rs.start()

    def stop(self) -> None:
        self._rs.stop()

    def send_duml(
        self,
        sender_type: int,
        sender_id: int,
        receiver_type: int,
        receiver_id: int,
        cmd_set: int,
        cmd_id: int,
        payload: bytes = b"",
        cmd_type: int = 0,
    ) -> int:
        return int(
            self._rs.send_duml(
                int(sender_type),
                int(sender_id),
                int(receiver_type),
                int(receiver_id),
                int(cmd_set),
                int(cmd_id),
                payload,
                int(cmd_type),
            )
        )

    def send_duml_req(
        self,
        receiver_type: int,
        receiver_id: int,
        cmd_set: int,
        cmd_id: int,
        payload: bytes = b"",
    ) -> int:
        return int(
            self._rs.send_duml_req(int(receiver_type), int(receiver_id), int(cmd_set), int(cmd_id), payload)
        )

    def send_duml_push(
        self,
        receiver_type: int,
        receiver_id: int,
        cmd_set: int,
        cmd_id: int,
        payload: bytes = b"",
    ) -> int:
        return int(
            self._rs.send_duml_push(int(receiver_type), int(receiver_id), int(cmd_set), int(cmd_id), payload)
        )

    def start_video(self) -> None:
        self._rs.start_video()
        logger.info("已启动 WiFi 视频心跳")

    def start_gimbal(self, hz: float) -> None:
        self._rs.start_gimbal(float(hz))

    def set_gimbal_velocity(self, yaw: float, pitch: float) -> None:
        self._rs.set_gimbal_velocity(float(yaw), float(pitch))

    def stop_gimbal(self) -> None:
        self._rs.stop_gimbal()
