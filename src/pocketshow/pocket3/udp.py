from __future__ import annotations

import logging
import random
import socket
import struct
import threading
import time

from pocketshow.pocket3.duml import (
    CMD_TYPE_PUSH,
    CMD_TYPE_REQ,
    CMD_TYPE_WRITE,
    build_duml,
    parse_duml,
)

logger = logging.getLogger(__name__)

TYPE_HANDSHAKE = 0x00
TYPE_TELEMETRY = 0x01
TYPE_VIDEO = 0x02
TYPE_ACK_TELEMETRY = 0x03
TYPE_ACK = 0x04
TYPE_COMMAND = 0x05

DEV_APP = 2
DEV_GIMBAL = 4
DEV_DM368 = 8

# Pocket 3 UDP handshake body: seq seed + window/capability fields used by the
# camera's port-9004 session (public DJI UDP framing, see dji_protocol).
_HANDSHAKE_TAIL = bytes(
    [
        0x64, 0x00, 0x64, 0x00, 0xC0, 0x05,
        0x14, 0x00, 0x00, 0x64, 0x00, 0x00, 0x01, 0x90,
        0x01, 0xC0, 0x05, 0x14, 0x00, 0x00, 0x64, 0x00,
        0x14, 0x00, 0x64, 0x00, 0xC0, 0x05, 0x14, 0x00,
        0x00, 0x64, 0x00, 0x01, 0x01, 0x04, 0x01, 0x02,
    ]
)


def _seq_ahead(new_seq: int, old_seq: int) -> bool:
    diff = (new_seq - old_seq) & 0xFFFF
    return 0 < diff < 0x8000


def _build_header(pkt_len: int, session_id: int, seq: int, pkt_type: int) -> bytes:
    buf = bytearray(8)
    encoded = pkt_len | 0x8000
    buf[0] = encoded & 0xFF
    buf[1] = (encoded >> 8) & 0xFF
    buf[2] = session_id & 0xFF
    buf[3] = (session_id >> 8) & 0xFF
    buf[4] = seq & 0xFF
    buf[5] = (seq >> 8) & 0xFF
    buf[6] = pkt_type & 0xFF
    xor = 0
    for i in range(7):
        xor ^= buf[i]
    buf[7] = xor
    return bytes(buf)


def _parse_header(data: bytes) -> dict | None:
    if len(data) < 8:
        return None
    pkt_len = (data[0] | (data[1] << 8)) & 0x7FFF
    xor = 0
    for i in range(7):
        xor ^= data[i]
    return {
        "length": pkt_len,
        "session_id": data[2] | (data[3] << 8),
        "seq": data[4] | (data[5] << 8),
        "type": data[6],
        "xor_ok": xor == data[7],
        "data": data[8:] if len(data) > 8 else b"",
    }


class DjiUdpClient:
    """Pocket 3 WiFi UDP（端口 9004）：握手、ACK、DUML 命令、H.264 收流。"""

    def __init__(self, camera_ip: str = "192.168.2.1", camera_port: int = 9004) -> None:
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.session_id = random.randint(0, 0xFFFF)
        self.sock: socket.socket | None = None
        self._running = False
        self._rx_thread: threading.Thread | None = None
        self._ack_thread: threading.Thread | None = None
        self._video_hb_thread: threading.Thread | None = None
        self._video_heartbeat_running = False
        self._cmd_seq = 0
        self._msg_seq = 1
        self._cmd_seq_lock = threading.Lock()
        self._video_rx_seq = 0
        self._acktelem_rx_seq = 0
        self._cmd_rx_seq = 0
        self._video_callbacks: list = []
        self._duml_callbacks: list[tuple[int, int, object]] = []
        self._video_hb_counter = 0

    def set_video_callback(self, callback) -> None:
        self._video_callbacks = [callback]

    def register_duml_callback(self, cmd_set: int, cmd_id: int, callback) -> None:
        self._duml_callbacks.append((cmd_set, cmd_id, callback))

    def _next_cmd_seq(self) -> int:
        with self._cmd_seq_lock:
            seq = self._cmd_seq
            self._cmd_seq = (self._cmd_seq + 8) & 0xFFFF
            return seq

    def _next_msg_seq(self) -> int:
        with self._cmd_seq_lock:
            seq = self._msg_seq
            self._msg_seq = (self._msg_seq + 1) & 0xFFFF
            return seq

    def connect(self, timeout: float = 10.0) -> bool:
        logger.info("UDP 连接 %s:%s session=0x%04x", self.camera_ip, self.camera_port, self.session_id)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self._bind_camera_iface()

        seed = random.randint(0, 0xFFFF) & 0xFFF8
        self._video_rx_seq = seed
        self._acktelem_rx_seq = seed
        self._cmd_seq = seed
        self._msg_seq = 1
        payload = bytes([seed & 0xFF, (seed >> 8) & 0xFF]) + _HANDSHAKE_TAIL
        packet = _build_header(8 + len(payload), self.session_id, 0, TYPE_HANDSHAKE) + payload
        try:
            self.sock.sendto(packet, (self.camera_ip, self.camera_port))
            data, _addr = self.sock.recvfrom(4096)
            hdr = _parse_header(data)
            if hdr and hdr["type"] == TYPE_HANDSHAKE:
                self.session_id = hdr["session_id"]
                logger.info("握手成功 camera_session=0x%04x seed=0x%04x", self.session_id, seed)
                return True
            logger.error("握手响应异常")
            return False
        except socket.timeout:
            logger.error("UDP 握手超时")
            return False

    def _bind_camera_iface(self) -> None:
        if self.sock is None:
            return
        for iface in ("en0", "en1"):
            try:
                import subprocess

                ip = subprocess.run(
                    ["ipconfig", "getifaddr", iface],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                if ip.startswith("192.168."):
                    self.sock.bind((ip, 0))
                    logger.debug("绑定网卡 %s (%s)", iface, ip)
                    return
            except OSError:
                continue

    def start(self) -> None:
        if self._running:
            return
        if self.sock is None:
            raise RuntimeError("先调用 connect()")
        self._running = True
        self.sock.settimeout(0.5)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="pocket3-rx")
        self._ack_thread = threading.Thread(target=self._ack_loop, daemon=True, name="pocket3-ack")
        self._rx_thread.start()
        self._ack_thread.start()

    def stop(self) -> None:
        self._running = False
        self._video_heartbeat_running = False
        for thread in (self._rx_thread, self._ack_thread, self._video_hb_thread):
            if thread:
                thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
            self.sock = None

    def _rx_loop(self) -> None:
        assert self.sock is not None
        while self._running:
            try:
                data, _addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            hdr = _parse_header(data)
            if not hdr:
                continue
            pkt_type = hdr["type"]
            if pkt_type == TYPE_VIDEO:
                if _seq_ahead(hdr["seq"], self._video_rx_seq):
                    self._video_rx_seq = hdr["seq"]
                raw = hdr["data"]
                if len(raw) > 12:
                    h264 = raw[12:]
                    for cb in self._video_callbacks:
                        try:
                            cb(h264)
                        except Exception:
                            logger.exception("video callback")
            elif pkt_type == TYPE_TELEMETRY:
                self._dispatch_duml(hdr["data"])
            elif pkt_type == TYPE_ACK_TELEMETRY:
                if _seq_ahead(hdr["seq"], self._acktelem_rx_seq):
                    self._acktelem_rx_seq = hdr["seq"]
                self._dispatch_duml(hdr["data"])
            elif pkt_type == TYPE_COMMAND:
                if _seq_ahead(hdr["seq"], self._cmd_rx_seq):
                    self._cmd_rx_seq = hdr["seq"]
                self._dispatch_duml(hdr["data"])

    def _dispatch_duml(self, data: bytes) -> None:
        for pkt in parse_duml(data):
            for cmd_set, cmd_id, callback in self._duml_callbacks:
                if cmd_set == pkt["cmd_set"] and cmd_id == pkt["cmd_id"]:
                    try:
                        callback(pkt)
                    except Exception:
                        logger.exception("duml callback")

    def _ack_loop(self) -> None:
        while self._running:
            time.sleep(0.02)
            try:
                self._send_ack()
            except Exception:
                pass

    def _send_ack(self) -> None:
        payload = bytearray()
        payload += struct.pack("<HHHH", self._video_rx_seq, self._video_rx_seq, 0, 0)
        payload += struct.pack("<HHHH", self._acktelem_rx_seq, self._acktelem_rx_seq, 0, 0)
        cmd_seq = self._cmd_seq
        payload += struct.pack("<HHHH", cmd_seq, cmd_seq, 0, 0)
        payload += struct.pack("<H", 0)
        packet = _build_header(8 + len(payload), self.session_id, 0, TYPE_ACK) + bytes(payload)
        if self.sock:
            try:
                self.sock.sendto(packet, (self.camera_ip, self.camera_port))
            except OSError:
                pass

    def send_duml(
        self,
        sender_type: int,
        sender_id: int,
        receiver_type: int,
        receiver_id: int,
        cmd_set: int,
        cmd_id: int,
        payload: bytes = b"",
        cmd_type: int = CMD_TYPE_REQ,
    ) -> int:
        seq = self._next_cmd_seq()
        duml_pkt = build_duml(
            sender_type=sender_type,
            sender_id=sender_id,
            receiver_type=receiver_type,
            receiver_id=receiver_id,
            cmd_set=cmd_set,
            cmd_id=cmd_id,
            payload=payload,
            seq=seq,
            cmd_type=cmd_type,
        )
        msg_seq = self._next_msg_seq()
        body = bytearray()
        win_start = (seq - 32) & 0xFFFF
        body += struct.pack("<HH", win_start, seq)
        body += struct.pack("<HH", 0, 0)
        body += struct.pack("<BBH", msg_seq & 0xFF, 0x01, 0x0060)
        body += duml_pkt
        packet = _build_header(8 + len(body), self.session_id, seq, TYPE_COMMAND) + bytes(body)
        if self.sock:
            self.sock.sendto(packet, (self.camera_ip, self.camera_port))
        return seq

    def send_duml_req(self, receiver_type: int, receiver_id: int, cmd_set: int, cmd_id: int, payload: bytes = b"") -> int:
        return self.send_duml(
            DEV_APP, 0, receiver_type, receiver_id, cmd_set, cmd_id, payload, CMD_TYPE_REQ
        )

    def send_duml_push(self, receiver_type: int, receiver_id: int, cmd_set: int, cmd_id: int, payload: bytes = b"") -> int:
        return self.send_duml(
            DEV_APP, 0, receiver_type, receiver_id, cmd_set, cmd_id, payload, CMD_TYPE_PUSH
        )

    def start_video(self) -> None:
        self.send_duml_push(
            DEV_DM368,
            1,
            0x00,
            0x88,
            bytes([0x17, 0x00, 0x46, 0x23, 0x73, 0x41, 0x50, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02]),
        )
        self._send_dm368_register()
        self._video_heartbeat_running = True
        self._video_hb_counter = 0
        self._video_hb_thread = threading.Thread(
            target=self._video_heartbeat_loop, daemon=True, name="pocket3-vhb"
        )
        self._video_hb_thread.start()
        logger.info("已启动 WiFi 视频心跳")

    def _send_dm368_register(self) -> None:
        app_payload = bytearray(64)
        app_payload[1:4] = b"APP"
        self.send_duml(DEV_APP, 0, DEV_DM368, 2, 0x00, 0x81, bytes(app_payload), CMD_TYPE_WRITE)
        self.send_duml(DEV_APP, 0, DEV_DM368, 2, 0x00, 0x82, bytes([0x00]), CMD_TYPE_WRITE)

    def _video_heartbeat_loop(self) -> None:
        tick = 0
        while self._running and self._video_heartbeat_running:
            try:
                counter = self._video_hb_counter & 0xFF
                hb = struct.pack("<BBBBBI", 0x01, 0x00, counter, 0x00, 0x00, 0xFFFFFFFF)
                self.send_duml_push(DEV_DM368, 2, 0x00, 0x4F, hb)
                if tick % 2 == 1:
                    self._video_hb_counter += 1
                if tick % 10 == 0 and tick > 0:
                    self._send_dm368_register()
                tick += 1
                time.sleep(0.2)
            except Exception:
                logger.exception("video heartbeat")
                time.sleep(0.5)
