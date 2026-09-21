from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
from urllib.parse import urlparse

OP_CONT = 0
OP_TEXT = 1
OP_BIN = 2
OP_CLOSE = 8
OP_PING = 9
OP_PONG = 10
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketError(RuntimeError):
    pass


def encode_frame(payload: bytes, opcode: int = OP_TEXT, *, masked: bool = True) -> bytes:
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))
    n = len(payload)
    mask_bit = 0x80 if masked else 0
    if n < 126:
        header.append(mask_bit | n)
    elif n < 65536:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", n))
    if not masked:
        return bytes(header) + payload
    key = os.urandom(4)
    header.extend(key)
    masked_payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return bytes(header) + masked_payload


def decode_frame_bytes(data: bytes) -> tuple[int, bytes, bool, bytes]:
    """解析一帧，返回 opcode, payload, fin, 剩余字节。"""
    if len(data) < 2:
        raise WebSocketError("帧太短")
    fin = bool(data[0] & 0x80)
    opcode = data[0] & 0x0F
    masked = bool(data[1] & 0x80)
    n = data[1] & 0x7F
    idx = 2
    if n == 126:
        if len(data) < 4:
            raise WebSocketError("帧太短")
        n = struct.unpack("!H", data[2:4])[0]
        idx = 4
    elif n == 127:
        if len(data) < 10:
            raise WebSocketError("帧太短")
        n = struct.unpack("!Q", data[2:10])[0]
        idx = 10
    mask = b""
    if masked:
        mask = data[idx : idx + 4]
        idx += 4
    end = idx + n
    if len(data) < end:
        raise WebSocketError("帧不完整")
    payload = data[idx:end]
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload, fin, data[end:]


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise WebSocketError("连接已关闭")
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(sock: socket.socket) -> tuple[int, bytes, bool]:
    header = _recv_exact(sock, 2)
    fin = bool(header[0] & 0x80)
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    n = header[1] & 0x7F
    if n == 126:
        n = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, n)
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload, fin


class WebSocketClient:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def send_text(self, text: str) -> None:
        self.sock.sendall(encode_frame(text.encode("utf-8"), OP_TEXT))

    def send_binary(self, payload: bytes) -> None:
        self.sock.sendall(encode_frame(payload, OP_BIN))

    def send_pong(self, payload: bytes = b"") -> None:
        self.sock.sendall(encode_frame(payload, OP_PONG))

    def recv(self, timeout: float | None = None) -> tuple[str, bytes | str]:
        self.sock.settimeout(timeout)
        while True:
            opcode, payload, fin = _read_frame(self.sock)
            if opcode == OP_PING:
                self.send_pong(payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                return "close", payload
            buf = bytearray(payload)
            orig = opcode
            while not fin:
                opcode, extra, fin = _read_frame(self.sock)
                if opcode != OP_CONT:
                    raise WebSocketError("分片 opcode 异常")
                buf.extend(extra)
            if orig == OP_TEXT:
                return "text", buf.decode("utf-8")
            if orig == OP_BIN:
                return "binary", bytes(buf)
            raise WebSocketError(f"未知 opcode {orig}")

    def close(self) -> None:
        try:
            self.sock.sendall(encode_frame(b"", OP_CLOSE))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def connect(url: str, timeout: float = 10.0) -> WebSocketClient:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"}:
        raise WebSocketError(f"不支持的 WebSocket 地址: {url}")
    host = parsed.hostname
    if not host:
        raise WebSocketError(f"WebSocket 地址缺少主机: {url}")
    secure = parsed.scheme == "wss"
    port = parsed.port or (443 if secure else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_header = host if (secure and port == 443) or (not secure and port == 80) else f"{host}:{port}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = sock.recv(4096)
            if not chunk:
                raise WebSocketError("握手失败：连接关闭")
            raw += chunk
            if len(raw) > 65536:
                raise WebSocketError("握手失败：响应过长")
        header_blob, _rest = raw.split(b"\r\n\r\n", 1)
        status_line = header_blob.split(b"\r\n", 1)[0].decode("ascii", "replace")
        if "101" not in status_line:
            raise WebSocketError(f"握手失败：{status_line}")
        headers = {}
        for line in header_blob.decode("iso-8859-1").split("\r\n")[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        expected = base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode("ascii")
        if headers.get("sec-websocket-accept") != expected:
            raise WebSocketError("握手失败：Accept 不匹配")
        sock.settimeout(timeout)
        return WebSocketClient(sock)
    except Exception:
        sock.close()
        raise
