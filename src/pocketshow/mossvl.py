from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from pocketshow.config import SceneConfig
from pocketshow.ws import WebSocketError, connect

logger = logging.getLogger("pocketshow.mossvl")

_CONTROL = (
    "<|response|>",
    "<|silence|>",
    "<|round_start|>",
    "<|round_end|>",
    "<|video|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|time_start|>",
    "<|time_end|>",
    "<|image|>",
)

DEFAULT_SYSTEM = (
    "你是办公室工位监控的实时助手。只描述画面里正在发生的事。"
    "信息不足或没有关键变化时保持沉默。"
    "不要编造姓名，只用用户消息里给出的名字；没认出的人写成「未知名」。"
)


class FrameSocket(Protocol):
    def send_text(self, text: str) -> None: ...
    def send_binary(self, payload: bytes) -> None: ...
    def recv(self, timeout: float | None = None) -> tuple[str, bytes | str]: ...
    def close(self) -> None: ...


def detect_protocol(url: str, protocol: str = "auto") -> str:
    if protocol and protocol != "auto":
        return protocol
    path = (urlparse(url).path or "").rstrip("/")
    if path.endswith("/v1/video/realtime") or path.endswith("/video/realtime"):
        return "sglang"
    return "hf"


def ws_url_from(cfg: SceneConfig) -> str:
    raw = (cfg.ws_url or "").strip()
    if raw:
        return raw
    base = (cfg.base_url or "").rstrip("/")
    if base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/v1/realtime"


def strip_moss_tokens(text: str) -> tuple[str, bool]:
    raw = text or ""
    silent = "<|silence|>" in raw
    for token in _CONTROL:
        raw = raw.replace(token, "")
    return raw.strip(), silent


def message_type(payload: dict) -> str:
    return str(payload.get("type") or payload.get("event") or "")


def message_text(payload: dict) -> str:
    for key in ("text", "delta", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class MossVLClient:
    """MOSS-VL 实时会话：HF `/v1/realtime` 或 SGLang-Omni `/v1/video/realtime`。"""

    def __init__(
        self,
        url: str,
        *,
        protocol: str = "auto",
        timeout_s: float = 25.0,
        max_tokens: int = 80,
        max_tokens_per_second: float = 12.0,
        connect_fn=connect,
    ) -> None:
        self.url = url
        self.protocol = detect_protocol(url, protocol)
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.max_tokens_per_second = max_tokens_per_second
        self._connect_fn = connect_fn
        self._ws: FrameSocket | None = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._created = threading.Event()
        self._ready = threading.Event()
        self._frame_ready = threading.Event()
        self._text = ""
        self._silent = False
        self._error = ""
        self._seq = 0
        self._started = 0.0
        self._thread: threading.Thread | None = None

    def stream_time(self, now: float | None = None) -> float:
        stamp = time.monotonic() if now is None else now
        if not self._started:
            return 0.0
        return max(0.0, stamp - self._started)

    def ensure(self, *, system_prompt: str = "", prompt: str = "") -> None:
        with self._lock:
            if self._ws is not None and not self._error:
                return
        self.close()
        self._stop.clear()
        self._created.clear()
        self._ready.clear()
        self._frame_ready.clear()
        self._text = ""
        self._silent = False
        self._error = ""
        self._seq = 0
        self._started = time.monotonic()
        ws = self._connect_fn(self.url, timeout=self.timeout_s)
        self._ws = ws
        self._thread = threading.Thread(target=self._loop, name="pocketshow-mossvl", daemon=True)
        self._thread.start()
        if self.protocol == "sglang":
            if not self._created.wait(timeout=min(self.timeout_s, 8.0)) and self._error:
                raise WebSocketError(self._error)
            self._send_json(
                {
                    "type": "session.configure",
                    "prompt": prompt or "",
                    "system_prompt": system_prompt or DEFAULT_SYSTEM,
                    "max_new_tokens": int(self.max_tokens),
                    "max_tokens_per_turn": int(self.max_tokens_per_second),
                    "do_sample": False,
                }
            )
        else:
            self._send_json(
                {
                    "type": "start",
                    "system_prompt": system_prompt or DEFAULT_SYSTEM,
                    "prompt": prompt or "描述重要变化。没有变化时保持沉默。",
                    "frame_queue_size": 256,
                    "max_tokens_per_second": float(self.max_tokens_per_second),
                    "max_new_tokens": max(int(self.max_tokens), 256),
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                }
            )
        if not self._ready.wait(timeout=self.timeout_s):
            raise WebSocketError(self._error or "MOSS-VL 会话没有就绪")
        if self._error:
            raise WebSocketError(self._error)

    def push_frame(self, jpeg: bytes, timestamp: float, prompt: str | None = None) -> None:
        if self._ws is None:
            raise WebSocketError("MOSS-VL 尚未连接")
        if self.protocol == "sglang":
            self._frame_ready.clear()
            body: dict[str, Any] = {
                "type": "input.frame",
                "seq_no": self._seq,
                "timestamp": float(timestamp),
                "mime_type": "image/jpeg",
            }
            if prompt:
                body["prompt"] = prompt
            self._seq += 1
            with self._send_lock:
                self._send_json(body)
                if not self._frame_ready.wait(timeout=self.timeout_s):
                    if self._error:
                        raise WebSocketError(self._error)
                    # 有的服务端不发 ready，直接跟二进制
                self._ws.send_binary(jpeg)
            return
        body = {"type": "frame", "timestamp": float(timestamp)}
        if prompt:
            body["prompt"] = prompt
        with self._send_lock:
            self._send_json(body)
            self._ws.send_binary(jpeg)

    def take_caption(self, timeout_s: float = 2.0) -> str:
        deadline = time.monotonic() + max(0.05, timeout_s)
        last = time.monotonic()
        seen = ""
        while time.monotonic() < deadline:
            with self._lock:
                if self._error:
                    raise WebSocketError(self._error)
                text = self._text
                silent = self._silent
            if silent and not text:
                self._clear_buffer()
                return "无事"
            if text:
                if text != seen:
                    seen = text
                    last = time.monotonic()
                elif time.monotonic() - last >= 0.35:
                    self._clear_buffer()
                    return text
            time.sleep(0.04)
        with self._lock:
            text = self._text
            silent = self._silent
        self._clear_buffer()
        if silent and not text:
            return "无事"
        return text

    def close(self) -> None:
        self._stop.set()
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                if self.protocol == "sglang":
                    self._send_json({"type": "session.abort"}, ws=ws)
                else:
                    self._send_json({"type": "stop"}, ws=ws)
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._created.set()
        self._ready.set()
        self._frame_ready.set()

    def _clear_buffer(self) -> None:
        with self._lock:
            self._text = ""
            self._silent = False

    def _send_json(self, payload: dict, ws: FrameSocket | None = None) -> None:
        target = ws if ws is not None else self._ws
        if target is None:
            raise WebSocketError("MOSS-VL 尚未连接")
        target.send_text(json.dumps(payload, ensure_ascii=False))

    def _loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        while not self._stop.is_set():
            try:
                kind, data = ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    with self._lock:
                        self._error = self._error or "MOSS-VL 连接断开"
                    self._created.set()
                    self._ready.set()
                    self._frame_ready.set()
                break
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("MOSS-VL 接收失败：%s", exc)
                    with self._lock:
                        self._error = str(exc)
                    self._created.set()
                    self._ready.set()
                    self._frame_ready.set()
                break
            if kind == "close":
                break
            if kind != "text" or not isinstance(data, str):
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            self._handle(payload)

    def _handle(self, payload: dict) -> None:
        kind = message_type(payload)
        if kind == "session.created":
            self._created.set()
            return
        if kind in {"ready", "session.ready"}:
            self._ready.set()
            return
        if kind == "session.configured":
            return
        if kind in {"input.frame.ready", "frame_ack"}:
            self._frame_ready.set()
            return
        if kind in {"input.frame.accepted", "input.frame.processed", "prompt_ack"}:
            return
        if kind in {"response.turn.silence", "silence"}:
            with self._lock:
                self._silent = True
            return
        if kind in {"output", "response.text.delta", "response.output_text.delta"}:
            chunk = message_text(payload)
            cleaned, silent = strip_moss_tokens(chunk)
            with self._lock:
                if silent and not cleaned:
                    self._silent = True
                if cleaned:
                    self._text = (self._text + cleaned).strip()
                    self._silent = False
            return
        if kind in {"error", "session.error"}:
            detail = str(payload.get("message") or payload.get("error") or payload.get("detail") or "MOSS-VL 报错")
            with self._lock:
                self._error = detail
            self._created.set()
            self._ready.set()
            self._frame_ready.set()
            logger.warning("MOSS-VL：%s", detail)
            return


def describe_with_moss(
    jpeg: bytes,
    prompt: str,
    cfg: SceneConfig,
    client: MossVLClient | None = None,
    *,
    timestamp: float | None = None,
) -> tuple[str, MossVLClient]:
    session = client
    if session is None:
        session = MossVLClient(
            ws_url_from(cfg),
            protocol=cfg.protocol,
            timeout_s=cfg.timeout_s,
            max_tokens=cfg.max_tokens,
            max_tokens_per_second=cfg.max_tokens_per_second,
        )
    session.ensure(system_prompt=cfg.system_prompt or DEFAULT_SYSTEM, prompt=prompt)
    session.push_frame(jpeg, session.stream_time(timestamp), prompt=prompt)
    wait = min(max(0.4, cfg.timeout_s), 8.0)
    text = session.take_caption(wait)
    return text, session
