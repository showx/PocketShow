import json
import queue
import threading
import time

from pocketshow.mossvl import (
    MossVLClient,
    detect_protocol,
    strip_moss_tokens,
    ws_url_from,
)
from pocketshow.config import SceneConfig
from pocketshow.scene import SceneNarrator
from pocketshow.types import Track
from pocketshow.ws import OP_TEXT, decode_frame_bytes, encode_frame


class QueueWS:
    def __init__(self) -> None:
        self.inbox: queue.Queue = queue.Queue()
        self.sent: list = []

    def push(self, payload: dict) -> None:
        self.inbox.put(("text", json.dumps(payload)))

    def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def send_binary(self, payload: bytes) -> None:
        self.sent.append(("bin", payload))

    def recv(self, timeout=None):
        try:
            return self.inbox.get(timeout=0.05 if timeout is None else timeout)
        except queue.Empty:
            raise TimeoutError()

    def close(self) -> None:
        self.inbox.put(("close", b""))


class FakeVL:
    def __init__(self, text: str = "小李站起来") -> None:
        self.text = text
        self.frames: list = []
        self.closed = False

    def ensure(self, **kwargs) -> None:
        return None

    def push_frame(self, jpeg: bytes, timestamp: float, prompt: str | None = None) -> None:
        self.frames.append((jpeg, timestamp, prompt))

    def take_caption(self, timeout_s: float = 2.0) -> str:
        return self.text

    def stream_time(self, now: float | None = None) -> float:
        return 1.25

    def close(self) -> None:
        self.closed = True


def test_ws_masked_roundtrip():
    raw = encode_frame(b"hello", OP_TEXT, masked=True)
    opcode, payload, fin, rest = decode_frame_bytes(raw)
    assert opcode == OP_TEXT
    assert payload == b"hello"
    assert fin is True
    assert rest == b""


def test_strip_and_protocol():
    text, silent = strip_moss_tokens("<|response|>小李在打电话")
    assert text == "小李在打电话"
    assert silent is False
    text, silent = strip_moss_tokens("<|silence|>")
    assert text == ""
    assert silent is True
    assert detect_protocol("ws://127.0.0.1:8000/v1/realtime") == "hf"
    assert detect_protocol("ws://127.0.0.1:18500/v1/video/realtime") == "sglang"
    cfg = SceneConfig(ws_url="", base_url="http://10.0.0.2:8000/v1")
    assert ws_url_from(cfg) == "ws://10.0.0.2:8000/v1/realtime"


def test_hf_session_caption():
    sock = QueueWS()
    sock.push({"type": "ready"})
    client = MossVLClient("ws://127.0.0.1:8000/v1/realtime", timeout_s=2.0, connect_fn=lambda url, timeout=0: sock)
    client.ensure(prompt="描述画面")
    assert sock.sent[0]["type"] == "start"
    client.push_frame(b"jpeg-bytes", 1.5, prompt="工位区1：小李")
    assert sock.sent[1]["type"] == "frame"
    assert any(item == ("bin", b"jpeg-bytes") for item in sock.sent)
    sock.push({"type": "output", "text": "<|response|>小李在打电话"})
    assert client.take_caption(1.2) == "小李在打电话"
    client.close()


def test_sglang_session_silence():
    sock = QueueWS()
    sock.push({"type": "session.created"})
    client = MossVLClient(
        "ws://127.0.0.1:18500/v1/video/realtime",
        protocol="sglang",
        timeout_s=2.0,
        connect_fn=lambda url, timeout=0: sock,
    )

    def later() -> None:
        time.sleep(0.05)
        sock.push({"type": "session.ready"})

    threading.Thread(target=later, daemon=True).start()
    client.ensure(prompt="描述画面")
    assert any(item.get("type") == "session.configure" for item in sock.sent if isinstance(item, dict))

    def frame_ready() -> None:
        time.sleep(0.05)
        sock.push({"type": "input.frame.ready"})

    threading.Thread(target=frame_ready, daemon=True).start()
    client.push_frame(b"frame", 0.0)
    sock.push({"type": "response.turn.silence"})
    assert client.take_caption(0.8) == "无事"
    client.close()


def test_narrator_uses_moss_client(tmp_path):
    import numpy as np

    fake = FakeVL("小李站起来离开座位")
    cfg = SceneConfig(
        enabled=True,
        backend="moss-vl",
        sample_fps=1.0,
        gap_s=0.0,
        status=str(tmp_path / "scene.json"),
        log=str(tmp_path / "scene.jsonl"),
    )
    narrator = SceneNarrator(cfg, inline=True, client=fake)
    frame = np.full((80, 120, 3), 30, dtype=np.uint8)
    tracks = [Track(id=1, bbox_xyxy=(10, 10, 40, 70), conf=0.9, person_name="小李", person_id="p001")]
    assert narrator.offer(frame, tracks, camera_id="office", camera_name="工位区1", now=1.0)
    assert narrator.line_for("office") == "小李站起来离开座位"
    assert fake.frames
    narrator.close()
    assert fake.closed
