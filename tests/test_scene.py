import json
from pathlib import Path

import numpy as np

from pocketshow.config import SceneConfig
from pocketshow.scene import (
    SceneLog,
    SceneNarrator,
    chat_vision,
    describe_stub,
    is_silence,
    people_from_tracks,
    same_caption,
)
from pocketshow.types import Track


def _frame():
    return np.full((120, 160, 3), 40, dtype=np.uint8)


def _track(name: str | None = "小李", track_id: int = 1) -> Track:
    return Track(
        id=track_id,
        bbox_xyxy=(10.0, 20.0, 60.0, 90.0),
        conf=0.9,
        person_name=name,
        person_id="p001" if name else None,
    )


def _cfg(tmp_path: Path, **kwargs) -> SceneConfig:
    data = dict(
        enabled=True,
        backend="stub",
        interval_s=8.0,
        gap_s=0.0,
        status=str(tmp_path / "scene.json"),
        log=str(tmp_path / "scene.jsonl"),
    )
    data.update(kwargs)
    return SceneConfig(**data)


def test_people_and_silence():
    assert people_from_tracks([_track("小李"), _track(None, 2)]) == ["小李", "未知名"]
    assert is_silence("无事")
    assert is_silence(" 无事。")
    assert not is_silence("小李站起来离开座位")
    assert same_caption("小李 在工位上。", "小李在工位上")
    assert describe_stub(["小李"], "工位区1") == "工位区1：小李 在工位上"
    assert describe_stub([], "") == "无事"


def test_stub_writes_hud_and_log(tmp_path):
    narrator = SceneNarrator(_cfg(tmp_path), inline=True)
    frame = _frame()
    tracks = [_track("小李")]
    assert narrator.offer(frame, tracks, camera_id="office", camera_name="工位区1", now=10.0)
    assert narrator.line_for("office") == "工位区1：小李 在工位上"
    assert narrator.offer(frame, tracks, camera_id="office", camera_name="工位区1", now=12.0) is False
    assert narrator.offer(frame, tracks, camera_id="office", camera_name="工位区1", now=20.0)
    events = SceneLog(tmp_path / "scene.json", tmp_path / "scene.jsonl").events()
    assert len(events) == 1
    assert events[0]["text"] == "工位区1：小李 在工位上"
    status = json.loads((tmp_path / "scene.json").read_text())
    assert status["enabled"] is True
    assert status["cameras"][0]["id"] == "office"
    narrator.close()


def test_silence_does_not_log(tmp_path):
    narrator = SceneNarrator(_cfg(tmp_path), inline=True)
    assert narrator.offer(_frame(), [], camera_id="office", camera_name="工位区1", now=1.0)
    assert narrator.line_for("office") == ""
    assert SceneLog(tmp_path / "scene.json", tmp_path / "scene.jsonl").events() == []
    narrator.close()


def test_disabled_is_noop(tmp_path):
    narrator = SceneNarrator(_cfg(tmp_path, enabled=False), inline=True)
    assert narrator.offer(_frame(), [_track()], camera_id="office", now=1.0) is False
    assert narrator.line_for("office") == ""
    narrator.close()


def test_chat_vision_payload(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "小李在打电话"}}]}
            ).encode()

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        captured["auth"] = request.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr("pocketshow.scene.urllib.request.urlopen", fake_urlopen)
    text = chat_vision(
        b"jpeg-bytes",
        "描述画面",
        base_url="http://127.0.0.1:30000/v1",
        model="microsoft/Mage-VL",
        api_key="secret",
        timeout_s=9.0,
        max_tokens=40,
    )
    assert text == "小李在打电话"
    assert captured["url"] == "http://127.0.0.1:30000/v1/chat/completions"
    assert captured["timeout"] == 9.0
    assert captured["auth"] == "Bearer secret"
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["text"] == "描述画面"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_scene_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, Settings, WatchConfig

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    log = tmp_path / "scene.jsonl"
    status = tmp_path / "scene.json"
    SceneLog(status, log).write_status(
        {
            "enabled": True,
            "backend": "openai",
            "model": "microsoft/Mage-VL",
            "updated": 1.0,
            "cameras": [{"id": "office", "name": "工位区1", "text": "小李在看屏幕", "silent": False}],
        }
    )
    SceneLog(status, log).append(
        {
            "event": "scene",
            "ts": "2026-09-11 17:00:00",
            "camera": "工位区1",
            "text": "小李在看屏幕",
            "people": ["小李"],
        }
    )
    settings = Settings(
        preview=str(tmp_path / "preview.jpg"),
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
        capture=CaptureConfig(source="auto"),
        rtsp=RtspConfig(settings=str(tmp_path / "capture.json")),
        scene=SceneConfig(enabled=True, status=str(status), log=str(log)),
    )
    client = TestClient(create_app(settings))
    page = client.get("/").text
    assert "sceneNow" in page
    assert "mapNow" in page
    assert "在干嘛" in page
    data = client.get("/api/scene").json()
    assert data["enabled"] is True
    assert data["events"][0]["text"] == "小李在看屏幕"
    assert client.get("/api/status").json()["scene"]["cameras"][0]["text"] == "小李在看屏幕"


def test_hud_chunks_keep_pid_out_of_scene():
    from pocketshow.overlay import _hud_chunks

    assert _hud_chunks("") == []
    assert _hud_chunks("小李在打电话") == ["小李在打电话"]
    long = "甲" * 80
    parts = _hud_chunks(long, width=34)
    assert len(parts) == 2
    assert parts[1].endswith("…")
