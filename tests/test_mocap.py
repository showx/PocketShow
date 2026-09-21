import json

import numpy as np

from pocketshow.config import MocapConfig, Settings
from pocketshow.mocap import (
    MocapClient,
    classify_activity,
    query_people,
    stub_infer,
    stub_skeleton,
)
from pocketshow.types import Track


def _frame():
    return np.full((80, 120, 3), 40, dtype=np.uint8)


def _track() -> Track:
    return Track(id=7, bbox_xyxy=(20.0, 10.0, 60.0, 70.0), conf=0.9, person_name="小李", person_id="p001")


def test_query_people_and_stub_infer():
    people = query_people([_track()], 120, 80)
    assert people[0]["id"] == "p001"
    assert people[0]["x1"] < people[0]["x2"]
    payload = stub_infer(people)
    assert payload["ready"] is True
    assert payload["people"][0]["keypoints"]
    assert payload["people"][0]["activity"] in {"坐着", "站着", "举手", ""}


def test_classify_sitting_and_raised_hand():
    sitting = stub_skeleton(0.3, 0.2, 0.7, 0.7)
    assert classify_activity(sitting) == "坐着"
    standing = stub_skeleton(0.4, 0.05, 0.6, 0.95)
    assert classify_activity(standing) == "站着"
    raised = [
        {"name": "left_shoulder", "u": 0.4, "v": 0.35, "conf": 1.0},
        {"name": "right_shoulder", "u": 0.6, "v": 0.35, "conf": 1.0},
        {"name": "left_hip", "u": 0.42, "v": 0.55, "conf": 1.0},
        {"name": "right_hip", "u": 0.58, "v": 0.55, "conf": 1.0},
        {"name": "left_knee", "u": 0.42, "v": 0.78, "conf": 1.0},
        {"name": "left_ankle", "u": 0.42, "v": 0.92, "conf": 1.0},
        {"name": "left_wrist", "u": 0.38, "v": 0.08, "conf": 1.0},
    ]
    assert classify_activity(raised) == "举手"


def test_stub_mocap_annotates_tracks(tmp_path):
    cfg = MocapConfig(
        enabled=True,
        backend="stub",
        interval_s=0.0,
        status=str(tmp_path / "mocap.json"),
        log=str(tmp_path / "mocap.jsonl"),
    )
    client = MocapClient(cfg, inline=True)
    tracks = [_track()]
    assert client.offer(_frame(), tracks, camera_id="office", camera_name="工位区1", now=1.0)
    client.annotate(tracks, "office")
    assert tracks[0].keypoints
    assert tracks[0].activity
    assert client.line_for("office").startswith("动捕")
    status = json.loads((tmp_path / "mocap.json").read_text())
    assert status["cameras"][0]["id"] == "office"
    client.close()


def test_mocap_http(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_post(url, payload, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "ready": True,
            "people": [
                {
                    "id": "p001",
                    "track_id": 7,
                    "name": "小李",
                    "keypoints": [
                        {"name": "nose", "u": 0.4, "v": 0.2, "conf": 0.9},
                        {"name": "left_shoulder", "u": 0.35, "v": 0.3, "conf": 0.9},
                        {"name": "right_shoulder", "u": 0.45, "v": 0.3, "conf": 0.9},
                        {"name": "left_hip", "u": 0.36, "v": 0.55, "conf": 0.9},
                        {"name": "right_hip", "u": 0.44, "v": 0.55, "conf": 0.9},
                        {"name": "left_knee", "u": 0.36, "v": 0.78, "conf": 0.9},
                        {"name": "left_ankle", "u": 0.36, "v": 0.92, "conf": 0.9},
                    ],
                }
            ],
        }

    monkeypatch.setattr("pocketshow.mocap.post_json", fake_post)
    cfg = MocapConfig(
        enabled=True,
        backend="http",
        base_url="http://127.0.0.1:8006",
        interval_s=0.0,
        status=str(tmp_path / "mocap.json"),
        log=str(tmp_path / "mocap.jsonl"),
    )
    client = MocapClient(cfg, inline=True)
    tracks = [_track()]
    assert client.offer(_frame(), tracks, camera_id="office", camera_name="工位区1", now=1.0)
    client.annotate(tracks, "office")
    assert captured["url"] == "http://127.0.0.1:8006/v1/mocap/frame"
    assert captured["payload"]["camera_id"] == "office"
    assert tracks[0].activity == "站着"
    assert tracks[0].keypoints[0]["name"] == "nose"
    client.close()


def test_mocap_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, WatchConfig
    from pocketshow.scene import SceneLog

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    status = tmp_path / "mocap.json"
    log = tmp_path / "mocap.jsonl"
    SceneLog(status, log).write_status(
        {
            "enabled": True,
            "backend": "http",
            "updated": 1.0,
            "cameras": [{"id": "office", "name": "工位区1", "line": "动捕 小李站着"}],
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
        mocap=MocapConfig(enabled=True, status=str(status), log=str(log)),
    )
    client = TestClient(create_app(settings))
    data = client.get("/api/mocap").json()
    assert data["enabled"] is True
    assert data["cameras"][0]["line"].startswith("动捕")
    assert client.get("/api/status").json()["mocap"]["cameras"][0]["id"] == "office"


def test_overlay_draws_skeleton():
    from pocketshow.overlay import draw_overlay
    from pocketshow.types import FollowCommand, FrameError

    frame = np.full((80, 120, 3), 12, dtype=np.uint8)
    track = _track()
    track.keypoints = stub_skeleton(20 / 120, 10 / 80, 60 / 120, 70 / 80)
    track.activity = "站着"
    vis = draw_overlay(
        frame,
        [track],
        FollowCommand(0.0, 0.0, True, FrameError(0.0, 0.0, 0.0), None),
        7,
        12.0,
        "stub",
        0.08,
        mocap_line="动捕 小李站着",
    )
    assert vis.shape == frame.shape
    assert not np.array_equal(vis, frame)
