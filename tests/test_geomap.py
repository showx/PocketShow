import json

import numpy as np

from pocketshow.config import MapConfig, Settings
from pocketshow.geomap import SceneMapper, query_points, stub_infer
from pocketshow.types import Track


def _frame():
    return np.full((80, 120, 3), 40, dtype=np.uint8)


def _track() -> Track:
    return Track(id=7, bbox_xyxy=(20.0, 10.0, 60.0, 70.0), conf=0.9, person_name="小李", person_id="p001")


def test_query_points_and_stub_infer():
    points = query_points([_track()], 120, 80)
    assert points[0]["id"] == "p001"
    assert 0.3 < points[0]["u"] < 0.4
    payload = stub_infer(points, 3)
    assert payload["ready"] is True
    assert payload["points"][0]["xyz"]


def test_stub_mapper_annotates_tracks(tmp_path):
    cfg = MapConfig(
        enabled=True,
        backend="stub",
        interval_s=0.0,
        status=str(tmp_path / "map.json"),
        log=str(tmp_path / "map.jsonl"),
    )
    mapper = SceneMapper(cfg, inline=True)
    tracks = [_track()]
    assert mapper.offer(_frame(), tracks, camera_id="office", camera_name="工位区1", now=1.0)
    assert mapper.offer(_frame(), tracks, camera_id="office", camera_name="工位区1", now=1.0)
    assert mapper.offer(_frame(), tracks, camera_id="office", camera_name="工位区1", now=1.0)
    mapper.annotate(tracks, "office")
    assert tracks[0].xyz is not None
    assert mapper.line_for("office").startswith("3D")
    status = json.loads((tmp_path / "map.json").read_text())
    assert status["cameras"][0]["id"] == "office"
    mapper.close()


def test_mapper_http(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_post(url, payload, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "ready": True,
            "frame_index": 4,
            "camera_xyz": [1.2, 1.5, 0.3],
            "points": [{"id": "p001", "track_id": 7, "xyz": [0.4, 1.4, 2.1], "depth": 2.1}],
        }

    monkeypatch.setattr("pocketshow.geomap.post_json", fake_post)
    cfg = MapConfig(
        enabled=True,
        backend="http",
        base_url="http://127.0.0.1:8090",
        interval_s=0.0,
        status=str(tmp_path / "map.json"),
        log=str(tmp_path / "map.jsonl"),
    )
    mapper = SceneMapper(cfg, inline=True)
    tracks = [_track()]
    assert mapper.offer(_frame(), tracks, camera_id="office", camera_name="工位区1", now=1.0)
    mapper.annotate(tracks, "office")
    assert captured["url"] == "http://127.0.0.1:8090/v1/map/frame"
    assert captured["payload"]["camera_id"] == "office"
    assert tracks[0].xyz == (0.4, 1.4, 2.1)
    mapper.close()


def test_map_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, WatchConfig
    from pocketshow.scene import SceneLog

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    status = tmp_path / "map.json"
    log = tmp_path / "map.jsonl"
    SceneLog(status, log).write_status(
        {
            "enabled": True,
            "backend": "http",
            "updated": 1.0,
            "cameras": [{"id": "office", "name": "工位区1", "line": "3D 12帧 相机(1.0,1.5,0.0) 1人", "ready": True}],
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
        geomap=MapConfig(enabled=True, status=str(status), log=str(log)),
    )
    client = TestClient(create_app(settings))
    data = client.get("/api/map").json()
    assert data["enabled"] is True
    assert data["cameras"][0]["line"].startswith("3D")
    assert client.get("/api/status").json()["geomap"]["cameras"][0]["id"] == "office"
