import time

import cv2
import numpy as np

from pocketshow.preview import PreviewHub, placeholder_jpeg


def test_preview_hub_writes_jpeg(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path)
    assert hub.read() is None
    assert hub.public()["fresh"] is False
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    assert hub.publish(frame, now=1.0) is True
    assert hub.publish(frame, now=1.05) is False
    data = hub.read()
    assert data is not None
    assert data[:2] == b"\xff\xd8"
    info = hub.public()
    assert info["fresh"] is True
    assert info["age_s"] is not None


def test_preview_hub_downscales(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path)
    wide = np.zeros((900, 2000, 3), dtype=np.uint8)
    assert hub.publish(wide, now=10.0)
    raw = np.frombuffer(hub.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] == 1600


def test_placeholder_jpeg():
    data = placeholder_jpeg()
    assert data[:2] == b"\xff\xd8"


def test_preview_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, Settings, WatchConfig

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    preview = tmp_path / "preview.jpg"
    settings = Settings(
        preview=str(preview),
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
        capture=CaptureConfig(source="auto"),
        rtsp=RtspConfig(settings=str(tmp_path / "capture.json")),
    )
    client = TestClient(create_app(settings))
    missing = client.get("/api/preview.jpg")
    assert missing.status_code == 404
    stale = client.get("/api/status").json()["preview"]
    assert stale["fresh"] is False
    PreviewHub(preview).publish(np.full((60, 80, 3), 12, dtype=np.uint8), now=time.monotonic())
    shot = client.get("/api/preview.jpg")
    assert shot.status_code == 200
    assert shot.headers["content-type"].startswith("image/jpeg")
    assert shot.content[:2] == b"\xff\xd8"
    assert client.get("/api/status").json()["preview"]["fresh"] is True
