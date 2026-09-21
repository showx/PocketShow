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
    assert info["cameras"] is None


def test_preview_hub_writes_cameras(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path)
    frame = np.full((40, 80, 3), 30, dtype=np.uint8)
    assert hub.publish(frame, now=1.0, cameras=[{"id": "office", "name": "工位区1"}])
    info = hub.public()
    assert info["cameras"] == [{"id": "office", "name": "工位区1"}]
    assert hub.publish(frame, now=1.05, cameras=[{"id": "office2", "name": "工位区2"}]) is False
    assert hub.public()["cameras"] == [{"id": "office2", "name": "工位区2"}]


def test_preview_freshness_uses_json_mtime(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path, fresh_s=2.0)
    frame = np.full((20, 30, 3), 10, dtype=np.uint8)
    assert hub.publish(frame, now=1.0, cameras=[{"id": "office", "name": "工位区1"}])
    old = time.time() - 10
    path.touch()
    import os

    os.utime(path, (old, old))
    os.utime(hub.meta_path(), (time.time(), time.time()))
    info = hub.public()
    assert info["fresh"] is True
    assert info["cameras"][0]["id"] == "office"


def test_preview_hub_writes_boxes(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path)
    frame = np.full((40, 80, 3), 30, dtype=np.uint8)
    assert hub.publish(
        frame,
        now=1.0,
        cameras=[
            {
                "id": "office",
                "name": "工位区1",
                "src_w": 80,
                "src_h": 40,
                "boxes": [{"id": 9, "x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.9, "name": "小周"}],
            }
        ],
    )
    boxes = hub.public()["cameras"][0]["boxes"]
    assert boxes[0]["id"] == 9
    assert boxes[0]["name"] == "小周"
    assert boxes[0]["x1"] == 0.1


def test_preview_hub_writes_native_panes(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path)
    a = np.full((48, 64, 3), 10, dtype=np.uint8)
    b = np.full((72, 80, 3), 20, dtype=np.uint8)
    assert hub.publish(
        np.zeros((72, 144, 3), dtype=np.uint8),
        now=2.0,
        cameras=[{"id": "office", "name": "工位区1"}, {"id": "office2", "name": "工位区2"}],
        panes=[
            {"id": "office", "image": a},
            {"id": "office2", "image": b},
        ],
    )
    one = cv2.imdecode(np.frombuffer(hub.read_pane("office"), dtype=np.uint8), cv2.IMREAD_COLOR)
    two = cv2.imdecode(np.frombuffer(hub.read_pane("office2"), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert one is not None and one.shape[:2] == (48, 64)
    assert two is not None and two.shape[:2] == (72, 80)


def test_preview_hub_keeps_native_width(tmp_path):
    path = tmp_path / "preview.jpg"
    hub = PreviewHub(path)
    wide = np.zeros((900, 2000, 3), dtype=np.uint8)
    assert hub.publish(wide, now=10.0)
    raw = np.frombuffer(hub.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] == 2000


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


def test_pin_seat_from_preview_click(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, Settings, WatchConfig
    from pocketshow.gallery import FaceGallery

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    preview = tmp_path / "preview.jpg"
    gallery_path = tmp_path / "f.json"
    photos = tmp_path / "faces"
    gallery = FaceGallery(gallery_path, photos)
    gallery.enroll(np.ones(4, dtype=np.float32), name="小李")
    other = gallery.enroll(np.ones(4, dtype=np.float32) * 0.2, name="小王")
    gallery.find(other["id"])["seats"] = {
        "office2": {"cx": 0.5, "cy": 0.5, "rx": 0.06, "ry": 0.1, "hits": 20, "camera_name": "工位区2"}
    }
    gallery.save()
    settings = Settings(
        preview=str(preview),
        recognize=RecognizeConfig(gallery=str(gallery_path), photos=str(photos)),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
        capture=CaptureConfig(source="auto"),
        rtsp=RtspConfig(settings=str(tmp_path / "capture.json")),
    )
    PreviewHub(preview).publish(
        np.full((60, 80, 3), 12, dtype=np.uint8),
        now=time.monotonic(),
        cameras=[{"id": "office", "name": "工位区1"}, {"id": "office2", "name": "工位区2"}],
    )
    client = TestClient(create_app(settings))
    pinned = client.post("/api/people/p001/seats", json={"nx": 0.75, "ny": 0.5})
    assert pinned.status_code == 200
    seats = pinned.json()["seats"]
    assert len(seats) == 1
    assert seats[0]["camera_id"] == "office2"
    assert seats[0]["locked"] is True
    assert client.get("/api/people/p002").json()["seats"] == []
    guest = client.patch("/api/people/p001", json={"guest": True})
    assert guest.status_code == 200
    denied = client.post("/api/people/p001/seats", json={"camera_id": "office", "cx": 0.2, "cy": 0.8})
    assert denied.status_code == 400


def test_mark_seat_creates_person_from_click(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, Settings, WatchConfig
    from pocketshow.gallery import FaceGallery

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    preview = tmp_path / "preview.jpg"
    gallery_path = tmp_path / "f.json"
    photos = tmp_path / "faces"
    FaceGallery(gallery_path, photos).save()
    settings = Settings(
        preview=str(preview),
        recognize=RecognizeConfig(gallery=str(gallery_path), photos=str(photos)),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
        capture=CaptureConfig(source="auto"),
        rtsp=RtspConfig(settings=str(tmp_path / "capture.json")),
    )
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    frame[80:160, 120:200] = 200
    PreviewHub(preview).publish(
        frame,
        now=time.monotonic(),
        cameras=[{"id": "office", "name": "工位区1", "src_w": 320, "src_h": 240}],
        panes=[{"id": "office", "image": frame}],
    )
    client = TestClient(create_app(settings))
    blank = client.post("/api/seats", json={"camera_id": "office", "cx": 0.5, "cy": 0.5})
    assert blank.status_code == 400
    created = client.post(
        "/api/seats",
        json={"name": "小周", "camera_id": "office", "camera_name": "工位区1", "cx": 0.5, "cy": 0.5},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["name"] == "小周"
    assert body["id"] == "p001"
    assert body["seats"][0]["camera_id"] == "office"
    assert body["seats"][0]["locked"] is True
    assert body["photo_url"]
    listed = client.get("/api/people").json()
    assert [p["name"] for p in listed] == ["小周"]
    mosaic = client.post("/api/seats", json={"name": "小吴", "nx": 0.5, "ny": 0.5})
    assert mosaic.status_code == 200
    assert mosaic.json()["seats"][0]["camera_id"] == "office"


def test_mark_seat_uses_person_box_on_camera(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.gallery import FaceGallery

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    preview = tmp_path / "preview.jpg"
    gallery_path = tmp_path / "f.json"
    photos = tmp_path / "faces"
    FaceGallery(gallery_path, photos).save()
    settings = _admin_settings(tmp_path, preview)
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    frame[80:160, 120:200] = 220
    box = {"id": 3, "x1": 0.375, "y1": 0.333, "x2": 0.625, "y2": 0.667}
    PreviewHub(preview).publish(
        frame,
        now=time.monotonic(),
        cameras=[{"id": "office", "name": "工位区1", "src_w": 320, "src_h": 240, "boxes": [box]}],
        panes=[{"id": "office", "image": frame}],
    )
    client = TestClient(create_app(settings))
    info = client.get("/api/status").json()["preview"]
    assert info["cameras"][0]["boxes"][0]["id"] == 3
    unnamed = client.post("/api/seats", json={"camera_id": "office", "cx": 0.5, "cy": 0.5})
    assert unnamed.status_code == 200
    body = unnamed.json()
    assert body["name"].startswith("人物")
    seat = body["seats"][0]
    assert seat["camera_id"] == "office"
    assert seat["camera_name"] == "工位区1"
    assert seat["rx"] > 0.1
    assert seat["ry"] > 0.1
    cover = cv2.imdecode(np.frombuffer(client.get(body["photo_url"]).content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert cover is not None
    assert int(cover.mean()) > 150
    away = client.post(
        "/api/seats",
        json={"name": "门口", "camera_id": "office", "cx": 0.05, "cy": 0.05},
    )
    assert away.status_code == 200
    assert away.json()["seats"][0]["rx"] <= 0.06
    boxed = client.post(
        "/api/seats",
        json={
            "name": "小周",
            "camera_id": "office",
            "camera_name": "工位区1",
            "cx": 0.1,
            "cy": 0.1,
            "x1": 0.2,
            "y1": 0.3,
            "x2": 0.5,
            "y2": 0.8,
        },
    )
    assert boxed.status_code == 200
    marked = boxed.json()["seats"][0]
    assert 0.34 < marked["cx"] < 0.36
    assert 0.54 < marked["cy"] < 0.56
    assert marked["rx"] > 0.1
    assert marked["ry"] > 0.2


def _admin_settings(tmp_path, preview):
    from pocketshow.config import CaptureConfig, GimbalConfig, RecognizeConfig, RtspConfig, Settings, WatchConfig

    return Settings(
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


def test_preview_mjpeg_stops(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    preview = tmp_path / "preview.jpg"
    app = create_app(_admin_settings(tmp_path, preview))
    PreviewHub(preview).publish(np.full((24, 32, 3), 9, dtype=np.uint8), now=time.monotonic())
    app.state.stopping.set()
    client = TestClient(app)
    resp = client.get("/api/preview")
    assert resp.status_code == 200
    assert "multipart" in resp.headers["content-type"]


def test_run_server_exits_on_second_sigint(tmp_path, monkeypatch):
    from pocketshow.admin import _run_server, create_app

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    app = create_app(_admin_settings(tmp_path, tmp_path / "preview.jpg"))
    captured: dict = {}

    class FakeConfig:
        def __init__(
            self,
            app=None,
            host="",
            port=0,
            log_level="info",
            timeout_keep_alive=5,
            timeout_graceful_shutdown=None,
            **kwargs,
        ):
            captured["kwargs"] = {
                "timeout_keep_alive": timeout_keep_alive,
                "timeout_graceful_shutdown": timeout_graceful_shutdown,
                **kwargs,
            }

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False
            self.handle_exit = self._inner

        def _inner(self, sig, frame):
            self.should_exit = True

        def run(self):
            self.handle_exit(2, None)
            assert app.state.stopping.is_set()
            self.handle_exit(2, None)

    def boom(code):
        raise SystemExit(code)

    monkeypatch.setattr("pocketshow.admin.uvicorn.Config", FakeConfig)
    monkeypatch.setattr("pocketshow.admin.uvicorn.Server", FakeServer)
    monkeypatch.setattr("pocketshow.admin.os._exit", boom)
    try:
        _run_server(app, "127.0.0.1", 8765)
        raise AssertionError("second SIGINT should exit")
    except SystemExit as exc:
        assert exc.code == 0
    assert captured["kwargs"]["timeout_keep_alive"] == 1
    assert captured["kwargs"]["timeout_graceful_shutdown"] == 1
