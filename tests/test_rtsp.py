import numpy as np

from pocketshow.capture import (
    CaptureStore,
    build_rtsp_url,
    camera_as_rtsp,
    compose_grid,
    grid_layout,
    hik_channel,
    pane_index,
    redact_rtsp_url,
    stream_label,
)
from pocketshow.config import CaptureConfig, RtspCamera, RtspConfig, Settings, load_settings


def test_hik_channel_and_label():
    assert hik_channel(1, "main") == 101
    assert hik_channel(1, "sub") == 102
    assert hik_channel(2, "sub") == 202
    assert "102" in stream_label("sub", 1)
    assert "第2路" in stream_label("sub", 2)


def test_build_and_redact_url():
    cfg = RtspConfig(host="192.168.0.177", username="admin", password="secret", stream="sub")
    url = build_rtsp_url(cfg)
    assert url == "rtsp://admin:secret@192.168.0.177:554/Streaming/Channels/102"
    hidden = redact_rtsp_url(url)
    assert "secret" not in hidden
    assert "192.168.0.177" in hidden
    assert hidden.endswith("/Streaming/Channels/102")


def test_password_special_chars_and_env(monkeypatch):
    monkeypatch.delenv("POCKETSHOW_RTSP_PASSWORD", raising=False)
    cfg = RtspConfig(host="10.0.0.8", username="admin", password="a@b:c", stream="main", channel=3)
    url = build_rtsp_url(cfg)
    assert "Streaming/Channels/301" in url
    assert "a%40b%3Ac" in url
    monkeypatch.setenv("POCKETSHOW_RTSP_PASSWORD", "from-env")
    env_url = build_rtsp_url(RtspConfig(host="10.0.0.8", password="yaml"))
    assert "from-env" in env_url


def test_url_override():
    raw = "rtsp://admin:x@192.168.0.177:554/h264/ch1/sub/av_stream"
    assert build_rtsp_url(RtspConfig(url=raw, host="ignored")) == raw


def test_capture_store_persists_stream(tmp_path, monkeypatch):
    monkeypatch.delenv("POCKETSHOW_RTSP_PASSWORD", raising=False)
    path = tmp_path / "capture.json"
    store = CaptureStore(
        CaptureConfig(source="auto"),
        RtspConfig(host="192.168.0.177", username="admin", password="pw", settings=str(path)),
    )
    public = store.save(source="rtsp", stream="main", channel=1)
    assert public["stream"] == "main"
    assert public["stream_id"] == 101
    assert public["has_password"] is True
    assert "pw" not in public["url"]

    other = CaptureStore(CaptureConfig(source="usb"), RtspConfig(settings=str(path)))
    assert other.capture.source == "rtsp"
    assert other.rtsp.stream == "main"
    assert other.rtsp.password == "pw"


def test_compose_grid_and_pane():
    a = np.full((100, 160, 3), 1, dtype=np.uint8)
    b = np.full((80, 90, 3), 2, dtype=np.uint8)
    grid = compose_grid([a, b], (400, 240))
    assert grid.shape == (240, 400, 3)
    assert pane_index(50, 40, 2, (400, 240)) == 0
    assert pane_index(250, 40, 2, (400, 240)) == 1
    assert grid_layout(8) == (2, 4)
    eight = compose_grid([np.zeros((10, 10, 3), dtype=np.uint8)] * 8, (1600, 800))
    assert eight.shape == (800, 1600, 3)


def test_camera_as_rtsp_uses_channel():
    shared = RtspConfig(password="pw", host="192.168.0.177")
    cam = RtspCamera(id="office2", name="工位区2", host="192.168.0.177", channel=2, stream="sub")
    url = build_rtsp_url(camera_as_rtsp(shared, cam))
    assert url.endswith("/Streaming/Channels/202")


def test_monitor_ids_filter_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("POCKETSHOW_RTSP_PASSWORD", raising=False)
    path = tmp_path / "capture.json"
    store = CaptureStore(
        CaptureConfig(source="rtsp"),
        RtspConfig(
            password="pw",
            settings=str(path),
            cameras=[
                RtspCamera(id="office", name="工位区1", host="192.168.0.177", stream="sub", monitor=True),
                RtspCamera(id="gate", name="门口", host="192.168.0.178", channel=2, stream="sub", monitor=True),
            ],
        ),
    )
    before = store.identity()
    public = store.save(monitor_ids=["office"])
    assert public["monitor_ids"] == ["office"]
    assert public["cameras"][0]["monitor"] is True
    assert public["cameras"][1]["monitor"] is False
    assert store.identity() != before
    assert [cam.id for cam in store.rtsp.cameras if cam.monitor] == ["office"]
    try:
        store.save(monitor_ids=[])
        raise AssertionError("empty monitor_ids should fail")
    except ValueError:
        pass


def test_switch_between_cameras(tmp_path, monkeypatch):
    monkeypatch.delenv("POCKETSHOW_RTSP_PASSWORD", raising=False)
    path = tmp_path / "capture.json"
    store = CaptureStore(
        CaptureConfig(source="rtsp"),
        RtspConfig(
            password="pw",
            settings=str(path),
            cameras=[
                RtspCamera(id="office", name="工位区", host="192.168.0.177", stream="sub"),
                RtspCamera(id="gate", name="门口", host="192.168.0.178", channel=2, stream="sub"),
            ],
            camera_id="office",
        ),
    )
    assert "Channels/102" in build_rtsp_url(store.rtsp)
    public = store.save(camera_id="gate")
    assert public["camera_id"] == "gate"
    assert public["host"] == "192.168.0.178"
    assert public["stream_id"] == 202
    assert "Channels/202" in build_rtsp_url(store.rtsp)
    other = CaptureStore(CaptureConfig(), RtspConfig(settings=str(path)))
    assert other.rtsp.camera_id == "gate"
    assert len(other.rtsp.cameras) == 2


def test_example_yaml_has_rtsp():
    from pathlib import Path

    settings = load_settings(Path(__file__).resolve().parents[1] / "configs" / "default.example.yaml")
    assert settings.rtsp.host == ""
    assert settings.rtsp.stream == "sub"
    assert settings.rtsp.password == ""


def test_capture_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig, WatchConfig

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
        capture=CaptureConfig(source="auto"),
        rtsp=RtspConfig(host="192.168.0.177", settings=str(tmp_path / "capture.json")),
        preview=str(tmp_path / "preview.jpg"),
    )
    client = TestClient(create_app(settings))
    data = client.get("/api/capture").json()
    assert data["host"] == "192.168.0.177"
    assert data["stream"] == "sub"
    saved = client.put("/api/capture", json={"source": "rtsp", "stream": "main", "password": "unit-test"})
    assert saved.status_code == 200
    body = saved.json()
    assert body["source"] == "rtsp"
    assert body["stream_id"] == 101
    assert body["has_password"] is True
    assert "unit-test" not in str(body)
    added = client.put(
        "/api/capture",
        json={"source": "rtsp", "camera": {"name": "门口", "host": "192.168.0.178", "channel": 2, "stream": "sub"}},
    )
    assert added.status_code == 200
    assert added.json()["camera_id"]
    assert len(added.json()["cameras"]) >= 2
    switched = client.put("/api/capture", json={"camera_id": added.json()["camera_id"]})
    assert switched.json()["host"] == "192.168.0.178"
    bad = client.put("/api/capture", json={"source": "rtsp", "stream": "foo"})
    assert bad.status_code == 400


def test_status_online_from_rtsp(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig
    from pocketshow.control import GimbalBus

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    command = tmp_path / "gimbal.json"
    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(command)),
        rtsp=RtspConfig(settings=str(tmp_path / "capture.json")),
    )
    GimbalBus(command).ack("stub", 0.0, 0.0, capture="rtsp", device="RTSP 192.168.0.177 子码流 102")
    client = TestClient(create_app(settings))
    data = client.get("/api/status").json()
    assert data["online"] is True
    assert data["kind"] == "rtsp"
    assert data["follow"] is True
    assert "局域网" in data["detail"] or "RTSP" in data["detail"]
