import time

from pocketshow.config import FollowConfig
from pocketshow.control import GimbalBus, mix_command
from pocketshow.follow import FollowController
from pocketshow.types import FollowCommand, FrameError


def _follow() -> FollowController:
    return FollowController(FollowConfig())


def _cmd(yaw=0.1) -> FollowCommand:
    return FollowCommand(yaw, 0.0, False, FrameError(0.1, 0.0, 1.0), 1)


def test_bus_manual_overrides_follow(tmp_path):
    bus = GimbalBus(tmp_path / "gimbal.json")
    bus.command(yaw=-0.4, pitch=0.0)
    follow = _follow()
    auto = _cmd(0.3)
    mixed, label, recenter = mix_command(auto, bus.read(), follow, now=time.time())
    assert label == "手动"
    assert mixed.yaw_rate == -0.4
    assert not recenter


def test_bus_stale_hold_stops(tmp_path):
    bus = GimbalBus(tmp_path / "gimbal.json")
    state = bus.command(yaw=0.5)
    state.updated = time.time() - 2.0
    bus.write(state)
    mixed, label, _ = mix_command(_cmd(), bus.read(), _follow(), now=time.time())
    assert label == "手动停"
    assert mixed.yaw_rate == 0.0


def test_bus_follow_keeps_auto(tmp_path):
    bus = GimbalBus(tmp_path / "gimbal.json")
    bus.command(mode="follow")
    auto = _cmd(0.22)
    mixed, label, _ = mix_command(auto, bus.read(), _follow(), now=time.time())
    assert label == "跟拍"
    assert mixed.yaw_rate == 0.22


def test_admin_gimbal_api(tmp_path):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig, Settings

    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
    )
    client = TestClient(create_app(settings))
    moved = client.post("/api/gimbal", json={"yaw": -0.5, "pitch": 0})
    assert moved.status_code == 200
    assert moved.json()["mode"] == "manual"
    assert moved.json()["yaw"] == -0.5
    back = client.post("/api/gimbal", json={"mode": "follow"})
    assert back.json()["mode"] == "follow"


def test_concurrent_writes_do_not_crash(tmp_path):
    path = tmp_path / "gimbal.json"
    a = GimbalBus(path)
    b = GimbalBus(path)
    errors: list[BaseException] = []

    def hammer(bus: GimbalBus, yaw: float) -> None:
        try:
            for _ in range(40):
                bus.command(yaw=yaw)
                bus.ack("stub", yaw, 0.0)
        except BaseException as exc:
            errors.append(exc)

    t1 = __import__("threading").Thread(target=hammer, args=(a, -0.4))
    t2 = __import__("threading").Thread(target=hammer, args=(b, 0.4))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []
    assert path.exists()
    assert a.read().mode in ("follow", "manual")


def test_status_offline_without_pocket(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig, Settings, WatchConfig

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
    )
    client = TestClient(create_app(settings))
    data = client.get("/api/status").json()
    assert data["online"] is False
    assert data["follow"] is False
    assert data["usb"] is False
    assert "未检测" in data["detail"]
    assert data["watch"]["state"] == "waiting"
    assert data["watch"]["alarm"] is False


def test_status_online_from_usb(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig, Settings

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: True)
    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
    )
    client = TestClient(create_app(settings))
    data = client.get("/api/status").json()
    assert data["online"] is True
    assert data["usb"] is True
    assert data["follow"] is False
    assert "USB" in data["detail"]


def test_status_online_from_follow_heartbeat(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig, Settings

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    command = tmp_path / "gimbal.json"
    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(command)),
    )
    bus = GimbalBus(command)
    bus.ack("stub", 0.0, 0.0, capture="usb", device="OsmoPocket3")
    client = TestClient(create_app(settings))
    data = client.get("/api/status").json()
    assert data["online"] is True
    assert data["follow"] is True
    assert data["device"] == "OsmoPocket3"
    assert "跟拍运行中" in data["detail"]


def test_watch_hours_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pocketshow.admin import create_app
    from pocketshow.config import GimbalConfig, RecognizeConfig, Settings, WatchConfig

    monkeypatch.setattr("pocketshow.admin.pocket3_usb_present", lambda ttl=5.0: False)
    settings = Settings(
        recognize=RecognizeConfig(gallery=str(tmp_path / "f.json"), photos=str(tmp_path / "faces")),
        gimbal=GimbalConfig(command=str(tmp_path / "gimbal.json")),
        watch=WatchConfig(
            status=str(tmp_path / "station.json"),
            settings=str(tmp_path / "watch.json"),
            log=str(tmp_path / "away.jsonl"),
        ),
    )
    client = TestClient(create_app(settings))
    data = client.get("/api/watch").json()
    assert data["settings"]["work_start"] == "09:00"
    assert data["settings"]["work_end"] == "18:30"
    assert "today" in data
    saved = client.put(
        "/api/watch",
        json={"work_start": "09:00", "work_end": "18:30", "workdays": [1, 2, 3, 4, 5], "away_s": 45},
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["settings"]["away_s"] == 45
    assert body["settings"]["hours_label"].startswith("工作日")
    bad = client.put("/api/watch", json={"work_start": "25:00"})
    assert bad.status_code == 400
