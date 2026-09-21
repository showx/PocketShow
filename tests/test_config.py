from pathlib import Path

from pocketshow.config import ensure_local_config, load_settings
from pocketshow.gimbal.stub import StubGimbal
from pocketshow.types import FollowCommand, FrameError


def test_load_default_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "default.example.yaml"
    settings = load_settings(path)
    assert settings.detect.model == "yolo11n.pt"
    assert settings.detect.imgsz == 1280
    assert settings.detect.conf == 0.2
    assert settings.detect.far_pass
    assert settings.detect.far_ratio == 1.0
    assert settings.detect.far_rows == 2
    assert settings.recognize.det_min_face == 12
    assert settings.recognize.liveness_min_face == 40
    assert settings.recognize.match_threshold == 0.50
    assert settings.recognize.match_margin == 0.06
    assert settings.recognize.match_min_face == 18
    assert settings.recognize.id_min_face == 28
    assert settings.recognize.id_confirm == 2
    assert settings.recognize.seats is True
    assert settings.recognize.seat_min_hits == 3
    assert settings.recognize.seat_confirm == 2
    assert settings.recognize.reid is True
    assert settings.recognize.reid_threshold == 0.48
    assert settings.gimbal.backend == "stub"
    assert settings.recognize.enabled
    assert settings.recognize.liveness
    assert settings.follow.deadzone > 0
    assert settings.watch.enabled
    assert settings.watch.away_s == 30
    assert settings.watch.work_start == "09:00"
    assert settings.watch.work_end == "18:30"
    assert settings.watch.workdays == [1, 2, 3, 4, 5]
    assert settings.preview == "data/preview.jpg"
    assert settings.scene.enabled is False
    assert settings.scene.backend == "moss-vl"
    assert settings.scene.model == "OpenMOSS-Team/MOSS-VL-Realtime"
    assert settings.scene.interval_s == 8.0
    assert settings.scene.sample_fps == 1.0
    assert settings.scene.ws_url == "ws://127.0.0.1:8000/v1/realtime"
    assert settings.geomap.enabled is False
    assert settings.geomap.backend == "http"
    assert settings.geomap.base_url == "http://127.0.0.1:8090"
    assert settings.mocap.enabled is False
    assert settings.mocap.backend == "http"
    assert settings.mocap.base_url == "http://127.0.0.1:8006"


def test_ensure_local_config_copies_example(tmp_path):
    example = tmp_path / "default.example.yaml"
    example.write_text((Path(__file__).resolve().parents[1] / "configs" / "default.example.yaml").read_text())
    target = tmp_path / "default.yaml"
    assert not target.exists()
    created = ensure_local_config(target)
    assert created == target
    assert target.exists()
    settings = load_settings(target)
    assert settings.rtsp.host == ""
    assert ensure_local_config(target) == target


def test_stub_gimbal_clamps_and_recenter():
    gimbal = StubGimbal()
    gimbal.set_velocity(2.0, -3.0)
    assert gimbal.yaw == 1.0
    assert gimbal.pitch == -1.0
    gimbal.apply(
        FollowCommand(
            yaw_rate=0.2,
            pitch_rate=-0.1,
            lost=False,
            error=FrameError(0.1, 0.0, 1.0),
            target_id=1,
        )
    )
    assert gimbal.yaw == 0.2
    gimbal.recenter()
    assert gimbal.yaw == 0.0
    assert gimbal.pitch == 0.0
    gimbal.close()
