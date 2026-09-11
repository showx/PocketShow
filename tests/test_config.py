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
    assert settings.recognize.det_min_face == 12
    assert settings.recognize.liveness_min_face == 40
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
