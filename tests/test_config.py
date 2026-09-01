from pathlib import Path

from pocketshow.config import load_settings
from pocketshow.gimbal.stub import StubGimbal
from pocketshow.types import FollowCommand, FrameError


def test_load_default_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    settings = load_settings(path)
    assert settings.detect.model == "yolo11n.pt"
    assert settings.gimbal.backend == "stub"
    assert settings.recognize.enabled
    assert settings.follow.deadzone > 0


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
