from pocketshow.config import FollowConfig
from pocketshow.follow import FollowController, compute_frame_error
from pocketshow.types import Track


def _track(x1, y1, x2, y2, tid=1, vx=0.0, vy=0.0) -> Track:
    return Track(id=tid, bbox_xyxy=(x1, y1, x2, y2), conf=0.9, vx=vx, vy=vy)


def test_right_side_positive_yaw():
    cfg = FollowConfig(deadzone=0.02, ki=0.0, kd=0.0, feedforward=0.0, max_rate=1.0)
    ctrl = FollowController(cfg)
    # 人物在画面右侧
    track = _track(700, 200, 900, 500)
    cmd = ctrl.update(track, 1000, 1000, dt=0.05)
    assert cmd.yaw_rate > 0
    assert not cmd.lost


def test_below_center_negative_pitch():
    cfg = FollowConfig(deadzone=0.02, ki=0.0, kd=0.0, feedforward=0.0, max_rate=1.0)
    ctrl = FollowController(cfg)
    track = _track(400, 600, 600, 900)
    cmd = ctrl.update(track, 1000, 1000, dt=0.05)
    assert cmd.pitch_rate < 0


def test_deadzone_holds_still():
    cfg = FollowConfig(deadzone=0.1, ki=0.0, kd=0.0, feedforward=0.0)
    ctrl = FollowController(cfg)
    track = _track(480, 480, 520, 520)
    cmd = ctrl.update(track, 1000, 1000, dt=0.05)
    assert cmd.yaw_rate == 0.0
    assert cmd.pitch_rate == 0.0


def test_max_rate_clamp():
    cfg = FollowConfig(deadzone=0.0, kp=20.0, ki=0.0, kd=0.0, feedforward=0.0, max_rate=0.4)
    ctrl = FollowController(cfg)
    track = _track(900, 100, 990, 400)
    cmd = ctrl.update(track, 1000, 1000, dt=0.05)
    assert abs(cmd.yaw_rate) <= 0.4 + 1e-9


def test_lost_decays_then_stops():
    cfg = FollowConfig(lost_hold_s=0.2, lost_timeout_s=0.5, deadzone=0.02, feedforward=0.0)
    ctrl = FollowController(cfg)
    track = _track(800, 200, 950, 500)
    first = ctrl.update(track, 1000, 1000, dt=0.05, now=1.0)
    assert first.yaw_rate != 0
    holding = ctrl.update(None, 1000, 1000, dt=0.05, now=1.1)
    assert holding.lost
    stopped = ctrl.update(None, 1000, 1000, dt=0.05, now=2.0)
    assert stopped.lost
    assert stopped.yaw_rate == 0.0
    assert stopped.pitch_rate == 0.0


def test_frame_error_center():
    track = _track(250, 250, 750, 750)
    err = compute_frame_error(track, 1000, 1000, target_area=0.25)
    assert abs(err.ex) < 1e-6
    assert abs(err.ey) < 1e-6
    assert abs(err.size_ratio - 1.0) < 1e-6
