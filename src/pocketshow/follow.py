from __future__ import annotations

import time

from pocketshow.config import FollowConfig
from pocketshow.types import FollowCommand, FrameError, Track


def compute_frame_error(
    track: Track,
    frame_w: int,
    frame_h: int,
    target_area: float,
) -> FrameError:
    cx, cy = track.center
    area = track.area / max(1.0, float(frame_w * frame_h))
    return FrameError(
        ex=cx / frame_w - 0.5,
        ey=cy / frame_h - 0.5,
        size_ratio=area / max(1e-6, target_area),
    )


class _AxisPid:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._i = 0.0
        self._prev: float | None = None

    def reset(self) -> None:
        self._i = 0.0
        self._prev = None

    def update(self, error: float, dt: float, deadzone: bool) -> float:
        if deadzone:
            self._i *= 0.5
            self._prev = 0.0
            return 0.0
        self._i = max(-1.0, min(1.0, self._i + error * dt))
        deriv = 0.0 if self._prev is None or dt <= 1e-6 else (error - self._prev) / dt
        self._prev = error
        return self.kp * error + self.ki * self._i + self.kd * deriv


class FollowController:
    """把人物画面误差转成云台速度：死区 + PID + 速度前馈 + 丢失衰减。"""

    def __init__(self, cfg: FollowConfig) -> None:
        self.cfg = cfg
        self._yaw = _AxisPid(cfg.kp, cfg.ki, cfg.kd)
        self._pitch = _AxisPid(cfg.kp, cfg.ki, cfg.kd)
        self._last_cmd = (0.0, 0.0)
        self._lost_since: float | None = None

    def reset(self) -> None:
        self._yaw.reset()
        self._pitch.reset()
        self._last_cmd = (0.0, 0.0)
        self._lost_since = None

    def update(
        self,
        target: Track | None,
        frame_w: int,
        frame_h: int,
        dt: float,
        now: float | None = None,
    ) -> FollowCommand:
        now = time.monotonic() if now is None else now
        dt = max(1e-3, dt)

        if target is None:
            return self._on_lost(now, dt)

        self._lost_since = None
        err = compute_frame_error(target, frame_w, frame_h, self.cfg.target_area)
        yaw = self._axis_output(
            self._yaw,
            err.ex,
            target.vx / max(1.0, frame_w) / dt,
            dt,
        )
        # 人物在画面下方 (ey>0) 时下俯，pitch 取反
        pitch = -self._axis_output(
            self._pitch,
            err.ey,
            target.vy / max(1.0, frame_h) / dt,
            dt,
        )
        yaw = self._clamp(yaw)
        pitch = self._clamp(pitch)
        self._last_cmd = (yaw, pitch)
        return FollowCommand(
            yaw_rate=yaw,
            pitch_rate=pitch,
            lost=False,
            error=err,
            target_id=target.id,
        )

    def _axis_output(self, pid: _AxisPid, error: float, vel_norm: float, dt: float) -> float:
        in_deadzone = abs(error) < self.cfg.deadzone
        shaped = 0.0 if in_deadzone else error
        pid_out = pid.update(shaped, dt, in_deadzone)
        if in_deadzone:
            return 0.0
        return pid_out + self.cfg.feedforward * vel_norm

    def _on_lost(self, now: float, dt: float) -> FollowCommand:
        if self._lost_since is None:
            self._lost_since = now
        elapsed = now - self._lost_since
        err = FrameError(ex=0.0, ey=0.0, size_ratio=0.0)
        if elapsed >= self.cfg.lost_timeout_s:
            self._yaw.reset()
            self._pitch.reset()
            self._last_cmd = (0.0, 0.0)
            return FollowCommand(0.0, 0.0, True, err, None)
        # 短暂遮挡：速度衰减，不乱扫
        decay = 0.85 ** max(1.0, elapsed / max(dt, 1e-3) * 0.25)
        if elapsed > self.cfg.lost_hold_s:
            decay *= 0.4
        yaw, pitch = self._last_cmd[0] * decay, self._last_cmd[1] * decay
        self._last_cmd = (yaw, pitch)
        return FollowCommand(yaw, pitch, True, err, None)

    def _clamp(self, value: float) -> float:
        limit = self.cfg.max_rate
        return max(-limit, min(limit, value))
