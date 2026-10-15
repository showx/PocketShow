"""画面误差 → 云台速度。计算在 Rust 控制核心里。"""

from __future__ import annotations

from pocketshow.config import FollowConfig
from pocketshow.native import core
from pocketshow.types import FollowCommand, FrameError, Track

_core = core()


def compute_frame_error(
    track: Track,
    frame_w: int,
    frame_h: int,
    target_area: float,
) -> FrameError:
    x1, y1, x2, y2 = track.bbox_xyxy
    ex, ey, size_ratio = _core.compute_frame_error(
        float(x1),
        float(y1),
        float(x2),
        float(y2),
        int(frame_w),
        int(frame_h),
        float(target_area),
    )
    return FrameError(ex=ex, ey=ey, size_ratio=size_ratio)


class FollowController:
    """把人物画面误差转成云台速度：死区 + PID + 速度前馈 + 丢失衰减。"""

    def __init__(self, cfg: FollowConfig) -> None:
        self.cfg = cfg
        self._core = _core.FollowController(
            cfg.deadzone,
            cfg.kp,
            cfg.ki,
            cfg.kd,
            cfg.feedforward,
            cfg.max_rate,
            cfg.lost_hold_s,
            cfg.lost_timeout_s,
            cfg.target_area,
        )

    def reset(self) -> None:
        self._core.reset()

    def update(
        self,
        target: Track | None,
        frame_w: int,
        frame_h: int,
        dt: float,
        now: float | None = None,
    ) -> FollowCommand:
        packed = None
        if target is not None:
            x1, y1, x2, y2 = target.bbox_xyxy
            packed = (
                int(target.id),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(target.vx),
                float(target.vy),
            )
        yaw, pitch, lost, ex, ey, size_ratio, target_id = self._core.update(
            packed,
            int(frame_w),
            int(frame_h),
            float(dt),
            now,
        )
        return FollowCommand(
            yaw_rate=yaw,
            pitch_rate=pitch,
            lost=lost,
            error=FrameError(ex, ey, size_ratio),
            target_id=None if target_id < 0 else int(target_id),
        )
