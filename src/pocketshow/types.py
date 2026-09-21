from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Track:
    id: int
    bbox_xyxy: tuple[float, float, float, float]
    conf: float
    vx: float = 0.0
    vy: float = 0.0
    person_id: str | None = None
    person_name: str | None = None
    face_bbox: tuple[float, float, float, float] | None = None
    face_score: float = 0.0
    live: bool | None = None
    live_score: float = 0.0
    by_seat: bool = False
    appearance: object | None = None
    reid_score: float = 0.0
    xyz: tuple[float, float, float] | None = None
    depth: float = 0.0
    keypoints: list[dict] | None = None
    activity: str = ""

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @property
    def label(self) -> str:
        if self.person_name:
            return self.person_name
        return f"ID {self.id}"


@dataclass(slots=True)
class FrameError:
    ex: float
    ey: float
    size_ratio: float


@dataclass(slots=True)
class FollowCommand:
    yaw_rate: float
    pitch_rate: float
    lost: bool
    error: FrameError
    target_id: int | None
