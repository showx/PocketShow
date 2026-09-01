from __future__ import annotations

import logging
import time

from pocketshow.config import DetectConfig
from pocketshow.types import Track

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class PersonTracker:
    """YOLO 只检 person，ByteTrack 赋稳定 ID。"""

    def __init__(self, cfg: DetectConfig) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        logger.info("加载 YOLO %s device=%s", cfg.model, self.device)
        self.model = YOLO(cfg.model)
        self._prev: dict[int, tuple[float, float, float]] = {}

    def track(self, frame) -> list[Track]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            classes=[PERSON_CLASS],
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )
        now = time.monotonic()
        tracks: list[Track] = []
        if not results:
            self._prev = {}
            return tracks

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracks

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        seen: dict[int, tuple[float, float, float]] = {}

        for bbox, tid, conf in zip(xyxy, ids, confs):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            vx = vy = 0.0
            prev = self._prev.get(int(tid))
            if prev is not None:
                pcx, pcy, pt = prev
                dt = max(1e-3, now - pt)
                vx = (cx - pcx) / dt
                vy = (cy - pcy) / dt
            tracks.append(
                Track(
                    id=int(tid),
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=float(conf),
                    vx=vx,
                    vy=vy,
                )
            )
            seen[int(tid)] = (cx, cy, now)

        self._prev = seen
        return tracks
