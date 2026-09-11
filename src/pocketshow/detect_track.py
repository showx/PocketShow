from __future__ import annotations

import logging
import time
from pathlib import Path

from pocketshow.config import DetectConfig
from pocketshow.types import Track

logger = logging.getLogger(__name__)

PERSON_CLASS = 0
EXTRA_ID_BASE = 800000
_BUNDLE_TRACKER = Path(__file__).with_name("bytetrack.yaml")


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


def resolve_tracker(path: str) -> str:
    raw = Path(path)
    if raw.is_file():
        return str(raw)
    if _BUNDLE_TRACKER.is_file() and raw.name in {"bytetrack.yaml", "bytetrack"}:
        return str(_BUNDLE_TRACKER)
    return path


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def far_tile_windows(
    height: int,
    width: int,
    ratio: float = 0.75,
    tiles: int = 2,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """画面上方远处带切成若干横向重叠窗口，让小人占更多推理像素。"""
    y2 = max(1, min(height, int(round(height * max(0.2, min(1.0, ratio))))))
    cols = max(1, int(tiles))
    if cols == 1:
        return [(0, 0, width, y2)]
    overlap = min(0.45, max(0.0, overlap))
    tile_w = width / cols
    overlap_px = tile_w * overlap
    windows: list[tuple[int, int, int, int]] = []
    for index in range(cols):
        x1 = 0 if index == 0 else int(index * tile_w - overlap_px)
        x2 = width if index == cols - 1 else int((index + 1) * tile_w + overlap_px)
        windows.append((max(0, x1), 0, min(width, x2), y2))
    return windows


def shift_box(
    xyxy: tuple[float, float, float, float],
    origin: tuple[int, int],
) -> tuple[float, float, float, float]:
    ox, oy = origin
    x1, y1, x2, y2 = xyxy
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)


def unmatched_boxes(
    existing: list[tuple[float, float, float, float]],
    candidates: list[tuple[tuple[float, float, float, float], float]],
    iou_thresh: float = 0.35,
) -> list[tuple[tuple[float, float, float, float], float]]:
    out: list[tuple[tuple[float, float, float, float], float]] = []
    for box, conf in candidates:
        if all(box_iou(box, other) < iou_thresh for other in existing):
            out.append((box, conf))
    return out


def nms_boxes(
    candidates: list[tuple[tuple[float, float, float, float], float]],
    iou_thresh: float = 0.55,
) -> list[tuple[tuple[float, float, float, float], float]]:
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for box, conf in ordered:
        if all(box_iou(box, other) < iou_thresh for other, _ in kept):
            kept.append((box, conf))
    return kept


def associate_by_iou(
    prev: dict[int, tuple[float, float, float, float]],
    detections: list[tuple[tuple[float, float, float, float], float]],
    *,
    iou_thresh: float = 0.3,
    next_id: int = EXTRA_ID_BASE,
    max_miss: int = 20,
    misses: dict[int, int] | None = None,
) -> tuple[list[tuple[int, tuple[float, float, float, float], float]], dict[int, tuple[float, float, float, float]], dict[int, int], int]:
    """把新检框接到上一帧 ID。未匹配的旧 ID 保留一段时间，避免远处目标闪烁。"""
    misses = dict(misses or {})
    pairs: list[tuple[float, int, int]] = []
    for di, (box, _conf) in enumerate(detections):
        for pid, prev_box in prev.items():
            iou = box_iou(box, prev_box)
            if iou >= iou_thresh:
                pairs.append((iou, pid, di))
    pairs.sort(reverse=True)
    used_ids: set[int] = set()
    used_det: set[int] = set()
    assigned: list[tuple[int, tuple[float, float, float, float], float]] = []
    alive: dict[int, tuple[float, float, float, float]] = {}
    alive_miss: dict[int, int] = {}

    for _iou, pid, di in pairs:
        if pid in used_ids or di in used_det:
            continue
        box, conf = detections[di]
        used_ids.add(pid)
        used_det.add(di)
        assigned.append((pid, box, conf))
        alive[pid] = box
        alive_miss[pid] = 0

    for di, (box, conf) in enumerate(detections):
        if di in used_det:
            continue
        assigned.append((next_id, box, conf))
        alive[next_id] = box
        alive_miss[next_id] = 0
        next_id += 1

    for pid, prev_box in prev.items():
        if pid in used_ids:
            continue
        miss = misses.get(pid, 0) + 1
        if miss <= max_miss:
            alive[pid] = prev_box
            alive_miss[pid] = miss

    return assigned, alive, alive_miss, next_id


class PersonTracker:
    """YOLO 只检 person，ByteTrack 赋稳定 ID；远处再切一块高分补检。"""

    def __init__(self, cfg: DetectConfig) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.tracker_path = resolve_tracker(cfg.tracker)
        logger.info("加载 YOLO %s device=%s imgsz=%s", cfg.model, self.device, cfg.imgsz)
        self.model = YOLO(cfg.model)
        self._far_model = YOLO(cfg.model) if cfg.far_pass else None
        self._prev: dict[int, tuple[float, float, float]] = {}
        self._extra_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._extra_miss: dict[int, int] = {}
        self._extra_next_id = EXTRA_ID_BASE

    def track(self, frame) -> list[Track]:
        now = time.monotonic()
        tracks = self._track_full(frame, now)
        if self._far_model is not None:
            extras = self._detect_far(frame)
            extras = unmatched_boxes([t.bbox_xyxy for t in tracks], extras)
            tracks.extend(self._update_extras(extras, now))
        return tracks

    def _track_full(self, frame, now: float) -> list[Track]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_path,
            classes=[PERSON_CLASS],
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
            save=False,
        )
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
        min_h = self.cfg.min_height

        for bbox, tid, conf in zip(xyxy, ids, confs):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            if (y2 - y1) < min_h:
                continue
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

    def _detect_far(self, frame) -> list[tuple[tuple[float, float, float, float], float]]:
        assert self._far_model is not None
        h, w = frame.shape[:2]
        found: list[tuple[tuple[float, float, float, float], float]] = []
        min_h = self.cfg.min_height
        for x1, y1, x2, y2 in far_tile_windows(h, w, self.cfg.far_ratio, self.cfg.far_tiles, self.cfg.tile_overlap):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results = self._far_model.predict(
                crop,
                classes=[PERSON_CLASS],
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                imgsz=self.cfg.imgsz,
                device=self.device,
                verbose=False,
                save=False,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for bbox, conf in zip(xyxy, confs):
                box = shift_box((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), (x1, y1))
                if (box[3] - box[1]) < min_h:
                    continue
                found.append((box, float(conf)))
        return nms_boxes(found)

    def _update_extras(
        self,
        detections: list[tuple[tuple[float, float, float, float], float]],
        now: float,
    ) -> list[Track]:
        assigned, self._extra_boxes, self._extra_miss, self._extra_next_id = associate_by_iou(
            self._extra_boxes,
            detections,
            next_id=self._extra_next_id,
            misses=self._extra_miss,
        )
        tracks: list[Track] = []
        for tid, box, conf in assigned:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            vx = vy = 0.0
            prev = self._prev.get(tid)
            if prev is not None:
                pcx, pcy, pt = prev
                dt = max(1e-3, now - pt)
                vx = (cx - pcx) / dt
                vy = (cy - pcy) / dt
            tracks.append(Track(id=tid, bbox_xyxy=box, conf=conf, vx=vx, vy=vy))
            self._prev[tid] = (cx, cy, now)
        return tracks
