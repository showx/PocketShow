from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from pocketshow.config import DetectConfig
from pocketshow.seats import box_seat_dist, seat_search_boxes
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


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = box_intersection(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_cover(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """小框被大框盖住的比例。头肩框套在全身框里时 IoU 往往不够高。"""
    inter = box_intersection(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(box_area(a), box_area(b))
    if smaller <= 0:
        return 0.0
    return inter / smaller


def far_tile_windows(
    height: int,
    width: int,
    ratio: float = 1.0,
    tiles: int = 2,
    overlap: float = 0.2,
    rows: int = 2,
) -> list[tuple[int, int, int, int]]:
    """把画面切成重叠窗口。默认整帧 2×2，近处坐着的人和远处小人都补得到。"""
    y_limit = max(1, min(height, int(round(height * max(0.2, min(1.0, ratio))))))
    cols = max(1, int(tiles))
    row_n = max(1, int(rows))
    overlap = min(0.45, max(0.0, overlap))
    tile_w = width / cols
    tile_h = y_limit / row_n
    overlap_x = tile_w * overlap
    overlap_y = tile_h * overlap
    windows: list[tuple[int, int, int, int]] = []
    for row in range(row_n):
        y1 = 0 if row == 0 else int(row * tile_h - overlap_y)
        y2 = y_limit if row == row_n - 1 else int((row + 1) * tile_h + overlap_y)
        for col in range(cols):
            x1 = 0 if col == 0 else int(col * tile_w - overlap_x)
            x2 = width if col == cols - 1 else int((col + 1) * tile_w + overlap_x)
            windows.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    return windows


def shift_box(
    xyxy: tuple[float, float, float, float],
    origin: tuple[int, int],
) -> tuple[float, float, float, float]:
    ox, oy = origin
    x1, y1, x2, y2 = xyxy
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)


def boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    iou_thresh: float,
    cover_thresh: float,
) -> bool:
    return box_iou(a, b) >= iou_thresh or box_cover(a, b) >= cover_thresh


def unmatched_boxes(
    existing: list[tuple[float, float, float, float]],
    candidates: list[tuple[tuple[float, float, float, float], float]],
    iou_thresh: float = 0.35,
    cover_thresh: float = 0.62,
) -> list[tuple[tuple[float, float, float, float], float]]:
    out: list[tuple[tuple[float, float, float, float], float]] = []
    for box, conf in candidates:
        if all(not boxes_overlap(box, other, iou_thresh=iou_thresh, cover_thresh=cover_thresh) for other in existing):
            out.append((box, conf))
    return out


def nms_boxes(
    candidates: list[tuple[tuple[float, float, float, float], float]],
    iou_thresh: float = 0.4,
    cover_thresh: float = 0.62,
) -> list[tuple[tuple[float, float, float, float], float]]:
    ordered = sorted(candidates, key=lambda item: (box_area(item[0]), item[1]), reverse=True)
    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for box, conf in ordered:
        if all(not boxes_overlap(box, other, iou_thresh=iou_thresh, cover_thresh=cover_thresh) for other, _ in kept):
            kept.append((box, conf))
    return kept


def crop_imgsz(height: int, width: int, cap: int = 1280) -> int:
    """远景切块：小窗不要硬拉到整帧的 1280。"""
    side = max(int(height), int(width))
    cap = max(320, int(cap or 640))
    target = max(320, min(cap, side * 2))
    return max(32, int(round(target / 32) * 32))


def enhance_crop(image: np.ndarray) -> np.ndarray:
    """坐姿小窗对比度差，CLAHE 后再检。"""
    if image is None or image.size == 0 or image.ndim != 3:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def seat_crop_imgsz(height: int, width: int, cap: int = 1280) -> int:
    """坐姿补检要把几十像素的头肩拉到至少 640，否则 YOLO 当椅子。"""
    side = max(int(height), int(width))
    cap = max(640, int(cap or 640))
    target = max(640, min(cap, max(side * 6, 640)))
    return max(32, int(round(target / 32) * 32))


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
        self._seat_hold: dict[str, tuple[tuple[float, float, float, float], float, float]] = {}

    def track(self, frame, regions: list[dict] | None = None) -> list[Track]:
        now = time.monotonic()
        tracks = self._track_full(frame, now)
        extras: list[tuple[tuple[float, float, float, float], float]] = []
        if self._far_model is not None:
            extras.extend(self._detect_far(frame))
        occupied = [t.bbox_xyxy for t in tracks] + [box for box, _ in extras]
        pending = self._empty_seat_regions(frame, occupied, regions or [])
        extras.extend(self._detect_regions(frame, pending))
        if extras:
            extras = unmatched_boxes([t.bbox_xyxy for t in tracks], extras)
            extras = nms_boxes(extras)
            tracks.extend(self._dedupe_extras(self._update_extras(extras, now), tracks))
        return tracks

    def _empty_seat_regions(
        self,
        frame,
        occupied: list[tuple[float, float, float, float]],
        regions: list[dict],
    ) -> list[dict]:
        if not regions:
            return []
        h, w = frame.shape[:2]
        empty: list[dict] = []
        for seat in regions:
            if any(box_seat_dist(seat, box, w, h) <= 1.25 for box in occupied):
                continue
            empty.append(seat)
        return empty

    def _dedupe_extras(self, extras: list[Track], tracks: list[Track]) -> list[Track]:
        existing = [t.bbox_xyxy for t in tracks]
        kept: list[Track] = []
        for track in sorted(extras, key=lambda item: (box_area(item.bbox_xyxy), item.conf), reverse=True):
            if any(boxes_overlap(track.bbox_xyxy, box, iou_thresh=0.4, cover_thresh=0.62) for box in existing):
                self._extra_boxes.pop(track.id, None)
                self._extra_miss.pop(track.id, None)
                continue
            kept.append(track)
            existing.append(track.bbox_xyxy)
        return kept

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

    def _crop_imgsz(self, crop) -> int:
        return crop_imgsz(crop.shape[0], crop.shape[1], self.cfg.imgsz)

    def _detect_far(self, frame) -> list[tuple[tuple[float, float, float, float], float]]:
        assert self._far_model is not None
        h, w = frame.shape[:2]
        found: list[tuple[tuple[float, float, float, float], float]] = []
        min_h = self.cfg.min_height
        for x1, y1, x2, y2 in far_tile_windows(
            h,
            w,
            self.cfg.far_ratio,
            self.cfg.far_tiles,
            self.cfg.tile_overlap,
            self.cfg.far_rows,
        ):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results = self._far_model.predict(
                crop,
                classes=[PERSON_CLASS],
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                imgsz=self._crop_imgsz(crop),
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

    def _detect_regions(self, frame, regions: list[dict]) -> list[tuple[tuple[float, float, float, float], float]]:
        """对指定工位窗口再检一次，坐着被桌子挡住的人整帧经常漏。"""
        if not regions:
            return []
        h, w = frame.shape[:2]
        model = self._far_model or self.model
        conf = min(float(self.cfg.conf), 0.08)
        min_h = max(16, int(self.cfg.min_height))
        min_w = 12.0
        min_area = 280.0
        found: list[tuple[tuple[float, float, float, float], float]] = []
        now = time.monotonic()
        for seat in regions:
            key = str(seat.get("person_id") or "") or f"{float(seat.get('cx') or 0):.3f}:{float(seat.get('cy') or 0):.3f}"
            hit: tuple[tuple[float, float, float, float], float] | None = None
            for window in seat_search_boxes(seat):
                x1 = int(window["x1"] * w)
                y1 = int(window["y1"] * h)
                x2 = int(window["x2"] * w)
                y2 = int(window["y2"] * h)
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                results = model.predict(
                    enhance_crop(crop),
                    classes=[PERSON_CLASS],
                    conf=conf,
                    iou=self.cfg.iou,
                    imgsz=seat_crop_imgsz(crop.shape[0], crop.shape[1], self.cfg.imgsz),
                    device=self.device,
                    verbose=False,
                    save=False,
                )
                if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                    continue
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for bbox, score in zip(xyxy, confs):
                    box = shift_box((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), (x1, y1))
                    bw = max(0.0, box[2] - box[0])
                    bh = max(0.0, box[3] - box[1])
                    if bh < min_h or bw < min_w or bw * bh < min_area:
                        continue
                    if box_seat_dist(seat, box, w, h) > 1.45:
                        continue
                    cand = (box, float(score))
                    if hit is None or cand[1] > hit[1]:
                        hit = cand
            if hit is not None:
                self._seat_hold[key] = (hit[0], hit[1], now)
                found.append(hit)
            else:
                held = self._seat_hold.get(key)
                if held is not None and now - held[2] <= 3.0:
                    found.append((held[0], held[1]))
                elif held is not None:
                    self._seat_hold.pop(key, None)
        return found

    def _update_extras(
        self,
        detections: list[tuple[tuple[float, float, float, float], float]],
        now: float,
    ) -> list[Track]:
        assigned, self._extra_boxes, self._extra_miss, self._extra_next_id = associate_by_iou(
            self._extra_boxes,
            detections,
            next_id=self._extra_next_id,
            max_miss=20,
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
