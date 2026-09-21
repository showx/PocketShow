from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from pocketshow.gallery import encode_jpeg
from pocketshow.seats import clean_box

DEFAULT_PREVIEW = "data/preview.jpg"
FRESH_S = 4.0
JPEG_QUALITY = 90
MIN_INTERVAL_S = 0.12
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class PreviewHub:
    """跟拍进程写下最新画布，管理页读出来给网页看。"""

    def __init__(self, path: str | Path = DEFAULT_PREVIEW, *, fresh_s: float = FRESH_S) -> None:
        self.path = Path(path)
        self.fresh_s = fresh_s
        self._last = 0.0

    def meta_path(self) -> Path:
        return self.path.with_name(self.path.stem + ".json")

    def pane_path(self, camera_id: str) -> Path:
        return self.path.with_name(f"{self.path.stem}.{safe_cam_id(camera_id)}.jpg")

    def _write_meta(self, payload: dict) -> None:
        path = self.meta_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _write_jpeg(self, path: Path, image: np.ndarray) -> None:
        data = encode_jpeg(image, quality=JPEG_QUALITY)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def _read_meta(self) -> dict:
        path = self.meta_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_cameras(self) -> list[dict] | None:
        data = self._read_meta()
        raw = data.get("cameras") if data else None
        if not isinstance(raw, list):
            return None
        cameras: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cam_id = str(item.get("id") or "")
            row = {"id": cam_id, "name": str(item.get("name") or cam_id)}
            try:
                src_w = int(item.get("src_w") or 0)
                src_h = int(item.get("src_h") or 0)
            except (TypeError, ValueError):
                src_w = src_h = 0
            if src_w > 0 and src_h > 0:
                row["src_w"] = src_w
                row["src_h"] = src_h
            raw_boxes = item.get("boxes")
            if isinstance(raw_boxes, list):
                boxes = []
                for box in raw_boxes:
                    if not isinstance(box, dict):
                        continue
                    cleaned = clean_box(box)
                    if cleaned is not None:
                        boxes.append(cleaned)
                if boxes:
                    row["boxes"] = boxes
            cameras.append(row)
        return cameras

    def _read_canvas(self) -> list[int] | None:
        raw = self._read_meta().get("canvas")
        if not isinstance(raw, list) or len(raw) < 2:
            return None
        try:
            width, height = int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return None
        if width < 1 or height < 1:
            return None
        return [width, height]

    def _meta_payload(self, cameras: list[dict] | None, canvas: list[int] | None) -> dict:
        payload: dict = {}
        if cameras is not None:
            payload["cameras"] = [
                {key: value for key, value in item.items() if key != "image"} for item in cameras
            ]
        else:
            current = self._read_cameras()
            if current is not None:
                payload["cameras"] = current
        if canvas is not None:
            payload["canvas"] = canvas
        else:
            current_canvas = self._read_canvas()
            if current_canvas is not None:
                payload["canvas"] = current_canvas
        return payload

    def _write_panes(self, panes: list[dict]) -> None:
        keep: set[str] = set()
        for item in panes:
            cam_id = str(item.get("id") or "")
            image = item.get("image")
            if not cam_id or image is None or getattr(image, "size", 0) == 0:
                continue
            safe = safe_cam_id(cam_id)
            keep.add(safe)
            self._write_jpeg(self.pane_path(safe), image)
        prefix = f"{self.path.stem}."
        suffix = ".jpg"
        for path in self.path.parent.glob(f"{self.path.stem}.*.jpg"):
            name = path.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            token = name[len(prefix) : -len(suffix)]
            if token and token not in keep:
                path.unlink(missing_ok=True)

    def publish(
        self,
        image: np.ndarray | None,
        *,
        now: float | None = None,
        cameras: list[dict] | None = None,
        panes: list[dict] | None = None,
    ) -> bool:
        if image is None or image.size == 0:
            if cameras is not None:
                self._write_meta(self._meta_payload(cameras, None))
            return False
        stamp = time.monotonic() if now is None else now
        if self._last and stamp - self._last < MIN_INTERVAL_S:
            if cameras is not None:
                self._write_meta(self._meta_payload(cameras, None))
            return False
        self._write_jpeg(self.path, image)
        if panes is not None:
            self._write_panes(panes)
        self._last = stamp
        if cameras is not None:
            self._write_meta(
                self._meta_payload(cameras, [int(image.shape[1]), int(image.shape[0])])
            )
        return True

    def read(self) -> bytes | None:
        return _read_bytes(self.path)

    def read_pane(self, camera_id: str) -> bytes | None:
        return _read_bytes(self.pane_path(camera_id))

    def _age_s(self) -> float | None:
        ages: list[float] = []
        for path in (self.path, self.meta_path()):
            if path.exists():
                ages.append(max(0.0, time.time() - path.stat().st_mtime))
        if not ages:
            return None
        return min(ages)

    def public(self) -> dict:
        cameras = self._read_cameras()
        info: dict = {"fresh": False, "age_s": None, "cameras": cameras, "canvas": self._read_canvas()}
        age = self._age_s()
        if age is None:
            return info
        info["fresh"] = age <= self.fresh_s
        info["age_s"] = round(age, 2)
        return info


def safe_cam_id(camera_id: str) -> str:
    text = _SAFE_ID.sub("_", (camera_id or "").strip()).strip("._")
    return text[:80] or "cam"


def _read_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data or None


def placeholder_jpeg(width: int = 640, height: int = 360) -> bytes:
    return encode_jpeg(np.zeros((height, width, 3), dtype=np.uint8), quality=40)
