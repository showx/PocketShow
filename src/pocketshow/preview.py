from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from pocketshow.gallery import encode_jpeg

DEFAULT_PREVIEW = "data/preview.jpg"
FRESH_S = 2.5
WEB_WIDTH = 1600
JPEG_QUALITY = 72
MIN_INTERVAL_S = 0.12


class PreviewHub:
    """跟拍进程写下最新画布，管理页读出来给网页看。"""

    def __init__(self, path: str | Path = DEFAULT_PREVIEW, *, fresh_s: float = FRESH_S) -> None:
        self.path = Path(path)
        self.fresh_s = fresh_s
        self._last = 0.0

    def publish(self, image: np.ndarray | None, *, now: float | None = None) -> bool:
        if image is None or image.size == 0:
            return False
        stamp = time.monotonic() if now is None else now
        if self._last and stamp - self._last < MIN_INTERVAL_S:
            return False
        frame = image
        height, width = frame.shape[:2]
        if width > WEB_WIDTH:
            scale = WEB_WIDTH / width
            frame = cv2.resize(
                frame,
                (WEB_WIDTH, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        data = encode_jpeg(frame, quality=JPEG_QUALITY)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(self.path)
        self._last = stamp
        return True

    def read(self) -> bytes | None:
        if not self.path.exists():
            return None
        try:
            data = self.path.read_bytes()
        except OSError:
            return None
        return data or None

    def public(self) -> dict:
        if not self.path.exists():
            return {"fresh": False, "age_s": None}
        age = max(0.0, time.time() - self.path.stat().st_mtime)
        return {"fresh": age <= self.fresh_s, "age_s": round(age, 2)}


def placeholder_jpeg(width: int = 640, height: int = 360) -> bytes:
    return encode_jpeg(np.zeros((height, width, 3), dtype=np.uint8), quality=40)
