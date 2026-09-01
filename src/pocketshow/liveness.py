from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path.home() / ".pocketshow" / "models"
_MODEL_NAME = "minifasnet_v2.onnx"
_MODEL_URLS = [
    "https://hf-mirror.com/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
    "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
]
_CROP_SCALE = 2.7
_INPUT = 80
LIVE_CLASS = 1


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    x = x - float(np.max(x))
    e = np.exp(x)
    return e / float(np.sum(e) + 1e-9)


def scaled_crop(image: np.ndarray, xyxy: tuple[float, float, float, float], scale: float, size: int) -> np.ndarray:
    """按 MiniFASNet 习惯：人脸框中心放大 scale 倍，再缩到 size×size。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    scale = min((h - 1) / box_h, (w - 1) / box_w, scale)
    cx = x1 + box_w * 0.5
    cy = y1 + box_h * 0.5
    nw, nh = box_w * scale, box_h * scale
    rx1 = max(0, int(cx - nw * 0.5))
    ry1 = max(0, int(cy - nh * 0.5))
    rx2 = min(w - 1, int(cx + nw * 0.5))
    ry2 = min(h - 1, int(cy + nh * 0.5))
    crop = image[ry1 : ry2 + 1, rx1 : rx2 + 1]
    if crop.size == 0:
        crop = image
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def geometry_penalty(person_xyxy: tuple[float, float, float, float], face_xyxy: tuple[float, float, float, float]) -> float:
    """头像照片常被 YOLO 当成整个人，人脸占比会异常大。"""
    px1, py1, px2, py2 = person_xyxy
    fx1, fy1, fx2, fy2 = face_xyxy
    person_area = max(1.0, (px2 - px1) * (py2 - py1))
    face_area = max(0.0, (fx2 - fx1) * (fy2 - fy1))
    ratio = face_area / person_area
    if ratio >= 0.62:
        return 0.35
    if ratio >= 0.48:
        return 0.18
    return 0.0


def is_live(score: float, threshold: float) -> bool:
    return score >= threshold


def ensure_model() -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = _MODEL_DIR / _MODEL_NAME
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    last_error: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for url in _MODEL_URLS:
        try:
            logger.info("下载活体模型 %s", url.split("/")[-1])
            urllib.request.urlretrieve(url, tmp)
            if tmp.exists() and tmp.stat().st_size > 100_000:
                tmp.replace(dest)
                return dest
        except Exception as exc:  # noqa: BLE001 — 镜像轮询
            last_error = exc
            logger.warning("活体模型下载失败 %s: %s", url, exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    raise RuntimeError(f"无法下载 MiniFASNet 活体模型: {last_error}")


class FaceLiveness:
    """MiniFASNetV2 静默活体：拦住纸质照片和屏幕翻拍。"""

    def __init__(self) -> None:
        path = ensure_model()
        self.net = cv2.dnn.readNetFromONNX(str(path))
        logger.info("已加载活体模型 %s", path.name)

    def score(
        self,
        frame: np.ndarray,
        face_xyxy: tuple[float, float, float, float],
        person_xyxy: tuple[float, float, float, float] | None = None,
    ) -> float:
        crop = scaled_crop(frame, face_xyxy, _CROP_SCALE, _INPUT)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (_INPUT, _INPUT), swapRB=False)
        self.net.setInput(blob)
        logits = self.net.forward()
        probs = softmax(logits)
        live = float(probs[LIVE_CLASS]) if probs.size > LIVE_CLASS else 0.0
        if person_xyxy is not None:
            live = max(0.0, live - geometry_penalty(person_xyxy, face_xyxy))
        return live
