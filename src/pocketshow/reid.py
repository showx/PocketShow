from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path.home() / ".pocketshow" / "models"
_XML_NAME = "person-reidentification-retail-0288.xml"
_BIN_NAME = "person-reidentification-retail-0288.bin"
_BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0288/FP32/"
_INPUT_H = 256
_INPUT_W = 128


def person_crop(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    *,
    pad: float = 0.08,
    min_side: int = 16,
) -> np.ndarray | None:
    """整框裁人，坐着只剩头肩也能抽外观。"""
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    rx1 = max(0, int(x1 - bw * pad))
    ry1 = max(0, int(y1 - bh * pad))
    rx2 = min(w, int(x2 + bw * pad))
    ry2 = min(h, int(y2 + bh * pad))
    if rx2 - rx1 < min_side or ry2 - ry1 < min_side:
        return None
    crop = frame[ry1:ry2, rx1:rx2]
    return crop if crop.size else None


def preprocess(image: np.ndarray) -> np.ndarray:
    """Intel 0288：BGR、1×3×256×128、0–255。"""
    resized = cv2.resize(image, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    blob = np.transpose(resized.astype(np.float32), (2, 0, 1))
    return np.ascontiguousarray(blob[None, ...])


def l2norm(vec: np.ndarray) -> np.ndarray | None:
    out = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(out))
    if n < 1e-6:
        return None
    return out / n


def _download(url: str, dest: Path, min_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("下载 ReID 模型 %s", dest.name)
    urllib.request.urlretrieve(url, tmp)
    if not tmp.exists() or tmp.stat().st_size < min_size:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ReID 模型不完整: {dest.name}")
    tmp.replace(dest)


def ensure_models() -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    xml = _MODEL_DIR / _XML_NAME
    bin_path = _MODEL_DIR / _BIN_NAME
    if not xml.exists() or xml.stat().st_size < 100_000:
        _download(_BASE + _XML_NAME, xml, 100_000)
    if not bin_path.exists() or bin_path.stat().st_size < 200_000:
        _download(_BASE + _BIN_NAME, bin_path, 200_000)
    return xml, bin_path


class PersonReID:
    """OSNet-x0.25 量级人体外观：Intel OmniScaleNet retail-0288，256 维余弦比对。"""

    def __init__(self) -> None:
        from openvino import Core

        xml, bin_path = ensure_models()
        core = Core()
        model = core.read_model(str(xml), str(bin_path))
        self._compiled = core.compile_model(model, "CPU")
        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)
        logger.info("已加载人体 ReID %s", xml.name)

    def embed_image(self, image: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0 or min(image.shape[:2]) < 8:
            return None
        blob = preprocess(image)
        raw = self._compiled([blob])[self._output]
        return l2norm(raw)

    def embed_box(
        self,
        frame: np.ndarray,
        xyxy: tuple[float, float, float, float],
        *,
        min_height: int = 24,
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = xyxy
        if (y2 - y1) < min_height:
            return None
        crop = person_crop(frame, xyxy)
        if crop is None:
            return None
        return self.embed_image(crop)
