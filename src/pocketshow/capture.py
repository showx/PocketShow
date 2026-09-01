from __future__ import annotations

import logging
import subprocess
import time
from typing import Protocol

import cv2
import numpy as np

from pocketshow.config import CaptureConfig, WifiConfig
from pocketshow.pocket3.udp import DjiUdpClient
from pocketshow.pocket3.video import H264FrameDecoder

logger = logging.getLogger(__name__)


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class OpenCvCapture:
    def __init__(self, cap: cv2.VideoCapture, label: str) -> None:
        self.cap = cap
        self.label = label

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        self.cap.release()


class WifiCapture:
    def __init__(self, client: DjiUdpClient, width: int, height: int) -> None:
        self.client = client
        self.label = "OsmoPocket3 WiFi"
        self.decoder = H264FrameDecoder(width, height)
        self.decoder.start()
        self.client.set_video_callback(self.decoder.feed)
        self.client.start_video()

    def read(self) -> np.ndarray | None:
        return self.decoder.latest()

    def close(self) -> None:
        self.decoder.close()


def _avfoundation_backend() -> int:
    backend = getattr(cv2, "CAP_AVFOUNDATION", None)
    return backend if backend is not None else cv2.CAP_ANY


def list_avfoundation_names() -> list[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    names: list[str] = []
    in_video = False
    for line in (result.stderr or "").splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if in_video and "]" in line:
            names.append(line.split("]", 1)[-1].strip())
    return names


_usb_cache: tuple[float, bool] = (0.0, False)


def looks_like_pocket(name: str, capture: str = "") -> bool:
    blob = f"{name} {capture}".lower()
    return capture == "wifi" or "osmo" in blob or "pocket" in blob


def pocket3_usb_present(ttl: float = 5.0) -> bool:
    """系统相机列表里是否有 Pocket 3。只列设备，不打开画面。"""
    global _usb_cache
    now = time.time()
    if now - _usb_cache[0] < ttl:
        return _usb_cache[1]
    names = list_avfoundation_names()
    present = any(looks_like_pocket(name) for name in names)
    _usb_cache = (now, present)
    return present


def capture_identity(source: FrameSource) -> tuple[str, str]:
    label = getattr(source, "label", "") or ""
    if isinstance(source, WifiCapture):
        return "wifi", label or "OsmoPocket3"
    lowered = label.lower()
    if any(lowered.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv")) or "/" in label:
        return "file", label
    return "usb", label


# Pocket 3 UVC 支持的边长。乱设 1920x1080 + MJPG 会得到 1080x608 撕裂帧。
_VALID_DIMS = {720, 1080, 1280, 1512, 1920, 2160, 2688, 3072, 3840}


def _frame_ok(frame: np.ndarray | None) -> bool:
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    return w in _VALID_DIMS and h in _VALID_DIMS


def _open_index(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, _avfoundation_backend())
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok or not _frame_ok(frame):
        cap.release()
        return None
    h, w = frame.shape[:2]
    if (w, h) != (width, height) and (h, w) != (width, height):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok2, frame2 = cap.read()
        if not (ok2 and _frame_ok(frame2)):
            logger.warning(
                "相机拒绝 %sx%s，保持原生 %sx%s",
                width,
                height,
                w,
                h,
            )
            cap.release()
            cap = cv2.VideoCapture(index, _avfoundation_backend())
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if not ok or not _frame_ok(frame):
                cap.release()
                return None
    return cap


def open_usb_or_camera(cfg: CaptureConfig) -> OpenCvCapture:
    names = list_avfoundation_names()
    preferred: list[int] = []
    for i, name in enumerate(names):
        lowered = name.lower()
        if "osmo" in lowered or "pocket" in lowered:
            preferred.append(i)
    order = preferred + [cfg.device_index] + [i for i in range(6) if i not in preferred]
    seen: set[int] = set()
    for index in order:
        if index in seen:
            continue
        seen.add(index)
        cap = _open_index(index, cfg.width, cfg.height, cfg.fps)
        if cap is not None:
            label = names[index] if index < len(names) else f"camera:{index}"
            logger.info("打开视频源 %s", label)
            return OpenCvCapture(cap, label)
    raise RuntimeError("打不开 USB/摄像头。检查 Pocket 3 是否处于 Webcam 模式，以及 macOS 摄像头权限。")


def open_file(path: str) -> OpenCvCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频文件: {path}")
    return OpenCvCapture(cap, path)


def open_capture(
    cfg: CaptureConfig,
    wifi_cfg: WifiConfig,
    client: DjiUdpClient | None = None,
) -> FrameSource:
    source = cfg.source
    if source == "file" or (cfg.file and source == "auto"):
        if not cfg.file:
            raise RuntimeError("capture.file 未设置")
        return open_file(cfg.file)
    if source == "wifi":
        if client is None:
            raise RuntimeError("WiFi 取流需要已连接的 Pocket3 UDP 会话")
        return WifiCapture(client, wifi_cfg.video_width, wifi_cfg.video_height)
    try:
        return open_usb_or_camera(cfg)
    except RuntimeError:
        if source == "auto" and client is not None:
            logger.warning("USB 取流失败，切到 WiFi 视频流")
            return WifiCapture(client, wifi_cfg.video_width, wifi_cfg.video_height)
        raise
