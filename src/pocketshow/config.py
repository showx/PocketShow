from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    source: Literal["auto", "usb", "camera", "file", "wifi"] = "auto"
    device_index: int = 0
    file: str | None = None
    width: int = 1920
    height: int = 1080
    fps: int = 30


class DetectConfig(BaseModel):
    model: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    device: str = "auto"
    tracker: str = "bytetrack.yaml"


class RecognizeConfig(BaseModel):
    enabled: bool = True
    match_threshold: float = 0.43
    soft_threshold: float = 0.38
    dup_threshold: float = 0.50
    det_score: float = 0.7
    enroll_score: float = 0.88
    enroll_min_face: int = 36
    enroll_confirm: int = 18
    auto_enroll: bool = True
    gallery: str = "data/faces.json"
    photos: str = "data/faces"


class FollowConfig(BaseModel):
    deadzone: float = 0.08
    kp: float = 1.35
    ki: float = 0.05
    kd: float = 0.18
    feedforward: float = 0.35
    max_rate: float = 0.55
    lost_hold_s: float = 0.8
    lost_timeout_s: float = 2.5
    target_area: float = 0.12


class GimbalConfig(BaseModel):
    backend: Literal["stub", "wifi"] = "stub"
    send_hz: float = 30.0
    command: str = "data/gimbal.json"


class WifiConfig(BaseModel):
    camera_ip: str = "192.168.2.1"
    camera_port: int = 9004
    ssid: str = ""
    password: str = ""
    join: bool = False
    ble: bool = False
    ble_timeout_s: float = 15.0
    video_width: int = 1280
    video_height: int = 720


class Settings(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    follow: FollowConfig = Field(default_factory=FollowConfig)
    recognize: RecognizeConfig = Field(default_factory=RecognizeConfig)
    gimbal: GimbalConfig = Field(default_factory=GimbalConfig)
    wifi: WifiConfig = Field(default_factory=WifiConfig)


def load_settings(path: str | Path | None) -> Settings:
    if path is None:
        return Settings()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text()) or {}
    return Settings.model_validate(data)
