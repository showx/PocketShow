from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    source: Literal["auto", "usb", "camera", "file", "wifi", "rtsp"] = "auto"
    device_index: int = 0
    file: str | None = None
    width: int = 1920
    height: int = 1080
    fps: int = 30


class RtspCamera(BaseModel):
    id: str = ""
    name: str = ""
    host: str = ""
    port: int = 554
    username: str = "admin"
    password: str = ""
    channel: int = 1
    stream: Literal["main", "sub", "third"] = "sub"
    url: str = ""
    transport: Literal["tcp", "udp"] = "tcp"
    monitor: bool = True


class RtspConfig(BaseModel):
    camera_id: str = ""
    cameras: list[RtspCamera] = Field(default_factory=list)
    host: str = ""
    port: int = 554
    username: str = "admin"
    password: str = ""
    channel: int = 1
    stream: Literal["main", "sub", "third"] = "sub"
    url: str = ""
    transport: Literal["tcp", "udp"] = "tcp"
    settings: str = "data/capture.json"


class DetectConfig(BaseModel):
    model: str = "yolo11n.pt"
    imgsz: int = 1280
    conf: float = 0.2
    iou: float = 0.5
    device: str = "auto"
    tracker: str = "bytetrack.yaml"
    min_height: int = 12
    far_pass: bool = True
    far_ratio: float = 0.75
    far_tiles: int = 2
    tile_overlap: float = 0.2


class RecognizeConfig(BaseModel):
    enabled: bool = True
    match_threshold: float = 0.43
    soft_threshold: float = 0.38
    dup_threshold: float = 0.70
    det_score: float = 0.5
    det_min_face: int = 12
    enroll_score: float = 0.88
    enroll_min_face: int = 36
    enroll_confirm: int = 18
    auto_enroll: bool = True
    liveness: bool = True
    liveness_threshold: float = 0.62
    liveness_confirm: int = 5
    liveness_min_face: int = 40
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


class WatchConfig(BaseModel):
    enabled: bool = True
    away_s: float = 30.0
    work_start: str = "09:00"
    work_end: str = "18:30"
    workdays: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    status: str = "data/station.json"
    settings: str = "data/watch.json"
    log: str = "data/away.jsonl"


class Settings(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    follow: FollowConfig = Field(default_factory=FollowConfig)
    recognize: RecognizeConfig = Field(default_factory=RecognizeConfig)
    gimbal: GimbalConfig = Field(default_factory=GimbalConfig)
    wifi: WifiConfig = Field(default_factory=WifiConfig)
    rtsp: RtspConfig = Field(default_factory=RtspConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    preview: str = "data/preview.jpg"


def ensure_local_config(path: str | Path) -> Path:
    """default.yaml 不进 git。缺失时从 default.example.yaml 复制一份到本地。"""
    cfg_path = Path(path)
    if cfg_path.exists():
        return cfg_path
    if cfg_path.name != "default.yaml":
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    example = cfg_path.with_name("default.example.yaml")
    if not example.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}（也没有 {example.name}）")
    cfg_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    return cfg_path


def load_settings(path: str | Path | None) -> Settings:
    if path is None:
        return Settings()
    cfg_path = ensure_local_config(path)
    data = yaml.safe_load(cfg_path.read_text()) or {}
    return Settings.model_validate(data)
