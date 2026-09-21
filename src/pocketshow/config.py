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
    far_ratio: float = 1.0
    far_tiles: int = 2
    far_rows: int = 2
    tile_overlap: float = 0.2


class RecognizeConfig(BaseModel):
    enabled: bool = True
    match_threshold: float = 0.50
    soft_threshold: float = 0.42
    match_margin: float = 0.06
    dup_threshold: float = 0.70
    det_score: float = 0.5
    det_min_face: int = 12
    match_min_face: int = 18
    match_min_score: float = 0.52
    match_max_yaw: float = 0.72
    id_min_face: int = 28
    id_min_score: float = 0.65
    id_max_yaw: float = 0.50
    id_confirm: int = 2
    update_min_sim: float = 0.55
    template_collide: float = 0.55
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
    seats: bool = True
    seat_min_hits: int = 3
    seat_confirm: int = 2
    reid: bool = True
    reid_threshold: float = 0.48
    reid_soft: float = 0.38
    reid_margin: float = 0.08
    reid_update: float = 0.50
    reid_collide: float = 0.58
    reid_min_height: int = 24


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


class SceneConfig(BaseModel):
    enabled: bool = False
    backend: Literal["moss-vl", "openai", "stub"] = "openai"
    protocol: Literal["auto", "hf", "sglang"] = "auto"
    base_url: str = "http://127.0.0.1:30000/v1"
    ws_url: str = "ws://127.0.0.1:8000/v1/realtime"
    api_key: str = ""
    model: str = "OpenMOSS-Team/MOSS-VL-Realtime"
    interval_s: float = 8.0
    sample_fps: float = 1.0
    gap_s: float = 2.0
    timeout_s: float = 25.0
    max_width: int = 768
    jpeg_quality: int = 70
    max_tokens: int = 80
    max_tokens_per_second: float = 12.0
    prompt: str = ""
    system_prompt: str = ""
    log: str = "data/scene.jsonl"
    status: str = "data/scene.json"


class MapConfig(BaseModel):
    enabled: bool = False
    backend: Literal["http", "stub"] = "http"
    base_url: str = "http://127.0.0.1:8090"
    camera_id: str = ""
    interval_s: float = 0.5
    timeout_s: float = 8.0
    max_width: int = 640
    jpeg_quality: int = 80
    log: str = "data/map.jsonl"
    status: str = "data/map.json"


class MocapConfig(BaseModel):
    enabled: bool = False
    backend: Literal["http", "stub"] = "http"
    base_url: str = "http://127.0.0.1:8006"
    camera_id: str = ""
    interval_s: float = 0.4
    timeout_s: float = 6.0
    max_width: int = 640
    jpeg_quality: int = 75
    log: str = "data/mocap.jsonl"
    status: str = "data/mocap.json"


class Settings(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    follow: FollowConfig = Field(default_factory=FollowConfig)
    recognize: RecognizeConfig = Field(default_factory=RecognizeConfig)
    gimbal: GimbalConfig = Field(default_factory=GimbalConfig)
    wifi: WifiConfig = Field(default_factory=WifiConfig)
    rtsp: RtspConfig = Field(default_factory=RtspConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    geomap: MapConfig = Field(default_factory=MapConfig)
    mocap: MocapConfig = Field(default_factory=MocapConfig)
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
