from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np

from pocketshow.config import CaptureConfig, RtspCamera, RtspConfig, WifiConfig
from pocketshow.pocket3.udp import DjiUdpClient
from pocketshow.pocket3.video import H264FrameDecoder

logger = logging.getLogger(__name__)

STREAM_CODES = {"main": 1, "sub": 2, "third": 3}
STREAM_LABELS = {"main": "主码流", "sub": "子码流", "third": "第三码流"}
STALE_FRAME_S = 2.5
_CV_LOCK = threading.Lock()


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


def looks_corrupt(frame: np.ndarray | None) -> bool:
    """H.264 没等到 I 帧、管道错位时会出现灰底彩噪花屏。"""
    if frame is None or frame.size == 0:
        return True
    if frame.ndim != 3 or frame.shape[2] < 3:
        return True
    height, width = frame.shape[:2]
    if height < 16 or width < 16:
        return True
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    contrast = float(gray.std())
    if contrast < 9.0:
        return True
    blue = small[:, :, 0].astype(np.int16)
    green = small[:, :, 1].astype(np.int16)
    red = small[:, :, 2].astype(np.int16)
    chroma = np.abs(red - green) + np.abs(green - blue) + np.abs(blue - red)
    speck = float((chroma > 70).mean())
    if contrast < 20.0 and speck > 0.05:
        return True
    if 80.0 <= mean <= 175.0 and contrast < 16.0:
        return True
    return False


def should_reconnect(miss: int, *, after: int = 12, every: int = 40) -> bool:
    if miss < after:
        return False
    return miss == after or miss % every == 0


def ffmpeg_rtsp_cmd(url: str, width: int, height: int, transport: str) -> list[str]:
    # nobuffer 会把尚未对齐的 P 帧直接吐出来，局域网监测宁可晚几帧也不要花屏。
    # VLC 按 limited range + 高质量色度放大；默认 scale 会把监控画面拉成一层雾。
    w, h = int(width), int(height)
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        transport,
        "-fflags",
        "+genpts+discardcorrupt",
        "-flags",
        "low_delay",
        "-i",
        url,
        "-an",
        "-sn",
        "-sws_flags",
        "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp",
        "-vf",
        f"scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int:in_range=tv:out_range=pc",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


class OpenCvCapture:
    def __init__(
        self,
        cap: cv2.VideoCapture,
        label: str,
        kind: str = "usb",
        seed: np.ndarray | None = None,
    ) -> None:
        self.cap = cap
        self.label = label
        self.kind = kind
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True
        self._thread: threading.Thread | None = None
        self._bad = 0
        if seed is not None and seed.size and not looks_corrupt(seed):
            self._frame = seed.copy()
            self._stamp = time.monotonic()
        if kind == "rtsp":
            self._thread = threading.Thread(target=self._drain, daemon=True, name="rtsp-cv")
            self._thread.start()

    def read(self) -> np.ndarray | None:
        if self._thread is None:
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame
        with self._lock:
            if self._frame is None:
                return None
            if time.monotonic() - self._stamp > STALE_FRAME_S:
                return None
            return self._frame.copy()

    def close(self) -> None:
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=0.8)

        def _release() -> None:
            try:
                self.cap.release()
            except Exception:
                logger.debug("%s 释放失败", self.label, exc_info=True)

        worker = threading.Thread(target=_release, daemon=True, name="cv-release")
        worker.start()
        worker.join(timeout=1.0)

    def _keep(self, frame: np.ndarray) -> None:
        if looks_corrupt(frame):
            self._bad += 1
            if self._bad == 1 or self._bad % 50 == 0:
                logger.warning("%s 解码花屏，丢弃坏帧", self.label)
            return
        self._bad = 0
        copied = frame.copy()
        with self._lock:
            self._frame = copied
            self._stamp = time.monotonic()

    def _drain(self) -> None:
        while self._running:
            try:
                with _CV_LOCK:
                    if not self._running:
                        break
                    ok, frame = self.cap.read()
            except Exception:
                break
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue
            self._keep(frame)


class FfmpegRtspCapture:
    """独立 ffmpeg 进程解 RTSP，只保留最新一帧完整画面。"""

    def __init__(self, url: str, width: int, height: int, transport: str, label: str) -> None:
        self.url = url
        self.width = width
        self.height = height
        self.label = label
        self.kind = "rtsp"
        self.frame_bytes = width * height * 3
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True
        self._bad = 0
        cmd = ffmpeg_rtsp_cmd(url, width, height, transport)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("需要 ffmpeg 才能拉取局域网 RTSP") from exc
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="rtsp-ffmpeg")
        self._reader.start()

    def read(self) -> np.ndarray | None:
        if not self._running:
            return None
        with self._lock:
            if self._frame is None:
                return None
            if time.monotonic() - self._stamp > STALE_FRAME_S:
                return None
            return self._frame.copy()

    def close(self) -> None:
        self._running = False
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        stdout = self._proc.stdout
        if stdout is not None:
            try:
                stdout.close()
            except Exception:
                pass
        self._reader.join(timeout=1.0)

    def _keep(self, frame: np.ndarray) -> None:
        if looks_corrupt(frame):
            self._bad += 1
            if self._bad == 1 or self._bad % 50 == 0:
                logger.warning("%s 解码花屏，丢弃坏帧", self.label)
            return
        self._bad = 0
        with self._lock:
            self._frame = frame
            self._stamp = time.monotonic()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        buf = bytearray()
        while self._running and self._proc.poll() is None:
            need = self.frame_bytes - len(buf)
            piece = self._proc.stdout.read(need)
            if not piece:
                break
            buf.extend(piece)
            if len(buf) < self.frame_bytes:
                continue
            frame = np.frombuffer(bytes(buf), dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            buf.clear()
            self._keep(frame)
        self._running = False


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
            timeout=3,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        logger.warning("列举摄像头超时")
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
    kind = getattr(source, "kind", "")
    if kind == "rtsp" or isinstance(source, FfmpegRtspCapture):
        return "rtsp", label or "RTSP"
    if isinstance(source, WifiCapture):
        return "wifi", label or "OsmoPocket3"
    if kind == "file":
        return "file", label
    lowered = label.lower()
    if any(lowered.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv")) or "/" in label:
        return "file", label
    return "usb", label


def hik_channel(channel: int, stream: str) -> int:
    if channel < 1:
        raise ValueError("通道号从 1 开始")
    if stream not in STREAM_CODES:
        raise ValueError("码流只能是 main / sub / third")
    return channel * 100 + STREAM_CODES[stream]


def camera_slug(name: str, host: str, channel: int, taken: set[str]) -> str:
    raw = (name or host or "cam").strip().lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-") or "cam"
    if channel > 1:
        slug = f"{slug}-ch{channel}"
    candidate = slug
    n = 2
    while candidate in taken:
        candidate = f"{slug}-{n}"
        n += 1
    return candidate


def list_cameras(cfg: RtspConfig) -> list[RtspCamera]:
    if cfg.cameras:
        return [cam.model_copy(deep=True) for cam in cfg.cameras]
    if (cfg.host or "").strip() or (cfg.url or "").strip():
        return [
            RtspCamera(
                id="default",
                name="摄像机",
                host=cfg.host,
                port=cfg.port,
                username=cfg.username,
                password=cfg.password,
                channel=cfg.channel,
                stream=cfg.stream,
                url=cfg.url,
                transport=cfg.transport,
            )
        ]
    return []


def pick_camera(cfg: RtspConfig) -> RtspCamera | None:
    cameras = list_cameras(cfg)
    if not cameras:
        return None
    return next((cam for cam in cameras if cam.id == cfg.camera_id), cameras[0])


def active_rtsp(cfg: RtspConfig) -> RtspConfig:
    cam = pick_camera(cfg)
    if cam is None:
        return cfg
    return cfg.model_copy(
        update={
            "camera_id": cam.id,
            "host": cam.host or cfg.host,
            "port": cam.port,
            "username": cam.username or cfg.username,
            "password": cam.password or cfg.password,
            "channel": cam.channel,
            "stream": cam.stream,
            "url": cam.url or cfg.url,
            "transport": cam.transport or cfg.transport,
        }
    )


def camera_public(cam: RtspCamera, shared: RtspConfig | None = None) -> dict:
    password = cam.password or (rtsp_password(shared) if shared is not None else "")
    host = cam.host or (shared.host if shared is not None else "")
    try:
        stream_id = hik_channel(cam.channel, cam.stream)
        label = stream_label(cam.stream, cam.channel)
    except ValueError:
        stream_id = 0
        label = cam.stream
    return {
        "id": cam.id,
        "name": cam.name or cam.host or cam.id,
        "host": host,
        "port": cam.port,
        "username": cam.username or (shared.username if shared is not None else "admin"),
        "has_password": bool(password),
        "channel": cam.channel,
        "stream": cam.stream,
        "stream_id": stream_id,
        "stream_label": label,
        "transport": cam.transport,
        "monitor": bool(cam.monitor),
    }


def stream_label(stream: str, channel: int = 1) -> str:
    name = STREAM_LABELS.get(stream, stream)
    stream_id = hik_channel(channel, stream) if stream in STREAM_CODES and channel >= 1 else 0
    if channel > 1:
        return f"第{channel}路 {name} {stream_id}"
    return f"{name} {stream_id}" if stream_id else name


def rtsp_password(cfg: RtspConfig) -> str:
    env = (os.environ.get("POCKETSHOW_RTSP_PASSWORD") or "").strip()
    return env or (cfg.password or "")


def redact_rtsp_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.username and not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = quote(parts.username or "", safe="")
    auth = f"{user}:***@" if user or parts.password else ""
    return urlunsplit((parts.scheme, f"{auth}{host}", parts.path, parts.query, parts.fragment))


def build_rtsp_url(cfg: RtspConfig) -> str:
    cfg = active_rtsp(cfg)
    if (cfg.url or "").strip():
        return cfg.url.strip()
    host = (cfg.host or "").strip()
    if not host:
        raise RuntimeError("还没填摄像机 IP。在配置或管理页里写上摄像机地址")
    password = rtsp_password(cfg)
    if not password:
        raise RuntimeError(
            "缺少 RTSP 密码。在管理页保存，或设置环境变量 POCKETSHOW_RTSP_PASSWORD，或写到 rtsp.password"
        )
    user = quote(cfg.username or "admin", safe="")
    pwd = quote(password, safe="")
    stream_id = hik_channel(cfg.channel, cfg.stream)
    return f"rtsp://{user}:{pwd}@{host}:{int(cfg.port)}/Streaming/Channels/{stream_id}"


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="capture.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class CaptureStore:
    """yaml 默认值 + data/capture.json。管理页改码流后，跟拍进程会重载。"""

    def __init__(self, capture: CaptureConfig, rtsp: RtspConfig) -> None:
        self.capture = capture.model_copy(deep=True)
        self.rtsp = rtsp.model_copy(deep=True)
        self.path = Path(rtsp.settings) if rtsp.settings else None
        self._mtime = 0.0
        self._seed_cameras = [cam.model_copy(deep=True) for cam in rtsp.cameras]
        self.reload(force=True)

    def identity(self) -> tuple[str, str]:
        if self.capture.source != "rtsp":
            return self.capture.source, ""
        parts: list[str] = []
        for cam in list_cameras(self.rtsp):
            flag = "on" if cam.monitor else "off"
            try:
                parts.append(f"{cam.id}:{flag}:{build_rtsp_url(camera_as_rtsp(self.rtsp, cam))}")
            except RuntimeError:
                parts.append(f"{cam.id}:{flag}")
        return "rtsp", "|".join(parts)

    def _sync_active(self) -> None:
        live = active_rtsp(self.rtsp)
        self.rtsp.camera_id = live.camera_id
        self.rtsp.host = live.host
        self.rtsp.port = live.port
        self.rtsp.username = live.username
        self.rtsp.password = live.password
        self.rtsp.channel = live.channel
        self.rtsp.stream = live.stream
        self.rtsp.url = live.url
        self.rtsp.transport = live.transport

    def public(self) -> dict:
        cameras = list_cameras(self.rtsp)
        live = active_rtsp(self.rtsp)
        configured = bool((live.host or "").strip() or (live.url or "").strip() or cameras)
        try:
            url = redact_rtsp_url(build_rtsp_url(self.rtsp)) if configured and rtsp_password(live) else ""
        except (RuntimeError, ValueError):
            url = ""
        return {
            "source": self.capture.source,
            "camera_id": live.camera_id,
            "cameras": [camera_public(cam, self.rtsp) for cam in cameras],
            "host": live.host,
            "port": live.port,
            "username": live.username,
            "has_password": bool(rtsp_password(live)),
            "channel": live.channel,
            "stream": live.stream,
            "stream_id": hik_channel(live.channel, live.stream) if live.host or live.url else 0,
            "stream_label": stream_label(live.stream, live.channel),
            "url": url,
            "transport": live.transport,
            "monitor_ids": [cam.id for cam in cameras if cam.monitor],
            "configured": configured,
        }

    def _upsert_camera(self, body: dict) -> RtspCamera:
        cameras = list_cameras(self.rtsp)
        camera_id = str(body.get("id") or "").strip()
        current = next((cam for cam in cameras if cam.id == camera_id), None) if camera_id else None
        if current is None and not camera_id and cameras and body.get("name") is None and body.get("host") is None:
            current = next((cam for cam in cameras if cam.id == self.rtsp.camera_id), cameras[0])
        if current is None:
            taken = {cam.id for cam in cameras}
            camera_id = camera_id or camera_slug(str(body.get("name") or ""), str(body.get("host") or ""), int(body.get("channel") or 1), taken)
            current = RtspCamera(id=camera_id)
            cameras.append(current)
        if body.get("name") is not None:
            current.name = str(body["name"]).strip()
        if body.get("host") is not None:
            current.host = str(body["host"]).strip()
        if body.get("port") is not None:
            port = int(body["port"])
            if not (1 <= port <= 65535):
                raise ValueError("RTSP 端口无效")
            current.port = port
        if body.get("username") is not None:
            current.username = str(body["username"]).strip() or "admin"
        if body.get("password"):
            current.password = str(body["password"])
        if body.get("channel") is not None:
            channel = int(body["channel"])
            if channel < 1:
                raise ValueError("通道号从 1 开始")
            current.channel = channel
        if body.get("stream") is not None:
            stream = str(body["stream"])
            if stream not in STREAM_CODES:
                raise ValueError("码流只能是主码流 / 子码流 / 第三码流")
            current.stream = stream
        if body.get("url") is not None:
            current.url = str(body["url"]).strip()
        if body.get("transport") is not None:
            transport = str(body["transport"])
            if transport not in ("tcp", "udp"):
                raise ValueError("传输只能是 tcp 或 udp")
            current.transport = transport
        if body.get("monitor") is not None:
            current.monitor = bool(body["monitor"])
        if not current.id:
            current.id = camera_slug(current.name, current.host, current.channel, {cam.id for cam in cameras if cam is not current})
        self.rtsp.cameras = cameras
        self.rtsp.camera_id = current.id
        return current

    def save(
        self,
        *,
        source: str | None = None,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        channel: int | None = None,
        stream: str | None = None,
        url: str | None = None,
        transport: str | None = None,
        camera_id: str | None = None,
        camera: dict | None = None,
        remove_camera_id: str | None = None,
        monitor_ids: list[str] | None = None,
    ) -> dict:
        if source is not None:
            if source not in ("auto", "usb", "camera", "file", "wifi", "rtsp"):
                raise ValueError("视频源只能是 auto / usb / camera / file / wifi / rtsp")
            self.capture.source = source
        if remove_camera_id:
            cameras = [cam for cam in list_cameras(self.rtsp) if cam.id != remove_camera_id]
            if not cameras and self.capture.source == "rtsp":
                raise ValueError("至少保留一台摄像机")
            self.rtsp.cameras = cameras
            if self.rtsp.camera_id == remove_camera_id:
                self.rtsp.camera_id = cameras[0].id if cameras else ""
        if camera is not None:
            self._upsert_camera(camera)
        elif any(v is not None for v in (host, port, username, password, channel, stream, url, transport)):
            self._upsert_camera(
                {
                    "id": camera_id or self.rtsp.camera_id,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "channel": channel,
                    "stream": stream,
                    "url": url,
                    "transport": transport,
                }
            )
        if camera_id is not None and camera is None and not remove_camera_id:
            ids = {cam.id for cam in list_cameras(self.rtsp)}
            if camera_id and camera_id not in ids:
                raise ValueError("找不到这台摄像机")
            self.rtsp.camera_id = camera_id
        if monitor_ids is not None:
            wanted = {str(item) for item in monitor_ids}
            cameras = list_cameras(self.rtsp)
            if self.capture.source == "rtsp" and cameras and not wanted:
                raise ValueError("至少选一路监测")
            unknown = wanted - {cam.id for cam in cameras}
            if unknown:
                raise ValueError("找不到摄像机 " + "、".join(sorted(unknown)))
            for cam in cameras:
                cam.monitor = cam.id in wanted
            self.rtsp.cameras = cameras
        self._sync_active()
        if self.capture.source == "rtsp" and not list_cameras(self.rtsp):
            raise ValueError("用局域网监测时请先添加摄像机")
        if self.capture.source == "rtsp" and not (self.rtsp.host or "").strip() and not (self.rtsp.url or "").strip():
            raise ValueError("用局域网监测时请填写摄像机 IP")
        payload = {
            "source": self.capture.source,
            "camera_id": self.rtsp.camera_id,
            "cameras": [cam.model_dump() for cam in list_cameras(self.rtsp)],
            "host": self.rtsp.host,
            "port": self.rtsp.port,
            "username": self.rtsp.username,
            "password": self.rtsp.password,
            "channel": self.rtsp.channel,
            "stream": self.rtsp.stream,
            "url": self.rtsp.url,
            "transport": self.rtsp.transport,
        }
        if self.path is not None:
            _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2))
            self._mtime = self.path.stat().st_mtime
        return self.public()

    def reload(self, force: bool = False) -> bool:
        if self.path is None or not self.path.exists():
            return False
        mtime = self.path.stat().st_mtime
        if not force and mtime == self._mtime:
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        before = self.identity()
        if data.get("source") in ("auto", "usb", "camera", "file", "wifi", "rtsp"):
            self.capture.source = data["source"]
        if data.get("camera_id") is not None:
            self.rtsp.camera_id = str(data["camera_id"]).strip()
        if isinstance(data.get("cameras"), list):
            cameras: list[RtspCamera] = []
            for item in data["cameras"]:
                if not isinstance(item, dict):
                    continue
                try:
                    cameras.append(RtspCamera.model_validate(item))
                except ValueError:
                    continue
            self.rtsp.cameras = cameras
        if "host" in data and data["host"] is not None:
            self.rtsp.host = str(data["host"]).strip()
        if data.get("port"):
            self.rtsp.port = int(data["port"])
        if data.get("username"):
            self.rtsp.username = str(data["username"]).strip()
        if data.get("password"):
            self.rtsp.password = str(data["password"])
        if data.get("channel"):
            self.rtsp.channel = int(data["channel"])
        if data.get("stream") in STREAM_CODES:
            self.rtsp.stream = data["stream"]
        if "url" in data and data["url"] is not None:
            self.rtsp.url = str(data["url"]).strip()
        if data.get("transport") in ("tcp", "udp"):
            self.rtsp.transport = data["transport"]
        self._merge_seed()
        self._sync_active()
        self._mtime = mtime
        return self.identity() != before

    def _merge_seed(self) -> None:
        have = {cam.id for cam in self.rtsp.cameras}
        extra = [cam.model_copy(deep=True) for cam in self._seed_cameras if cam.id and cam.id not in have]
        if extra:
            self.rtsp.cameras = list_cameras(self.rtsp) + extra

    def maybe_reload(self) -> bool:
        return self.reload(force=False)


# Pocket 3 UVC 支持的边长。乱设 1920x1080 + MJPG 会得到 1080x608 撕裂帧。
_VALID_DIMS = {720, 1080, 1280, 1512, 1920, 2160, 2688, 3072, 3840}


def _frame_ok(frame: np.ndarray | None) -> bool:
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    return w in _VALID_DIMS and h in _VALID_DIMS


def _read_ok(cap: cv2.VideoCapture, attempts: int = 15, delay_s: float = 0.05) -> np.ndarray | None:
    """macOS AVFoundation 刚打开时前几帧 read() 经常失败，不能只试一次。"""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and _frame_ok(frame):
            return frame
        time.sleep(delay_s)
    return None


def _open_index(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, _avfoundation_backend())
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = _read_ok(cap)
    if frame is None:
        cap.release()
        return None
    h, w = frame.shape[:2]
    if (w, h) != (width, height) and (h, w) != (width, height):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        frame2 = _read_ok(cap)
        if frame2 is None:
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
            if _read_ok(cap) is None:
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


def try_open_pocket(
    cfg: CaptureConfig,
    current_label: str = "",
    *,
    force: bool = False,
) -> OpenCvCapture | None:
    """Pocket 3 刚开机出现在系统相机列表时，切过去。"""
    if looks_like_pocket(current_label) and not force:
        return None
    if not pocket3_usb_present(ttl=2.0):
        return None
    try:
        return open_usb_or_camera(cfg)
    except RuntimeError:
        logger.warning("检测到 Pocket 3 但暂时打不开")
        return None


def open_file(path: str) -> OpenCvCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频文件: {path}")
    return OpenCvCapture(cap, path, kind="file")


def _read_any(cap: cv2.VideoCapture, attempts: int = 50, delay_s: float = 0.1) -> np.ndarray | None:
    """多等几秒，避开海康 GOP 开头那些没 I 帧的花屏。"""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size and not looks_corrupt(frame):
            return frame
        time.sleep(delay_s)
    return None


def _ffprobe_size(url: str, transport: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-rtsp_transport",
        transport,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=12)
    except FileNotFoundError as exc:
        raise RuntimeError("需要 ffprobe（随 ffmpeg 安装）才能回退拉取 RTSP") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("探测 RTSP 分辨率超时，检查地址、账号和网络") from exc
    text = (result.stdout or "").strip()
    if "x" not in text:
        raise RuntimeError("探测不到 RTSP 分辨率，检查地址或改用 OpenCV 能打开的码流")
    width_s, height_s = text.split("x", 1)
    width, height = int(width_s), int(height_s)
    if width < 16 or height < 16:
        raise RuntimeError(f"RTSP 分辨率异常 {width}x{height}")
    return width, height


def camera_as_rtsp(shared: RtspConfig, cam: RtspCamera) -> RtspConfig:
    return shared.model_copy(
        update={
            "camera_id": cam.id,
            "cameras": [cam],
            "host": cam.host or shared.host,
            "port": cam.port,
            "username": cam.username or shared.username,
            "password": cam.password or shared.password,
            "channel": cam.channel,
            "stream": cam.stream,
            "url": cam.url or shared.url,
            "transport": cam.transport or shared.transport,
        }
    )


def reopen_rtsp(cfg: RtspConfig, camera_id: str) -> FrameSource:
    cameras = [cam for cam in list_cameras(cfg) if cam.id == camera_id]
    if not cameras:
        raise RuntimeError(f"找不到摄像机 {camera_id}")
    return open_rtsp(camera_as_rtsp(cfg, cameras[0]))


def open_rtsp_many(cfg: RtspConfig, only_id: str | None = None) -> list[tuple[RtspCamera, FrameSource]]:
    cameras = list_cameras(cfg)
    if only_id:
        cameras = [cam for cam in cameras if cam.id == only_id]
        if not cameras:
            raise RuntimeError(f"找不到摄像机 {only_id}")
    else:
        cameras = [cam for cam in cameras if cam.monitor]
        if not cameras:
            raise RuntimeError("还没选要监测的摄像机")
    opened: list[tuple[RtspCamera, FrameSource]] = []
    errors: list[str] = []
    for cam in cameras:
        try:
            opened.append((cam, open_rtsp(camera_as_rtsp(cfg, cam))))
        except Exception as exc:
            errors.append(f"{cam.name or cam.id}: {exc}")
            logger.exception("打不开 %s", cam.name or cam.id)
    if not opened:
        raise RuntimeError("没有一路摄像机能打开。" + (" ".join(errors) if errors else ""))
    if errors:
        logger.warning("部分摄像机未打开：%s", "；".join(errors))
    return opened


def grid_layout(count: int) -> tuple[int, int]:
    """返回 (行, 列)，路数多时优先横向铺开，避免宽屏挤在左边。"""
    n = max(1, count)
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 3, 4
    return 3, 5


def native_mosaic_size(
    images: list[np.ndarray],
    fallback: tuple[int, int] = (1280, 720),
) -> tuple[int, int]:
    """按源画面拼宫格，不放大、不跟着窗口拉伸。"""
    sizes = [(im.shape[1], im.shape[0]) for im in images if im is not None and getattr(im, "size", 0)]
    if not sizes:
        return fallback
    cell_w = max(width for width, _height in sizes)
    cell_h = max(height for _width, height in sizes)
    rows, cols = grid_layout(len(images))
    return max(16, cell_w) * cols, max(16, cell_h) * rows


def pane_from_norm(
    nx: float,
    ny: float,
    count: int,
    *,
    canvas: tuple[int, int] | list[int] | None = None,
    cameras: list[dict] | None = None,
) -> tuple[int, float, float] | None:
    """宫格归一化点击 → (路序号, 源画面 nx, 源画面 ny)。"""
    if count <= 0:
        return None
    try:
        nx = float(nx)
        ny = float(ny)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
        return None
    rows, cols = grid_layout(count)
    col = cols - 1 if nx >= 1.0 else min(cols - 1, int(nx * cols))
    row = rows - 1 if ny >= 1.0 else min(rows - 1, int(ny * rows))
    index = row * cols + col
    if index >= count:
        return None
    local_x = nx * cols - col
    local_y = ny * rows - row
    cam = cameras[index] if cameras and index < len(cameras) else None
    src_w = int((cam or {}).get("src_w") or 0)
    src_h = int((cam or {}).get("src_h") or 0)
    if src_w > 0 and src_h > 0:
        if canvas is not None and len(canvas) >= 2:
            cell_w, cell_h = grid_cell_size(count, (int(canvas[0]), int(canvas[1])))
        else:
            cell_w, cell_h = grid_cell_size(count)
        mapped = letterbox_to_source(local_x * cell_w, local_y * cell_h, cell_w, cell_h, src_w, src_h)
        if mapped is None:
            return None
        return index, mapped[0] / src_w, mapped[1] / src_h
    return index, max(0.0, min(1.0, local_x)), max(0.0, min(1.0, local_y))


def grid_cell_size(count: int, canvas: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    rows, cols = grid_layout(count)
    width = max(16, int(canvas[0]) // cols)
    height = max(16, int(canvas[1]) // rows)
    return width, height


def grid_origin(count: int, canvas: tuple[int, int] = (1920, 1080)) -> tuple[int, int, int, int]:
    """宫格左上角和单格尺寸。余数居中，避免整幅再拉伸一次。"""
    rows, cols = grid_layout(count)
    cell_w, cell_h = grid_cell_size(count, canvas)
    canvas_w = max(320, int(canvas[0]))
    canvas_h = max(240, int(canvas[1]))
    ox = (canvas_w - cell_w * cols) // 2
    oy = (canvas_h - cell_h * rows) // 2
    return ox, oy, cell_w, cell_h


def letterbox_geometry(
    src_w: int,
    src_h: int,
    cell_w: int,
    cell_h: int,
) -> tuple[int, int, int, int]:
    """源画面放进格子的位置。只缩小不放大，空余留黑边。"""
    cell_w = max(1, int(cell_w))
    cell_h = max(1, int(cell_h))
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    scale = min(1.0, cell_w / src_w, cell_h / src_h)
    nw = max(1, min(cell_w, int(round(src_w * scale))))
    nh = max(1, min(cell_h, int(round(src_h * scale))))
    return (cell_w - nw) // 2, (cell_h - nh) // 2, nw, nh


def letterbox_into(image: np.ndarray, cell_w: int, cell_h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """按原分辨率放进格子，大于格子才缩小。返回 (格子图, (x0, y0, nw, nh))。"""
    cell_w = max(1, int(cell_w))
    cell_h = max(1, int(cell_h))
    tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if image is None or image.size == 0:
        return tile, (0, 0, 0, 0)
    src_h, src_w = image.shape[:2]
    if src_w < 1 or src_h < 1:
        return tile, (0, 0, 0, 0)
    x0, y0, nw, nh = letterbox_geometry(src_w, src_h, cell_w, cell_h)
    if nw == src_w and nh == src_h:
        resized = image
    else:
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    tile[y0 : y0 + nh, x0 : x0 + nw] = resized
    return tile, (x0, y0, nw, nh)


def letterbox_to_source(
    local_x: float,
    local_y: float,
    cell_w: int,
    cell_h: int,
    src_w: int,
    src_h: int,
) -> tuple[int, int] | None:
    """格子内像素 → 源画面像素。点到黑边返回 None。"""
    if src_w < 1 or src_h < 1 or cell_w < 1 or cell_h < 1:
        return None
    x0, y0, nw, nh = letterbox_geometry(src_w, src_h, cell_w, cell_h)
    if not (x0 <= local_x < x0 + nw and y0 <= local_y < y0 + nh):
        return None
    ox = int((local_x - x0) * src_w / nw)
    oy = int((local_y - y0) * src_h / nh)
    return max(0, min(src_w - 1, ox)), max(0, min(src_h - 1, oy))


def compose_grid(images: list[np.ndarray], canvas: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    canvas_w = max(320, int(canvas[0]))
    canvas_h = max(240, int(canvas[1]))
    grid = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    if not images:
        return grid
    ox, oy, cell_w, cell_h = grid_origin(len(images), (canvas_w, canvas_h))
    cols = grid_layout(len(images))[1]
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        tile, _ = letterbox_into(image, cell_w, cell_h)
        y = oy + row * cell_h
        x = ox + col * cell_w
        grid[y : y + cell_h, x : x + cell_w] = tile
    return grid


def pane_index(x: int, y: int, count: int, canvas: tuple[int, int] = (1920, 1080)) -> int | None:
    if count <= 0:
        return None
    rows, cols = grid_layout(count)
    ox, oy, cell_w, cell_h = grid_origin(count, canvas)
    if x < ox or y < oy:
        return None
    col = (x - ox) // cell_w
    row = (y - oy) // cell_h
    if not (0 <= col < cols and 0 <= row < rows):
        return None
    index = row * cols + col
    if 0 <= index < count:
        return index
    return None


def pane_source_xy(
    x: int,
    y: int,
    count: int,
    canvas: tuple[int, int],
    src_w: int,
    src_h: int,
) -> tuple[int, int, int] | None:
    """窗口点击 → (路序号, 源图 x, 源图 y)。"""
    index = pane_index(x, y, count, canvas)
    if index is None:
        return None
    ox, oy, cell_w, cell_h = grid_origin(count, canvas)
    col = (x - ox) // max(cell_w, 1)
    row = (y - oy) // max(cell_h, 1)
    mapped = letterbox_to_source(x - ox - col * cell_w, y - oy - row * cell_h, cell_w, cell_h, src_w, src_h)
    if mapped is None:
        return None
    return index, mapped[0], mapped[1]


def open_rtsp(cfg: RtspConfig) -> FrameSource:
    live = active_rtsp(cfg)
    url = build_rtsp_url(cfg)
    cam = pick_camera(cfg)
    name = (cam.name if cam else "") or live.host or "url"
    label = f"RTSP {name} {stream_label(live.stream, live.channel)}"
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{live.transport}")
    cap = cv2.VideoCapture(url, getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = _read_any(cap)
    if frame is not None:
        height, width = frame.shape[:2]
        logger.info("打开 %s %sx%s", redact_rtsp_url(url), width, height)
        return OpenCvCapture(cap, label, kind="rtsp", seed=frame)
    cap.release()
    logger.warning("OpenCV 打不开 RTSP，改用 ffmpeg %s", redact_rtsp_url(url))
    width, height = _ffprobe_size(url, live.transport)
    return FfmpegRtspCapture(url, width, height, live.transport, label)


def open_capture(
    cfg: CaptureConfig,
    wifi_cfg: WifiConfig,
    client: DjiUdpClient | None = None,
    rtsp_cfg: RtspConfig | None = None,
) -> FrameSource:
    source = cfg.source
    if source == "file" or (cfg.file and source == "auto"):
        if not cfg.file:
            raise RuntimeError("capture.file 未设置")
        return open_file(cfg.file)
    if source == "rtsp":
        if rtsp_cfg is None:
            raise RuntimeError("RTSP 取流需要 rtsp 配置")
        return open_rtsp(rtsp_cfg)
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
