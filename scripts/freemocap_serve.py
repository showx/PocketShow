#!/usr/bin/env python3
"""FreeMoCap / skellytracker HTTP 适配，给 PocketShow mocap 客户端用。

在 FreeMoCap 的 uv 环境里运行（不要装进 PocketShow 自己的 venv）：

    # 先按 https://github.com/freemocap/freemocap 装好 skellytracker
    python /path/to/PocketShow/scripts/freemocap_serve.py --port 8006

默认 MediaPipe 姿态（CPU / Apple Silicon 都行）。NVIDIA GPU 可改：

    python scripts/freemocap_serve.py --tracker rtmpose --port 8006
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

logger = logging.getLogger("freemocap_serve")

_MIN_CROP = 48
_PAD = 0.18
_MIN_CONF = 0.2

BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]


def _decode_jpeg(jpeg: bytes) -> np.ndarray:
    from PIL import Image

    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _keypoints_from(kpts, width: int, height: int, ox: float = 0.0, oy: float = 0.0) -> list[dict]:
    if kpts is None:
        return []
    xyz = np.asarray(getattr(kpts, "xyz", kpts))
    names = list(getattr(kpts, "names", []) or [])
    vis = getattr(kpts, "visibility", None)
    vis = np.asarray(vis) if vis is not None else None
    if xyz.ndim == 2:
        xyz = xyz[None, ...]
    if xyz.ndim != 3:
        return []
    people, count, _ = xyz.shape
    if not names:
        names = [f"kpt_{i}" for i in range(count)]
    best = 0
    if people > 1 and vis is not None:
        raw = vis[None, ...] if vis.ndim == 1 else vis
        scores = [float(np.nanmean(raw[i])) for i in range(min(people, raw.shape[0]))]
        if scores:
            best = int(np.argmax(scores))
    row = xyz[min(best, people - 1)]
    conf_row = None
    if vis is not None:
        raw = vis[None, ...] if vis.ndim == 1 else vis
        if raw.ndim >= 2:
            conf_row = raw[min(best, raw.shape[0] - 1)]
        elif raw.ndim == 1:
            conf_row = raw
    xs, ys = row[:, 0].astype(float), row[:, 1].astype(float)
    if xs.size == 0:
        return []
    normalized = bool(np.nanmax(xs) <= 1.5 and np.nanmax(ys) <= 1.5)
    out: list[dict] = []
    for i, name in enumerate(names[:count]):
        conf = 1.0
        if conf_row is not None and i < len(conf_row):
            try:
                conf = float(conf_row[i])
            except (TypeError, ValueError):
                conf = 1.0
        if conf < _MIN_CONF:
            continue
        x, y = float(xs[i]), float(ys[i])
        px, py = (x * width, y * height) if normalized else (x, y)
        out.append(
            {
                "name": str(name),
                "u": round(px + ox, 6),
                "v": round(py + oy, 6),
                "conf": round(conf, 3),
            }
        )
    return out


def _normalize_to_frame(points: list[dict], frame_w: int, frame_h: int) -> list[dict]:
    out = []
    for item in points:
        u, v = float(item["u"]), float(item["v"])
        # 若还是像素，除以整帧尺寸
        if u > 1.5 or v > 1.5:
            u = u / max(frame_w, 1)
            v = v / max(frame_h, 1)
        out.append(
            {
                "name": item["name"],
                "u": round(max(0.0, min(1.0, u)), 4),
                "v": round(max(0.0, min(1.0, v)), 4),
                "conf": item["conf"],
            }
        )
    return out


def _crop_box(frame: np.ndarray, person: dict) -> tuple[np.ndarray, int, int] | None:
    h, w = frame.shape[:2]
    try:
        x1 = float(person["x1"]) * w
        y1 = float(person["y1"]) * h
        x2 = float(person["x2"]) * w
        y2 = float(person["y2"]) * h
    except (KeyError, TypeError, ValueError):
        return None
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad_x, pad_y = bw * _PAD, bh * _PAD
    xa = max(0, int(x1 - pad_x))
    ya = max(0, int(y1 - pad_y))
    xb = min(w, int(x2 + pad_x))
    yb = min(h, int(y2 + pad_y))
    if xb - xa < _MIN_CROP or yb - ya < _MIN_CROP:
        return None
    return frame[ya:yb, xa:xb].copy(), xa, ya


class PoseEngine:
    def __init__(self, tracker_name: str) -> None:
        self.tracker_name = tracker_name
        self.lock = threading.Lock()
        self.frames = 0
        self.camera_id = ""
        self._tracker = None
        self._session = None
        self._state_cls = None
        self._states: dict[str, object] = {}
        self._build(tracker_name)

    def _build(self, tracker_name: str) -> None:
        try:
            from skellytracker.core import TrackerState
        except ImportError as exc:
            raise SystemExit(
                "找不到 skellytracker。请在 FreeMoCap 环境里运行本脚本：\n"
                "  pip install 'skellytracker[recommended-cpu]'\n"
                "或按 https://github.com/freemocap/freemocap 用 uv sync"
            ) from exc
        self._state_cls = TrackerState
        name = (tracker_name or "mediapipe").strip().lower()
        if name == "rtmpose":
            self._build_rtmpose()
        else:
            self._build_mediapipe()
        logger.info("已加载 skellytracker：%s", name)

    def _build_mediapipe(self) -> None:
        from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig

        try:
            from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
                MediaPipeSession,
                MediaPipeSessionConfig,
                MediapipePoseDetectorConfig,
            )
        except ImportError:
            from skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector import (
                MediapipePoseDetectorConfig,
            )
            from skellytracker.core.sessions.mediapipe_session import (
                MediaPipeSession,
                MediaPipeSessionConfig,
            )

        session = MediaPipeSession.create(MediaPipeSessionConfig())
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="body",
                    keypoint_detectors=[MediapipePoseDetectorConfig()],
                )
            ]
        )
        self._session = session
        self._tracker = Tracker.create(config, sessions={"mediapipe": session})

    def _build_rtmpose(self) -> None:
        from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig
        from skellytracker.core.detectors.keypoint_detectors.rtmpose import (
            RTMPoseDetectorConfig,
            RTMPoseKeypointDetector,
        )
        from skellytracker.core.detectors.object_detectors.yolox import (
            YoloxPersonDetector,
            YoloxPersonDetectorConfig,
        )
        from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig

        session = OnnxSession.create(
            OnnxSessionConfig(
                batch_size=1,
                models=[
                    YoloxPersonDetector.model_spec("yolox-m"),
                    RTMPoseKeypointDetector.model_spec("rtmw-x-l_256x192"),
                ],
            )
        )
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="body",
                    object_detector=YoloxPersonDetectorConfig(),
                    keypoint_detectors=[RTMPoseDetectorConfig()],
                )
            ]
        )
        self._session = session
        self._tracker = Tracker.create(config, sessions={"onnx": session})

    def _pose(self, image: np.ndarray, key: str) -> object:
        state = self._states.get(key)
        if state is None:
            state = self._state_cls()
        observation, state = self._tracker.process_image(image, frame_number=self.frames, state=state)
        self._states[key] = state
        if len(self._states) > 64:
            extra = list(self._states)[:16]
            for old in extra:
                self._states.pop(old, None)
        return observation

    def step(self, jpeg: bytes, camera_id: str, people: list[dict]) -> dict:
        frame = _decode_jpeg(jpeg)
        h, w = frame.shape[:2]
        with self.lock:
            if camera_id:
                self.camera_id = camera_id
            self.frames += 1
            posed: list[dict] = []
            if people:
                for person in people:
                    crop = _crop_box(frame, person)
                    if crop is None:
                        continue
                    patch, ox, oy = crop
                    ph, pw = patch.shape[:2]
                    key = f"{camera_id}:{person.get('id') or person.get('track_id') or len(posed)}"
                    try:
                        obs = self._pose(patch, key)
                    except Exception:
                        logger.exception("姿态推理失败")
                        continue
                    body = getattr(obs, "stages", {}).get("body") if obs is not None else None
                    kpts = getattr(body, "keypoints", None) if body is not None else None
                    raw = _keypoints_from(kpts, pw, ph, ox, oy)
                    mapped = _normalize_to_frame(raw, w, h)
                    if not mapped:
                        continue
                    posed.append(
                        {
                            "id": person.get("id") or "",
                            "track_id": person.get("track_id"),
                            "name": person.get("name") or "",
                            "keypoints": mapped,
                            "bones": [list(pair) for pair in BONES],
                        }
                    )
            else:
                obs = self._pose(frame, camera_id or "full")
                body = getattr(obs, "stages", {}).get("body") if obs is not None else None
                kpts = getattr(body, "keypoints", None) if body is not None else None
                mapped = _normalize_to_frame(_keypoints_from(kpts, w, h), w, h)
                if mapped:
                    posed.append({"id": "", "track_id": None, "name": "", "keypoints": mapped, "bones": [list(p) for p in BONES]})
            return {"ready": True, "people": posed, "frame_index": self.frames}

    def close(self) -> None:
        with self.lock:
            if self._tracker is not None:
                try:
                    self._tracker.close()
                except Exception:
                    logger.exception("关闭 tracker 失败")


def make_handler(engine: PoseEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] in {"/health", "/v1/mocap/health"}:
                self._send(200, {"ok": True, "camera_id": engine.camera_id, "frames": engine.frames})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json()
                if path == "/v1/mocap/frame":
                    b64 = body.get("jpeg_b64") or ""
                    jpeg = base64.b64decode(b64)
                    if not jpeg:
                        self._send(400, {"error": "缺少 jpeg_b64"})
                        return
                    result = engine.step(
                        jpeg,
                        str(body.get("camera_id") or ""),
                        [item for item in (body.get("people") or []) if isinstance(item, dict)],
                    )
                    self._send(200, result)
                    return
            except Exception as exc:
                logger.exception("处理失败")
                self._send(500, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FreeMoCap / skellytracker HTTP 服务（PocketShow 旁路）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--tracker", choices=["mediapipe", "rtmpose"], default="mediapipe")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    engine = PoseEngine(args.tracker)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    logger.info("FreeMoCap 姿态服务 http://%s:%s/v1/mocap/frame （%s）", args.host, args.port, args.tracker)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
