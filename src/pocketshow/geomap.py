from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from pocketshow.config import MapConfig
from pocketshow.gallery import encode_jpeg
from pocketshow.scene import SceneLog
from pocketshow.types import Track

logger = logging.getLogger("pocketshow.geomap")

_FRESH_S = 8.0
_TRAIL = 80


def xyz_tuple(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError):
        return None


def query_points(tracks: list[Track], width: int, height: int) -> list[dict]:
    width = max(1, int(width))
    height = max(1, int(height))
    out: list[dict] = []
    for track in tracks:
        if track.live is False:
            continue
        x1, y1, x2, y2 = (float(v) for v in track.bbox_xyxy)
        out.append(
            {
                "id": track.person_id or str(track.id),
                "track_id": track.id,
                "name": track.person_name or "",
                "u": max(0.0, min(1.0, ((x1 + x2) * 0.5) / width)),
                "v": max(0.0, min(1.0, y2 / height)),
            }
        )
    return out


def describe_pose(note: "MapNote | None") -> str:
    if note is None or not note.ready:
        if note is not None and note.warming:
            return f"三维重建预热 {note.buffered}/{max(note.need, 1)}"
        return ""
    x, y, z = note.camera_xyz or (0.0, 0.0, 0.0)
    people = len(note.people)
    return f"3D {note.frames}帧 相机({x:.1f},{y:.1f},{z:.1f}) {people}人"


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body) if body else {}
    if not isinstance(data, dict):
        raise RuntimeError("LingBot-Map 返回不是对象")
    return data


def encode_frame(frame: np.ndarray, *, max_width: int, quality: int) -> bytes:
    vis = frame
    h, w = vis.shape[:2]
    if w > max_width:
        scale = max_width / w
        import cv2

        vis = cv2.resize(
            vis,
            (max_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return encode_jpeg(vis, quality=quality)


def stub_infer(points: list[dict], frame_index: int) -> dict:
    people = []
    for item in points:
        u = float(item.get("u") or 0.5)
        v = float(item.get("v") or 0.5)
        xyz = (round((u - 0.5) * 4.0, 3), 1.4, round(1.6 + v * 2.4, 3))
        people.append(
            {
                "id": item.get("id") or "",
                "track_id": item.get("track_id"),
                "name": item.get("name") or "",
                "u": u,
                "v": v,
                "xyz": list(xyz),
                "depth": xyz[2],
                "conf": 2.0,
            }
        )
    t = frame_index * 0.05
    return {
        "ready": frame_index >= 2,
        "warming": frame_index < 2,
        "buffered": min(frame_index + 1, 2),
        "need": 2,
        "frame_index": frame_index,
        "keyframe": True,
        "camera_xyz": [round(t, 3), 1.5, 0.0],
        "extrinsic": [[1, 0, 0, t], [0, 1, 0, 1.5], [0, 0, 1, 0]],
        "intrinsic": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "points": people,
    }


@dataclass
class MapNote:
    camera_id: str
    camera: str
    ready: bool = False
    warming: bool = False
    buffered: int = 0
    need: int = 0
    frames: int = 0
    camera_xyz: tuple[float, float, float] | None = None
    people: list[dict] = field(default_factory=list)
    trail: list[list[float]] = field(default_factory=list)
    t: float = 0.0
    error: str = ""

    def public(self) -> dict:
        return {
            "id": self.camera_id,
            "name": self.camera,
            "ready": self.ready,
            "warming": self.warming,
            "buffered": self.buffered,
            "need": self.need,
            "frames": self.frames,
            "camera_xyz": list(self.camera_xyz) if self.camera_xyz else None,
            "people": list(self.people),
            "trail": list(self.trail[-40:]),
            "t": self.t,
            "error": self.error,
            "line": describe_pose(self),
        }


@dataclass
class _Job:
    camera_id: str
    camera: str
    jpeg: bytes
    points: list[dict]


class SceneMapper:
    """旁路三维重建：把帧送给 LingBot-Map，位姿/深度只进 HUD，不进跟拍。"""

    def __init__(self, cfg: MapConfig, *, inline: bool = False) -> None:
        self.cfg = cfg
        self.log = SceneLog(cfg.status, cfg.log)
        self._inline = inline
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._notes: dict[str, MapNote] = {}
        self._last_offer: dict[str, float] = {}
        self._home = ""
        self._busy = False
        self._error = ""
        self._frames: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if cfg.enabled and not inline:
            self._thread = threading.Thread(target=self._loop, name="pocketshow-map", daemon=True)
            self._thread.start()

    def wants(self, camera_id: str) -> bool:
        if not self.cfg.enabled:
            return False
        wanted = (self.cfg.camera_id or "").strip()
        if wanted:
            return camera_id == wanted
        with self._lock:
            if not self._home:
                self._home = camera_id
            return camera_id == self._home

    def offer(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        *,
        camera_id: str = "",
        camera_name: str = "",
        now: float | None = None,
    ) -> bool:
        if not self.cfg.enabled or frame is None or frame.size == 0:
            return False
        cam_id = camera_id or camera_name or "live"
        if not self.wants(cam_id):
            return False
        stamp = time.monotonic() if now is None else now
        with self._lock:
            if self._busy and not self._inline:
                return False
            last = self._last_offer.get(cam_id)
            if last is not None and stamp - last < self.cfg.interval_s:
                return False
        h, w = frame.shape[:2]
        try:
            jpeg = encode_frame(frame, max_width=self.cfg.max_width, quality=self.cfg.jpeg_quality)
        except Exception:
            logger.exception("三维重建抽帧失败")
            return False
        job = _Job(
            camera_id=cam_id,
            camera=camera_name or cam_id,
            jpeg=jpeg,
            points=query_points(tracks, w, h),
        )
        if self._inline:
            with self._lock:
                self._last_offer[cam_id] = stamp
            self._run(job)
            return True
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            return False
        with self._lock:
            self._last_offer[cam_id] = stamp
        return True

    def annotate(self, tracks: list[Track], camera_id: str = "") -> None:
        with self._lock:
            note = self._notes.get(camera_id or self._home or "live")
            people = list(note.people) if note is not None else []
        if not people:
            return
        by_id = {str(item.get("id") or ""): item for item in people if item.get("id") is not None}
        by_track = {}
        for item in people:
            try:
                by_track[int(item.get("track_id"))] = item
            except (TypeError, ValueError):
                continue
        for track in tracks:
            item = None
            if track.person_id:
                item = by_id.get(track.person_id)
            if item is None:
                item = by_track.get(track.id)
            if item is None:
                continue
            xyz = xyz_tuple(item.get("xyz"))
            if xyz is None:
                continue
            track.xyz = xyz
            try:
                track.depth = float(item.get("depth") or xyz[2])
            except (TypeError, ValueError):
                track.depth = xyz[2]

    def line_for(self, camera_id: str = "") -> str:
        with self._lock:
            note = self._notes.get(camera_id or self._home or "live")
            if note is None and len(self._notes) == 1:
                note = next(iter(self._notes.values()))
            return describe_pose(note)

    def public(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            cameras = [note.public() for note in self._notes.values()]
            error = self._error
            busy = self._busy
            updated = max((note.t for note in self._notes.values()), default=0.0)
        age = max(0.0, now - updated) if updated else None
        return {
            "enabled": self.cfg.enabled,
            "backend": self.cfg.backend,
            "fresh": bool(updated and age is not None and age <= _FRESH_S),
            "updated": updated,
            "age_s": round(age, 1) if age is not None else None,
            "error": error,
            "busy": busy,
            "cameras": cameras,
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._run(job)
            except Exception:
                logger.exception("三维重建线程异常")
            finally:
                self._queue.task_done()

    def _run(self, job: _Job) -> None:
        with self._lock:
            self._busy = True
            index = self._frames.get(job.camera_id, 0)
            self._frames[job.camera_id] = index + 1
        wall = time.time()
        try:
            payload = self._infer(job, index)
            error = ""
        except Exception as exc:
            payload = {}
            error = str(exc)
            logger.warning("三维重建失败：%s", error)
        note = self._note_from(job, payload, wall, error)
        with self._lock:
            self._notes[job.camera_id] = note
            self._error = error
            self._busy = False
        snapshot = self.public(now=wall)
        self.log.write_status(snapshot)
        if error or not note.ready:
            return
        if index % 20 == 0:
            self.log.append(
                {
                    "event": "map",
                    "t": wall,
                    "camera_id": job.camera_id,
                    "camera": job.camera,
                    "frames": note.frames,
                    "camera_xyz": list(note.camera_xyz) if note.camera_xyz else None,
                    "people": len(note.people),
                }
            )

    def _infer(self, job: _Job, index: int) -> dict:
        if self.cfg.backend == "stub":
            return stub_infer(job.points, index)
        url = self.cfg.base_url.rstrip("/") + "/v1/map/frame"
        return post_json(
            url,
            {
                "camera_id": job.camera_id,
                "timestamp": index * max(self.cfg.interval_s, 0.05),
                "jpeg_b64": base64.b64encode(job.jpeg).decode("ascii"),
                "points": job.points,
            },
            self.cfg.timeout_s,
        )

    def _note_from(self, job: _Job, payload: dict, wall: float, error: str) -> MapNote:
        with self._lock:
            prev = self._notes.get(job.camera_id)
        xyz = xyz_tuple(payload.get("camera_xyz"))
        trail = list(prev.trail) if prev is not None else []
        if xyz is not None:
            trail.append([round(xyz[0], 3), round(xyz[1], 3), round(xyz[2], 3)])
            trail = trail[-_TRAIL:]
        people = [item for item in (payload.get("points") or []) if isinstance(item, dict)]
        return MapNote(
            camera_id=job.camera_id,
            camera=job.camera,
            ready=bool(payload.get("ready")),
            warming=bool(payload.get("warming")),
            buffered=int(payload.get("buffered") or 0),
            need=int(payload.get("need") or 0),
            frames=int(payload.get("frame_index") or 0) + 1,
            camera_xyz=xyz,
            people=people,
            trail=trail,
            t=wall,
            error=error,
        )
