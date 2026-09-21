from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from pocketshow.config import MocapConfig
from pocketshow.geomap import encode_frame, post_json
from pocketshow.scene import SceneLog
from pocketshow.types import Track

logger = logging.getLogger("pocketshow.mocap")

_FRESH_S = 6.0
_MIN_CONF = 0.25


def query_people(tracks: list[Track], width: int, height: int) -> list[dict]:
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
                "v": max(0.0, min(1.0, ((y1 + y2) * 0.5) / height)),
                "x1": max(0.0, min(1.0, x1 / width)),
                "y1": max(0.0, min(1.0, y1 / height)),
                "x2": max(0.0, min(1.0, x2 / width)),
                "y2": max(0.0, min(1.0, y2 / height)),
            }
        )
    return out


def _named(keypoints: list[dict]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for item in keypoints:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            conf = float(item.get("conf") if item.get("conf") is not None else 1.0)
            out[name] = (float(item["u"]), float(item["v"]), conf)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _avg_y(named: dict[str, tuple[float, float, float]], *names: str) -> float | None:
    ys = [named[n][1] for n in names if n in named and named[n][2] >= _MIN_CONF]
    if not ys:
        return None
    return sum(ys) / len(ys)


def classify_activity(keypoints: list[dict]) -> str:
    named = _named(keypoints)
    shoulder = _avg_y(named, "left_shoulder", "right_shoulder")
    hip = _avg_y(named, "left_hip", "right_hip")
    knee = _avg_y(named, "left_knee", "right_knee")
    ankle = _avg_y(named, "left_ankle", "right_ankle")
    lw = named.get("left_wrist")
    rw = named.get("right_wrist")
    ls = named.get("left_shoulder")
    rs = named.get("right_shoulder")
    torso = abs(hip - shoulder) if hip is not None and shoulder is not None else 0.12
    torso = max(torso, 0.04)
    if (lw and ls and lw[2] >= _MIN_CONF and lw[1] < ls[1] - torso * 0.35) or (
        rw and rs and rw[2] >= _MIN_CONF and rw[1] < rs[1] - torso * 0.35
    ):
        return "举手"
    if hip is not None and knee is not None and abs(knee - hip) < torso * 0.75:
        return "坐着"
    if hip is not None and ankle is not None and abs(ankle - hip) < torso * 1.15:
        return "坐着"
    if hip is not None and (knee is not None or ankle is not None):
        return "站着"
    return ""


def stub_skeleton(x1: float, y1: float, x2: float, y2: float) -> list[dict]:
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    sitting = h / w < 1.65
    cx = (x1 + x2) * 0.5
    head = y1 + h * 0.08
    sh = y1 + h * 0.22
    hip_y = y1 + h * (0.55 if sitting else 0.48)
    knee_y = y1 + h * (0.74 if sitting else 0.74)
    ank_y = y2 - h * 0.03
    sw, hw, kw = w * 0.22, w * 0.16, w * 0.14
    points = {
        "nose": (cx, head),
        "left_eye": (cx - w * 0.04, head - h * 0.01),
        "right_eye": (cx + w * 0.04, head - h * 0.01),
        "left_ear": (cx - w * 0.07, head),
        "right_ear": (cx + w * 0.07, head),
        "left_shoulder": (cx - sw, sh),
        "right_shoulder": (cx + sw, sh),
        "left_elbow": (cx - sw - w * 0.04, (sh + hip_y) * 0.55),
        "right_elbow": (cx + sw + w * 0.04, (sh + hip_y) * 0.55),
        "left_wrist": (cx - sw - w * 0.02, hip_y - h * 0.02),
        "right_wrist": (cx + sw + w * 0.02, hip_y - h * 0.02),
        "left_hip": (cx - hw, hip_y),
        "right_hip": (cx + hw, hip_y),
        "left_knee": (cx - kw, knee_y),
        "right_knee": (cx + kw, knee_y),
        "left_ankle": (cx - kw, ank_y),
        "right_ankle": (cx + kw, ank_y),
    }
    return [
        {"name": name, "u": round(u, 4), "v": round(v, 4), "conf": 1.0}
        for name, (u, v) in points.items()
    ]


def stub_infer(people: list[dict]) -> dict:
    out = []
    for item in people:
        x1 = float(item.get("x1") or 0.2)
        y1 = float(item.get("y1") or 0.1)
        x2 = float(item.get("x2") or 0.8)
        y2 = float(item.get("y2") or 0.9)
        keypoints = stub_skeleton(x1, y1, x2, y2)
        out.append(
            {
                "id": item.get("id") or "",
                "track_id": item.get("track_id"),
                "name": item.get("name") or "",
                "activity": classify_activity(keypoints),
                "keypoints": keypoints,
            }
        )
    return {"ready": True, "people": out}


def describe_mocap(note: "MocapNote | None") -> str:
    if note is None:
        return ""
    if note.error and not note.people:
        return ""
    bits: list[str] = []
    for person in note.people:
        who = str(person.get("name") or "")
        act = str(person.get("activity") or "")
        if who and act:
            bits.append(f"{who}{act}")
        elif act:
            bits.append(act)
        elif who:
            bits.append(who)
    if bits:
        return "动捕 " + " ".join(bits)
    if note.people:
        return f"动捕 {len(note.people)}人"
    return ""


def _person_keypoints(item: dict) -> list[dict]:
    raw = item.get("keypoints") or []
    out: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name:
            continue
        try:
            out.append(
                {
                    "name": name,
                    "u": float(row["u"]),
                    "v": float(row["v"]),
                    "conf": float(row.get("conf") if row.get("conf") is not None else 1.0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


@dataclass
class MocapNote:
    camera_id: str
    camera: str
    people: list[dict] = field(default_factory=list)
    t: float = 0.0
    error: str = ""

    def public(self) -> dict:
        return {
            "id": self.camera_id,
            "name": self.camera,
            "people": list(self.people),
            "t": self.t,
            "error": self.error,
            "line": describe_mocap(self),
        }


@dataclass
class _Job:
    camera_id: str
    camera: str
    jpeg: bytes
    people: list[dict]


class MocapClient:
    """旁路动作捕捉：把帧送给 FreeMoCap / skellytracker，骨架只进 HUD，不进跟拍。"""

    def __init__(self, cfg: MocapConfig, *, inline: bool = False) -> None:
        self.cfg = cfg
        self.log = SceneLog(cfg.status, cfg.log)
        self._inline = inline
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._notes: dict[str, MocapNote] = {}
        self._last_offer: dict[str, float] = {}
        self._busy = False
        self._error = ""
        self._frames = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if cfg.enabled and not inline:
            self._thread = threading.Thread(target=self._loop, name="pocketshow-mocap", daemon=True)
            self._thread.start()

    def wants(self, camera_id: str) -> bool:
        if not self.cfg.enabled:
            return False
        wanted = (self.cfg.camera_id or "").strip()
        return not wanted or camera_id == wanted

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
            logger.exception("动作捕捉抽帧失败")
            return False
        job = _Job(
            camera_id=cam_id,
            camera=camera_name or cam_id,
            jpeg=jpeg,
            people=query_people(tracks, w, h),
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
            note = self._notes.get(camera_id or "live")
            people = list(note.people) if note is not None else []
        if not people:
            return
        by_id = {str(item.get("id") or ""): item for item in people if item.get("id") is not None}
        by_track: dict[int, dict] = {}
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
            keypoints = _person_keypoints(item)
            if not keypoints:
                continue
            track.keypoints = keypoints
            track.activity = str(item.get("activity") or classify_activity(keypoints))

    def line_for(self, camera_id: str = "") -> str:
        with self._lock:
            note = self._notes.get(camera_id or "live")
            if note is None and len(self._notes) == 1:
                note = next(iter(self._notes.values()))
            return describe_mocap(note)

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
                logger.exception("动作捕捉线程异常")
            finally:
                self._queue.task_done()

    def _run(self, job: _Job) -> None:
        with self._lock:
            self._busy = True
            self._frames += 1
            index = self._frames
        wall = time.time()
        try:
            payload = self._infer(job)
            error = ""
        except Exception as exc:
            payload = {}
            error = str(exc)
            logger.warning("动作捕捉失败：%s", error)
        people = []
        for item in payload.get("people") or []:
            if not isinstance(item, dict):
                continue
            keypoints = _person_keypoints(item)
            if not keypoints:
                continue
            people.append(
                {
                    "id": item.get("id") or "",
                    "track_id": item.get("track_id"),
                    "name": item.get("name") or "",
                    "activity": str(item.get("activity") or classify_activity(keypoints)),
                    "keypoints": keypoints,
                }
            )
        note = MocapNote(
            camera_id=job.camera_id,
            camera=job.camera,
            people=people,
            t=wall,
            error=error,
        )
        with self._lock:
            self._notes[job.camera_id] = note
            self._error = error
            self._busy = False
        snapshot = self.public(now=wall)
        self.log.write_status(snapshot)
        if error or not people:
            return
        if index % 20 == 0:
            self.log.append(
                {
                    "event": "mocap",
                    "t": wall,
                    "camera_id": job.camera_id,
                    "camera": job.camera,
                    "people": [
                        {"id": p.get("id"), "name": p.get("name"), "activity": p.get("activity")}
                        for p in people
                    ],
                }
            )

    def _infer(self, job: _Job) -> dict:
        if self.cfg.backend == "stub":
            return stub_infer(job.people)
        url = self.cfg.base_url.rstrip("/") + "/v1/mocap/frame"
        import base64

        return post_json(
            url,
            {
                "camera_id": job.camera_id,
                "jpeg_b64": base64.b64encode(job.jpeg).decode("ascii"),
                "people": job.people,
            },
            self.cfg.timeout_s,
        )
