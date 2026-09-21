from __future__ import annotations

import base64
import json
import logging
import os
import queue
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pocketshow.config import SceneConfig
from pocketshow.gallery import encode_jpeg
from pocketshow.overlay import _draw_texts
from pocketshow.types import Track

logger = logging.getLogger("pocketshow.scene")

DEFAULT_PROMPT = (
    "这是办公室工位监控画面。镜头：{camera}。画面里已识别到的人：{people}。"
    "用一两句中文说明他们正在做什么（坐着工作、打电话、站着交谈、离开座位、低头看手机等）。"
    "不要编造姓名，只用我给出的名字；没认出的人写成「未知名」。"
    "若没有值得说的变化，只回复「无事」。"
)
_SILENCE = ("无事", "无明显变化", "没有值得", "silence", "nothing noteworthy", "no noteworthy")
_FRESH_S = 30.0
_MAX_EVENTS = 400


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="scene.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def people_from_tracks(tracks: list[Track]) -> list[str]:
    names: list[str] = []
    unknown = 0
    for track in tracks:
        if track.live is False:
            continue
        if track.person_name:
            names.append(track.person_name)
        else:
            unknown += 1
    if unknown == 1:
        names.append("未知名")
    elif unknown > 1:
        names.append(f"{unknown} 名未知名")
    return names


def is_silence(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    folded = raw.strip("。．.！!？? \n").lower()
    if folded in {"无事", "silence", "ok", "none", "n/a"}:
        return True
    if len(folded) <= 24 and any(key in folded for key in _SILENCE):
        return True
    return False


def _norm(text: str) -> str:
    return "".join((text or "").lower().split()).strip("。．.！!？?,，、 ")


def same_caption(a: str, b: str) -> bool:
    left, right = _norm(a), _norm(b)
    return bool(left) and left == right


def chat_vision(
    jpeg: bytes,
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 25.0,
    max_tokens: int = 80,
) -> str:
    """OpenAI 兼容的多模态 chat/completions。Mage-VL / SGLang / 其它视觉模型都能用。"""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    b64 = base64.b64encode(jpeg).decode("ascii")
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("POCKETSHOW_SCENE_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("视觉模型没有返回内容")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if isinstance(text, list):
        text = "".join(
            part.get("text") or ""
            for part in text
            if isinstance(part, dict)
        )
    return str(text).strip()


def describe_stub(people: list[str], camera: str) -> str:
    if not people:
        return "无事"
    who = "、".join(people)
    prefix = f"{camera}：" if camera else ""
    return f"{prefix}{who} 在工位上"


def annotate_people(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    vis = frame.copy()
    texts: list[tuple[str, tuple[int, int], tuple[int, int, int], int]] = []
    cream = (210, 220, 232)
    for track in tracks:
        if track.live is False:
            continue
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), cream, 2)
        texts.append((track.label, (x1, max(8, y1 - 22)), cream, 18))
    return _draw_texts(vis, texts)


def prepare_jpeg(
    frame: np.ndarray,
    tracks: list[Track],
    *,
    max_width: int = 768,
    quality: int = 70,
) -> bytes:
    vis = annotate_people(frame, tracks)
    h, w = vis.shape[:2]
    if w > max_width:
        scale = max_width / w
        vis = cv2.resize(
            vis,
            (max_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return encode_jpeg(vis, quality=quality)


@dataclass
class SceneNote:
    camera_id: str
    camera: str
    text: str
    people: list[str] = field(default_factory=list)
    silent: bool = False
    t: float = 0.0
    ts: str = ""
    error: str = ""

    def public(self) -> dict:
        return {
            "id": self.camera_id,
            "name": self.camera,
            "text": self.text,
            "people": list(self.people),
            "silent": self.silent,
            "t": self.t,
            "ts": self.ts,
            "error": self.error,
        }


@dataclass
class _Job:
    camera_id: str
    camera: str
    jpeg: bytes
    people: list[str]


class SceneLog:
    """管理页读最新一句和流水。跟拍进程写入 status / jsonl。"""

    def __init__(self, status: str | Path, log: str | Path) -> None:
        self.status_path = Path(status)
        self.log_path = Path(log)

    def public(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        empty = {
            "enabled": False,
            "fresh": False,
            "updated": 0.0,
            "age_s": None,
            "error": "",
            "busy": False,
            "backend": "",
            "model": "",
            "cameras": [],
        }
        if not self.status_path.exists():
            return empty
        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(data, dict):
            return empty
        updated = float(data.get("updated") or 0)
        age = max(0.0, now - updated) if updated else None
        data["age_s"] = round(age, 1) if age is not None else None
        data["fresh"] = bool(updated and age is not None and age <= _FRESH_S)
        data.setdefault("cameras", [])
        data.setdefault("error", "")
        return data

    def events(self, limit: int = 40) -> list[dict]:
        if not self.log_path.exists():
            return []
        rows: list[dict] = []
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines[-max(1, limit) :]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        rows.reverse()
        return rows

    def append(self, event: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._trim()

    def write_status(self, payload: dict) -> None:
        _atomic_write(self.status_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _trim(self) -> None:
        try:
            lines = [line for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return
        if len(lines) <= _MAX_EVENTS:
            return
        tmp = self.log_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines[-_MAX_EVENTS:]) + "\n", encoding="utf-8")
        tmp.replace(self.log_path)


class SceneNarrator:
    """旁路场景理解：抽关键帧、后台问视觉模型，结果只进 HUD / 日志，不进跟拍。"""

    def __init__(self, cfg: SceneConfig, *, inline: bool = False, client=None) -> None:
        self.cfg = cfg
        self.log = SceneLog(cfg.status, cfg.log)
        self._inline = inline
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._notes: dict[str, SceneNote] = {}
        self._last_offer: dict[str, float] = {}
        self._last_any = 0.0
        self._busy = False
        self._error = ""
        self._vl = client
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if cfg.enabled and not inline:
            self._thread = threading.Thread(target=self._loop, name="pocketshow-scene", daemon=True)
            self._thread.start()

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
        stamp = time.monotonic() if now is None else now
        cam_id = camera_id or camera_name or "live"
        interval = self._interval()
        with self._lock:
            if self._busy and not self._inline:
                return False
            last = self._last_offer.get(cam_id)
            if last is not None and stamp - last < interval:
                return False
            if self._last_any and stamp - self._last_any < self.cfg.gap_s:
                return False
        people = people_from_tracks(tracks)
        try:
            jpeg = prepare_jpeg(
                frame,
                tracks,
                max_width=self.cfg.max_width,
                quality=self.cfg.jpeg_quality,
            )
        except Exception:
            logger.exception("场景抽帧失败")
            return False
        job = _Job(camera_id=cam_id, camera=camera_name or cam_id, jpeg=jpeg, people=people)
        if self._inline:
            with self._lock:
                self._last_offer[cam_id] = stamp
                self._last_any = stamp
            self._run(job)
            return True
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            return False
        with self._lock:
            self._last_offer[cam_id] = stamp
            self._last_any = stamp
        return True

    def line_for(self, camera_id: str = "") -> str:
        with self._lock:
            note = self._notes.get(camera_id or "live")
            if note is None and len(self._notes) == 1:
                note = next(iter(self._notes.values()))
            if note is None or note.silent or not note.text:
                return ""
            return note.text

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
            "model": self.cfg.model,
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
        if self._vl is not None:
            try:
                self._vl.close()
            except Exception:
                pass
            self._vl = None

    def _interval(self) -> float:
        if self.cfg.backend == "moss-vl" and self.cfg.sample_fps > 0:
            return max(0.2, 1.0 / self.cfg.sample_fps)
        return self.cfg.interval_s

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._run(job)
            except Exception:
                logger.exception("场景理解线程异常")
            finally:
                self._queue.task_done()

    def _run(self, job: _Job) -> None:
        with self._lock:
            self._busy = True
        wall = time.time()
        try:
            text = self._describe(job)
            error = ""
        except Exception as exc:
            text = ""
            error = str(exc)
            logger.warning("场景理解失败：%s", error)
        silent = (not error) and is_silence(text)
        note = SceneNote(
            camera_id=job.camera_id,
            camera=job.camera,
            text="" if silent else text,
            people=list(job.people),
            silent=silent,
            t=wall,
            ts=_fmt(wall),
            error=error,
        )
        changed = True
        with self._lock:
            prev = self._notes.get(job.camera_id)
            if prev is not None and not error and same_caption(prev.text, note.text) and prev.silent == note.silent:
                changed = False
                prev.t = wall
                prev.ts = note.ts
                prev.people = note.people
                prev.error = ""
                note = prev
            else:
                self._notes[job.camera_id] = note
            self._error = error
            self._busy = False
        snapshot = self.public(now=wall)
        self.log.write_status(snapshot)
        if error or not changed or silent:
            return
        self.log.append(
            {
                "event": "scene",
                "t": wall,
                "ts": note.ts,
                "camera_id": job.camera_id,
                "camera": job.camera,
                "text": note.text,
                "people": list(job.people),
            }
        )

    def _describe(self, job: _Job) -> str:
        if self.cfg.backend == "stub":
            return describe_stub(job.people, job.camera)
        people = "、".join(job.people) if job.people else "无人"
        prompt = (self.cfg.prompt or DEFAULT_PROMPT).format(camera=job.camera or "当前镜头", people=people)
        if self.cfg.backend == "moss-vl":
            from pocketshow.mossvl import describe_with_moss

            text, self._vl = describe_with_moss(job.jpeg, prompt, self.cfg, self._vl)
            return text
        return chat_vision(
            job.jpeg,
            prompt,
            base_url=self.cfg.base_url,
            model=self.cfg.model,
            api_key=self.cfg.api_key,
            timeout_s=self.cfg.timeout_s,
            max_tokens=self.cfg.max_tokens,
        )
