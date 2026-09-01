from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes} 分"


class AppearanceLog:
    """入镜/离镜流水。JSONL 追加写入，短时遮挡不拆成两次。"""

    def __init__(self, path: str | Path, gap_s: float = 1.6, max_events: int = 4000) -> None:
        self.path = Path(path)
        self.present_path = self.path.with_name("present.json")
        self.gap_s = gap_s
        self.max_events = max_events
        self.live: dict[str, dict] = {}
        self._writes = 0
        self._last_present_write = 0.0
        self._last_present_ids: tuple[str, ...] | None = None

    def tick(self, present: dict[str, dict], now: float | None = None) -> None:
        now = time.time() if now is None else now
        seen = set(present)
        for pid, info in present.items():
            name = info.get("name") or pid
            photo = info.get("photo") or ""
            guest = bool(info.get("guest"))
            session = self.live.get(pid)
            if session is None:
                session = {
                    "person_id": pid,
                    "name": name,
                    "photo": photo,
                    "guest": guest,
                    "start": now,
                    "last": now,
                    "frames": 1,
                }
                self.live[pid] = session
                self._append(
                    {
                        "event": "enter",
                        "t": now,
                        "ts": _fmt(now),
                        "person_id": pid,
                        "name": name,
                        "photo": photo,
                    }
                )
            else:
                session["last"] = now
                session["frames"] = int(session.get("frames") or 0) + 1
                session["name"] = name
                session["guest"] = guest
                if photo:
                    session["photo"] = photo

        gone = [pid for pid in list(self.live) if pid not in seen]
        for pid in gone:
            session = self.live[pid]
            if now - float(session["last"]) < self.gap_s:
                continue
            start = float(session["start"])
            end = float(session["last"])
            self._append(
                {
                    "event": "leave",
                    "t": end,
                    "ts": _fmt(end),
                    "person_id": pid,
                    "name": session.get("name") or pid,
                    "photo": session.get("photo") or "",
                    "start_ts": _fmt(start),
                    "duration_s": round(end - start, 1),
                    "frames": int(session.get("frames") or 0),
                }
            )
            del self.live[pid]
        self._write_present(now)

    def present(self, now: float | None = None) -> dict:
        """跟拍进程写出的当前在镜名单，管理页用来做「在镜中」栏目。"""
        now = time.time() if now is None else now
        path = self.present_path
        if not path.exists():
            return {"updated": 0.0, "fresh": False, "count": 0, "people": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"updated": 0.0, "fresh": False, "count": 0, "people": []}
        updated = float(data.get("updated") or 0)
        stale = (now - updated) > 3.5
        people = [] if stale else list(data.get("people") or [])
        for row in people:
            start = float(row.get("start") or updated)
            row["duration_s"] = round(max(0.0, now - start), 1)
            row["duration"] = _duration(now - start)
            row["live"] = True
        return {
            "updated": updated,
            "fresh": not stale and updated > 0,
            "count": len(people),
            "people": people,
        }

    def _write_present(self, now: float) -> None:
        monotonic = time.monotonic()
        ids = tuple(sorted(self.live))
        if ids == self._last_present_ids and monotonic - self._last_present_write < 0.25:
            return
        self._last_present_write = monotonic
        self._last_present_ids = ids
        people = []
        for pid, session in self.live.items():
            start = float(session["start"])
            last = float(session.get("last") or now)
            people.append(
                {
                    "person_id": pid,
                    "name": session.get("name") or pid,
                    "photo": session.get("photo") or "",
                    "guest": bool(session.get("guest")),
                    "start": start,
                    "start_ts": _fmt(start),
                    "last": last,
                    "duration_s": round(last - start, 1),
                    "duration": _duration(last - start),
                    "frames": int(session.get("frames") or 0),
                    "live": True,
                }
            )
        payload = json.dumps({"updated": now, "people": people}, ensure_ascii=False, indent=2)
        path = self.present_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="present.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def relabel(self, source_id: str, keep_id: str, keep_name: str) -> None:
        if source_id in self.live:
            src = self.live.pop(source_id)
            keep = self.live.get(keep_id)
            if keep is None:
                src["person_id"] = keep_id
                src["name"] = keep_name
                self.live[keep_id] = src
            else:
                keep["start"] = min(float(keep["start"]), float(src["start"]))
                keep["last"] = max(float(keep["last"]), float(src["last"]))
                keep["frames"] = int(keep.get("frames") or 0) + int(src.get("frames") or 0)
                keep["name"] = keep_name
        events = self._read_raw()
        changed = False
        for event in events:
            if event.get("person_id") == source_id:
                event["person_id"] = keep_id
                event["name"] = keep_name
                event["merged_from"] = source_id
                changed = True
        if changed:
            self._rewrite(events)

    def visits(self, person_id: str | None = None, limit: int = 200) -> list[dict]:
        events = self._read_raw()
        open_visits: dict[str, dict] = {}
        closed: list[dict] = []
        for event in events:
            pid = str(event.get("person_id") or "")
            if person_id and pid != person_id:
                continue
            if event.get("event") == "enter":
                open_visits[pid] = {
                    "person_id": pid,
                    "name": event.get("name") or pid,
                    "photo": event.get("photo") or "",
                    "start": event.get("ts") or "",
                    "start_t": float(event.get("t") or 0),
                    "end": None,
                    "duration_s": None,
                    "duration": "进行中",
                    "live": True,
                    "frames": 0,
                }
            elif event.get("event") == "leave":
                row = open_visits.pop(pid, None)
                duration_s = event.get("duration_s")
                if duration_s is None and row is not None:
                    duration_s = round(float(event.get("t") or 0) - float(row.get("start_t") or 0), 1)
                closed.append(
                    {
                        "person_id": pid,
                        "name": event.get("name") or (row or {}).get("name") or pid,
                        "photo": event.get("photo") or (row or {}).get("photo") or "",
                        "start": event.get("start_ts") or (row or {}).get("start") or "",
                        "end": event.get("ts") or "",
                        "duration_s": duration_s,
                        "duration": _duration(float(duration_s or 0)),
                        "live": False,
                        "frames": int(event.get("frames") or 0),
                    }
                )
        live_rows = []
        for pid, session in self.live.items():
            if person_id and pid != person_id:
                continue
            start = float(session["start"])
            last = float(session["last"])
            live_rows.append(
                {
                    "person_id": pid,
                    "name": session.get("name") or pid,
                    "photo": session.get("photo") or "",
                    "start": _fmt(start),
                    "end": None,
                    "duration_s": round(last - start, 1),
                    "duration": _duration(last - start) + " · 在镜中",
                    "live": True,
                    "frames": int(session.get("frames") or 0),
                }
            )
        leftover = []
        now = time.time()
        for pid, row in open_visits.items():
            if pid in self.live:
                continue
            age = now - float(row.get("start_t") or 0)
            if age > 180:
                row["live"] = False
                row["duration"] = "中断"
                row["end"] = ""
            leftover.append(row)
        all_rows = closed + leftover + live_rows
        all_rows.sort(key=lambda r: r.get("start") or "", reverse=True)
        return all_rows[:limit]

    def raw_events(self) -> list[dict]:
        return self._read_raw()

    def visit_intervals(self) -> dict[str, list[tuple[float, float]]]:
        intervals: dict[str, list[tuple[float, float]]] = {}
        open_t: dict[str, float] = {}
        for event in self._read_raw():
            pid = str(event.get("person_id") or "")
            if not pid:
                continue
            t = float(event.get("t") or 0)
            if event.get("event") == "enter":
                open_t[pid] = t
            elif event.get("event") == "leave":
                start = open_t.pop(pid, t - float(event.get("duration_s") or 0))
                intervals.setdefault(pid, []).append((start, t))
        snapshot = self.present()
        if snapshot.get("fresh"):
            updated = float(snapshot.get("updated") or time.time())
            for row in snapshot.get("people") or []:
                pid = str(row.get("person_id") or "")
                if not pid:
                    continue
                start = float(row.get("start") or updated)
                intervals.setdefault(pid, []).append((start, updated))
        return intervals

    def concurrent_pairs(self, min_overlap_s: float = 8.0) -> set[tuple[str, str]]:
        """同框超过 min_overlap_s 秒的两人，不可能是同一人。"""
        intervals = self.visit_intervals()
        out: set[tuple[str, str]] = set()
        ids = list(intervals)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                overlap = 0.0
                for a0, a1 in intervals[a]:
                    for b0, b1 in intervals[b]:
                        overlap += max(0.0, min(a1, b1) - max(a0, b0))
                        if overlap >= min_overlap_s:
                            out.add((a, b) if a < b else (b, a))
                            break
                    if overlap >= min_overlap_s:
                        break
        return out

    def _append(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._writes += 1
        if self._writes % 80 == 0:
            self._trim()

    def _read_raw(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _rewrite(self, events: list[dict]) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
        tmp.replace(self.path)

    def _trim(self) -> None:
        events = self._read_raw()
        if len(events) <= self.max_events:
            return
        self._rewrite(events[-self.max_events :])
