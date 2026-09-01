from __future__ import annotations

import json
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
        self.gap_s = gap_s
        self.max_events = max_events
        self.live: dict[str, dict] = {}
        self._writes = 0

    def tick(self, present: dict[str, dict], now: float | None = None) -> None:
        now = time.time() if now is None else now
        seen = set(present)
        for pid, info in present.items():
            name = info.get("name") or pid
            photo = info.get("photo") or ""
            session = self.live.get(pid)
            if session is None:
                session = {
                    "person_id": pid,
                    "name": name,
                    "photo": photo,
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
