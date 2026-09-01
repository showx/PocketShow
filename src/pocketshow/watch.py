from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pocketshow.types import Track

_STATES = {
    "waiting": "等待有人入镜",
    "at_desk": "在岗",
    "away": "暂离",
    "alarm": "离岗报警",
    "no_camera": "镜头中断",
    "off_hours": "非上班时间",
}

_WEEK = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "日"}
_DEFAULT_DAYS = [1, 2, 3, 4, 5]


def parse_hhmm(text: str) -> tuple[int, int]:
    raw = (text or "").strip().replace("：", ":")
    if not raw:
        raise ValueError("时间不能为空")
    parts = raw.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"无效时间 {text}")
    return hour, minute


def format_hhmm(text: str) -> str:
    hour, minute = parse_hhmm(text)
    return f"{hour:02d}:{minute:02d}"


def normalize_workdays(days: list[int] | None) -> list[int]:
    if days is None:
        return list(_DEFAULT_DAYS)
    out = sorted({int(d) for d in days if 1 <= int(d) <= 7})
    return out


def hours_label(start: str, end: str, workdays: list[int] | None) -> str:
    days = normalize_workdays(workdays)
    if days == [1, 2, 3, 4, 5]:
        who = "工作日"
    elif days == [1, 2, 3, 4, 5, 6, 7]:
        who = "每天"
    elif not days:
        who = "未选工作日"
    else:
        who = "周" + "、".join(_WEEK[d] for d in days)
    return f"{who} {format_hhmm(start)}–{format_hhmm(end)}"


def in_work_hours(
    now: float | datetime | None = None,
    *,
    start: str = "09:00",
    end: str = "18:30",
    workdays: list[int] | None = None,
) -> bool:
    dt = now if isinstance(now, datetime) else datetime.fromtimestamp(now if now is not None else time.time())
    days = normalize_workdays(workdays)
    if not days or dt.isoweekday() not in days:
        return False
    cur = dt.hour * 60 + dt.minute
    sh, sm = parse_hhmm(start)
    eh, em = parse_hhmm(end)
    begin = sh * 60 + sm
    finish = eh * 60 + em
    if begin == finish:
        return True
    if begin < finish:
        return begin <= cur < finish
    return cur >= begin or cur < finish


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="station.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


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


@dataclass
class StationState:
    occupied: bool = False
    armed: bool = False
    away_s: float = 0.0
    alarm: bool = False
    people: int = 0
    names: list[str] = field(default_factory=list)
    last_seen: float = 0.0
    state: str = "waiting"
    away_limit_s: float = 30.0
    updated: float = 0.0
    camera: bool = True
    on_duty: bool = True
    work_start: str = "09:00"
    work_end: str = "18:30"
    workdays: list[int] = field(default_factory=lambda: list(_DEFAULT_DAYS))


def occupied_from_tracks(tracks: list[Track]) -> tuple[int, list[str]]:
    """工位上有活人（照片/屏幕不算）。"""
    names: list[str] = []
    count = 0
    for track in tracks:
        if track.live is False:
            continue
        count += 1
        if track.person_name:
            names.append(track.person_name)
        else:
            names.append(f"ID {track.id}")
    return count, names


class AwayLog:
    """上班时段内的离岗流水。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._open: dict | None = None

    def start(self, now: float, names: list[str]) -> None:
        if self._open is not None:
            return
        self._open = {"start": now, "names": list(names), "alarm": False}

    def note_alarm(self) -> None:
        if self._open is not None:
            self._open["alarm"] = True

    def close(self, now: float) -> None:
        session = self._open
        self._open = None
        if session is None:
            return
        start = float(session["start"])
        duration = max(0.0, now - start)
        if duration < 2.0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "start": start,
            "end": now,
            "start_ts": _fmt(start),
            "end_ts": _fmt(now),
            "duration_s": round(duration, 1),
            "duration": _duration(duration),
            "alarm": bool(session.get("alarm")),
            "names": session.get("names") or [],
            "date": datetime.fromtimestamp(start).strftime("%Y-%m-%d"),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def today(self, now: float | None = None, live: dict | None = None) -> dict:
        now = time.time() if now is None else now
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        rows: list[dict] = []
        if self.path.exists():
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("date") == day:
                        rows.append(item)
            except (OSError, json.JSONDecodeError):
                rows = []
        if live is not None:
            rows.append(live)
        away_s = sum(float(r.get("duration_s") or 0) for r in rows)
        alarm_count = sum(1 for r in rows if r.get("alarm"))
        return {
            "date": day,
            "sessions": list(reversed(rows[-80:])),
            "away_count": len(rows),
            "alarm_count": alarm_count,
            "away_s": round(away_s, 1),
            "away_label": _duration(away_s),
        }


class StationWatch:
    """离岗计时：仅上班时段检测；有人在岗则清零，离开超过 away_limit_s 报警。"""

    def __init__(
        self,
        path: str | Path,
        away_s: float = 30.0,
        *,
        settings_path: str | Path | None = None,
        log_path: str | Path | None = None,
        work_start: str = "09:00",
        work_end: str = "18:30",
        workdays: list[int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.settings_path = Path(settings_path) if settings_path else None
        self.away_limit_s = max(3.0, float(away_s))
        self.work_start = format_hhmm(work_start)
        self.work_end = format_hhmm(work_end)
        self.workdays = normalize_workdays(workdays)
        self.last_seen = 0.0
        self.armed = False
        self._last_write = 0.0
        self._last_alarm = False
        self._settings_mtime = 0.0
        self.log = AwayLog(log_path) if log_path else AwayLog(self.path.with_name("away.jsonl"))

    def save_settings(
        self,
        *,
        work_start: str | None = None,
        work_end: str | None = None,
        workdays: list[int] | None = None,
        away_s: float | None = None,
    ) -> dict:
        if work_start is not None:
            self.work_start = format_hhmm(work_start)
        if work_end is not None:
            self.work_end = format_hhmm(work_end)
        if workdays is not None:
            self.workdays = normalize_workdays(workdays)
        if away_s is not None:
            self.away_limit_s = max(3.0, float(away_s))
        payload = self.settings_public()
        if self.settings_path is not None:
            _atomic_write(self.settings_path, json.dumps(payload, ensure_ascii=False, indent=2))
            self._settings_mtime = self.settings_path.stat().st_mtime
        return payload

    def settings_public(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        on_duty = in_work_hours(now, start=self.work_start, end=self.work_end, workdays=self.workdays)
        return {
            "work_start": self.work_start,
            "work_end": self.work_end,
            "workdays": list(self.workdays),
            "away_s": self.away_limit_s,
            "on_duty": on_duty,
            "hours_label": hours_label(self.work_start, self.work_end, self.workdays),
        }

    def tick(
        self,
        tracks: list[Track],
        *,
        now: float | None = None,
        camera_ok: bool = True,
        write: bool = True,
        on_duty: bool | None = None,
    ) -> StationState:
        now = time.time() if now is None else now
        self._reload_settings()
        if on_duty is None:
            on_duty = in_work_hours(now, start=self.work_start, end=self.work_end, workdays=self.workdays)
        people, names = occupied_from_tracks(tracks)
        occupied = people > 0
        state = StationState(
            away_limit_s=self.away_limit_s,
            updated=now,
            camera=camera_ok,
            on_duty=on_duty,
            work_start=self.work_start,
            work_end=self.work_end,
            workdays=list(self.workdays),
        )
        state.people = people
        state.names = names
        state.occupied = occupied

        if not on_duty:
            self.log.close(now)
            self.armed = False
            self.last_seen = 0.0
            state.state = "off_hours"
            state.alarm = False
            self._maybe_write(state, write, force=True)
            self._last_alarm = False
            return state

        if not camera_ok:
            state.state = "no_camera"
            state.armed = self.armed
            state.last_seen = self.last_seen
            if self.armed and self.last_seen:
                state.away_s = max(0.0, now - self.last_seen)
            self._maybe_write(state, write, force=True)
            return state

        if occupied:
            self.log.close(now)
            self.armed = True
            self.last_seen = now
            state.armed = True
            state.last_seen = now
            state.state = "at_desk"
            state.away_s = 0.0
            state.alarm = False
        elif not self.armed:
            state.state = "waiting"
        else:
            self.log.start(self.last_seen or now, names)
            state.armed = True
            state.last_seen = self.last_seen
            state.away_s = max(0.0, now - self.last_seen)
            if state.away_s >= self.away_limit_s:
                state.state = "alarm"
                state.alarm = True
                self.log.note_alarm()
            else:
                state.state = "away"
        self._maybe_write(state, write, force=state.alarm != self._last_alarm)
        self._last_alarm = state.alarm
        return state

    def public(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        self._reload_settings()
        state = StationState(
            away_limit_s=self.away_limit_s,
            work_start=self.work_start,
            work_end=self.work_end,
            workdays=list(self.workdays),
        )
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for key in asdict(state):
                    if key in data:
                        setattr(state, key, data[key])
            except (OSError, json.JSONDecodeError):
                pass
        stale = (now - float(state.updated or 0)) > 3.5
        on_duty = in_work_hours(now, start=self.work_start, end=self.work_end, workdays=self.workdays)
        out = asdict(state)
        out["on_duty"] = on_duty
        out["work_start"] = self.work_start
        out["work_end"] = self.work_end
        out["workdays"] = list(self.workdays)
        out["hours_label"] = hours_label(self.work_start, self.work_end, self.workdays)
        if not on_duty:
            out["alarm"] = False
            if not stale and state.state != "no_camera":
                out["state"] = "off_hours"
        out["fresh"] = not stale and float(state.updated or 0) > 0
        out["label"] = _STATES.get(out["state"], out["state"])
        out["detail"] = _detail(out["state"], state, stale, out["hours_label"], on_duty)
        return out

    def report(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        status = self.public(now)
        live = None
        if status.get("fresh") and status.get("on_duty") and status.get("state") in {"away", "alarm"}:
            start = float(status.get("last_seen") or now)
            duration = float(status.get("away_s") or 0)
            live = {
                "start": start,
                "end": None,
                "start_ts": _fmt(start),
                "end_ts": "进行中",
                "duration_s": round(duration, 1),
                "duration": _duration(duration),
                "alarm": bool(status.get("alarm")),
                "names": status.get("names") or [],
                "date": datetime.fromtimestamp(now).strftime("%Y-%m-%d"),
                "live": True,
            }
        return {
            "settings": self.settings_public(now),
            "state": status,
            "today": self.log.today(now, live=live),
        }

    def _reload_settings(self) -> None:
        if self.settings_path is None or not self.settings_path.exists():
            return
        mtime = self.settings_path.stat().st_mtime
        if mtime == self._settings_mtime:
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._settings_mtime = mtime
        try:
            if data.get("work_start"):
                self.work_start = format_hhmm(str(data["work_start"]))
            if data.get("work_end"):
                self.work_end = format_hhmm(str(data["work_end"]))
            if "workdays" in data:
                self.workdays = normalize_workdays(list(data["workdays"]))
            if data.get("away_s") is not None:
                self.away_limit_s = max(3.0, float(data["away_s"]))
        except (TypeError, ValueError):
            return

    def _maybe_write(self, state: StationState, write: bool, force: bool) -> None:
        if not write:
            return
        now = time.monotonic()
        if not force and now - self._last_write < 0.25:
            return
        self._last_write = now
        _atomic_write(self.path, json.dumps(asdict(state), ensure_ascii=False, indent=2))


def _work_window(now: float, start: str, end: str) -> tuple[float, float]:
    dt = datetime.fromtimestamp(now)
    sh, sm = parse_hhmm(start)
    eh, em = parse_hhmm(end)
    begin = datetime(dt.year, dt.month, dt.day, sh, sm).timestamp()
    finish = datetime(dt.year, dt.month, dt.day, eh, em).timestamp()
    if finish <= begin:
        finish += 24 * 3600
    return begin, finish


def person_away_today(
    events: list[dict],
    *,
    now: float | None = None,
    work_start: str = "09:00",
    work_end: str = "18:30",
    workdays: list[int] | None = None,
    min_s: float = 15.0,
    alarm_s: float = 30.0,
    present_ids: set[str] | None = None,
    skip_ids: set[str] | None = None,
) -> dict:
    """按人统计上班时段离开镜头。工位上还有别人时，也会记这个人自己的离岗。客人（skip_ids）不计入。"""
    now = time.time() if now is None else now
    day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    empty = {
        "date": day,
        "sessions": [],
        "away_count": 0,
        "alarm_count": 0,
        "away_s": 0.0,
        "away_label": "0 秒",
        "alarm_s": 0.0,
        "alarm_label": "0 秒",
        "longest_s": 0.0,
        "longest_label": "0 秒",
        "people_count": 0,
        "by_person": [],
        "alarms": [],
    }
    if datetime.fromtimestamp(now).isoweekday() not in normalize_workdays(workdays):
        return empty
    begin, finish = _work_window(now, work_start, work_end)
    cap = min(now, finish)
    present_ids = present_ids or set()
    skip_ids = skip_ids or set()
    per: dict[str, dict] = {}
    ordered = sorted(
        (e for e in events if float(e.get("t") or 0) > 0),
        key=lambda e: float(e.get("t") or 0),
    )
    sessions: list[dict] = []

    def emit(pid: str, info: dict, start: float, end: float, live: bool) -> None:
        if pid in skip_ids:
            return
        duration = max(0.0, end - start)
        if duration < min_s:
            return
        name = info.get("name") or pid
        sessions.append(
            {
                "person_id": pid,
                "name": name,
                "names": [name],
                "photo": info.get("photo") or "",
                "start": start,
                "end": None if live else end,
                "start_ts": _fmt(start),
                "end_ts": "进行中" if live else _fmt(end),
                "duration_s": round(duration, 1),
                "duration": _duration(duration),
                "alarm": duration >= alarm_s,
                "live": live,
                "date": day,
            }
        )

    for event in ordered:
        t = float(event.get("t") or 0)
        if t < begin or t > finish:
            continue
        pid = str(event.get("person_id") or "")
        if not pid:
            continue
        info = per.setdefault(pid, {"in": False, "seen": False, "away_start": None, "name": pid, "photo": ""})
        if event.get("name"):
            info["name"] = event["name"]
        if event.get("photo"):
            info["photo"] = event["photo"]
        kind = event.get("event")
        if kind == "enter":
            info["seen"] = True
            if info["away_start"] is not None:
                emit(pid, info, float(info["away_start"]), t, False)
                info["away_start"] = None
            info["in"] = True
        elif kind == "leave" and info["seen"]:
            info["in"] = False
            if info["away_start"] is None:
                info["away_start"] = t

    for pid, info in per.items():
        start = info.get("away_start")
        if start is None:
            continue
        if pid in present_ids:
            continue
        emit(pid, info, float(start), cap, True)

    sessions.sort(key=lambda r: float(r.get("start") or 0), reverse=True)
    away_s = sum(float(r.get("duration_s") or 0) for r in sessions)
    stats = _alarm_stats(sessions)
    return {
        "date": day,
        "sessions": sessions[:80],
        "away_count": len(sessions),
        "away_s": round(away_s, 1),
        "away_label": _duration(away_s),
        **stats,
    }


def _alarm_stats(sessions: list[dict]) -> dict:
    alarms = [row for row in sessions if row.get("alarm")]
    by: dict[str, dict] = {}
    for row in alarms:
        pid = str(row.get("person_id") or "")
        item = by.setdefault(
            pid,
            {
                "person_id": pid,
                "name": row.get("name") or pid,
                "photo": row.get("photo") or "",
                "alarm_count": 0,
                "alarm_s": 0.0,
                "longest_s": 0.0,
                "last_ts": "",
            },
        )
        dur = float(row.get("duration_s") or 0)
        item["alarm_count"] += 1
        item["alarm_s"] += dur
        if dur > item["longest_s"]:
            item["longest_s"] = dur
        ts = str(row.get("start_ts") or "")
        if ts >= item["last_ts"]:
            item["last_ts"] = ts
            item["name"] = row.get("name") or item["name"]
            if row.get("photo"):
                item["photo"] = row["photo"]
    people = []
    for item in by.values():
        item["alarm_s"] = round(float(item["alarm_s"]), 1)
        item["longest_s"] = round(float(item["longest_s"]), 1)
        item["alarm_label"] = _duration(item["alarm_s"])
        item["longest_label"] = _duration(item["longest_s"])
        people.append(item)
    people.sort(key=lambda x: (x["alarm_s"], x["alarm_count"]), reverse=True)
    alarm_s = sum(float(r.get("duration_s") or 0) for r in alarms)
    longest = max((float(r.get("duration_s") or 0) for r in alarms), default=0.0)
    return {
        "alarm_count": len(alarms),
        "alarm_s": round(alarm_s, 1),
        "alarm_label": _duration(alarm_s),
        "longest_s": round(longest, 1),
        "longest_label": _duration(longest),
        "people_count": len(people),
        "by_person": people,
        "alarms": alarms[:80],
    }


def _detail(state_name: str, state: StationState, stale: bool, hours: str, on_duty: bool) -> str:
    if not on_duty:
        return f"{hours}，现在不计离岗"
    if stale:
        return "跟拍未在更新工位状态"
    if state_name == "at_desk":
        who = "、".join(state.names[:3]) or "有人"
        return f"{who}在岗"
    if state_name == "away":
        return f"暂离 {int(state.away_s)} 秒"
    if state_name == "alarm":
        return f"已离开工位 {int(state.away_s)} 秒"
    if state_name == "no_camera":
        return "画面中断，不计离岗"
    if state_name == "off_hours":
        return f"{hours}，现在不计离岗"
    return "还没人入镜，不计离岗"


def hud_line(state: StationState) -> str:
    if state.state == "off_hours":
        return f"非上班时间 {state.work_start}–{state.work_end}"
    if state.state == "alarm":
        return f"离岗报警 已离开 {int(state.away_s)} 秒"
    if state.state == "away":
        return f"暂离 {int(state.away_s)} / {int(state.away_limit_s)} 秒"
    if state.state == "at_desk":
        return "在岗 " + (" ".join(state.names[:2]) or "")
    if state.state == "no_camera":
        return "镜头中断"
    return "工位等待入镜"
