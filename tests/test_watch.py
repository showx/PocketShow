from datetime import datetime

from pocketshow.types import Track
from pocketshow.watch import (
    StationWatch,
    hud_line,
    in_work_hours,
    occupied_from_tracks,
)


def _track(*, tid=1, name="小明", live: bool | None = True) -> Track:
    return Track(id=tid, bbox_xyxy=(0, 0, 40, 80), conf=0.9, person_name=name, live=live)


def _watch(tmp_path, **kwargs) -> StationWatch:
    return StationWatch(
        tmp_path / "station.json",
        away_s=kwargs.pop("away_s", 30),
        settings_path=tmp_path / "watch.json",
        log_path=tmp_path / "away.jsonl",
        **kwargs,
    )


def test_photo_does_not_count_as_occupied():
    people, names = occupied_from_tracks([_track(live=False, name="海报")])
    assert people == 0
    assert names == []


def test_unknown_liveness_counts_as_occupied():
    people, names = occupied_from_tracks([_track(live=None)])
    assert people == 1
    assert names == ["小明"]


def test_at_desk_clears_away(tmp_path):
    watch = _watch(tmp_path, away_s=30)
    first = watch.tick([_track()], now=1000.0, on_duty=True)
    assert first.state == "at_desk"
    assert first.away_s == 0
    assert not first.alarm

    gone = watch.tick([], now=1010.0, on_duty=True)
    assert gone.state == "away"
    assert 9.9 <= gone.away_s <= 10.1
    assert not gone.alarm

    back = watch.tick([_track()], now=1012.0, on_duty=True)
    assert back.state == "at_desk"
    assert back.away_s == 0
    assert not back.alarm


def test_away_beyond_limit_alarms(tmp_path):
    watch = _watch(tmp_path, away_s=30)
    watch.tick([_track()], now=0.0, on_duty=True, write=False)
    still = watch.tick([], now=29.0, on_duty=True, write=False)
    assert still.state == "away"
    assert not still.alarm

    fired = watch.tick([], now=30.0, on_duty=True)
    assert fired.state == "alarm"
    assert fired.alarm
    assert int(fired.away_s) == 30
    assert "离开" in hud_line(fired)


def test_waiting_until_someone_appears(tmp_path):
    watch = _watch(tmp_path, away_s=10)
    idle = watch.tick([], now=50.0, on_duty=True, write=False)
    assert idle.state == "waiting"
    assert not idle.alarm
    assert idle.away_s == 0


def test_photo_only_does_not_arm(tmp_path):
    watch = _watch(tmp_path, away_s=5)
    photo = watch.tick([_track(live=False)], now=1.0, on_duty=True, write=False)
    assert photo.state == "waiting"
    assert not photo.occupied
    later = watch.tick([], now=20.0, on_duty=True, write=False)
    assert later.state == "waiting"
    assert not later.alarm


def test_camera_drop_does_not_alarm(tmp_path):
    watch = _watch(tmp_path, away_s=8)
    watch.tick([_track()], now=10.0, on_duty=True, write=False)
    dropped = watch.tick([], now=40.0, camera_ok=False, on_duty=True)
    assert dropped.state == "no_camera"
    assert not dropped.alarm
    assert "中断" in hud_line(dropped)


def test_public_missing_file_has_labels(tmp_path):
    watch = StationWatch(tmp_path / "missing.json", away_s=30)
    info = watch.public()
    assert info["fresh"] is False
    assert info["alarm"] is False
    assert info["hours_label"]


def test_public_reads_written_state(tmp_path):
    watch = _watch(tmp_path)
    noon = datetime(2026, 9, 1, 10, 0, 0).timestamp()
    watch.tick([_track(name="阿强")], now=noon, on_duty=True)
    info = watch.public(now=noon + 0.2)
    assert info["fresh"] is True
    assert info["state"] == "at_desk"
    assert info["label"] == "在岗"
    assert "阿强" in info["detail"]


def test_weekdays_nine_to_six_thirty():
    tue_noon = datetime(2026, 9, 1, 12, 0)  # Tuesday
    tue_early = datetime(2026, 9, 1, 8, 59)
    tue_late = datetime(2026, 9, 1, 18, 30)
    tue_end = datetime(2026, 9, 1, 18, 29)
    sat = datetime(2026, 9, 5, 12, 0)
    assert in_work_hours(tue_noon, start="09:00", end="18:30")
    assert not in_work_hours(tue_early, start="09:00", end="18:30")
    assert in_work_hours(tue_end, start="09:00", end="18:30")
    assert not in_work_hours(tue_late, start="09:00", end="18:30")
    assert not in_work_hours(sat, start="09:00", end="18:30")


def test_off_hours_does_not_alarm(tmp_path):
    watch = _watch(tmp_path, away_s=5)
    watch.tick([_track()], now=10.0, on_duty=True, write=False)
    off = watch.tick([], now=80.0, on_duty=False)
    assert off.state == "off_hours"
    assert not off.alarm
    assert "非上班" in hud_line(off)


def test_settings_file_reloads(tmp_path):
    watch = _watch(tmp_path, work_start="09:00", work_end="18:30")
    watch.save_settings(work_start="08:30", work_end="17:00", workdays=[1, 2, 3, 4, 5, 6])
    other = StationWatch(
        tmp_path / "station.json",
        settings_path=tmp_path / "watch.json",
        log_path=tmp_path / "away.jsonl",
    )
    other.tick([], now=1.0, on_duty=True, write=False)
    assert other.work_start == "08:30"
    assert other.work_end == "17:00"
    assert 6 in other.workdays


def test_away_log_and_report(tmp_path):
    watch = _watch(tmp_path, away_s=5)
    watch.tick([_track(name="小明")], now=100.0, on_duty=True)
    watch.tick([], now=108.0, on_duty=True)
    watch.tick([_track(name="小明")], now=110.0, on_duty=True)
    report = watch.report(now=110.0)
    assert report["today"]["away_count"] == 1
    assert report["today"]["alarm_count"] == 1
    session = report["today"]["sessions"][0]
    assert session["alarm"] is True
    assert session["duration_s"] >= 8


def test_person_away_counts_even_if_someone_else_stays():
    from pocketshow.watch import person_away_today

    def ts(hour, minute, second=0):
        return datetime(2026, 9, 1, hour, minute, second).timestamp()

    events = [
        {"event": "enter", "t": ts(10, 0), "person_id": "p001", "name": "小明"},
        {"event": "enter", "t": ts(10, 0, 1), "person_id": "p002", "name": "小红"},
        {"event": "leave", "t": ts(10, 5), "person_id": "p001", "name": "小明"},
        {"event": "enter", "t": ts(10, 12), "person_id": "p001", "name": "小明"},
        {"event": "leave", "t": ts(10, 20), "person_id": "p001", "name": "小明"},
    ]
    out = person_away_today(
        events,
        now=ts(10, 30),
        min_s=15,
        alarm_s=30,
        present_ids={"p002"},
    )
    assert out["away_count"] == 2
    names = {row["name"] for row in out["sessions"]}
    assert names == {"小明"}
    long = [row for row in out["sessions"] if row["duration_s"] >= 400][0]
    assert long["alarm"] is True
    live = [row for row in out["sessions"] if row.get("live")][0]
    assert live["live"] is True
    assert out["people_count"] == 1
    assert out["by_person"][0]["name"] == "小明"
    assert out["by_person"][0]["alarm_count"] == 2
    assert out["alarm_count"] == 2
    assert out["alarm_s"] >= 400
    assert out["longest_s"] >= 400
    assert out["alarms"]


def test_person_away_skips_guests():
    from pocketshow.watch import person_away_today

    def ts(hour, minute, second=0):
        return datetime(2026, 9, 1, hour, minute, second).timestamp()

    events = [
        {"event": "enter", "t": ts(10, 0), "person_id": "p001", "name": "小明"},
        {"event": "leave", "t": ts(10, 5), "person_id": "p001", "name": "小明"},
        {"event": "enter", "t": ts(10, 0), "person_id": "p002", "name": "访客"},
        {"event": "leave", "t": ts(10, 20), "person_id": "p002", "name": "访客"},
    ]
    out = person_away_today(
        events,
        now=ts(10, 30),
        min_s=15,
        alarm_s=30,
        skip_ids={"p002"},
    )
    names = {row["name"] for row in out["sessions"]}
    assert names == {"小明"}
    assert out["people_count"] == 1
    assert out["by_person"][0]["name"] == "小明"
    assert all(row["person_id"] != "p002" for row in out["alarms"])


def test_person_away_ignores_one_frame_peek():
    from pocketshow.watch import person_away_today

    def ts(hour, minute, second=0):
        return datetime(2026, 9, 1, hour, minute, second).timestamp()

    events = [
        {"event": "enter", "t": ts(10, 0), "person_id": "p005", "name": "徐璐"},
        {
            "event": "leave",
            "t": ts(10, 0, 1),
            "person_id": "p005",
            "name": "徐璐",
            "duration_s": 0.0,
            "frames": 1,
        },
    ]
    out = person_away_today(events, now=ts(10, 30), min_s=15, alarm_s=5)
    assert out["away_count"] == 0
    assert out["sessions"] == []
    assert out["people_count"] == 0


def test_person_away_peek_does_not_clear_away():
    from pocketshow.watch import person_away_today

    def ts(hour, minute, second=0):
        return datetime(2026, 9, 1, hour, minute, second).timestamp()

    events = [
        {"event": "enter", "t": ts(10, 0), "person_id": "p005", "name": "徐璐"},
        {"event": "leave", "t": ts(10, 5), "person_id": "p005", "name": "徐璐", "duration_s": 300.0, "frames": 200},
        {"event": "enter", "t": ts(10, 12), "person_id": "p005", "name": "徐璐"},
        {
            "event": "leave",
            "t": ts(10, 12, 1),
            "person_id": "p005",
            "name": "徐璐",
            "duration_s": 0.2,
            "frames": 2,
        },
    ]
    out = person_away_today(events, now=ts(10, 30), min_s=15, alarm_s=30)
    assert out["away_count"] == 1
    live = out["sessions"][0]
    assert live["live"] is True
    assert live["name"] == "徐璐"
    assert live["duration_s"] >= 1400


def test_on_duty_roster_lists_present_and_empty_seats():
    from pocketshow.watch import on_duty_roster

    people = [
        {
            "id": "p001",
            "name": "阿强",
            "photo": "p001/cover.jpg",
            "seats": {"office": {"camera_name": "工位区1"}},
        },
        {
            "id": "p002",
            "name": "小美",
            "photo": "p002/cover.jpg",
            "seats": {"office": {"camera_name": "工位区1"}},
        },
        {"id": "p003", "name": "路人", "guest": True},
        {"id": "p009", "name": "没工位"},
    ]
    snapshot = {
        "fresh": True,
        "people": [
            {"person_id": "p001", "name": "阿强", "photo": "p001/cover.jpg", "duration_s": 12, "start_ts": "10:00:00"},
            {"person_id": "p003", "name": "路人", "guest": True, "duration_s": 3},
        ],
    }
    away = {
        "sessions": [
            {"person_id": "p002", "live": True, "alarm": True, "duration_s": 40, "start_ts": "10:01:00"},
        ]
    }
    out = on_duty_roster(people, snapshot, away, on_duty=True)
    assert out["count"] == 1
    assert out["guest_count"] == 1
    assert out["empty_count"] == 1
    assert out["seated_count"] == 2
    by = {row["person_id"]: row for row in out["people"]}
    assert by["p001"]["status"] == "at_desk"
    assert by["p001"]["label"] == "在岗"
    assert by["p002"]["status"] == "alarm"
    assert by["p002"]["seat_label"] == "工位区1"
    assert by["p003"]["guest"] is True
    assert "p009" not in by


def test_on_duty_roster_occupied_seat_counts_as_present():
    from pocketshow.watch import on_duty_roster

    people = [
        {"id": "p001", "name": "阿强", "seats": {"office": {"camera_name": "工位区1"}}},
        {"id": "p002", "name": "恒瑞", "seats": {"office": {"camera_name": "工位区1"}}},
    ]
    snapshot = {
        "fresh": True,
        "people": [{"person_id": "p001", "name": "阿强", "duration_s": 12, "start_ts": "18:01:00"}],
    }
    away = {
        "sessions": [
            {"person_id": "p002", "live": True, "alarm": False, "duration_s": 40, "start_ts": "18:04:05"},
        ]
    }
    out = on_duty_roster(people, snapshot, away, on_duty=True, occupied_ids={"p002"})
    assert out["count"] == 2
    assert out["empty_count"] == 0
    by = {row["person_id"]: row for row in out["people"]}
    assert by["p002"]["status"] == "at_desk"
    assert by["p002"]["label"] == "在岗"


def test_on_duty_roster_stale_snapshot_marks_empty_seats():
    from pocketshow.watch import on_duty_roster

    people = [{"id": "p001", "name": "阿强", "seats": {"cam": {"camera_name": "A"}}}]
    out = on_duty_roster(people, {"fresh": False, "people": [{"person_id": "p001"}]}, on_duty=True)
    assert out["fresh"] is False
    assert out["count"] == 0
    assert out["people"][0]["status"] == "stale"
    assert out["people"][0]["label"] == "等待监测"


def test_on_duty_roster_off_hours_hides_empty_seats():
    from pocketshow.watch import on_duty_roster

    people = [
        {"id": "p001", "name": "阿强", "seats": {"office": {"camera_name": "工位区1"}}},
        {"id": "p003", "name": "shengsheng", "seats": {"office": {"camera_name": "工位区1"}}},
        {"id": "p005", "name": "徐璐", "seats": {"office": {"camera_name": "工位区1"}}},
    ]
    snapshot = {
        "fresh": True,
        "people": [{"person_id": "p001", "name": "阿强", "duration_s": 12, "start_ts": "18:40:00"}],
    }
    out = on_duty_roster(people, snapshot, on_duty=False, occupied_ids={"p003"})
    assert out["count"] == 2
    assert out["empty_count"] == 0
    by = {row["person_id"]: row for row in out["people"]}
    assert by["p001"]["status"] == "at_desk"
    assert by["p003"]["status"] == "at_desk"
    assert by["p003"]["label"] == "在岗"
    assert "p005" not in by
