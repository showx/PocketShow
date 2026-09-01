from pocketshow.appear import AppearanceLog


def test_appearance_enter_and_leave(tmp_path):
    log = AppearanceLog(tmp_path / "appear.jsonl", gap_s=1.0)
    log.tick({"p001": {"name": "人物A", "photo": "p001/cover.jpg"}}, now=1000.0)
    log.tick({"p001": {"name": "人物A", "photo": "p001/cover.jpg"}}, now=1000.5)
    assert log.live["p001"]["frames"] == 2
    log.tick({}, now=1000.8)
    assert "p001" in log.live
    log.tick({}, now=1002.2)
    assert "p001" not in log.live
    visits = log.visits()
    assert visits
    row = visits[0]
    assert row["person_id"] == "p001"
    assert row["live"] is False
    assert row["duration_s"] == 0.5


def test_appearance_filter_by_person(tmp_path):
    log = AppearanceLog(tmp_path / "appear.jsonl", gap_s=0.1)
    log.tick({"p001": {"name": "A"}, "p002": {"name": "B"}}, now=10.0)
    log.tick({}, now=11.0)
    only_a = log.visits(person_id="p001")
    assert all(v["person_id"] == "p001" for v in only_a)
    assert only_a


def test_present_snapshot(tmp_path):
    log = AppearanceLog(tmp_path / "appear.jsonl", gap_s=1.0)
    log.tick({"p001": {"name": "人物A", "photo": "p001/cover.jpg"}}, now=1000.0)
    data = log.present(now=1000.2)
    assert data["fresh"] is True
    assert data["count"] == 1
    assert data["people"][0]["name"] == "人物A"
    assert data["people"][0]["live"] is True
    log.tick({}, now=1002.2)
    gone = log.present(now=1002.3)
    assert gone["count"] == 0
    stale = log.present(now=1010.0)
    assert stale["fresh"] is False
    assert stale["count"] == 0


def test_concurrent_pairs_from_overlapping_visits(tmp_path):
    log = AppearanceLog(tmp_path / "appear.jsonl", gap_s=0.5)
    log.tick({"p001": {"name": "A"}, "p002": {"name": "B"}}, now=1000.0)
    log.tick({"p001": {"name": "A"}, "p002": {"name": "B"}}, now=1010.0)
    log.tick({}, now=1020.0)
    assert ("p001", "p002") in log.concurrent_pairs(min_overlap_s=8.0)
    assert log.concurrent_pairs(min_overlap_s=30.0) == set()
