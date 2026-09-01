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
