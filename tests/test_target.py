from pocketshow.target import TargetLock
from pocketshow.types import Track


def _person(tid: int, area_scale: float, conf: float = 0.9) -> Track:
    w = 40 * area_scale
    return Track(id=tid, bbox_xyxy=(10, 10, 10 + w, 10 + w), conf=conf)


def test_auto_selects_largest():
    lock = TargetLock()
    tracks = [_person(1, 1.0), _person(2, 3.0), _person(3, 1.2)]
    chosen = lock.update(tracks, dt=0.03)
    assert chosen is not None
    assert chosen.id == 2
    assert lock.locked_id == 2


def test_stickiness_while_missing():
    lock = TargetLock(lost_timeout_s=1.0)
    lock.update([_person(7, 2.0)], dt=0.03)
    assert lock.locked_id == 7
    missing = lock.update([_person(1, 5.0)], dt=0.03)
    assert missing is None
    assert lock.locked_id == 7


def test_reselect_after_timeout_if_auto():
    lock = TargetLock(lost_timeout_s=0.2)
    lock.update([_person(7, 2.0)], dt=0.03)
    gone = lock.update([_person(1, 5.0)], dt=0.25)
    assert gone is not None
    assert gone.id == 1


def test_manual_lock_does_not_steal():
    lock = TargetLock(lost_timeout_s=0.2)
    lock.lock(3)
    tracks = [_person(1, 9.0), _person(3, 1.0)]
    chosen = lock.update(tracks, dt=0.03)
    assert chosen is not None
    assert chosen.id == 3


def test_click_lock():
    lock = TargetLock()
    tracks = [
        Track(id=1, bbox_xyxy=(0, 0, 20, 20), conf=0.9),
        Track(id=2, bbox_xyxy=(50, 50, 90, 90), conf=0.9),
    ]
    assert lock.lock_at(60, 60, tracks) == 2
    assert lock.lock_at(0, 80, tracks) is None


def test_relock_by_person_id():
    lock = TargetLock(lost_timeout_s=2.0)
    first = Track(
        id=1,
        bbox_xyxy=(0, 0, 10, 10),
        conf=0.9,
        person_id="p001",
        person_name="人物A",
    )
    lock.lock(first)
    swapped = Track(
        id=9,
        bbox_xyxy=(0, 0, 12, 12),
        conf=0.9,
        person_id="p001",
        person_name="人物A",
    )
    other = Track(id=2, bbox_xyxy=(20, 20, 40, 40), conf=1.0)
    chosen = lock.update([swapped, other], dt=0.03)
    assert chosen is not None
    assert chosen.id == 9
    assert lock.locked_person_id == "p001"
