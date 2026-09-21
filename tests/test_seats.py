from pocketshow.seats import (
    blend_seat,
    box_pose,
    box_seat_dist,
    box_to_norm,
    boxes_from_tracks,
    camera_seats,
    can_name_at_seat,
    contains,
    ellipse_dist,
    hit_box,
    is_stationary,
    locked_away_ids,
    merge_seats,
    occupied_seat_ids,
    other_owns,
    pick_seat,
    pinned_seat,
    seat_search_box,
    seat_search_boxes,
    vacate_overlapping,
    xyz_dist,
)
from pocketshow.types import Track


def test_box_to_norm_and_contains():
    seat = box_to_norm((100, 200, 180, 360), 1000, 800)
    assert 0.13 < seat["cx"] < 0.15
    assert 0.34 < seat["cy"] < 0.36
    assert contains(seat, seat["cx"], seat["cy"])
    assert not contains(seat, 0.9, 0.9)


def test_blend_seat_moves_slowly():
    first = blend_seat(None, {"cx": 0.2, "cy": 0.7, "rx": 0.05, "ry": 0.1}, camera_name="工位区1")
    assert first["hits"] == 1
    assert first["camera_name"] == "工位区1"
    moved = {"cx": 0.8, "cy": 0.2, "rx": 0.05, "ry": 0.1}
    later = first
    for _ in range(8):
        later = blend_seat(later, moved)
    assert later["hits"] == 9
    assert later["cx"] < 0.78
    assert later["cx"] > 0.45


def test_pick_seat_unique_and_ambiguous():
    a = {"person_id": "p001", "name": "甲", "cx": 0.2, "cy": 0.7, "rx": 0.06, "ry": 0.1, "hits": 20}
    b = {"person_id": "p002", "name": "乙", "cx": 0.8, "cy": 0.7, "rx": 0.06, "ry": 0.1, "hits": 20}
    picked = pick_seat(0.21, 0.71, [a, b], set())
    assert picked is not None and picked["person_id"] == "p001"
    assert pick_seat(0.21, 0.71, [a, b], {"p001"}) is None
    close = {"person_id": "p003", "name": "丙", "cx": 0.22, "cy": 0.71, "rx": 0.06, "ry": 0.1, "hits": 20}
    assert pick_seat(0.21, 0.71, [a, close], set()) is None
    assert pick_seat(0.5, 0.5, [a, b], set()) is None
    assert pick_seat(0.21, 0.71, [a], set(), min_hits=50) is None


def test_other_owns_and_merge():
    people = [
        {"id": "p001", "name": "甲", "seats": {"office": {"cx": 0.2, "cy": 0.7, "rx": 0.06, "ry": 0.1, "hits": 20}}},
        {"id": "p002", "name": "乙", "seats": {"office": {"cx": 0.8, "cy": 0.7, "rx": 0.06, "ry": 0.1, "hits": 3}}},
    ]
    assert other_owns(0.2, 0.7, people, "office", "p002", min_hits=8) is True
    assert other_owns(0.8, 0.7, people, "office", "p001", min_hits=8) is False
    keep = {"id": "p001", "seats": {"office": {"cx": 0.2, "cy": 0.7, "hits": 5}}}
    src = {"id": "p009", "seats": {"office": {"cx": 0.21, "cy": 0.71, "hits": 40}, "office2": {"cx": 0.4, "cy": 0.4, "hits": 2}}}
    merge_seats(keep, src)
    assert keep["seats"]["office"]["hits"] == 40
    assert "office2" in keep["seats"]


def test_stationary_and_camera_seats():
    still = Track(id=1, bbox_xyxy=(0, 0, 40, 80), conf=0.9, vx=2.0, vy=1.0)
    moving = Track(id=2, bbox_xyxy=(0, 0, 40, 80), conf=0.9, vx=400.0, vy=10.0)
    assert is_stationary(still, 1920, 1080)
    assert not is_stationary(moving, 1920, 1080)
    people = [
        {"id": "p001", "name": "甲", "guest": False, "seats": {"office": {"cx": 0.2, "cy": 0.7, "rx": 0.05, "ry": 0.1, "hits": 12}}},
        {"id": "p002", "name": "客人", "guest": True, "seats": {"office": {"cx": 0.4, "cy": 0.4, "rx": 0.05, "ry": 0.1, "hits": 12}}},
    ]
    seats = camera_seats(people, "office", min_hits=8)
    assert [s["person_id"] for s in seats] == ["p001"]
    assert ellipse_dist(seats[0], 0.2, 0.7) < 0.01


def test_pinned_seat_overrides_and_matches():
    locked = pinned_seat(0.21, 0.72, camera_name="工位区1")
    assert locked["locked"] is True
    assert locked["hits"] >= 100
    people = [
        {"id": "p001", "name": "甲", "seats": {"office": {"cx": 0.2, "cy": 0.7, "rx": 0.06, "ry": 0.1, "hits": 20}}},
        {"id": "p002", "name": "乙", "seats": {}},
    ]
    vacate_overlapping(people, "office", locked, "p002")
    assert "office" not in people[0]["seats"]
    people[1]["seats"]["office"] = locked
    seats = camera_seats(people, "office", min_hits=8)
    assert [s["person_id"] for s in seats] == ["p002"]
    picked = pick_seat(0.22, 0.73, seats, set(), min_hits=8)
    assert picked is not None and picked["person_id"] == "p002"


def test_locked_seat_works_without_face_embedding():
    people = [
        {
            "id": "p001",
            "name": "小周",
            "guest": False,
            "embedding": [],
            "seats": {"office": pinned_seat(0.2, 0.7, camera_name="工位区1")},
        }
    ]
    seats = camera_seats(people, "office", min_hits=8)
    assert [s["person_id"] for s in seats] == ["p001"]
    picked = pick_seat(0.21, 0.71, seats, set(), min_hits=8)
    assert picked is not None and picked["person_id"] == "p001"


def test_hit_box_picks_smallest_and_pose():
    boxes = [
        {"id": 1, "x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9, "name": "大框"},
        {"id": 2, "x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.7, "name": "小框"},
    ]
    hit = hit_box(boxes, 0.5, 0.5)
    assert hit is not None and hit["id"] == 2
    assert hit_box(boxes, 0.05, 0.05) is None
    cx, cy, rx, ry = box_pose(hit)
    assert 0.49 < cx < 0.51
    assert 0.54 < cy < 0.56
    assert abs(rx - 0.1) < 1e-9
    assert abs(ry - 0.15) < 1e-9
    tracks = [
        Track(id=7, bbox_xyxy=(100, 80, 180, 240), conf=0.9, person_name="小周"),
    ]
    published = boxes_from_tracks(tracks, 320, 240)
    assert published[0]["id"] == 7
    assert published[0]["name"] == "小周"
    assert published[0]["x1"] == 0.3125


def test_sitting_head_box_still_hits_chair_seat():
    seat = {
        "person_id": "p002",
        "name": "恒瑞",
        "cx": 0.433,
        "cy": 0.808,
        "rx": 0.05,
        "ry": 0.10,
        "hits": 999,
        "locked": True,
    }
    width, height = 640, 360
    bbox = (0.40 * width, 0.62 * height, 0.52 * width, 0.78 * height)
    cx = 0.46
    cy = 0.70
    assert pick_seat(cx, cy, [seat], set(), slack=1.0) is None
    assert box_seat_dist(seat, bbox, width, height) < 1.0
    picked = pick_seat(cx, cy, [seat], set(), bbox=bbox, width=width, height=height)
    assert picked is not None and picked["person_id"] == "p002"
    neighbor = {
        "person_id": "p001",
        "name": "喜明",
        "cx": 0.618,
        "cy": 0.759,
        "rx": 0.058,
        "ry": 0.14,
        "hits": 999,
        "locked": True,
    }
    picked = pick_seat(cx, cy, [seat, neighbor], set(), bbox=bbox, width=width, height=height)
    assert picked is not None and picked["person_id"] == "p002"
    far = (0.80 * width, 0.30 * height, 0.88 * width, 0.42 * height)
    assert pick_seat(0.84, 0.36, [seat], set(), bbox=far, width=width, height=height) is None


def test_head_above_monitor_hits_own_seat_not_front_row():
    hengrui = {
        "person_id": "p002",
        "name": "恒瑞",
        "cx": 0.433,
        "cy": 0.808,
        "rx": 0.05,
        "ry": 0.10,
        "hits": 999,
        "locked": True,
    }
    xulu = {
        "person_id": "p005",
        "name": "徐璐",
        "cx": 0.409,
        "cy": 0.510,
        "rx": 0.025,
        "ry": 0.05,
        "hits": 999,
        "locked": True,
    }
    sheng = {
        "person_id": "p003",
        "name": "shengsheng",
        "cx": 0.879,
        "cy": 0.701,
        "rx": 0.061,
        "ry": 0.135,
        "hits": 999,
        "locked": True,
    }
    head = (0.40, 0.55, 0.50, 0.68)
    picked = pick_seat(0.45, 0.615, [hengrui, xulu], set(), bbox=head, width=1, height=1)
    assert picked is not None and picked["person_id"] == "p002"
    xulu_box = (0.399, 0.470, 0.426, 0.532)
    picked = pick_seat(0.413, 0.501, [hengrui, xulu], set(), bbox=xulu_box, width=1, height=1)
    assert picked is not None and picked["person_id"] == "p005"
    back = (0.421, 0.401, 0.498, 0.528)
    assert pick_seat(0.46, 0.465, [hengrui], set(), bbox=back, width=1, height=1) is None
    sheng_head = (0.84, 0.48, 0.94, 0.62)
    picked = pick_seat(0.89, 0.55, [sheng, hengrui], set(), bbox=sheng_head, width=1, height=1)
    assert picked is not None and picked["person_id"] == "p003"
    far = (0.828, 0.381, 0.867, 0.475)
    assert pick_seat(0.848, 0.428, [sheng], set(), bbox=far, width=1, height=1) is None
    hoodie = (0.769, 0.378, 0.812, 0.512)
    picked = pick_seat(0.791, 0.445, [sheng, hengrui], set(), bbox=hoodie, width=1, height=1)
    assert picked is not None and picked["person_id"] == "p003"


def test_seat_search_box_covers_head_above_chair():
    window = seat_search_box({"cx": 0.433, "cy": 0.808, "rx": 0.05, "ry": 0.10})
    assert window["x1"] < 0.433 < window["x2"]
    assert window["y1"] < 0.65 < window["y2"]
    assert window["y1"] < 0.808 < window["y2"]
    assert window["y2"] > 0.808
    assert window["x2"] - window["x1"] <= 0.14
    assert window["y1"] >= 0.60
    assert len(seat_search_boxes({"cx": 0.433, "cy": 0.808, "rx": 0.05, "ry": 0.10})) == 2


def test_occupied_seat_ids_needs_named_person_box():
    from pocketshow.seats import occupied_seat_ids, person_like_box, pinned_seat

    assert person_like_box({"x1": 0.40, "y1": 0.55, "x2": 0.50, "y2": 0.68})
    assert not person_like_box({"x1": 0.417, "y1": 0.778, "x2": 0.425, "y2": 0.828})
    people = [
        {"id": "p002", "name": "恒瑞", "seats": {"office": pinned_seat(0.433, 0.808)}},
        {"id": "p004", "name": "宇翔", "seats": {"office": pinned_seat(0.669, 0.451)}},
        {"id": "p005", "name": "徐璐", "seats": {"office": pinned_seat(0.409, 0.510, rx=0.025, ry=0.05)}},
    ]
    cameras = [
        {
            "id": "office",
            "boxes": [
                {"x1": 0.64, "y1": 0.36, "x2": 0.70, "y2": 0.54, "person_id": "p004", "name": "宇翔"},
                {"x1": 0.40, "y1": 0.55, "x2": 0.50, "y2": 0.68},
                {"x1": 0.417, "y1": 0.778, "x2": 0.425, "y2": 0.828, "person_id": "p005", "name": "徐璐"},
            ],
        }
    ]
    ids = occupied_seat_ids(people, cameras)
    assert ids == {"p004"}


def test_locked_person_only_named_on_own_seat():
    hengrui = {"id": "p002", "name": "恒瑞", "seats": {"office": pinned_seat(0.433, 0.808)}}
    ximing = {"id": "p001", "name": "喜明", "seats": {"office": pinned_seat(0.618, 0.759)}}
    guest = {"id": "g1", "name": "路人", "guest": True}
    assert can_name_at_seat(hengrui, "office", "p002") is True
    assert can_name_at_seat(hengrui, "office", "p001") is False
    assert can_name_at_seat(hengrui, "office", "") is False
    assert can_name_at_seat(ximing, "office", "p002") is False
    assert can_name_at_seat(guest, "office", "") is True
    assert locked_away_ids([hengrui, ximing, guest], "office", "p002") == {"p001"}
    assert locked_away_ids([hengrui, ximing], "office", "") == {"p001", "p002"}


def test_pick_seat_uses_xyz_when_both_have_it():
    near = {
        "person_id": "p001",
        "name": "甲",
        "cx": 0.8,
        "cy": 0.8,
        "rx": 0.06,
        "ry": 0.1,
        "hits": 20,
        "xyz": [0.1, 1.4, 2.0],
    }
    far = {
        "person_id": "p002",
        "name": "乙",
        "cx": 0.21,
        "cy": 0.71,
        "rx": 0.06,
        "ry": 0.1,
        "hits": 20,
        "xyz": [3.0, 1.4, 2.0],
    }
    picked = pick_seat(0.21, 0.71, [near, far], set(), xyz=(0.12, 1.41, 2.02))
    assert picked is not None and picked["person_id"] == "p001"
    assert xyz_dist((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0
    blended = blend_seat(None, {"cx": 0.2, "cy": 0.7, "rx": 0.05, "ry": 0.1, "xyz": [1.0, 1.5, 2.0]})
    assert blended["xyz"] == [1.0, 1.5, 2.0]
    pinned = pinned_seat(0.2, 0.7, xyz=(1.1, 1.4, 2.2))
    assert pinned["xyz"] == [1.1, 1.4, 2.2]
