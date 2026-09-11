from pocketshow.detect_track import (
    EXTRA_ID_BASE,
    associate_by_iou,
    box_iou,
    far_tile_windows,
    nms_boxes,
    resolve_tracker,
    shift_box,
    unmatched_boxes,
)


def test_box_iou_identical_and_disjoint():
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == 1.0
    assert box_iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_far_tile_windows_cover_upper_band():
    windows = far_tile_windows(1000, 1920, ratio=0.75, tiles=2, overlap=0.2)
    assert len(windows) == 2
    assert windows[0][1] == 0 and windows[1][1] == 0
    assert windows[0][3] == 750 and windows[1][3] == 750
    assert windows[0][0] == 0
    assert windows[1][2] == 1920
    assert windows[0][2] > windows[1][0]


def test_shift_and_unmatched():
    shifted = shift_box((5.0, 6.0, 15.0, 26.0), (100, 40))
    assert shifted == (105.0, 46.0, 115.0, 66.0)
    existing = [(0.0, 0.0, 50.0, 80.0)]
    near = ((10.0, 10.0, 40.0, 70.0), 0.9)
    far = ((200.0, 20.0, 230.0, 90.0), 0.4)
    leftover = unmatched_boxes(existing, [near, far], iou_thresh=0.3)
    assert leftover == [far]


def test_nms_keeps_higher_score():
    a = ((0.0, 0.0, 20.0, 40.0), 0.9)
    b = ((2.0, 2.0, 22.0, 42.0), 0.4)
    c = ((80.0, 10.0, 100.0, 50.0), 0.5)
    kept = nms_boxes([a, b, c], iou_thresh=0.5)
    assert kept == [a, c]


def test_associate_reuses_id_and_drops_after_misses():
    prev = {EXTRA_ID_BASE: (10.0, 10.0, 30.0, 50.0)}
    detections = [((12.0, 12.0, 32.0, 52.0), 0.6)]
    assigned, alive, misses, next_id = associate_by_iou(prev, detections, next_id=EXTRA_ID_BASE + 3)
    assert assigned[0][0] == EXTRA_ID_BASE
    assert next_id == EXTRA_ID_BASE + 3
    assert misses[EXTRA_ID_BASE] == 0

    assigned, alive, misses, next_id = associate_by_iou(alive, [], next_id=next_id, max_miss=2, misses=misses)
    assert assigned == []
    assert EXTRA_ID_BASE in alive
    assigned, alive, misses, next_id = associate_by_iou(alive, [], next_id=next_id, max_miss=2, misses=misses)
    assigned, alive, misses, next_id = associate_by_iou(alive, [], next_id=next_id, max_miss=2, misses=misses)
    assert EXTRA_ID_BASE not in alive


def test_resolve_tracker_uses_bundled_yaml():
    from pathlib import Path

    path = resolve_tracker("bytetrack.yaml")
    assert Path(path).name == "bytetrack.yaml"
    assert Path(path).is_file()
    assert Path(path).parent.name == "pocketshow"
