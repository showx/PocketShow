import numpy as np

from pocketshow.gallery import FaceGallery
from pocketshow.reid import l2norm, person_crop, preprocess
from pocketshow.types import Track


def test_person_crop_pads_and_rejects_tiny():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:80, 40:90] = 180
    crop = person_crop(frame, (40.0, 20.0, 90.0, 80.0), pad=0.0)
    assert crop is not None
    assert crop.shape[0] == 60
    assert crop.shape[1] == 50
    assert person_crop(frame, (0.0, 0.0, 4.0, 4.0), min_side=16) is None


def test_preprocess_bgr_nchw():
    image = np.zeros((80, 40, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    blob = preprocess(image)
    assert blob.shape == (1, 3, 256, 128)
    assert blob.dtype == np.float32
    assert blob[0, 0].mean() == 10.0
    assert blob[0, 1].mean() == 20.0
    assert blob[0, 2].mean() == 30.0


def test_l2norm_and_zero():
    vec = l2norm(np.array([3.0, 4.0], dtype=np.float32))
    assert vec is not None
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    assert l2norm(np.zeros(8, dtype=np.float32)) is None


def test_gallery_appearance_score_and_collide(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    a = gallery.enroll(None, name="甲")
    b = gallery.enroll(None, name="乙")
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    gallery.update_appearance(a, vec_a, force=True)
    gallery.update_appearance(b, vec_b, force=True)
    query = np.array([0.97, 0.24, 0.0], dtype=np.float32)
    query = query / float(np.linalg.norm(query))
    assert gallery.score_appearance(a, query) > gallery.score_appearance(b, query)
    assert gallery.appearance_collides(vec_a, "p002", 0.9) is True
    assert gallery.appearance_collides(vec_a, "p001", 0.9) is False
    keep = gallery.merge(a["id"], b["id"])
    assert gallery.find(b["id"]) is None
    assert gallery.match_appearances(keep)


def test_track_can_hold_appearance():
    track = Track(id=3, bbox_xyxy=(1, 2, 10, 40), conf=0.5)
    track.appearance = np.ones(4, dtype=np.float32)
    track.reid_score = 0.61
    assert track.reid_score == 0.61
    assert np.allclose(track.appearance, 1.0)
