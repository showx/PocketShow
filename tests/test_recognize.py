import numpy as np
import pytest

from pocketshow.config import RecognizeConfig
from pocketshow.recognize import (
    can_match_face,
    cosine_sim,
    enhance_for_face,
    face_in_person,
    face_yaw_ratio,
    fuse_identity,
    hits_needed,
    is_identity_face,
    map_xyxy,
    match_is_confident,
    person_head_crop,
    should_count_toward_enroll,
    should_skip_liveness,
    should_update_template,
)
from pocketshow.types import Track


def test_cosine_identical():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_sim(a, b) == 0.0


def test_cosine_zero_vector():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert cosine_sim(a, b) == 0.0


def test_face_in_upper_body():
    person = Track(id=1, bbox_xyxy=(0, 0, 100, 200), conf=0.9)
    assert face_in_person((30, 10, 70, 60), person)
    assert not face_in_person((30, 140, 70, 190), person)
    assert not face_in_person((200, 10, 240, 50), person)


def test_should_not_enroll_near_existing():
    cfg = RecognizeConfig()
    assert should_count_toward_enroll(cfg, best_sim=cfg.soft_threshold, face_score=0.95, face_px=80) is False
    assert should_count_toward_enroll(cfg, best_sim=0.10, face_score=0.95, face_px=20) is False
    assert should_count_toward_enroll(cfg, best_sim=0.10, face_score=0.95, face_px=80) is True


def test_skip_liveness_on_tiny_face():
    cfg = RecognizeConfig()
    assert should_skip_liveness(cfg, 18) is True
    assert should_skip_liveness(cfg, 80) is False
    assert should_skip_liveness(RecognizeConfig(liveness=False), 80) is True


def test_head_crop_upscales_and_maps_back():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cropped = person_head_crop(frame, (20.0, 40.0, 50.0, 120.0), min_side=128)
    assert cropped is not None
    crop, origin, scale = cropped
    assert origin == (15, 28)
    assert scale > 1.0
    assert min(crop.shape[0], crop.shape[1]) >= 128
    mapped = map_xyxy((10.0, 8.0, 40.0, 48.0), origin, scale)
    assert mapped[0] == 15 + 10.0 / scale
    assert mapped[2] > mapped[0]


def test_identity_requires_frontal_and_size():
    cfg = RecognizeConfig()
    frontal = np.array([0, 0, 100, 120, 30, 40, 70, 42, 50, 70, 35, 95, 65, 96, 0.9], dtype=np.float32)
    profile = np.array([0, 0, 100, 120, 20, 40, 38, 41, 85, 70, 30, 95, 45, 96, 0.9], dtype=np.float32)
    assert face_yaw_ratio(frontal) < 0.2
    assert face_yaw_ratio(profile) > 0.5
    assert is_identity_face(cfg, 80, 0.9, frontal) is True
    assert is_identity_face(cfg, 20, 0.9, frontal) is False
    assert is_identity_face(cfg, 80, 0.4, frontal) is False
    assert is_identity_face(cfg, 80, 0.9, profile) is False
    assert match_is_confident(0.62, 0.40, 0.50, 0.06) is True
    assert match_is_confident(0.51, 0.49, 0.50, 0.06) is False
    assert match_is_confident(0.44, -1.0, 0.50, 0.06) is False
    assert can_match_face(cfg, 22, 0.6, frontal) is True
    mild = np.array([0, 0, 100, 120, 30, 40, 70, 42, 70, 70, 35, 95, 65, 96, 0.9], dtype=np.float32)
    assert 0.45 < face_yaw_ratio(mild) < 0.72
    assert can_match_face(cfg, 22, 0.6, mild) is True
    assert can_match_face(cfg, 22, 0.6, profile) is False
    assert can_match_face(cfg, 10, 0.9, frontal) is False
    assert is_identity_face(cfg, 22, 0.9, frontal) is False


def test_head_crop_keeps_short_sitting_box():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cropped = person_head_crop(frame, (80.0, 140.0, 130.0, 190.0), min_side=128)
    assert cropped is not None
    crop, origin, scale = cropped
    assert origin[1] <= 140
    assert crop.shape[0] >= 128
    assert scale >= 1.0


def test_enhance_for_face_keeps_shape():
    image = np.linspace(20, 200, 80 * 60 * 3, dtype=np.float32).reshape(80, 60, 3).astype(np.uint8)
    out = enhance_for_face(image)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_fuse_identity_prefers_locked_seat_without_face():
    owner = {"id": "p003", "name": "shengsheng"}
    other = {"id": "p002", "name": "恒瑞"}
    person, sim, source = fuse_identity(
        None, -1.0, -1.0, owner, -1.0, match_threshold=0.50, soft_threshold=0.42, match_margin=0.06
    )
    assert source == "" and person is None
    person, sim, source = fuse_identity(
        other, 0.63, 0.40, owner, -1.0, match_threshold=0.50, soft_threshold=0.42, match_margin=0.06
    )
    assert source == "face" and person is other
    person, sim, source = fuse_identity(
        owner, 0.46, 0.20, owner, 0.46, match_threshold=0.50, soft_threshold=0.42, match_margin=0.06
    )
    assert source == "face_seat" and person is owner
    person, sim, source = fuse_identity(
        other, 0.58, 0.52, owner, 0.55, match_threshold=0.50, soft_threshold=0.42, match_margin=0.06
    )
    assert source == "" and person is None


def test_hits_and_template_update_gates():
    cfg = RecognizeConfig()
    assert hits_needed(cfg, 0.0, "seat") == 1
    assert hits_needed(cfg, 0.70, "face") == 1
    assert hits_needed(cfg, 0.52, "face") == 2
    assert should_update_template(cfg, "seat", True, 0.0, False) is True
    assert should_update_template(cfg, "seat", False, 0.8, False) is False
    assert should_update_template(cfg, "face", True, 0.40, True) is False
    assert should_update_template(cfg, "face_seat", True, 0.62, True) is True
    assert hits_needed(cfg, 0.62, "reid_seat") == 1
    assert hits_needed(cfg, 0.70, "reid") == 1


def test_fuse_identity_uses_reid_with_seat():
    owner = {"id": "p003", "name": "shengsheng"}
    other = {"id": "p002", "name": "恒瑞"}
    person, sim, source = fuse_identity(
        None,
        -1.0,
        -1.0,
        owner,
        -1.0,
        match_threshold=0.50,
        soft_threshold=0.42,
        match_margin=0.06,
        reid_person=owner,
        reid_sim=0.62,
        reid_second=0.30,
        seat_reid_sim=0.62,
    )
    assert source == "reid_seat" and person is owner
    person, sim, source = fuse_identity(
        None,
        -1.0,
        -1.0,
        owner,
        -1.0,
        match_threshold=0.50,
        soft_threshold=0.42,
        match_margin=0.06,
        reid_person=other,
        reid_sim=0.70,
        reid_second=0.30,
        seat_reid_sim=0.20,
    )
    assert source == "" and person is None
    person, sim, source = fuse_identity(
        None,
        -1.0,
        -1.0,
        None,
        -1.0,
        match_threshold=0.50,
        soft_threshold=0.42,
        match_margin=0.06,
        reid_person=other,
        reid_sim=0.70,
        reid_second=0.30,
    )
    assert source == "reid" and person is other
    person, sim, source = fuse_identity(
        None,
        -1.0,
        -1.0,
        owner,
        -1.0,
        match_threshold=0.50,
        soft_threshold=0.42,
        match_margin=0.06,
        reid_person=other,
        reid_sim=0.55,
        reid_second=0.30,
        seat_reid_sim=0.50,
    )
    assert source == "seat" and person is owner
    person, sim, source = fuse_identity(
        other,
        0.63,
        0.40,
        owner,
        -1.0,
        match_threshold=0.50,
        soft_threshold=0.42,
        match_margin=0.06,
        reid_person=owner,
        reid_sim=0.60,
        reid_second=0.20,
        seat_reid_sim=0.60,
        reid_soft=0.38,
    )
    assert source == "seat" and person is owner
