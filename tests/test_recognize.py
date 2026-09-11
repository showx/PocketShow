import numpy as np
import pytest

from pocketshow.config import RecognizeConfig
from pocketshow.recognize import cosine_sim, face_in_person, map_xyxy, person_head_crop, should_count_toward_enroll, should_skip_liveness
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
    assert should_count_toward_enroll(cfg, best_sim=0.40, face_score=0.95, face_px=80) is False
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
