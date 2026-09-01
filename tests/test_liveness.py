import numpy as np

from pocketshow.config import RecognizeConfig
from pocketshow.liveness import geometry_penalty, is_live, scaled_crop, softmax
from pocketshow.recognize import PersonRecognizer
from pocketshow.types import Track


def test_softmax_peaks_on_live_class():
    probs = softmax([-2.0, 3.0, 0.1])
    assert abs(float(np.sum(probs)) - 1.0) < 1e-5
    assert int(np.argmax(probs)) == 1
    assert is_live(float(probs[1]), 0.5) is True
    assert is_live(0.2, 0.62) is False


def test_geometry_penalty_flags_headshot_photo():
    person = (0.0, 0.0, 100.0, 120.0)
    huge_face = (5.0, 5.0, 95.0, 110.0)
    small_face = (35.0, 10.0, 65.0, 50.0)
    assert geometry_penalty(person, huge_face) >= 0.3
    assert geometry_penalty(person, small_face) == 0.0


def test_scaled_crop_shape():
    image = np.zeros((200, 160, 3), dtype=np.uint8)
    crop = scaled_crop(image, (40, 20, 90, 80), scale=2.7, size=80)
    assert crop.shape == (80, 80, 3)


def test_live_needs_consecutive_passes():
    rec = object.__new__(PersonRecognizer)
    rec.cfg = RecognizeConfig(liveness_threshold=0.5, liveness_confirm=3)
    rec._live_hits = {}

    class Fake:
        def __init__(self, value: float) -> None:
            self.value = value

        def score(self, *_args, **_kwargs) -> float:
            return self.value

    rec.liveness = Fake(0.9)
    track = Track(id=7, bbox_xyxy=(0, 0, 100, 200), conf=0.9)
    face = {"xyxy": (30.0, 10.0, 70.0, 50.0)}
    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    assert rec._update_live(track, face, frame) is False
    assert track.live is None
    assert rec._update_live(track, face, frame) is False
    assert rec._update_live(track, face, frame) is True
    rec.liveness = Fake(0.1)
    assert rec._update_live(track, face, frame) is False
    assert track.live is False
