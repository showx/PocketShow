import numpy as np
from fastapi.testclient import TestClient

from pocketshow.admin import create_app
from pocketshow.config import RecognizeConfig, Settings
from pocketshow.gallery import FaceGallery, crop_face, encode_jpeg


def test_crop_face_keeps_head_region():
    frame = np.zeros((240, 180, 3), dtype=np.uint8)
    frame[20:90, 50:120] = (0, 200, 255)
    crop = crop_face(frame, (50, 20, 120, 90))
    assert crop is not None
    assert crop.shape[0] >= 70
    assert crop.shape[1] >= 70


def test_crop_face_rejects_tiny_box():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert crop_face(frame, (0, 0, 2, 2), pad=0.0) is None


def test_gallery_enroll_rename_photo_delete(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    person = gallery.enroll(np.ones(8, dtype=np.float32), name="小李")
    assert person["id"] == "p001"
    renamed = gallery.rename(person["id"], "老李", note="左侧")
    assert renamed["name"] == "老李"
    assert renamed["note"] == "左侧"

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20:80, 30:90] = 180
    saved = gallery.save_crop(person, frame, (30, 20, 90, 80), force_cover=True)
    assert saved is not None
    assert saved.exists()
    cover = gallery.public(person)
    assert cover["photo_url"] == "/media/p001/cover.jpg"
    assert cover["photos"][0]["cover"] is True

    gallery.add_image_bytes(person, encode_jpeg(frame), as_cover=False)
    assert len(gallery.list_photos(person["id"])) == 2

    gallery.delete(person["id"])
    assert gallery.find("p001") is None
    assert not (tmp_path / "faces" / "p001").exists()


def test_next_id_skips_existing(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    gallery.enroll(np.ones(4, dtype=np.float32), name="A")
    gallery.enroll(np.ones(4, dtype=np.float32), name="B")
    gallery.delete("p001")
    person = gallery.enroll(np.ones(4, dtype=np.float32), name="C")
    assert person["id"] == "p003"


def test_admin_list_and_rename(tmp_path):
    json_path = tmp_path / "faces.json"
    photos = tmp_path / "faces"
    gallery = FaceGallery(json_path, photos)
    gallery.enroll(np.ones(4, dtype=np.float32), name="人物A")
    settings = Settings(recognize=RecognizeConfig(gallery=str(json_path), photos=str(photos)))
    client = TestClient(create_app(settings))
    listed = client.get("/api/people")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "人物A"
    patched = client.patch("/api/people/p001", json={"name": "阿强", "note": "蓝白条纹"})
    assert patched.status_code == 200
    assert patched.json()["name"] == "阿强"
    assert patched.json()["note"] == "蓝白条纹"
    page = client.get("/")
    assert page.status_code == 200
    assert "人物库" in page.text
    gone = client.delete("/api/people/p001")
    assert gone.status_code == 200
    assert client.get("/api/people").json() == []


def test_gallery_merge_and_twins(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    a = gallery.enroll(np.array([1.0, 0.0, 0.0], dtype=np.float32), name="人物B")
    b = gallery.enroll(np.array([0.92, 0.39, 0.0], dtype=np.float32), name="人物D")
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    gallery.save_crop(b, frame, (10, 10, 60, 60), force_cover=True)
    pairs = gallery.similar_pairs(threshold=0.3)
    assert any({p["a"], p["b"]} == {a["id"], b["id"]} for p in pairs)
    keep = gallery.merge(a["id"], b["id"])
    assert gallery.find(b["id"]) is None
    assert keep["samples"] >= 2
    assert (tmp_path / "faces" / a["id"] / "cover.jpg").exists() or list((tmp_path / "faces" / a["id"]).glob("*.jpg"))
    public = gallery.public_all(dup_threshold=0.3)
    assert public[0]["twins"] == []


def test_admin_merge(tmp_path):
    json_path = tmp_path / "faces.json"
    photos = tmp_path / "faces"
    gallery = FaceGallery(json_path, photos)
    gallery.enroll(np.array([1.0, 0.0], dtype=np.float32), name="人物B")
    gallery.enroll(np.array([0.99, 0.1], dtype=np.float32), name="人物D")
    settings = Settings(recognize=RecognizeConfig(gallery=str(json_path), photos=str(photos)))
    client = TestClient(create_app(settings))
    merged = client.post("/api/people/p001/merge", json={"source_id": "p002"})
    assert merged.status_code == 200
    listed = client.get("/api/people").json()
    assert len(listed) == 1
    assert listed[0]["name"] == "人物B"
    log = client.get("/api/appear")
    assert log.status_code == 200
    assert isinstance(log.json(), list)
