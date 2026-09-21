import numpy as np
import pytest
from fastapi.testclient import TestClient

from pocketshow.admin import create_app
from pocketshow.config import RecognizeConfig, Settings
from pocketshow.gallery import FaceGallery, crop_around, crop_face, encode_jpeg


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


def test_crop_around_keeps_click_region():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[70:130, 70:130] = (0, 200, 255)
    crop = crop_around(frame, 0.5, 0.5, rx=0.2, ry=0.2)
    assert crop is not None
    assert crop.shape[0] >= 70
    assert crop.shape[1] >= 70
    assert crop_around(frame, 0.5, 0.5, rx=0.01, ry=0.01) is None


def test_gallery_enroll_without_face(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    person = gallery.enroll(None, name="小周")
    assert person["id"] == "p001"
    assert person["embedding"] == []
    assert person["samples"] == 0
    pinned = gallery.pin_seat(person["id"], "office", 0.22, 0.71, camera_name="工位区1")
    assert pinned["seats"]["office"]["locked"] is True
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    saved = gallery.save_point_crop(person, frame, 0.22, 0.71)
    assert saved is not None
    assert gallery.public(person)["photo_url"] == "/media/p001/cover.jpg"
    faced = gallery.enroll(np.ones(4, dtype=np.float32), name="小李")
    keep = gallery.merge(person["id"], faced["id"])
    assert gallery.find(faced["id"]) is None
    assert keep["embedding"]
    assert keep["seats"]["office"]["locked"] is True


def test_gallery_enroll_rename_photo_delete(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    person = gallery.enroll(np.ones(8, dtype=np.float32), name="小李")
    assert person["id"] == "p001"
    renamed = gallery.rename(person["id"], "老李", note="左侧")
    assert renamed["name"] == "老李"
    assert renamed["note"] == "左侧"
    assert renamed.get("guest") is False
    guested = gallery.rename(person["id"], "老李", guest=True)
    assert guested["guest"] is True
    assert gallery.public(guested)["guest"] is True

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
    assert listed.json()[0]["seats"] == []
    gallery.find("p001")["seats"] = {"office": {"cx": 0.3, "cy": 0.6, "hits": 9, "camera_name": "工位区1"}}
    gallery.save()
    cleared = client.delete("/api/people/p001/seats", params={"camera_id": "office"})
    assert cleared.status_code == 200
    assert cleared.json()["seats"] == []
    patched = client.patch("/api/people/p001", json={"name": "阿强", "note": "蓝白条纹"})
    assert patched.status_code == 200
    assert patched.json()["name"] == "阿强"
    assert patched.json()["note"] == "蓝白条纹"
    assert patched.json()["guest"] is False
    guested = client.patch("/api/people/p001", json={"guest": True})
    assert guested.status_code == 200
    assert guested.json()["guest"] is True
    assert guested.json()["name"] == "阿强"
    page = client.get("/")
    assert page.status_code == 200
    assert "人物库" in page.text
    assert "离岗分析" in page.text
    assert "在岗人员" in page.text
    assert "离岗记录" in page.text
    assert 'id="presentPanel"' in page.text
    assert 'id="recordsPanel"' in page.text
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
    assert public[0]["seats"] == []


def test_gallery_seats_public_and_clear(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    person = gallery.enroll(np.ones(4, dtype=np.float32), name="靓仔培")
    person["seats"] = {"office": {"cx": 0.4, "cy": 0.7, "rx": 0.05, "ry": 0.1, "hits": 12, "camera_name": "工位区1"}}
    gallery.save()
    info = gallery.public(person)
    assert info["seats"][0]["camera_id"] == "office"
    assert info["seats"][0]["camera_name"] == "工位区1"
    gallery.clear_seat(person["id"], "office")
    assert gallery.public(gallery.find(person["id"]))["seats"] == []


def test_similar_pairs_uses_median_not_one_lucky_template(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    a = gallery.enroll(np.array([1.0, 0.0, 0.0], dtype=np.float32), name="人物A")
    b = gallery.enroll(np.array([0.0, 1.0, 0.0], dtype=np.float32), name="人物H")
    lucky = [0.70, 0.71, 0.0]
    a["templates"] = [[0.99, 0.01, 0.0], [0.98, 0.02, 0.0], [0.97, 0.03, 0.0], lucky]
    b["templates"] = [[0.01, 0.99, 0.0], [0.02, 0.98, 0.0], [0.03, 0.97, 0.0], lucky]
    assert gallery.pair_score(a, b) < 0.40
    assert gallery.similar_pairs(threshold=0.50) == []
    twins = gallery.public_all()
    assert twins[0]["twins"] == []
    assert twins[1]["twins"] == []


def test_similar_pairs_skips_people_who_shared_the_frame(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    a = gallery.enroll(np.array([1.0, 0.0], dtype=np.float32), name="人物A")
    b = gallery.enroll(np.array([0.99, 0.14], dtype=np.float32), name="人物H")
    assert gallery.pair_score(a, b) >= 0.70
    assert gallery.similar_pairs(threshold=0.70)
    gallery.appear.tick({a["id"]: {"name": "人物A"}, b["id"]: {"name": "人物H"}}, now=1000.0)
    for offset in range(1, 9):
        gallery.appear.tick({a["id"]: {"name": "人物A"}, b["id"]: {"name": "人物H"}}, now=1000.0 + offset)
    gallery.appear.tick({}, now=1020.0)
    assert gallery.similar_pairs(threshold=0.70) == []


def test_score_uses_templates_not_centroid(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    person = gallery.enroll(np.array([1.0, 0.0, 0.0], dtype=np.float32), name="人物A")
    person["templates"] = [[0.0, 1.0, 0.0]]
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert gallery.score(person, query) == pytest.approx(1.0, abs=1e-5)
    assert gallery.score(person, np.array([1.0, 0.0, 0.0], dtype=np.float32)) < 0.2


def test_prune_and_skip_colliding_templates(tmp_path):
    gallery = FaceGallery(tmp_path / "faces.json", tmp_path / "faces")
    a = gallery.enroll(np.array([1.0, 0.0, 0.0], dtype=np.float32), name="人物A")
    b = gallery.enroll(np.array([0.0, 1.0, 0.0], dtype=np.float32), name="人物B")
    twin = np.array([0.0, 0.99, 0.1], dtype=np.float32)
    own = np.array([0.98, 0.1, 0.0], dtype=np.float32)
    a["templates"] = [own.tolist(), twin.tolist()]
    assert gallery.add_template(b, twin, collide=0.55) is False
    dropped = gallery.prune_colliding_templates(0.55)
    assert dropped == 1
    assert len(a["templates"]) == 1
    assert gallery.score(a, own) > 0.9


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


def test_admin_watch_roster_includes_present_people(tmp_path):
    import json
    import time

    json_path = tmp_path / "faces.json"
    photos = tmp_path / "faces"
    gallery = FaceGallery(json_path, photos)
    person = gallery.enroll(np.ones(4, dtype=np.float32), name="阿强")
    gallery.pin_seat(person["id"], "office", 0.22, 0.71, camera_name="工位区1")
    now = time.time()
    (tmp_path / "present.json").write_text(
        json.dumps(
            {
                "updated": now,
                "people": [
                    {
                        "person_id": person["id"],
                        "name": "阿强",
                        "photo": person.get("photo") or "",
                        "start": now - 20,
                        "start_ts": "2026-09-14 10:00:00",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(recognize=RecognizeConfig(gallery=str(json_path), photos=str(photos)))
    client = TestClient(create_app(settings))
    roster = client.get("/api/watch").json()["roster"]
    assert roster["fresh"] is True
    assert roster["count"] == 1
    assert roster["people"][0]["status"] == "at_desk"
    assert roster["people"][0]["name"] == "阿强"
    assert roster["people"][0]["seat_label"] == "工位区1"
