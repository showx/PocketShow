from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from pocketshow.config import RecognizeConfig
from pocketshow.gallery import FaceGallery, cosine_sim
from pocketshow.liveness import FaceLiveness, is_live
from pocketshow.types import Track

logger = logging.getLogger(__name__)

_MODEL_DIR = Path.home() / ".pocketshow" / "models"
_YUNET = (
    "face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
_SFACE = (
    "face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)


def should_count_toward_enroll(cfg: RecognizeConfig, best_sim: float, face_score: float, face_px: float) -> bool:
    if not cfg.auto_enroll:
        return False
    if best_sim >= cfg.soft_threshold:
        return False
    if face_score < cfg.enroll_score:
        return False
    return face_px >= cfg.enroll_min_face


def face_in_person(face_xyxy: tuple[float, float, float, float], person: Track) -> bool:
    fx1, fy1, fx2, fy2 = face_xyxy
    fcx, fcy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
    px1, py1, px2, py2 = person.bbox_xyxy
    if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
        return False
    return fcy <= py1 + 0.55 * (py2 - py1)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("下载人脸模型 %s", dest.name)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_models() -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in (_YUNET, _SFACE):
        path = _MODEL_DIR / name
        if not path.exists() or path.stat().st_size < 10_000:
            _download(url, path)
        paths.append(path)
    return paths[0], paths[1]


class PersonRecognizer:
    """YuNet 人脸检测 + SFace 特征，把人脸挂到人体框并给出稳定身份。"""

    def __init__(
        self,
        cfg: RecognizeConfig,
        gallery_path: Path | None = None,
        gallery: FaceGallery | None = None,
    ) -> None:
        self.cfg = cfg
        if gallery is not None:
            self.gallery = gallery
        else:
            json_path = Path(cfg.gallery) if gallery_path is None else gallery_path
            photos_dir = Path(cfg.photos) if cfg.photos else json_path.parent / "faces"
            self.gallery = FaceGallery(json_path, photos_dir)
        yunet, sface = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), cfg.det_score, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        self._input_size: tuple[int, int] | None = None
        self._id_memory: dict[int, tuple[str, str, float]] = {}
        self._pending_enroll: dict[int, int] = {}
        self._live_hits: dict[int, int] = {}
        self.liveness: FaceLiveness | None = None
        if cfg.liveness:
            try:
                self.liveness = FaceLiveness()
            except Exception:
                logger.exception("活体模型加载失败，照片可能仍会被认成真人")
        logger.info("已加载 %s 个登记人物", len(self.gallery.people))

    @property
    def people(self) -> list[dict]:
        return self.gallery.people

    @property
    def gallery_path(self) -> Path:
        return self.gallery.path

    def save(self) -> None:
        self.gallery.save()

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)
        _retval, faces = self.detector.detect(frame)
        if faces is None:
            return []
        out: list[dict] = []
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            score = float(row[-1])
            if score < self.cfg.det_score or bw < 24 or bh < 24:
                continue
            xyxy = (x, y, x + bw, y + bh)
            feat = self._embed(frame, row)
            if feat is None:
                continue
            out.append({"xyxy": xyxy, "score": score, "embedding": feat, "raw": row})
        return out

    def _embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        try:
            aligned = self.recognizer.alignCrop(frame, face_row)
            feat = self.recognizer.feature(aligned)
            vec = np.asarray(feat, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(vec))
            if n < 1e-6:
                return None
            return vec / n
        except cv2.error:
            return None

    def embed_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        faces = self.detect_faces(image)
        if not faces:
            return None
        best = max(
            faces,
            key=lambda f: f["score"] * (f["xyxy"][2] - f["xyxy"][0]) * (f["xyxy"][3] - f["xyxy"][1]),
        )
        return best["embedding"], best["xyxy"]

    def rank(self, embedding: np.ndarray, exclude_ids: set[str] | None = None) -> tuple[dict | None, float]:
        exclude_ids = exclude_ids or set()
        best, best_sim = None, -1.0
        for person in self.gallery.people:
            if person.get("id") in exclude_ids or not person.get("embedding"):
                continue
            sim = self.gallery.score(person, embedding)
            if sim > best_sim:
                best, best_sim = person, sim
        return best, best_sim

    def match(
        self,
        embedding: np.ndarray,
        exclude_ids: set[str] | None = None,
        threshold: float | None = None,
    ) -> tuple[dict | None, float]:
        person, sim = self.rank(embedding, exclude_ids)
        cutoff = self.cfg.match_threshold if threshold is None else threshold
        if person is None or sim < cutoff:
            return None, sim
        return person, sim

    def enroll(self, embedding: np.ndarray, name: str | None = None) -> dict:
        person = self.gallery.enroll(embedding, name=name)
        logger.info("登记 %s (%s)", person["name"], person["id"])
        return person

    def update_embedding(self, person: dict, embedding: np.ndarray, sim: float = 1.0) -> None:
        n = int(person.get("samples", 1))
        person["samples"] = n + 1
        if sim >= self.cfg.match_threshold:
            old = np.asarray(person["embedding"], dtype=np.float32)
            blended = (old * n + embedding) / (n + 1)
            norm = float(np.linalg.norm(blended))
            if norm > 1e-6:
                blended = blended / norm
            person["embedding"] = blended.astype(float).tolist()
        self.gallery.add_template(person, embedding)
        if n % 8 == 0:
            self.gallery.save()

    def _assign(self, track: Track, person: dict, sim: float, frame: np.ndarray, face: dict) -> None:
        track.person_id = person["id"]
        track.person_name = person["name"]
        track.face_score = sim
        self._id_memory[track.id] = (person["id"], person["name"], sim)
        self._pending_enroll.pop(track.id, None)
        self.update_embedding(person, face["embedding"], sim)
        self.gallery.maybe_save_live(person, frame, face["xyxy"])

    def _update_live(self, track: Track, face: dict, frame: np.ndarray) -> bool:
        if self.liveness is None:
            track.live = True
            track.live_score = 1.0
            return True
        score = self.liveness.score(frame, face["xyxy"], track.bbox_xyxy)
        track.live_score = score
        hits = self._live_hits.get(track.id, 0)
        if is_live(score, self.cfg.liveness_threshold):
            hits += 1
        else:
            hits = 0
        self._live_hits[track.id] = hits
        if hits >= self.cfg.liveness_confirm:
            track.live = True
        elif hits == 0:
            track.live = False
        else:
            track.live = None
        return track.live is True

    def apply(self, frame: np.ndarray, tracks: list[Track]) -> list[Track]:
        self.gallery.maybe_reload()
        if not tracks:
            self._id_memory.clear()
            self._pending_enroll.clear()
            self._live_hits.clear()
            self.gallery.appear.tick({})
            return tracks
        faces = self.detect_faces(frame)
        used_faces: set[int] = set()
        used_people: set[str] = set()
        paired: list[tuple[Track, dict]] = []
        for track in tracks:
            best_i, best_area = -1, -1.0
            for i, face in enumerate(faces):
                if i in used_faces:
                    continue
                if not face_in_person(face["xyxy"], track):
                    continue
                fx1, fy1, fx2, fy2 = face["xyxy"]
                area = (fx2 - fx1) * (fy2 - fy1) * face["score"]
                if area > best_area:
                    best_area, best_i = area, i
            if best_i < 0:
                continue
            used_faces.add(best_i)
            track.face_bbox = faces[best_i]["xyxy"]
            paired.append((track, faces[best_i]))
        paired.sort(
            key=lambda item: (item[1]["xyxy"][2] - item[1]["xyxy"][0]) * (item[1]["xyxy"][3] - item[1]["xyxy"][1]),
            reverse=True,
        )

        for track, face in paired:
            embedding = face["embedding"]
            fx1, fy1, fx2, fy2 = face["xyxy"]
            face_px = min(fx2 - fx1, fy2 - fy1)
            if not self._update_live(track, face, frame):
                self._pending_enroll.pop(track.id, None)
                continue
            person, sim = self.match(embedding, exclude_ids=used_people, threshold=self.cfg.soft_threshold)
            remembered = self._id_memory.get(track.id)

            if person is not None and sim >= self.cfg.match_threshold:
                self._assign(track, person, sim, frame, face)
                used_people.add(person["id"])
                continue

            if remembered is not None:
                mem_person = self.gallery.find(remembered[0])
                if mem_person is not None and mem_person["id"] not in used_people:
                    mem_sim = self.gallery.score(mem_person, embedding)
                    if mem_sim >= self.cfg.soft_threshold:
                        if mem_sim >= self.cfg.match_threshold:
                            self._assign(track, mem_person, mem_sim, frame, face)
                        else:
                            track.person_id = mem_person["id"]
                            track.person_name = mem_person["name"]
                            track.face_score = mem_sim
                            self._id_memory[track.id] = (mem_person["id"], mem_person["name"], mem_sim)
                            self._pending_enroll.pop(track.id, None)
                        used_people.add(mem_person["id"])
                        continue

            if person is not None and sim >= self.cfg.soft_threshold:
                self._assign(track, person, sim, frame, face)
                used_people.add(person["id"])
                continue

            _, all_sim = self.rank(embedding)
            if all_sim >= self.cfg.soft_threshold:
                self._pending_enroll.pop(track.id, None)
                continue

            if should_count_toward_enroll(self.cfg, sim if person else all_sim, face["score"], face_px):
                pending = self._pending_enroll.get(track.id, 0) + 1
                self._pending_enroll[track.id] = pending
                if pending >= self.cfg.enroll_confirm:
                    person = self.enroll(embedding)
                    track.person_id = person["id"]
                    track.person_name = person["name"]
                    track.face_score = 1.0
                    self._id_memory[track.id] = (person["id"], person["name"], 1.0)
                    self._pending_enroll.pop(track.id, None)
                    used_people.add(person["id"])
                    self.gallery.save_crop(person, frame, face["xyxy"], force_cover=True)
            else:
                self._pending_enroll.pop(track.id, None)

        alive = {t.id for t in tracks}
        spoofed = {t.id for t in tracks if t.face_bbox is not None and t.live is not True}
        for track in tracks:
            if track.person_id or track.id not in self._id_memory or track.id in spoofed:
                continue
            pid, name, score = self._id_memory[track.id]
            if pid in used_people:
                continue
            track.person_id = pid
            track.person_name = name
            track.face_score = score
            used_people.add(pid)
        self._id_memory = {k: v for k, v in self._id_memory.items() if k in alive}
        self._pending_enroll = {k: v for k, v in self._pending_enroll.items() if k in alive}
        self._live_hits = {k: v for k, v in self._live_hits.items() if k in alive}
        present: dict[str, dict] = {}
        for track in tracks:
            if not track.person_id:
                continue
            person = self.gallery.find(track.person_id)
            present[track.person_id] = {
                "name": track.person_name or (person.get("name") if person else track.person_id),
                "photo": (person.get("photo") if person else "") or "",
                "guest": bool(person.get("guest")) if person else False,
            }
        self.gallery.appear.tick(present)
        return tracks

    def enroll_track(self, frame: np.ndarray, track: Track, name: str | None = None) -> str | None:
        faces = self.detect_faces(frame)
        candidates = [f for f in faces if face_in_person(f["xyxy"], track)]
        if not candidates:
            logger.warning("当前目标没有可用人脸，无法登记")
            return None
        face = max(candidates, key=lambda f: f["score"])
        if self.liveness is not None:
            score = self.liveness.score(frame, face["xyxy"], track.bbox_xyxy)
            track.live_score = score
            if not is_live(score, self.cfg.liveness_threshold):
                track.live = False
                track.face_bbox = face["xyxy"]
                logger.warning("当前目标像照片/屏幕，未登记")
                return None
        track.live = True
        matched, sim = self.match(face["embedding"], threshold=self.cfg.soft_threshold)
        if matched is not None:
            if name:
                matched["name"] = name
            self.update_embedding(matched, face["embedding"], sim)
            self.gallery.save_crop(matched, frame, face["xyxy"], force_cover=True)
            self.gallery.save()
            track.person_id = matched["id"]
            track.person_name = matched["name"]
            track.face_bbox = face["xyxy"]
            self._id_memory[track.id] = (matched["id"], matched["name"], 1.0)
            return matched["name"]
        if track.person_id:
            for person in self.gallery.people:
                if person["id"] == track.person_id:
                    if name:
                        person["name"] = name
                    self.update_embedding(person, face["embedding"])
                    self.gallery.save_crop(person, frame, face["xyxy"], force_cover=True)
                    self.gallery.save()
                    track.person_name = person["name"]
                    track.face_bbox = face["xyxy"]
                    self._id_memory[track.id] = (person["id"], person["name"], 1.0)
                    return person["name"]
        person = self.enroll(face["embedding"], name=name)
        track.person_id = person["id"]
        track.person_name = person["name"]
        track.face_bbox = face["xyxy"]
        self._id_memory[track.id] = (person["id"], person["name"], 1.0)
        self.gallery.save_crop(person, frame, face["xyxy"], force_cover=True)
        return person["name"]
