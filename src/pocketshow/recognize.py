from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from pocketshow.config import RecognizeConfig
from pocketshow.gallery import FaceGallery, cosine_sim
from pocketshow.liveness import FaceLiveness, is_live
from pocketshow.reid import PersonReID
from pocketshow.seats import (
    blend_seat,
    box_to_norm,
    camera_seats,
    can_name_at_seat,
    is_stationary,
    locked_away_ids,
    other_owns,
    person_like_box,
    person_seats,
    pick_seat,
)
from pocketshow.types import Track

logger = logging.getLogger(__name__)

_SEAT_HIT_HOLD_S = 4.0

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


def should_skip_liveness(cfg: RecognizeConfig, face_px: float) -> bool:
    """远处小脸做不了活体，拦掉反而认不出人。"""
    if not cfg.liveness:
        return True
    return face_px < cfg.liveness_min_face


def face_yaw_ratio(row: np.ndarray | None) -> float:
    """0 为正脸，越大越侧。缺关键点时当作不可认。"""
    if row is None:
        return 1.0
    vals = np.asarray(row, dtype=np.float32).reshape(-1)
    if vals.size < 10:
        return 1.0
    re_x, re_y = float(vals[4]), float(vals[5])
    le_x, le_y = float(vals[6]), float(vals[7])
    nose_x = float(vals[8])
    eye_span = abs(le_x - re_x)
    if eye_span < 3:
        return 1.0
    if abs(le_y - re_y) > eye_span * 0.85:
        return 1.0
    mid = (re_x + le_x) * 0.5
    return abs(nose_x - mid) / eye_span


def is_identity_face(cfg: RecognizeConfig, face_px: float, face_score: float, raw: np.ndarray | None) -> bool:
    """高质量正脸才写入底库，避免糊脸把档案冲坏。"""
    if face_px < cfg.id_min_face:
        return False
    if face_score < cfg.id_min_score:
        return False
    return face_yaw_ratio(raw) <= cfg.id_max_yaw


def can_match_face(cfg: RecognizeConfig, face_px: float, face_score: float, raw: np.ndarray | None) -> bool:
    """比对可以比入库松：子码流小脸、略侧也先拿去跟底库对。"""
    if face_px < cfg.match_min_face:
        return False
    if face_score < cfg.match_min_score:
        return False
    return face_yaw_ratio(raw) <= cfg.match_max_yaw


def match_is_confident(best_sim: float, second_sim: float, threshold: float, margin: float) -> bool:
    if best_sim < threshold:
        return False
    if second_sim >= 0 and best_sim - second_sim < margin:
        return False
    return True


def fuse_identity(
    matched: dict | None,
    sim: float,
    second_sim: float,
    seat_person: dict | None,
    seat_sim: float,
    *,
    match_threshold: float,
    soft_threshold: float,
    match_margin: float,
    reid_person: dict | None = None,
    reid_sim: float = -1.0,
    reid_second: float = -1.0,
    seat_reid_sim: float = -1.0,
    reid_threshold: float = 0.48,
    reid_soft: float = 0.38,
    reid_margin: float = 0.08,
) -> tuple[dict | None, float, str]:
    """人脸、人体外观、锁定工位一起看。工位上坐了别人时，不能因为钉点就把名字套给主人。"""
    seat_id = str(seat_person.get("id") or "") if seat_person else ""
    match_id = str(matched.get("id") or "") if matched else ""
    reid_id = str(reid_person.get("id") or "") if reid_person else ""
    face_ok = bool(matched) and match_is_confident(sim, second_sim, match_threshold, match_margin)
    reid_ok = bool(reid_person) and match_is_confident(reid_sim, reid_second, reid_threshold, reid_margin)
    same_seat = bool(matched) and bool(seat_id) and match_id == seat_id and sim >= soft_threshold
    reid_same = bool(reid_person) and bool(seat_id) and reid_id == seat_id and reid_sim >= reid_soft
    if face_ok and seat_id and match_id != seat_id:
        if reid_ok and reid_id == match_id:
            return matched, sim, "face"
        if reid_same or seat_reid_sim >= reid_soft:
            return seat_person, max(float(seat_reid_sim), 0.0), "seat"
        if seat_sim >= 0 and sim - seat_sim < match_margin:
            if seat_sim >= soft_threshold:
                return seat_person, seat_sim, "seat"
            return None, sim, ""
        return matched, sim, "face"
    if face_ok and same_seat:
        return matched, sim, "face_seat"
    if face_ok:
        return matched, sim, "face"
    if same_seat:
        return matched, sim, "face_seat"
    if reid_same:
        return reid_person, reid_sim, "reid_seat"
    if seat_person is not None:
        if seat_reid_sim >= reid_soft:
            return seat_person, seat_reid_sim, "seat"
        if reid_ok and reid_id != seat_id:
            return None, reid_sim, ""
        return None, max(float(seat_reid_sim), 0.0), ""
    if reid_ok:
        return reid_person, reid_sim, "reid"
    return None, sim, ""


def hits_needed(cfg: RecognizeConfig, sim: float, source: str) -> int:
    if source in {"seat", "face_seat", "reid_seat"}:
        return 1
    if source == "reid" and sim >= cfg.reid_threshold + 0.08:
        return 1
    if sim >= cfg.match_threshold + 0.10:
        return 1
    return max(1, int(cfg.id_confirm))


def should_update_template(cfg: RecognizeConfig, source: str, can_learn: bool, sim: float, has_embedding: bool) -> bool:
    if not can_learn:
        return False
    if source == "seat" and not has_embedding:
        return True
    return sim >= cfg.update_min_sim


def face_in_person(face_xyxy: tuple[float, float, float, float], person: Track) -> bool:
    fx1, fy1, fx2, fy2 = face_xyxy
    fcx, fcy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
    px1, py1, px2, py2 = person.bbox_xyxy
    if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
        return False
    return fcy <= py1 + 0.70 * (py2 - py1)


_CLAHE: cv2.CLAHE | None = None


def enhance_for_face(image: np.ndarray) -> np.ndarray:
    """监控子码流对比度差，CLAHE 后再交给 YuNet / SFace。"""
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return image
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_CLAHE.apply(lightness), a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def person_head_crop(
    frame: np.ndarray,
    person_xyxy: tuple[float, float, float, float],
    *,
    head_ratio: float = 0.58,
    pad: float = 0.14,
    min_side: int = 128,
) -> tuple[np.ndarray, tuple[int, int], float] | None:
    """裁人体上半并放大，让远处小脸够 YuNet 检。坐着只检出头肩时把整框都留下。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_xyxy
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if bh / max(h, 1) <= 0.32 or bh / bw <= 1.4:
        head_ratio = max(head_ratio, 1.08)
    rx1 = max(0, int(x1 - bw * pad))
    ry1 = max(0, int(y1 - bh * pad))
    rx2 = min(w, int(x2 + bw * pad))
    ry2 = min(h, int(y1 + bh * head_ratio + bh * pad))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = 1.0
    shortest = min(ch, cw)
    if shortest < min_side:
        scale = min_side / max(shortest, 1)
        crop = cv2.resize(crop, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), interpolation=cv2.INTER_CUBIC)
    return enhance_for_face(crop), (rx1, ry1), scale


def map_xyxy(
    xyxy: tuple[float, float, float, float],
    origin: tuple[int, int],
    scale: float,
) -> tuple[float, float, float, float]:
    ox, oy = origin
    scale = scale if scale > 1e-6 else 1.0
    x1, y1, x2, y2 = xyxy
    return (x1 / scale + ox, y1 / scale + oy, x2 / scale + ox, y2 / scale + oy)


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
        self._id_hits: dict[str, tuple[float, int]] = {}
        self._seat_hits: dict[str, tuple[float, int]] = {}
        self._app_seed_tried: set[str] = set()
        self.liveness: FaceLiveness | None = None
        self.reid: PersonReID | None = None
        if cfg.liveness:
            try:
                self.liveness = FaceLiveness()
            except Exception:
                logger.exception("活体模型加载失败，照片可能仍会被认成真人")
        if cfg.reid:
            try:
                self.reid = PersonReID()
            except Exception:
                logger.exception("人体 ReID 加载失败，后脑勺只能靠工位")
        dropped = self.gallery.prune_colliding_templates(cfg.template_collide)
        if dropped:
            logger.info("已清理 %s 个互相撞车的人脸模板", dropped)
        dropped_app = self.gallery.prune_colliding_appearances(cfg.reid_collide)
        if dropped_app:
            logger.info("已清理 %s 个互相撞车的外观模板", dropped_app)
        seeded = self.seed_missing_embeddings()
        if seeded:
            logger.info("已从留底相片补了 %s 个人的人脸特征", seeded)
        seeded_app = self.seed_missing_appearances()
        if seeded_app:
            logger.info("已从留底相片补了 %s 个人的外观特征", seeded_app)
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
        work = enhance_for_face(frame) if 240 <= min(h, w) <= 960 else frame
        self.detector.setInputSize((w, h))
        self._input_size = (w, h)
        _retval, faces = self.detector.detect(work)
        if faces is None:
            return []
        out: list[dict] = []
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            score = float(row[-1])
            if score < self.cfg.det_score or bw < self.cfg.det_min_face or bh < self.cfg.det_min_face:
                continue
            xyxy = (x, y, x + bw, y + bh)
            feat = self._embed(work, row)
            if feat is None:
                continue
            out.append({"xyxy": xyxy, "score": score, "embedding": feat, "raw": row})
        return out

    def _detect_face_in_track(self, frame: np.ndarray, track: Track) -> dict | None:
        cropped = person_head_crop(frame, track.bbox_xyxy)
        if cropped is None:
            return None
        crop, origin, scale = cropped
        best, best_area = None, -1.0
        for face in self.detect_faces(crop):
            mapped = dict(face)
            mapped["xyxy"] = map_xyxy(face["xyxy"], origin, scale)
            if not face_in_person(mapped["xyxy"], track):
                continue
            fx1, fy1, fx2, fy2 = mapped["xyxy"]
            area = (fx2 - fx1) * (fy2 - fy1) * mapped["score"]
            if area > best_area:
                best, best_area = mapped, area
        return best

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
        best, best_sim, _second = self.rank_top(embedding, exclude_ids)
        return best, best_sim

    def rank_top(
        self,
        embedding: np.ndarray,
        exclude_ids: set[str] | None = None,
    ) -> tuple[dict | None, float, float]:
        exclude_ids = exclude_ids or set()
        best, best_sim = None, -1.0
        second_sim = -1.0
        for person in self.gallery.people:
            if person.get("id") in exclude_ids or not person.get("embedding"):
                continue
            sim = self.gallery.score(person, embedding)
            if sim > best_sim:
                second_sim = best_sim
                best, best_sim = person, sim
            elif sim > second_sim:
                second_sim = sim
        return best, best_sim, second_sim

    def match(
        self,
        embedding: np.ndarray,
        exclude_ids: set[str] | None = None,
        threshold: float | None = None,
        *,
        require_margin: bool = True,
    ) -> tuple[dict | None, float]:
        person, sim, second = self.rank_top(embedding, exclude_ids)
        cutoff = self.cfg.match_threshold if threshold is None else threshold
        if person is None or not match_is_confident(sim, second, cutoff, self.cfg.match_margin if require_margin else 0.0):
            return None, sim
        return person, sim

    def enroll(self, embedding: np.ndarray, name: str | None = None) -> dict:
        person = self.gallery.enroll(embedding, name=name)
        logger.info("登记 %s (%s)", person["name"], person["id"])
        return person

    def update_embedding(self, person: dict, embedding: np.ndarray, sim: float = 1.0, *, force: bool = False) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        n = int(person.get("samples", 0) or 0)
        old = np.asarray(person.get("embedding") or [], dtype=np.float32)
        if old.size == 0:
            person["embedding"] = vec.astype(float).tolist()
            person["samples"] = max(1, n + 1)
            self.gallery.add_template(
                person,
                vec,
                collide=self.cfg.template_collide,
                force=True,
            )
            self.gallery.save()
            return
        person["samples"] = n + 1
        if force or sim >= self.cfg.update_min_sim:
            alpha = 1.0 if force else 0.08
            blended = (1.0 - alpha) * old + alpha * vec
            norm = float(np.linalg.norm(blended))
            if norm > 1e-6:
                blended = blended / norm
            person["embedding"] = blended.astype(float).tolist()
            self.gallery.add_template(
                person,
                vec,
                collide=self.cfg.template_collide,
                force=force,
            )
        if person["samples"] % 8 == 0:
            self.gallery.save()

    def seed_missing_embeddings(self) -> int:
        """点工位登记的人没有脸向量。留底图里能抠出脸就补上。"""
        filled = 0
        for person in self.gallery.people:
            if np.asarray(person.get("embedding") or [], dtype=np.float32).size:
                continue
            rel = str(person.get("photo") or "")
            path = self.gallery.resolve_media(rel) if rel else None
            if path is None:
                continue
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                continue
            found = self.embed_image(image)
            if found is None:
                continue
            self.update_embedding(person, found[0], force=True)
            filled += 1
        if filled:
            self.gallery.save()
        return filled

    def seed_missing_appearances(self) -> int:
        """点工位留下的封面图正好能当人体外观底库。"""
        if self.reid is None:
            return 0
        filled = 0
        for person in self.gallery.people:
            pid = str(person.get("id") or "")
            if not pid or pid in self._app_seed_tried:
                continue
            self._app_seed_tried.add(pid)
            if self.gallery.match_appearances(person):
                continue
            rel = str(person.get("photo") or "")
            path = self.gallery.resolve_media(rel) if rel else None
            if path is None:
                continue
            if float(person.get("photo_area") or 0) < 4000:
                continue
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                continue
            vec = self.reid.embed_image(image)
            if vec is None:
                continue
            if self.gallery.appearance_collides(vec, pid, self.cfg.reid_collide):
                continue
            self.gallery.update_appearance(person, vec, force=True, collide=self.cfg.reid_collide)
            filled += 1
        return filled

    def rank_appearance(
        self,
        embedding: np.ndarray,
        exclude_ids: set[str] | None = None,
    ) -> tuple[dict | None, float, float]:
        exclude_ids = exclude_ids or set()
        best, best_sim = None, -1.0
        second = -1.0
        for person in self.gallery.people:
            if person.get("id") in exclude_ids or not self.gallery.match_appearances(person):
                continue
            sim = self.gallery.score_appearance(person, embedding)
            if sim > best_sim:
                second = best_sim
                best, best_sim = person, sim
            elif sim > second:
                second = sim
        return best, best_sim, second

    def _assign(
        self,
        track: Track,
        person: dict,
        sim: float,
        frame: np.ndarray,
        face: dict | None = None,
        *,
        update: bool = True,
        force: bool = False,
        appearance: np.ndarray | None = None,
        update_app: bool = False,
        by_seat: bool = False,
    ) -> None:
        track.person_id = person["id"]
        track.person_name = person["name"]
        track.face_score = sim
        track.by_seat = by_seat
        if appearance is not None:
            track.reid_score = max(float(self.gallery.score_appearance(person, appearance)), 0.0)
        self._id_memory[track.id] = (person["id"], person["name"], sim)
        self._pending_enroll.pop(track.id, None)
        if update and face is not None:
            self.update_embedding(person, face["embedding"], sim, force=force)
            self.gallery.maybe_save_live(person, frame, face["xyxy"])
        if update_app and appearance is not None:
            has = bool(self.gallery.match_appearances(person))
            if has and not force and self.gallery.appearance_collides(appearance, str(person["id"]), self.cfg.reid_collide):
                return
            self.gallery.update_appearance(
                person,
                appearance,
                sim=max(track.reid_score, float(sim)),
                force=force or not has,
                collide=self.cfg.reid_collide,
            )

    def _prune_id_hits(self) -> None:
        now = time.monotonic()
        self._id_hits = {pid: item for pid, item in self._id_hits.items() if now - item[0] <= 8.0}

    def _bump_id_hits(self, _track_id: int, person_id: str) -> int:
        now = time.monotonic()
        prev = self._id_hits.get(person_id)
        hits = prev[1] + 1 if prev is not None and now - prev[0] <= 8.0 else 1
        self._id_hits[person_id] = (now, hits)
        return hits

    def _update_live(self, track: Track, face: dict, frame: np.ndarray) -> bool:
        fx1, fy1, fx2, fy2 = face["xyxy"]
        face_px = min(fx2 - fx1, fy2 - fy1)
        if should_skip_liveness(self.cfg, face_px):
            track.live = True
            track.live_score = 1.0
            return True
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

    def present_from_tracks(self, tracks: list[Track]) -> dict[str, dict]:
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
        return present

    def flush_appear(self, groups: list[list[Track]]) -> None:
        present: dict[str, dict] = {}
        alive: set[int] = set()
        for tracks in groups:
            alive.update(track.id for track in tracks)
            present.update(self.present_from_tracks(tracks))
        self._id_memory = {key: value for key, value in self._id_memory.items() if key in alive}
        self._pending_enroll = {key: value for key, value in self._pending_enroll.items() if key in alive}
        self._live_hits = {key: value for key, value in self._live_hits.items() if key in alive}
        self._prune_id_hits()
        self._prune_seat_hits()
        self.gallery.appear.tick(present)

    def apply(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        *,
        tick_appear: bool = True,
        camera_id: str | None = None,
        camera_name: str = "",
    ) -> list[Track]:
        reloaded = self.gallery.maybe_reload()
        if reloaded:
            self._app_seed_tried.clear()
            self.seed_missing_appearances()
        if not tracks:
            if tick_appear:
                self._id_memory.clear()
                self._pending_enroll.clear()
                self._live_hits.clear()
                self._id_hits.clear()
                self._seat_hits.clear()
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
        paired_ids = {track.id for track, _ in paired}
        for track in tracks:
            if track.id in paired_ids:
                continue
            face = self._detect_face_in_track(frame, track)
            if face is None:
                continue
            track.face_bbox = face["xyxy"]
            paired.append((track, face))
            paired_ids.add(track.id)
        paired.sort(
            key=lambda item: (item[1]["xyxy"][2] - item[1]["xyxy"][0]) * (item[1]["xyxy"][3] - item[1]["xyxy"][1]),
            reverse=True,
        )
        h, w = frame.shape[:2]
        seats = (
            camera_seats(self.gallery.people, camera_id, min_hits=self.cfg.seat_min_hits)
            if camera_id and self.cfg.seats
            else []
        )
        face_by_id = {track.id: face for track, face in paired}
        if self.reid is not None:
            for track in tracks:
                track.appearance = self.reid.embed_box(frame, track.bbox_xyxy, min_height=self.cfg.reid_min_height)

        for track in tracks:
            face = face_by_id.get(track.id)
            appearance = track.appearance if isinstance(track.appearance, np.ndarray) else None
            if face is not None and not self._update_live(track, face, frame):
                face = None
            owner = self._pick_for_track(track, seats, used_people, w, h)
            seat_person = self.gallery.find(str(owner["person_id"])) if owner else None
            seat_owner_id = str(owner["person_id"]) if owner else ""
            blocked = used_people | locked_away_ids(self.gallery.people, camera_id, seat_owner_id)
            embedding = face["embedding"] if face is not None else None
            face_px = 0.0
            can_match = False
            can_learn = False
            if face is not None:
                fx1, fy1, fx2, fy2 = face["xyxy"]
                face_px = min(fx2 - fx1, fy2 - fy1)
                can_match = can_match_face(self.cfg, face_px, face["score"], face.get("raw"))
                can_learn = is_identity_face(self.cfg, face_px, face["score"], face.get("raw"))
            ranked, sim, second = None, -1.0, -1.0
            seat_sim = -1.0
            if can_match and embedding is not None:
                ranked, sim, second = self.rank_top(embedding, blocked)
                if seat_person is not None:
                    seat_sim = self.gallery.score(seat_person, embedding)
            reid_ranked, reid_sim, reid_second = None, -1.0, -1.0
            seat_reid_sim = -1.0
            if appearance is not None:
                reid_ranked, reid_sim, reid_second = self.rank_appearance(appearance, blocked)
                if seat_person is not None:
                    seat_reid_sim = self.gallery.score_appearance(seat_person, appearance)
                    track.reid_score = max(float(seat_reid_sim), float(reid_sim))
            chosen, chosen_sim, source = fuse_identity(
                ranked,
                sim,
                second,
                seat_person,
                seat_sim,
                match_threshold=self.cfg.match_threshold,
                soft_threshold=self.cfg.soft_threshold,
                match_margin=self.cfg.match_margin,
                reid_person=reid_ranked,
                reid_sim=reid_sim,
                reid_second=reid_second,
                seat_reid_sim=seat_reid_sim,
                reid_threshold=self.cfg.reid_threshold,
                reid_soft=self.cfg.reid_soft,
                reid_margin=self.cfg.reid_margin,
            )
            if chosen is not None and not can_name_at_seat(chosen, camera_id, seat_owner_id):
                chosen, chosen_sim, source = None, -1.0, ""
            remembered = self._id_memory.get(track.id)

            if chosen is not None:
                hits = self._bump_id_hits(track.id, chosen["id"])
                if hits >= hits_needed(self.cfg, chosen_sim, source):
                    has_vec = bool(self.gallery.match_vectors(chosen))
                    update = bool(face) and should_update_template(self.cfg, source, can_learn, chosen_sim, has_vec)
                    if (
                        update
                        and embedding is not None
                        and source == "seat"
                        and not has_vec
                        and self.gallery.template_collides(embedding, str(chosen["id"]), self.cfg.template_collide)
                    ):
                        update = False
                    has_app = bool(self.gallery.match_appearances(chosen))
                    update_app = False
                    if appearance is not None and source in {"reid_seat", "face_seat", "face"}:
                        owner_app = max(float(seat_reid_sim), float(reid_sim) if reid_ranked is chosen else -1.0)
                        if not has_app:
                            update_app = not self.gallery.appearance_collides(
                                appearance, str(chosen["id"]), self.cfg.reid_collide
                            )
                        elif source in {"reid_seat", "face_seat", "face"} and owner_app >= self.cfg.reid_update:
                            update_app = True
                    self._assign(
                        track,
                        chosen,
                        chosen_sim,
                        frame,
                        face,
                        update=update,
                        force=update and source in {"seat", "face_seat"} and not has_vec,
                        appearance=appearance,
                        update_app=bool(update_app),
                        by_seat=source == "seat",
                    )
                    used_people.add(chosen["id"])
                continue

            if remembered is not None:
                mem_person = self.gallery.find(remembered[0])
                if (
                    mem_person is not None
                    and mem_person["id"] not in used_people
                    and can_name_at_seat(mem_person, camera_id, seat_owner_id)
                ):
                    mem_sim = self.gallery.score(mem_person, embedding) if can_match and embedding is not None else -1.0
                    mem_app = self.gallery.score_appearance(mem_person, appearance) if appearance is not None else -1.0
                    if mem_sim >= self.cfg.soft_threshold or mem_app >= self.cfg.reid_soft:
                        track.person_id = mem_person["id"]
                        track.person_name = mem_person["name"]
                        track.face_score = mem_sim if mem_sim > 0 else mem_app
                        track.reid_score = max(mem_app, 0.0)
                        self._id_memory[track.id] = (mem_person["id"], mem_person["name"], track.face_score)
                        self._pending_enroll.pop(track.id, None)
                        used_people.add(mem_person["id"])
                        continue

            if face is None:
                continue
            _, all_sim = self.rank(embedding) if can_match and embedding is not None else (None, -1.0)
            if all_sim >= self.cfg.soft_threshold:
                self._pending_enroll.pop(track.id, None)
                continue

            if should_count_toward_enroll(self.cfg, sim if ranked else all_sim, face["score"], face_px):
                if owner and owner.get("locked"):
                    self._pending_enroll.pop(track.id, None)
                    continue
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
                    if appearance is not None:
                        self.gallery.update_appearance(person, appearance, force=True, collide=self.cfg.reid_collide)
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
            mem_person = self.gallery.find(pid)
            owner = self._pick_for_track(track, seats, used_people, w, h) if seats else None
            seat_owner_id = str(owner["person_id"]) if owner else ""
            if mem_person is not None and not can_name_at_seat(mem_person, camera_id, seat_owner_id):
                continue
            track.person_id = pid
            track.person_name = name
            track.face_score = score
            used_people.add(pid)
        if camera_id and self.cfg.seats:
            self._learn_seats(frame, tracks, camera_id, camera_name)
            self._apply_seats(frame, tracks, camera_id, used_people)
        if tick_appear:
            self._id_memory = {k: v for k, v in self._id_memory.items() if k in alive}
            self._pending_enroll = {k: v for k, v in self._pending_enroll.items() if k in alive}
            self._live_hits = {k: v for k, v in self._live_hits.items() if k in alive}
            self._prune_id_hits()
            self._prune_seat_hits()
            self.gallery.appear.tick(self.present_from_tracks(tracks))
        return tracks

    def seats_for(self, camera_id: str) -> list[dict]:
        if not camera_id or not self.cfg.seats:
            return []
        return camera_seats(self.gallery.people, camera_id)

    def _prune_seat_hits(self) -> None:
        now = time.monotonic()
        self._seat_hits = {pid: item for pid, item in self._seat_hits.items() if now - item[0] <= _SEAT_HIT_HOLD_S}

    def _bump_seat_hits(self, pid: str) -> int:
        now = time.monotonic()
        prev = self._seat_hits.get(pid)
        hits = prev[1] + 1 if prev is not None and now - prev[0] <= _SEAT_HIT_HOLD_S else 1
        self._seat_hits[pid] = (now, hits)
        return hits

    def _pick_for_track(
        self,
        track: Track,
        seats: list[dict],
        used_people: set[str],
        width: int,
        height: int,
    ) -> dict | None:
        obs = box_to_norm(track.bbox_xyxy, width, height)
        return pick_seat(
            obs["cx"],
            obs["cy"],
            seats,
            used_people,
            min_hits=self.cfg.seat_min_hits,
            bbox=track.bbox_xyxy,
            width=width,
            height=height,
            xyz=track.xyz,
        )

    def _learn_seats(self, frame: np.ndarray, tracks: list[Track], camera_id: str, camera_name: str) -> None:
        h, w = frame.shape[:2]
        for track in tracks:
            if not track.person_id or track.by_seat or track.live is False:
                continue
            person = self.gallery.find(track.person_id)
            if person is None or person.get("guest"):
                continue
            if not is_stationary(track, w, h):
                continue
            obs = box_to_norm(track.bbox_xyxy, w, h)
            if track.xyz is not None:
                obs["xyz"] = list(track.xyz)
            if other_owns(obs["cx"], obs["cy"], self.gallery.people, camera_id, track.person_id, min_hits=self.cfg.seat_min_hits):
                continue
            prev = person_seats(person).get(camera_id)
            if prev and prev.get("locked"):
                continue
            seat = blend_seat(prev, obs, camera_name=camera_name)
            seats = person_seats(person)
            seats[camera_id] = seat
            person["seats"] = seats
            hits = int(seat.get("hits") or 0)
            if hits <= 3 or hits % 10 == 0:
                self.gallery.save()

    def _apply_seats(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        camera_id: str,
        used_people: set[str],
    ) -> None:
        h, w = frame.shape[:2]
        seats = camera_seats(self.gallery.people, camera_id, min_hits=self.cfg.seat_min_hits)
        if not seats:
            return
        for track in tracks:
            if track.person_id:
                continue
            if not person_like_box(track.bbox_xyxy, w, h):
                continue
            seat = self._pick_for_track(track, seats, used_people, w, h)
            if seat is None:
                continue
            if not seat.get("locked") and not is_stationary(track, w, h):
                continue
            pid = str(seat["person_id"])
            appearance = track.appearance if isinstance(track.appearance, np.ndarray) else None
            person = self.gallery.find(pid)
            owner_sim = self.gallery.score_appearance(person, appearance) if person is not None and appearance is not None else -1.0
            if person is None or not self.gallery.match_appearances(person) or owner_sim < self.cfg.reid_soft:
                continue
            if appearance is not None:
                ranked, reid_sim, reid_second = self.rank_appearance(appearance, used_people)
                if (
                    ranked is not None
                    and ranked["id"] != pid
                    and match_is_confident(reid_sim, reid_second, self.cfg.reid_threshold, self.cfg.reid_margin)
                    and reid_sim - max(owner_sim, 0.0) >= self.cfg.reid_margin
                ):
                    continue
            hits = self._bump_seat_hits(pid)
            needed = 1 if seat.get("locked") and owner_sim >= self.cfg.reid_soft else max(1, int(self.cfg.seat_confirm))
            if hits < needed:
                continue
            name = (person.get("name") if person else None) or str(seat.get("name") or pid)
            track.person_id = pid
            track.person_name = name
            track.by_seat = True
            track.face_score = 0.0
            if appearance is not None:
                track.reid_score = max(owner_sim, 0.0)
            self._id_memory[track.id] = (pid, name, 0.0)
            used_people.add(pid)

    def _capture_appearance(self, frame: np.ndarray, track: Track, person: dict) -> None:
        if self.reid is None:
            return
        vec = self.reid.embed_box(frame, track.bbox_xyxy, min_height=self.cfg.reid_min_height)
        if vec is None:
            return
        self.gallery.update_appearance(person, vec, force=True, collide=self.cfg.reid_collide)

    def enroll_track(self, frame: np.ndarray, track: Track, name: str | None = None) -> str | None:
        faces = self.detect_faces(frame)
        candidates = [f for f in faces if face_in_person(f["xyxy"], track)]
        if not candidates:
            cropped = self._detect_face_in_track(frame, track)
            if cropped is not None:
                candidates = [cropped]
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
        matched, sim = self.match(face["embedding"], threshold=self.cfg.soft_threshold, require_margin=False)
        if matched is not None:
            if name:
                matched["name"] = name
            self.update_embedding(matched, face["embedding"], sim, force=True)
            self.gallery.save_crop(matched, frame, face["xyxy"], force_cover=True)
            self.gallery.save()
            track.person_id = matched["id"]
            track.person_name = matched["name"]
            track.face_bbox = face["xyxy"]
            self._id_memory[track.id] = (matched["id"], matched["name"], 1.0)
            self._capture_appearance(frame, track, matched)
            return matched["name"]
        if track.person_id:
            for person in self.gallery.people:
                if person["id"] == track.person_id:
                    if name:
                        person["name"] = name
                    self.update_embedding(person, face["embedding"], force=True)
                    self.gallery.save_crop(person, frame, face["xyxy"], force_cover=True)
                    self.gallery.save()
                    track.person_name = person["name"]
                    track.face_bbox = face["xyxy"]
                    self._id_memory[track.id] = (person["id"], person["name"], 1.0)
                    self._capture_appearance(frame, track, person)
                    return person["name"]
        person = self.enroll(face["embedding"], name=name)
        track.person_id = person["id"]
        track.person_name = person["name"]
        track.face_bbox = face["xyxy"]
        self._id_memory[track.id] = (person["id"], person["name"], 1.0)
        self.gallery.save_crop(person, frame, face["xyxy"], force_cover=True)
        self._capture_appearance(frame, track, person)
        return person["name"]
