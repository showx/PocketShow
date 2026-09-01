from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pocketshow.appear import AppearanceLog

_NAME_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_MAX_SNAPS = 8
_COVER = "cover.jpg"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def crop_face(frame: np.ndarray, xyxy: tuple[float, float, float, float], pad: float = 0.42) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad * 1.25))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad * 0.55))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    longest = max(ch, cw)
    if longest > 480:
        scale = 480 / longest
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    return crop


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("无法编码相片")
    return buf.tobytes()


class FaceGallery:
    """人物名册 + 相片目录。JSON 与封面图都写在 data/ 下。"""

    def __init__(self, path: str | Path, photos_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.photos_dir = Path(photos_dir) if photos_dir is not None else self.path.parent / "faces"
        self.appear = AppearanceLog(self.path.with_name("appear.jsonl"))
        self.people: list[dict] = []
        self.mtime: float = 0.0
        self.load()

    def load(self) -> None:
        self.people = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.people = data.get("people", [])
            self.mtime = self.path.stat().st_mtime
        else:
            self.mtime = 0.0

    def maybe_reload(self) -> bool:
        if not self.path.exists():
            if self.people:
                self.people = []
                self.mtime = 0.0
                return True
            return False
        mtime = self.path.stat().st_mtime
        if mtime != self.mtime:
            self.load()
            return True
        return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"people": self.people}, ensure_ascii=False, indent=2))
        tmp.replace(self.path)
        self.mtime = self.path.stat().st_mtime

    def find(self, person_id: str) -> dict | None:
        return next((p for p in self.people if p.get("id") == person_id), None)

    def person_vectors(self, person: dict) -> list[np.ndarray]:
        vecs = [np.asarray(person["embedding"], dtype=np.float32)]
        for extra in person.get("templates") or []:
            vecs.append(np.asarray(extra, dtype=np.float32))
        return vecs

    def score(self, person: dict, embedding: np.ndarray) -> float:
        best = -1.0
        for vec in self.person_vectors(person):
            sim = cosine_sim(embedding, vec)
            if sim > best:
                best = sim
        return best

    def add_template(self, person: dict, embedding: np.ndarray, *, max_templates: int = 6) -> None:
        vec = embedding.astype(float)
        existing = self.person_vectors(person)
        if existing and max(cosine_sim(embedding, v) for v in existing) >= 0.88:
            return
        templates = list(person.get("templates") or [])
        templates.append(vec.tolist())
        person["templates"] = templates[-max_templates:]

    def pair_score(self, a: dict, b: dict) -> float:
        """两条档案有多像。取全部模板对的中位数，避免一张糊脸/侧脸把不同的人拉在一起。"""
        av = self.person_vectors(a)
        bv = self.person_vectors(b)
        if not av or not bv:
            return 0.0
        sims = [cosine_sim(x, y) for x in av for y in bv]
        return float(np.median(np.asarray(sims, dtype=np.float32)))

    def similar_pairs(self, threshold: float = 0.70, min_overlap_s: float = 8.0) -> list[dict]:
        pairs: list[dict] = []
        skip = self.appear.concurrent_pairs(min_overlap_s=min_overlap_s)
        people = [p for p in self.people if p.get("embedding")]
        for i, a in enumerate(people):
            for b in people[i + 1 :]:
                key = (a["id"], b["id"]) if a["id"] < b["id"] else (b["id"], a["id"])
                if key in skip:
                    continue
                sim = self.pair_score(a, b)
                if sim >= threshold:
                    pairs.append(
                        {
                            "a": a["id"],
                            "b": b["id"],
                            "a_name": a.get("name") or a["id"],
                            "b_name": b.get("name") or b["id"],
                            "sim": round(float(sim), 3),
                        }
                    )
        pairs.sort(key=lambda p: p["sim"], reverse=True)
        return pairs

    def merge(self, keep_id: str, source_id: str) -> dict:
        if keep_id == source_id:
            raise ValueError("不能和自己合并")
        keep = self.find(keep_id)
        src = self.find(source_id)
        if keep is None or src is None:
            raise KeyError(source_id if keep is not None else keep_id)
        kn = max(1, int(keep.get("samples") or 1))
        sn = max(1, int(src.get("samples") or 1))
        ke = np.asarray(keep["embedding"], dtype=np.float32)
        se = np.asarray(src["embedding"], dtype=np.float32)
        blended = (ke * kn + se * sn) / (kn + sn)
        norm = float(np.linalg.norm(blended))
        if norm > 1e-6:
            blended = blended / norm
        keep["embedding"] = blended.astype(float).tolist()
        keep["samples"] = kn + sn
        self.add_template(keep, se)
        for extra in src.get("templates") or []:
            self.add_template(keep, np.asarray(extra, dtype=np.float32))
        note_bits = [keep.get("note") or "", src.get("note") or "", f"已合并 {src.get('name') or source_id}"]
        keep["note"] = "；".join(x for x in note_bits if x)
        keep["updated_at"] = now_iso()

        src_dir = self.photos_dir / source_id
        keep_dir = self.person_dir(keep_id)
        if src_dir.exists():
            for path in sorted(src_dir.glob("*.jpg")):
                dest_name = path.name if path.name != _COVER else f"snap_merge_{source_id}.jpg"
                if dest_name == _COVER:
                    dest_name = f"snap_merge_{source_id}.jpg"
                dest = keep_dir / dest_name
                if dest.exists():
                    dest = keep_dir / f"snap_merge_{source_id}_{path.stem}.jpg"
                shutil.copy2(path, dest)
            if not (keep_dir / _COVER).exists():
                fallback = next(iter(keep_dir.glob("*.jpg")), None)
                if fallback is not None:
                    shutil.copy2(fallback, keep_dir / _COVER)
                    keep["photo"] = f"{keep_id}/{_COVER}"
            elif not keep.get("photo"):
                keep["photo"] = f"{keep_id}/{_COVER}"
            shutil.rmtree(src_dir)
            self._prune_snaps(keep_dir)

        self.people = [p for p in self.people if p.get("id") != source_id]
        self.appear.relabel(source_id, keep_id, str(keep.get("name") or keep_id))
        self.save()
        return keep

    def next_name(self) -> str:
        used = {p.get("name") for p in self.people}
        for ch in _NAME_CHARS:
            name = f"人物{ch}"
            if name not in used:
                return name
        return f"人物{len(self.people) + 1}"

    def next_id(self) -> str:
        nums: list[int] = []
        for person in self.people:
            pid = str(person.get("id", ""))
            if pid.startswith("p") and pid[1:].isdigit():
                nums.append(int(pid[1:]))
        return f"p{max(nums, default=0) + 1:03d}"

    def enroll(self, embedding: np.ndarray, name: str | None = None) -> dict:
        stamp = now_iso()
        person = {
            "id": self.next_id(),
            "name": name or self.next_name(),
            "note": "",
            "guest": False,
            "embedding": embedding.astype(float).tolist(),
            "samples": 1,
            "created_at": stamp,
            "updated_at": stamp,
            "photo": "",
            "photo_area": 0.0,
            "photo_at": 0.0,
            "templates": [],
        }
        self.people.append(person)
        self.save()
        return person

    def person_dir(self, person_id: str) -> Path:
        path = self.photos_dir / person_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_crop(
        self,
        person: dict,
        frame: np.ndarray,
        xyxy: tuple[float, float, float, float],
        *,
        force_cover: bool = False,
        as_snap: bool = False,
    ) -> Path | None:
        crop = crop_face(frame, xyxy)
        if crop is None:
            return None
        fx1, fy1, fx2, fy2 = xyxy
        area = max(0.0, (fx2 - fx1) * (fy2 - fy1))
        folder = self.person_dir(person["id"])
        cover = folder / _COVER
        need_cover = force_cover or not cover.exists() or not person.get("photo")
        if need_cover:
            cover.write_bytes(encode_jpeg(crop))
            person["photo"] = f"{person['id']}/{_COVER}"
            person["photo_area"] = area
            person["photo_at"] = datetime.now().timestamp()
            person["updated_at"] = now_iso()
            self.save()
            return cover
        if as_snap:
            name = datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
            dest = folder / name
            dest.write_bytes(encode_jpeg(crop))
            person["photo_at"] = datetime.now().timestamp()
            self._prune_snaps(folder)
            self.save()
            return dest
        return None

    def maybe_save_live(self, person: dict, frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> None:
        folder = self.person_dir(person["id"])
        cover = folder / _COVER
        fx1, fy1, fx2, fy2 = xyxy
        area = max(0.0, (fx2 - fx1) * (fy2 - fy1))
        now = datetime.now().timestamp()
        last = float(person.get("photo_at") or 0)
        prev_area = float(person.get("photo_area") or 0)
        if not cover.exists() or not person.get("photo"):
            self.save_crop(person, frame, xyxy, force_cover=True)
            return
        if area > prev_area * 1.25 and now - last >= 8:
            self.save_crop(person, frame, xyxy, force_cover=True)
            return
        if now - last >= 75:
            self.save_crop(person, frame, xyxy, as_snap=True)

    def add_image_bytes(self, person: dict, data: bytes, *, as_cover: bool = False) -> Path:
        folder = self.person_dir(person["id"])
        if as_cover or not (folder / _COVER).exists():
            dest = folder / _COVER
            dest.write_bytes(data)
            person["photo"] = f"{person['id']}/{_COVER}"
            person["updated_at"] = now_iso()
            person["photo_at"] = datetime.now().timestamp()
            self.save()
            return dest
        name = datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
        dest = folder / name
        dest.write_bytes(data)
        self._prune_snaps(folder)
        person["updated_at"] = now_iso()
        self.save()
        return dest

    def set_cover(self, person_id: str, filename: str) -> None:
        person = self.find(person_id)
        if person is None:
            raise KeyError(person_id)
        src = (self.person_dir(person_id) / Path(filename).name).resolve()
        folder = self.person_dir(person_id).resolve()
        if folder not in src.parents or not src.exists():
            raise FileNotFoundError(filename)
        cover = folder / _COVER
        if src != cover:
            shutil.copy2(src, cover)
        person["photo"] = f"{person_id}/{_COVER}"
        person["updated_at"] = now_iso()
        self.save()

    def rename(
        self,
        person_id: str,
        name: str,
        note: str | None = None,
        guest: bool | None = None,
    ) -> dict:
        person = self.find(person_id)
        if person is None:
            raise KeyError(person_id)
        person["name"] = name.strip() or person["name"]
        if note is not None:
            person["note"] = note
        if guest is not None:
            person["guest"] = bool(guest)
        person["updated_at"] = now_iso()
        self.save()
        return person

    def delete(self, person_id: str) -> None:
        person = self.find(person_id)
        if person is None:
            raise KeyError(person_id)
        self.people = [p for p in self.people if p.get("id") != person_id]
        self.save()
        folder = self.photos_dir / person_id
        if folder.exists():
            shutil.rmtree(folder)

    def list_photos(self, person_id: str) -> list[dict]:
        folder = self.photos_dir / person_id
        if not folder.exists():
            return []
        items = []
        for path in sorted(folder.glob("*.jpg")):
            items.append(
                {
                    "name": path.name,
                    "url": f"/media/{person_id}/{path.name}",
                    "cover": path.name == _COVER,
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
        items.sort(key=lambda x: (not x["cover"], x["name"]))
        return items

    def public(self, person: dict) -> dict:
        pid = person.get("id", "")
        photos = self.list_photos(pid)
        cover = next((p["url"] for p in photos if p["cover"]), None)
        return {
            "id": pid,
            "name": person.get("name") or pid,
            "note": person.get("note") or "",
            "guest": bool(person.get("guest")),
            "samples": int(person.get("samples") or 0),
            "created_at": person.get("created_at") or "",
            "updated_at": person.get("updated_at") or "",
            "photo_url": cover,
            "photos": photos,
        }

    def public_all(self, dup_threshold: float = 0.70) -> list[dict]:
        pairs = self.similar_pairs(dup_threshold)
        twins: dict[str, list[dict]] = {}
        for pair in pairs:
            twins.setdefault(pair["a"], []).append(pair)
            twins.setdefault(pair["b"], []).append(pair)
        out = []
        for person in self.people:
            item = self.public(person)
            item["twins"] = twins.get(person["id"], [])
            out.append(item)
        return out

    def resolve_media(self, rel: str) -> Path | None:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return None
        path = (self.photos_dir / rel_path).resolve()
        root = self.photos_dir.resolve()
        if not path.is_file() or not path.is_relative_to(root):
            return None
        return path

    @staticmethod
    def _prune_snaps(folder: Path) -> None:
        snaps = sorted(folder.glob("snap_*.jpg"))
        extra = snaps[:-_MAX_SNAPS] if len(snaps) > _MAX_SNAPS else []
        for path in extra:
            path.unlink(missing_ok=True)
