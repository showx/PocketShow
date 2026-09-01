from __future__ import annotations

from pocketshow.types import Track


class TargetLock:
    """锁定 ByteTrack ID；已识别身份时，换 ID 后仍按同一人物接回。"""

    def __init__(self, lost_timeout_s: float = 2.5) -> None:
        self.lost_timeout_s = lost_timeout_s
        self.locked_id: int | None = None
        self.locked_person_id: str | None = None
        self._manual = False
        self._missing_s = 0.0

    def lock(self, track: Track | int, person_id: str | None = None) -> None:
        if isinstance(track, Track):
            self.locked_id = track.id
            self.locked_person_id = track.person_id
        else:
            self.locked_id = track
            self.locked_person_id = person_id
        self._manual = True
        self._missing_s = 0.0

    def clear(self) -> None:
        self.locked_id = None
        self.locked_person_id = None
        self._manual = False
        self._missing_s = 0.0

    def lock_at(self, x: float, y: float, tracks: list[Track]) -> int | None:
        hits = [
            t
            for t in tracks
            if t.bbox_xyxy[0] <= x <= t.bbox_xyxy[2] and t.bbox_xyxy[1] <= y <= t.bbox_xyxy[3]
        ]
        if not hits:
            return None
        hits.sort(key=lambda t: t.area, reverse=True)
        self.lock(hits[0])
        return hits[0].id

    def cycle(self, tracks: list[Track], step: int = 1) -> int | None:
        if not tracks:
            return self.locked_id
        ordered = sorted(tracks, key=lambda t: t.id)
        ids = [t.id for t in ordered]
        if self.locked_id in ids:
            idx = (ids.index(self.locked_id) + step) % len(ids)
        else:
            idx = 0
        self.lock(ordered[idx])
        return ordered[idx].id

    def _by_person(self, tracks: list[Track]) -> Track | None:
        if not self.locked_person_id:
            return None
        matches = [t for t in tracks if t.person_id == self.locked_person_id]
        if not matches:
            return None
        return max(matches, key=lambda t: t.area * t.conf)

    def update(self, tracks: list[Track], dt: float) -> Track | None:
        by_id = {t.id: t for t in tracks}
        if self.locked_id is not None and self.locked_id in by_id:
            current = by_id[self.locked_id]
            if current.person_id:
                self.locked_person_id = current.person_id
            self._missing_s = 0.0
            return current

        identified = self._by_person(tracks)
        if identified is not None:
            self.locked_id = identified.id
            self._missing_s = 0.0
            return identified

        if self.locked_id is not None:
            self._missing_s += dt
            if self._missing_s < self.lost_timeout_s:
                return None
            if self._manual:
                return None
            self.locked_id = None
            self.locked_person_id = None
            self._missing_s = 0.0

        chosen = self._auto_select(tracks)
        if chosen is not None:
            self.locked_id = chosen.id
            self.locked_person_id = chosen.person_id
            self._missing_s = 0.0
        return chosen

    @staticmethod
    def _auto_select(tracks: list[Track]) -> Track | None:
        if not tracks:
            return None
        return max(tracks, key=lambda t: t.area * t.conf)
