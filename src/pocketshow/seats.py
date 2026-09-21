from __future__ import annotations

from pocketshow.types import Track

DEFAULT_RX = 0.05
DEFAULT_RY = 0.10
MIN_RX = 0.025
MIN_RY = 0.05
MAX_RX = 0.16
MAX_RY = 0.28
PIN_HITS = 999


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _xyz(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError):
        return None


def xyz_dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def box_to_norm(bbox: tuple[float, float, float, float], width: int, height: int) -> dict[str, float]:
    width = max(1, int(width))
    height = max(1, int(height))
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    rx = max((x2 - x1) * 0.5 / width, MIN_RX)
    ry = max((y2 - y1) * 0.5 / height, MIN_RY)
    return {
        "cx": clamp(cx, 0.0, 1.0),
        "cy": clamp(cy, 0.0, 1.0),
        "rx": clamp(rx, MIN_RX, MAX_RX),
        "ry": clamp(ry, MIN_RY, MAX_RY),
    }


def clean_box(item: dict) -> dict | None:
    try:
        x1, y1 = float(item["x1"]), float(item["y1"])
        x2, y2 = float(item["x2"]), float(item["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if x2 == x1 or y2 == y1:
        return None
    row = {
        "x1": round(clamp(min(x1, x2), 0.0, 1.0), 4),
        "y1": round(clamp(min(y1, y2), 0.0, 1.0), 4),
        "x2": round(clamp(max(x1, x2), 0.0, 1.0), 4),
        "y2": round(clamp(max(y1, y2), 0.0, 1.0), 4),
    }
    try:
        row["id"] = int(item["id"])
    except (KeyError, TypeError, ValueError):
        pass
    person_id = str(item.get("person_id") or "")
    if person_id:
        row["person_id"] = person_id
    name = str(item.get("name") or "")
    if name:
        row["name"] = name
    return row


def boxes_from_tracks(tracks: list[Track], width: int, height: int) -> list[dict]:
    width = max(1, int(width))
    height = max(1, int(height))
    out: list[dict] = []
    for track in tracks:
        x1, y1, x2, y2 = (float(v) for v in track.bbox_xyxy)
        box = clean_box(
            {
                "id": track.id,
                "x1": x1 / width,
                "y1": y1 / height,
                "x2": x2 / width,
                "y2": y2 / height,
                "person_id": track.person_id or "",
                "name": track.person_name or "",
            }
        )
        if box is not None:
            out.append(box)
    return out


def hit_box(boxes: list[dict], cx: float, cy: float) -> dict | None:
    hits: list[tuple[float, dict]] = []
    for item in boxes:
        box = clean_box(item) if "x1" in item else None
        if box is None:
            continue
        if not (box["x1"] <= cx <= box["x2"] and box["y1"] <= cy <= box["y2"]):
            continue
        area = max(1e-6, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))
        hits.append((area, box))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    return dict(hits[0][1])


def box_pose(box: dict) -> tuple[float, float, float, float]:
    cleaned = clean_box(box) or box
    x1, y1 = float(cleaned["x1"]), float(cleaned["y1"])
    x2, y2 = float(cleaned["x2"]), float(cleaned["y2"])
    return (
        clamp((x1 + x2) * 0.5, 0.0, 1.0),
        clamp((y1 + y2) * 0.5, 0.0, 1.0),
        max(abs(x2 - x1) * 0.5, MIN_RX),
        max(abs(y2 - y1) * 0.5, MIN_RY),
    )


def ellipse_dist(seat: dict, cx: float, cy: float) -> float:
    rx = max(float(seat.get("rx") or DEFAULT_RX), 1e-4)
    ry = max(float(seat.get("ry") or DEFAULT_RY), 1e-4)
    dx = (cx - float(seat.get("cx") or 0.0)) / rx
    dy = (cy - float(seat.get("cy") or 0.0)) / ry
    return (dx * dx + dy * dy) ** 0.5


def contains(seat: dict, cx: float, cy: float, slack: float = 1.0) -> bool:
    return ellipse_dist(seat, cx, cy) <= slack


def _norm_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    width = max(1, int(width))
    height = max(1, int(height))
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (
        clamp(min(x1, x2) / width, 0.0, 1.0),
        clamp(min(y1, y2) / height, 0.0, 1.0),
        clamp(max(x1, x2) / width, 0.0, 1.0),
        clamp(max(y1, y2) / height, 0.0, 1.0),
    )


def box_seat_dist(
    seat: dict,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> float:
    """人框到工位的椭圆距离。坐着时检测框常只有头肩，中心会偏上，所以框向下探一点再取最近点。"""
    nx1, ny1, nx2, orig_ny2 = _norm_box(bbox, width, height)
    box_h = max(0.0, orig_ny2 - ny1)
    ry = max(float(seat.get("ry") or DEFAULT_RY), 1e-4)
    max_drop = max(0.10, box_h * 1.15 + 0.05, ry * 1.3)
    reach = max(0.045, box_h * 0.5)
    if box_h <= 0.22:
        reach = max(reach, min(0.16, max_drop))
    ny2 = clamp(orig_ny2 + reach, 0.0, 1.0)
    sx = float(seat.get("cx") or 0.0)
    sy = float(seat.get("cy") or 0.0)
    dist = ellipse_dist(seat, clamp(sx, nx1, nx2), clamp(sy, ny1, ny2))
    gap = sy - orig_ny2
    if gap > max_drop:
        return 9.0
    return dist


def is_stationary(track: Track, width: int, height: int, max_speed: float = 0.16) -> bool:
    span = max(1.0, float(min(width, height)))
    speed = (track.vx * track.vx + track.vy * track.vy) ** 0.5
    return speed < max_speed * span


def blend_seat(prev: dict | None, obs: dict, *, alpha: float = 0.12, camera_name: str = "") -> dict:
    xyz = _xyz(obs.get("xyz")) if "xyz" in obs else (_xyz(prev.get("xyz")) if prev else None)
    if prev is None:
        row = {
            "cx": float(obs["cx"]),
            "cy": float(obs["cy"]),
            "rx": float(obs["rx"]),
            "ry": float(obs["ry"]),
            "hits": 1,
            "locked": False,
            "camera_name": camera_name or str(obs.get("camera_name") or ""),
        }
        if xyz is not None:
            row["xyz"] = list(xyz)
        return row
    hits = int(prev.get("hits") or 0) + 1
    mix = 0.35 if hits <= 4 else alpha
    name = str(prev.get("camera_name") or camera_name or "")
    if xyz is None:
        xyz = _xyz(prev.get("xyz"))
    elif prev.get("xyz"):
        old = _xyz(prev.get("xyz"))
        if old is not None:
            xyz = (
                (1.0 - mix) * old[0] + mix * xyz[0],
                (1.0 - mix) * old[1] + mix * xyz[1],
                (1.0 - mix) * old[2] + mix * xyz[2],
            )
    row = {
        "cx": (1.0 - mix) * float(prev.get("cx") or obs["cx"]) + mix * float(obs["cx"]),
        "cy": (1.0 - mix) * float(prev.get("cy") or obs["cy"]) + mix * float(obs["cy"]),
        "rx": clamp((1.0 - mix) * float(prev.get("rx") or obs["rx"]) + mix * float(obs["rx"]), MIN_RX, MAX_RX),
        "ry": clamp((1.0 - mix) * float(prev.get("ry") or obs["ry"]) + mix * float(obs["ry"]), MIN_RY, MAX_RY),
        "hits": hits,
        "locked": bool(prev.get("locked")),
        "camera_name": name,
    }
    if xyz is not None:
        row["xyz"] = [round(xyz[0], 3), round(xyz[1], 3), round(xyz[2], 3)]
    return row


def person_seats(person: dict) -> dict[str, dict]:
    raw = person.get("seats")
    if isinstance(raw, dict):
        return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}
    return {}


def seat_ready(seat: dict | None, min_hits: int) -> bool:
    if not isinstance(seat, dict):
        return False
    if seat.get("locked"):
        return True
    return int(seat.get("hits") or 0) >= int(min_hits)


def pinned_seat(
    cx: float,
    cy: float,
    *,
    camera_name: str = "",
    rx: float | None = None,
    ry: float | None = None,
    xyz: tuple[float, float, float] | None = None,
) -> dict:
    row = {
        "cx": clamp(float(cx), 0.0, 1.0),
        "cy": clamp(float(cy), 0.0, 1.0),
        "rx": clamp(float(rx if rx is not None else DEFAULT_RX), MIN_RX, MAX_RX),
        "ry": clamp(float(ry if ry is not None else DEFAULT_RY), MIN_RY, MAX_RY),
        "hits": PIN_HITS,
        "locked": True,
        "camera_name": camera_name,
    }
    if xyz is not None:
        row["xyz"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return row


def vacate_overlapping(people: list[dict], camera_id: str, seat: dict, except_id: str) -> None:
    camera_id = str(camera_id)
    for person in people:
        pid = str(person.get("id") or "")
        if not pid or pid == except_id:
            continue
        seats = person_seats(person)
        other = seats.get(camera_id)
        if other is None:
            continue
        other_cx = float(other.get("cx") or 0.0)
        other_cy = float(other.get("cy") or 0.0)
        if contains(other, float(seat["cx"]), float(seat["cy"]), slack=1.05) or contains(
            seat, other_cx, other_cy, slack=1.05
        ):
            seats.pop(camera_id, None)
            person["seats"] = seats


def other_owns(
    cx: float,
    cy: float,
    people: list[dict],
    camera_id: str,
    except_id: str,
    *,
    min_hits: int = 8,
) -> bool:
    for person in people:
        pid = str(person.get("id") or "")
        if not pid or pid == except_id or person.get("guest"):
            continue
        seat = person_seats(person).get(camera_id)
        if not seat_ready(seat, min_hits):
            continue
        if contains(seat, cx, cy, slack=1.05):
            return True
    return False


def camera_seats(people: list[dict], camera_id: str, *, min_hits: int = 0) -> list[dict]:
    out: list[dict] = []
    for person in people:
        pid = str(person.get("id") or "")
        if not pid or person.get("guest"):
            continue
        seat = person_seats(person).get(camera_id)
        if not seat_ready(seat, min_hits):
            continue
        item = dict(seat)
        item["person_id"] = pid
        item["name"] = str(person.get("name") or pid)
        out.append(item)
    return out


def can_name_at_seat(person: dict, camera_id: str | None, seat_owner_id: str) -> bool:
    """钉死工位的人只能在自己座位上亮名，不能被外观认到别的坐标。"""
    if not camera_id or person.get("guest"):
        return True
    seat = person_seats(person).get(camera_id)
    if not seat or not seat.get("locked"):
        return True
    return str(person.get("id") or "") == str(seat_owner_id or "")


def locked_away_ids(people: list[dict], camera_id: str | None, seat_owner_id: str) -> set[str]:
    blocked: set[str] = set()
    for person in people:
        pid = str(person.get("id") or "")
        if pid and not can_name_at_seat(person, camera_id, seat_owner_id):
            blocked.add(pid)
    return blocked


def pick_seat(
    cx: float,
    cy: float,
    seats: list[dict],
    used_ids: set[str],
    *,
    min_hits: int = 8,
    bbox: tuple[float, float, float, float] | None = None,
    width: int = 1,
    height: int = 1,
    slack: float = 1.25,
    xyz: tuple[float, float, float] | None = None,
    xyz_slack: float = 0.85,
) -> dict | None:
    scored: list[tuple[float, dict]] = []
    for seat in seats:
        pid = str(seat.get("person_id") or "")
        if not pid or pid in used_ids or not seat_ready(seat, min_hits):
            continue
        seat_xyz = _xyz(seat.get("xyz"))
        if xyz is not None and seat_xyz is not None:
            dist = xyz_dist(xyz, seat_xyz)
            if dist > xyz_slack:
                continue
        else:
            dist = (
                box_seat_dist(seat, bbox, width, height)
                if bbox is not None
                else ellipse_dist(seat, cx, cy)
            )
            if dist > slack:
                continue
        scored.append((dist, seat))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    best_d, best = scored[0]
    if len(scored) > 1:
        second_d = scored[1][0]
        if second_d <= 1.0 and second_d / max(best_d, 0.08) < 1.35:
            return None
    return best


def seat_search_box(seat: dict) -> dict[str, float]:
    """工位补检窗口：只圈椅子附近。扩太大，坐着的人会被桌子、显示器淹没。"""
    cx = clamp(float(seat.get("cx") or 0.0), 0.0, 1.0)
    cy = clamp(float(seat.get("cy") or 0.0), 0.0, 1.0)
    rx = max(float(seat.get("rx") or DEFAULT_RX), MIN_RX)
    ry = max(float(seat.get("ry") or DEFAULT_RY), MIN_RY)
    half_w = max(rx * 1.2, 0.05)
    return {
        "x1": clamp(cx - half_w, 0.0, 1.0),
        "y1": clamp(cy - max(ry * 1.6, 0.16), 0.0, 1.0),
        "x2": clamp(cx + half_w, 0.0, 1.0),
        "y2": clamp(cy + max(ry * 0.8, 0.08), 0.0, 1.0),
    }


def seat_search_boxes(seat: dict) -> list[dict[str, float]]:
    """椅子一窗、头肩再一窗。坐着的人有时只在其中一块被检到。"""
    tight = seat_search_box(seat)
    cx = clamp(float(seat.get("cx") or 0.0), 0.0, 1.0)
    cy = clamp(float(seat.get("cy") or 0.0), 0.0, 1.0)
    tall = {
        "x1": clamp(cx - 0.07, 0.0, 1.0),
        "y1": clamp(cy - 0.22, 0.0, 1.0),
        "x2": clamp(cx + 0.07, 0.0, 1.0),
        "y2": clamp(cy + 0.10, 0.0, 1.0),
    }
    if abs(tall["y1"] - tight["y1"]) < 0.02 and abs(tall["x1"] - tight["x1"]) < 0.02:
        return [tight]
    return [tight, tall]


def person_like_box(
    box: dict | tuple[float, float, float, float],
    width: int = 1,
    height: int = 1,
    *,
    min_w: float = 0.022,
    min_h: float = 0.05,
    min_area: float = 0.0012,
) -> bool:
    """椅子、显示器上的小噪点不算人。归一化后大约 14×18 像素起。"""
    if isinstance(box, dict):
        cleaned = clean_box(box)
        if cleaned is None:
            return False
        bw = cleaned["x2"] - cleaned["x1"]
        bh = cleaned["y2"] - cleaned["y1"]
    else:
        x1, y1, x2, y2 = box
        bw = (x2 - x1) / max(1, int(width))
        bh = (y2 - y1) / max(1, int(height))
    return bw >= min_w and bh >= min_h and bw * bh >= min_area


def occupied_seat_ids(
    people: list[dict],
    cameras: list[dict] | None,
    *,
    min_hits: int = 0,
    slack: float = 1.25,
) -> set[str]:
    """只有认出名字、并且框够大，才算本人在岗。工位上坐了别人或挂了件衣服不算。"""
    occupied: set[str] = set()
    for camera in cameras or []:
        if not isinstance(camera, dict):
            continue
        for box in camera.get("boxes") or []:
            if not isinstance(box, dict):
                continue
            pid = str(box.get("person_id") or "")
            if not pid or not person_like_box(box):
                continue
            occupied.add(pid)
    return occupied


def merge_seats(keep: dict, src: dict) -> dict:
    seats = person_seats(keep)
    for camera_id, seat in person_seats(src).items():
        current = seats.get(camera_id)
        if current is None or int(seat.get("hits") or 0) > int(current.get("hits") or 0):
            seats[camera_id] = dict(seat)
    keep["seats"] = seats
    return keep
