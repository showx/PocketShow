from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pocketshow.types import FollowCommand, Track

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_texts(
    vis: np.ndarray,
    items: list[tuple[str, tuple[int, int], tuple[int, int, int], int]],
) -> np.ndarray:
    if not items:
        return vis
    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for text, (x, y), color, size in items:
        font = cache.get(size)
        if font is None:
            font = _font(size)
            cache[size] = font
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x + 1, y + 1), text, font=font, fill=(20, 20, 20))
        draw.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    command: FollowCommand,
    locked_id: int | None,
    fps: float,
    gimbal_name: str,
    deadzone: float,
    locked_name: str | None = None,
    control_mode: str = "跟拍",
    watch_line: str = "",
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    cx, cy = w // 2, h // 2
    dz_w, dz_h = int(w * deadzone), int(h * deadzone)
    cream = (210, 220, 232)
    mute = (150, 150, 150)
    cv2.rectangle(vis, (cx - dz_w, cy - dz_h), (cx + dz_w, cy + dz_h), (70, 70, 70), 1)
    cv2.drawMarker(vis, (cx, cy), cream, cv2.MARKER_CROSS, 14, 1)

    texts: list[tuple[str, tuple[int, int], tuple[int, int, int], int]] = []
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        is_target = track.id == locked_id
        color = cream if is_target else mute
        thickness = 2 if is_target else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if track.face_bbox is not None:
            fx1, fy1, fx2, fy2 = (int(v) for v in track.face_bbox)
            face_color = (168, 96, 210) if track.live is False else (140, 190, 90)
            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), face_color, 1)
        if track.live is False:
            name = "照片"
            color = (168, 96, 210)
        else:
            name = track.label
            if track.person_name and track.face_score > 0:
                name = f"{track.person_name} {track.face_score:.2f}"
        texts.append((name, (x1, max(8, y1 - 26)), color, 20))

    if command.target_id is not None:
        tx = int((0.5 + command.error.ex) * w)
        ty = int((0.5 + command.error.ey) * h)
        cv2.arrowedLine(vis, (cx, cy), (tx, ty), cream, 2, tipLength=0.12)

    status = "LOST" if command.lost else "LOCK"
    who = locked_name or (str(locked_id) if locked_id is not None else "-")
    hud = [
        f"FPS {fps:.1f}",
        f"{status} {who}",
        f"err x={command.error.ex:+.3f} y={command.error.ey:+.3f} size={command.error.size_ratio:.2f}",
        f"gimbal {gimbal_name} {control_mode} yaw={command.yaw_rate:+.2f} pitch={command.pitch_rate:+.2f}",
    ]
    if watch_line:
        hud.append(watch_line)
    hud.append("点选锁定  e登记人脸  n/p切换  c自动  q退出")
    plate = vis.copy()
    hud_h = 12 + 22 * len(hud) + 8
    hud_w = min(w - 12, 540)
    cv2.rectangle(plate, (8, 6), (hud_w, hud_h), (16, 16, 16), -1)
    vis = cv2.addWeighted(plate, 0.42, vis, 0.58, 0)
    y = 12
    for line in hud:
        color = (90, 90, 230) if line.startswith("离岗") else (236, 236, 232)
        texts.append((line, (16, y), color, 16))
        y += 22

    vis = _draw_texts(vis, texts)
    bar_x = w - 28
    _draw_rate_bar(vis, bar_x, cy, command.yaw_rate, vertical=False)
    _draw_rate_bar(vis, cx, 18, command.pitch_rate, vertical=True)
    return vis


def _draw_rate_bar(img: np.ndarray, x: int, y: int, value: float, vertical: bool) -> None:
    length = 70
    value = max(-1.0, min(1.0, value))
    color = (210, 220, 232) if abs(value) > 0.02 else (80, 80, 80)
    if vertical:
        cv2.line(img, (x - length, y), (x + length, y), (60, 60, 60), 2)
        cv2.line(img, (x, y), (x + int(value * length), y), color, 3)
    else:
        cv2.line(img, (x, y - length), (x, y + length), (60, 60, 60), 2)
        cv2.line(img, (x, y), (x, y - int(value * length)), color, 3)
