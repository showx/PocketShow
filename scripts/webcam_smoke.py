#!/usr/bin/env python3
"""Pocket 3 Webcam 冒烟测试：取流 + YOLO + ByteTrack + 跟拍决策。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketshow.capture import open_usb_or_camera
from pocketshow.config import CaptureConfig, DetectConfig, FollowConfig
from pocketshow.detect_track import PersonTracker
from pocketshow.follow import FollowController
from pocketshow.gimbal.stub import StubGimbal
from pocketshow.overlay import draw_overlay
from pocketshow.target import TargetLock

OUT = Path("/tmp/pocketshow_live")
OUT.mkdir(parents=True, exist_ok=True)
FRAMES = 24
SAVE_EVERY = 8


def main() -> int:
    cap_cfg = CaptureConfig(source="usb", device_index=0, width=1280, height=720, fps=30)
    print("opening camera...", flush=True)
    capture = open_usb_or_camera(cap_cfg)
    print(f"opened: {capture.label}", flush=True)
    tracker = PersonTracker(DetectConfig(device="auto", imgsz=640, conf=0.35))
    locker = TargetLock()
    follow = FollowController(FollowConfig())
    gimbal = StubGimbal()

    prev = time.monotonic()
    stats = {
        "source": capture.label,
        "frames": 0,
        "with_person": 0,
        "ids": [],
        "yaw": [],
        "pitch": [],
        "fps": [],
        "shape": None,
    }
    t0 = time.monotonic()
    while stats["frames"] < FRAMES:
        frame = capture.read()
        if frame is None:
            continue
        now = time.monotonic()
        dt = max(1e-3, now - prev)
        prev = now
        stats["frames"] += 1
        stats["shape"] = list(frame.shape)
        tracks = tracker.track(frame)
        target = locker.update(tracks, dt)
        h, w = frame.shape[:2]
        cmd = follow.update(target, w, h, dt, now)
        gimbal.apply(cmd)
        fps = 1.0 / dt
        stats["fps"].append(round(fps, 2))
        if tracks:
            stats["with_person"] += 1
            stats["ids"].append([t.id for t in tracks])
        stats["yaw"].append(round(cmd.yaw_rate, 3))
        stats["pitch"].append(round(cmd.pitch_rate, 3))
        print(
            f"f={stats['frames']:02d} {w}x{h} fps={fps:.1f} "
            f"n={len(tracks)} lock={locker.locked_id} "
            f"yaw={cmd.yaw_rate:+.2f} pitch={cmd.pitch_rate:+.2f} lost={cmd.lost}",
            flush=True,
        )
        if stats["frames"] % SAVE_EVERY == 0 or stats["frames"] == 1:
            vis = draw_overlay(frame, tracks, cmd, locker.locked_id, fps, "stub", 0.08)
            path = OUT / f"frame_{stats['frames']:03d}.jpg"
            cv2.imwrite(str(path), vis)
            print(f"saved {path}", flush=True)

    capture.close()
    gimbal.close()
    elapsed = time.monotonic() - t0
    summary = {
        "source": stats["source"],
        "frames": stats["frames"],
        "elapsed_s": round(elapsed, 2),
        "avg_fps": round(stats["frames"] / elapsed, 2) if elapsed else 0,
        "with_person": stats["with_person"],
        "unique_ids": sorted({i for group in stats["ids"] for i in group}),
        "shape": stats["shape"],
        "yaw_last": stats["yaw"][-1] if stats["yaw"] else None,
        "pitch_last": stats["pitch"][-1] if stats["pitch"] else None,
        "yaw_nonzero": sum(1 for y in stats["yaw"] if abs(y) > 0.02),
        "pitch_nonzero": sum(1 for p in stats["pitch"] if abs(p) > 0.02),
    }
    (OUT / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
