from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pocketshow.follow import FollowController
from pocketshow.types import FollowCommand

HOLD_TIMEOUT_S = 0.8


@dataclass
class RemoteGimbal:
    mode: str = "follow"  # follow | manual
    yaw: float = 0.0
    pitch: float = 0.0
    recenter: bool = False
    updated: float = 0.0
    ack: float = 0.0
    backend: str = ""
    applied_yaw: float = 0.0
    applied_pitch: float = 0.0
    capture: str = ""
    device: str = ""


class GimbalBus:
    """网页与跟拍进程之间的云台指令：写 JSON，跟拍循环读取。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def read(self) -> RemoteGimbal:
        if not self.path.exists():
            return RemoteGimbal()
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return RemoteGimbal()
        state = RemoteGimbal()
        for key in asdict(state):
            if key in data:
                setattr(state, key, data[key])
        state.yaw = max(-1.0, min(1.0, float(state.yaw)))
        state.pitch = max(-1.0, min(1.0, float(state.pitch)))
        if state.mode not in ("follow", "manual"):
            state.mode = "follow"
        return state

    def write(self, state: RemoteGimbal) -> RemoteGimbal:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), ensure_ascii=False, indent=2)
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(prefix="gimbal.", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.replace(tmp_name, self.path)
            except Exception:
                Path(tmp_name).unlink(missing_ok=True)
                raise
        return state

    def command(
        self,
        *,
        mode: str | None = None,
        yaw: float | None = None,
        pitch: float | None = None,
        recenter: bool = False,
        stop: bool = False,
    ) -> RemoteGimbal:
        state = self.read()
        if stop:
            state.yaw = 0.0
            state.pitch = 0.0
            state.mode = "manual"
        if yaw is not None:
            state.yaw = max(-1.0, min(1.0, float(yaw)))
        if pitch is not None:
            state.pitch = max(-1.0, min(1.0, float(pitch)))
        if recenter:
            state.recenter = True
            state.mode = "manual"
            state.yaw = 0.0
            state.pitch = 0.0
        elif mode:
            state.mode = mode
        elif yaw is not None or pitch is not None:
            state.mode = "manual"
        state.updated = time.time()
        return self.write(state)

    def ack(
        self,
        backend: str,
        yaw: float,
        pitch: float,
        *,
        clear_recenter: bool = False,
        capture: str = "",
        device: str = "",
    ) -> None:
        state = self.read()
        state.ack = time.time()
        state.backend = backend
        state.applied_yaw = yaw
        state.applied_pitch = pitch
        if capture:
            state.capture = capture
        if device:
            state.device = device
        if clear_recenter:
            state.recenter = False
        self.write(state)

    def public(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        state = self.read()
        holding = state.mode == "manual" and (now - float(state.updated or 0)) < HOLD_TIMEOUT_S
        connected = (now - float(state.ack or 0)) < 2.5
        return {
            "mode": state.mode,
            "yaw": state.yaw if holding else 0.0,
            "pitch": state.pitch if holding else 0.0,
            "recenter": state.recenter,
            "backend": state.backend,
            "applied_yaw": state.applied_yaw,
            "applied_pitch": state.applied_pitch,
            "connected": connected,
            "holding": holding,
            "capture": state.capture,
            "device": state.device,
        }


def mix_command(
    auto: FollowCommand,
    remote: RemoteGimbal,
    follow: FollowController,
    now: float | None = None,
) -> tuple[FollowCommand, str, bool]:
    """手动优先。返回 (指令, 模式文案, 是否回中)。"""
    now = time.time() if now is None else now
    if remote.recenter:
        follow.reset()
        idle = FollowCommand(0.0, 0.0, auto.lost, auto.error, auto.target_id)
        return idle, "回中", True
    fresh = (now - float(remote.updated or 0)) < HOLD_TIMEOUT_S
    if remote.mode == "manual" and fresh:
        follow.reset()
        return (
            FollowCommand(remote.yaw, remote.pitch, False, auto.error, auto.target_id),
            "手动",
            False,
        )
    if remote.mode == "manual":
        follow.reset()
        return FollowCommand(0.0, 0.0, auto.lost, auto.error, auto.target_id), "手动停", False
    return auto, "跟拍", False
