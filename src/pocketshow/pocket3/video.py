from __future__ import annotations

import logging
import queue
import subprocess
import threading

import numpy as np

logger = logging.getLogger(__name__)


class H264FrameDecoder:
    """把 Pocket 3 的 Annex-B H.264 解成 BGR 帧，供视觉 pipeline 使用。"""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._proc: subprocess.Popen | None = None
        self._in_q: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "-sn",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("需要 ffmpeg 才能解码 Pocket 3 WiFi 视频流") from exc
        self._running = True
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="h264-in")
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="h264-out")
        self._writer.start()
        self._reader.start()

    def feed(self, data: bytes) -> None:
        try:
            self._in_q.put_nowait(data)
        except queue.Full:
            pass

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _write_loop(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        while self._running:
            try:
                chunk = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            try:
                self._proc.stdin.write(chunk)
            except BrokenPipeError:
                break

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        while self._running:
            piece = self._proc.stdout.read(self.frame_bytes - len(buf))
            if not piece:
                break
            buf += piece
            if len(buf) < self.frame_bytes:
                continue
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
            buf = b""
            with self._lock:
                self._frame = frame

    def close(self) -> None:
        self._running = False
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        if self._proc:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
