from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pocketshow.capture import (
    CaptureStore,
    FrameSource,
    capture_identity,
    compose_grid,
    grid_cell_size,
    looks_like_pocket,
    open_capture,
    open_rtsp_many,
    pane_index,
    redact_rtsp_url,
    try_open_pocket,
)
from pocketshow.config import Settings, load_settings
from pocketshow.control import GimbalBus, mix_command
from pocketshow.detect_track import PersonTracker
from pocketshow.follow import FollowController
from pocketshow.gimbal.stub import StubGimbal
from pocketshow.overlay import draw_overlay
from pocketshow.pocket3.udp import DjiUdpClient
from pocketshow.preview import PreviewHub
from pocketshow.recognize import PersonRecognizer
from pocketshow.target import TargetLock
from pocketshow.types import FollowCommand, FrameError, Track
from pocketshow.watch import StationWatch, hud_line

logger = logging.getLogger("pocketshow")
WINDOW = "PocketShow"
TRACK_ID_GAP = 100000


def preview_canvas(window: str = WINDOW) -> tuple[int, int]:
    try:
        rect = cv2.getWindowImageRect(window)
        if rect is not None and len(rect) >= 4 and int(rect[2]) >= 320 and int(rect[3]) >= 240:
            return int(rect[2]), int(rect[3])
    except Exception:
        pass
    return 1920, 1080


@dataclass
class CameraView:
    cam_id: str
    name: str
    capture: FrameSource
    tracker: PersonTracker
    locker: TargetLock
    offset: int
    miss: int = 0
    tracks: list[Track] = field(default_factory=list)
    frame: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pocket 3 智能跟拍闭环")
    parser.add_argument("--config", default=str(Path("configs/default.yaml")))
    parser.add_argument("--source", choices=["auto", "usb", "camera", "file", "wifi", "rtsp"])
    parser.add_argument("--file")
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--stream", choices=["main", "sub", "third"], help="海康主/子/第三码流")
    parser.add_argument("--rtsp-host")
    parser.add_argument("--rtsp-channel", type=int)
    parser.add_argument("--rtsp-camera", help="多路摄像机 id，对应 rtsp.cameras")
    parser.add_argument("--gimbal", choices=["stub", "wifi"])
    parser.add_argument("--ssid")
    parser.add_argument("--password")
    parser.add_argument("--ble", action="store_true", help="BLE 配对以唤醒相机 WiFi AP")
    parser.add_argument("--join-wifi", action="store_true", help="用 networksetup/nmcli 加入相机热点")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.source:
        settings.capture.source = args.source
    if args.file:
        settings.capture.file = args.file
        settings.capture.source = "file"
    if args.device_index is not None:
        settings.capture.device_index = args.device_index
    if args.gimbal:
        settings.gimbal.backend = args.gimbal
    if args.ssid:
        settings.wifi.ssid = args.ssid
    if args.password:
        settings.wifi.password = args.password
    if args.ble:
        settings.wifi.ble = True
    if args.join_wifi:
        settings.wifi.join = True
    if args.stream:
        settings.rtsp.stream = args.stream
    if args.rtsp_host:
        settings.rtsp.host = args.rtsp_host
        if settings.capture.source == "auto":
            settings.capture.source = "rtsp"
    if args.rtsp_channel is not None:
        settings.rtsp.channel = args.rtsp_channel
    if getattr(args, "rtsp_camera", None):
        settings.rtsp.camera_id = args.rtsp_camera
    return settings


def connect_pocket3(settings: Settings) -> DjiUdpClient:
    wifi = settings.wifi
    if wifi.ble:
        from pocketshow.pocket3.ble import activate_wifi_ap_sync

        if not activate_wifi_ap_sync(wifi.ble_timeout_s):
            raise RuntimeError("BLE 配对失败。可在相机上手动打开 WiFi，然后去掉 --ble。")
        time.sleep(18)

    if wifi.join:
        from pocketshow.pocket3.wifi_join import join_wifi, wait_for_camera

        if not join_wifi(wifi.ssid, wifi.password):
            raise RuntimeError("加入相机 WiFi 失败，检查 ssid/password。")
        wait_for_camera(wifi.camera_ip)

    client = DjiUdpClient(wifi.camera_ip, wifi.camera_port)
    if not client.connect():
        raise RuntimeError(
            "UDP 握手失败。确认已连上 OsmoPocket3 热点、没有其它程序占用 9004，"
            "且相机未停留在纯 Webcam 互斥状态。"
        )
    client.start()
    return client


def _idle_command() -> FollowCommand:
    return FollowCommand(0.0, 0.0, True, FrameError(0.0, 0.0, 0.0), None)


def open_camera_views(store: CaptureStore, settings: Settings, only_id: str | None) -> list[CameraView]:
    opened = open_rtsp_many(store.rtsp, only_id)
    views: list[CameraView] = []
    for index, (cam, source) in enumerate(opened):
        views.append(
            CameraView(
                cam_id=cam.id,
                name=cam.name or cam.host or cam.id,
                capture=source,
                tracker=PersonTracker(settings.detect),
                locker=TargetLock(settings.follow.lost_timeout_s),
                offset=(index + 1) * TRACK_ID_GAP,
            )
        )
    logger.info("同时监测 %s 路：%s", len(views), "、".join(view.name for view in views))
    return views


def close_camera_views(views: list[CameraView]) -> None:
    for view in views:
        view.capture.close()


def run_rtsp_monitor(
    args: argparse.Namespace,
    settings: Settings,
    store: CaptureStore,
    watch: StationWatch | None,
    gimbal,
    gimbal_name: str,
    bus: GimbalBus,
    recognizer: PersonRecognizer | None,
    only_id: str | None,
    preview: PreviewHub,
) -> int:
    views = open_camera_views(store, settings, only_id)
    idle = _idle_command()
    latest: dict[str, list[Track]] = {"tracks": []}
    focus = 0
    layout = {"canvas": (1920, 1080)}
    if not args.no_preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1920, 1080)

        def on_mouse(event, x, y, _flags, _param) -> None:
            nonlocal focus
            if event != cv2.EVENT_LBUTTONDOWN or not views:
                return
            canvas = layout["canvas"]
            index = pane_index(x, y, len(views), canvas)
            if index is None:
                return
            focus = index
            view = views[index]
            if view.frame is None:
                return
            cell_w, cell_h = grid_cell_size(len(views), canvas)
            local_x = x % cell_w
            local_y = y % cell_h
            vh, vw = view.frame.shape[:2]
            ox = int(local_x * vw / max(cell_w, 1))
            oy = int(local_y * vh / max(cell_h, 1))
            locked = view.locker.lock_at(ox, oy, view.tracks)
            if locked is not None:
                name = next((t.person_name for t in view.tracks if t.id == locked), None)
                logger.info("锁定 %s ID %s %s", view.name, locked, name or "")

        cv2.setMouseCallback(WINDOW, on_mouse)

    prev = time.monotonic()
    fps = 0.0
    last_store_check = 0.0
    logger.info("多路监测已启动。q 退出。")
    try:
        while True:
            now_wall = time.monotonic()
            if now_wall - last_store_check > 1.0:
                last_store_check = now_wall
                if store.maybe_reload():
                    try:
                        refreshed = open_camera_views(store, settings, only_id)
                    except Exception:
                        logger.exception("摄像机名单已改，但重连失败，仍用当前画面")
                    else:
                        close_camera_views(views)
                        views = refreshed
                        focus = 0
            panes: list[np.ndarray] = []
            groups: list[list[Track]] = []
            any_ok = False
            for view in views:
                frame = view.capture.read()
                if frame is None:
                    view.miss += 1
                    if view.miss == 1 or view.miss % 80 == 0:
                        logger.warning("%s 中断，稍后重试", view.name)
                    groups.append([])
                    panes.append(np.zeros((180, 320, 3), dtype=np.uint8))
                    continue
                view.miss = 0
                any_ok = True
                view.frame = frame
                tracks = view.tracker.track(frame)
                for track in tracks:
                    track.id += view.offset
                if recognizer is not None:
                    tracks = recognizer.apply(frame, tracks, tick_appear=False)
                view.tracks = tracks
                view.locker.update(tracks, max(1e-3, now_wall - prev))
                groups.append(tracks)
                panes.append(
                    draw_overlay(
                        frame,
                        tracks,
                        idle,
                        view.locker.locked_id,
                        fps,
                        gimbal_name,
                        settings.follow.deadzone,
                        monitor=True,
                        title=view.name,
                    )
                )
            if recognizer is not None:
                recognizer.flush_appear(groups)
            merged = [track for group in groups for track in group]
            latest["tracks"] = merged
            station = None
            if watch is not None:
                station = watch.tick(merged, camera_ok=any_ok)
            dt = now_wall - prev
            prev = now_wall
            fps = fps * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1
            bus.ack(
                gimbal_name,
                0.0,
                0.0,
                capture="rtsp",
                device=" + ".join(view.name for view in views),
            )
            if not any_ok:
                time.sleep(0.02)
            layout["canvas"] = preview_canvas(WINDOW) if not args.no_preview else (1920, 1080)
            grid = compose_grid(panes, layout["canvas"])
            preview.publish(grid)
            if args.no_preview:
                continue
            cv2.imshow(WINDOW, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("c") and views:
                views[focus].locker.clear()
            if key == ord("e") and recognizer is not None and views:
                view = views[focus]
                target = next((t for t in view.tracks if t.id == view.locker.locked_id), None)
                if target is None and view.tracks:
                    target = view.locker.update(view.tracks, 0.0)
                if target is not None and view.frame is not None:
                    name = recognizer.enroll_track(view.frame, target)
                    if name:
                        logger.info("已登记 %s", name)
        return 0
    finally:
        close_camera_views(views)


def build_gimbal(settings: Settings, client: DjiUdpClient | None):
    if settings.gimbal.backend == "stub":
        return StubGimbal(), "stub"
    if client is None:
        raise RuntimeError("--gimbal wifi 需要 Pocket 3 UDP 连接")
    from pocketshow.gimbal.wifi import WifiGimbal

    gimbal = WifiGimbal(client, settings.gimbal.send_hz)
    gimbal.start()
    return gimbal, "wifi"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    settings = apply_overrides(settings, args)

    need_link = settings.gimbal.backend == "wifi" or settings.capture.source == "wifi"
    client: DjiUdpClient | None = None
    if need_link:
        client = connect_pocket3(settings)

    capture: FrameSource | None = None
    gimbal = None
    store = CaptureStore(settings.capture, settings.rtsp)
    try:
        recognizer: PersonRecognizer | None = None
        if settings.recognize.enabled:
            recognizer = PersonRecognizer(settings.recognize)
        gimbal, gimbal_name = build_gimbal(settings, client)
        bus = GimbalBus(settings.gimbal.command)
        watch = (
            StationWatch(
                settings.watch.status,
                settings.watch.away_s,
                settings_path=settings.watch.settings,
                log_path=settings.watch.log,
                work_start=settings.watch.work_start,
                work_end=settings.watch.work_end,
                workdays=settings.watch.workdays,
            )
            if settings.watch.enabled
            else None
        )
        preview = PreviewHub(settings.preview)
        if store.capture.source == "rtsp":
            return run_rtsp_monitor(
                args,
                settings,
                store,
                watch,
                gimbal,
                gimbal_name,
                bus,
                recognizer,
                args.rtsp_camera,
                preview,
            )
        capture = open_capture(store.capture, settings.wifi, client, store.rtsp)
        tracker = PersonTracker(settings.detect)
        locker = TargetLock(settings.follow.lost_timeout_s)
        follow = FollowController(settings.follow)
        capture_kind, capture_name = capture_identity(capture)

        latest: dict[str, list[Track]] = {"tracks": []}
        if not args.no_preview:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

            def on_mouse(event, x, y, _flags, _param) -> None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    locked = locker.lock_at(x, y, latest["tracks"])
                    if locked is not None:
                        name = next((t.person_name for t in latest["tracks"] if t.id == locked), None)
                        logger.info("手动锁定 ID %s %s", locked, name or "")

            cv2.setMouseCallback(WINDOW, on_mouse)

        prev = time.monotonic()
        fps = 0.0
        miss = 0
        last_pocket_probe = 0.0
        last_store_check = 0.0
        allow_pocket = store.capture.source in ("auto", "usb", "camera")
        logger.info("跟拍循环已启动。q 退出。")
        while True:
            now_wall = time.monotonic()
            if now_wall - last_store_check > 1.0:
                last_store_check = now_wall
                if store.maybe_reload():
                    allow_pocket = store.capture.source in ("auto", "usb", "camera")
                    try:
                        switched = open_capture(store.capture, settings.wifi, client, store.rtsp)
                    except Exception:
                        logger.exception("监测源已改，但重连失败，仍用当前画面")
                    else:
                        if capture is not None:
                            capture.close()
                        capture = switched
                        capture_kind, capture_name = capture_identity(capture)
                        miss = 0
                        shown = store.identity()[1]
                        logger.info(
                            "已切换监测源 %s",
                            capture_name if capture_kind != "rtsp" else redact_rtsp_url(shown) or capture_name,
                        )
            frame = capture.read() if capture is not None else None
            if frame is None:
                miss += 1
                if watch is not None:
                    watch.tick([], camera_ok=False)
                if allow_pocket and (miss == 1 or miss % 60 == 0):
                    switched = try_open_pocket(store.capture, capture_name, force=miss > 20)
                    if switched is not None:
                        if capture is not None:
                            capture.close()
                        capture = switched
                        capture_kind, capture_name = capture_identity(capture)
                        logger.info("已自动切到 %s", capture_name)
                        miss = 0
                elif capture_kind == "rtsp" and (miss == 1 or miss % 80 == 0):
                    try:
                        switched = open_capture(store.capture, settings.wifi, client, store.rtsp)
                    except Exception:
                        logger.warning("RTSP 中断，稍后重试")
                    else:
                        if capture is not None:
                            capture.close()
                        capture = switched
                        capture_kind, capture_name = capture_identity(capture)
                        logger.info("已重连 %s", capture_name)
                        miss = 0
                time.sleep(0.02)
                continue
            if miss:
                miss = 0
            if (
                allow_pocket
                and capture is not None
                and not looks_like_pocket(capture_name, capture_kind)
                and now_wall - last_pocket_probe > 3.0
            ):
                last_pocket_probe = now_wall
                switched = try_open_pocket(store.capture, capture_name)
                if switched is not None:
                    capture.close()
                    capture = switched
                    capture_kind, capture_name = capture_identity(capture)
                    logger.info("检测到 Pocket 3，已切换视频源 %s", capture_name)
                    continue
            now = now_wall
            dt = now - prev
            prev = now
            fps = fps * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1

            tracks = tracker.track(frame)
            if recognizer is not None:
                tracks = recognizer.apply(frame, tracks)
            latest["tracks"] = tracks
            station = None
            if watch is not None:
                station = watch.tick(tracks, camera_ok=True)
            target = locker.update(tracks, dt)
            h, w = frame.shape[:2]
            command: FollowCommand = follow.update(target, w, h, dt, now)
            remote = bus.read()
            command, control_mode, do_recenter = mix_command(command, remote, follow)
            if do_recenter:
                gimbal.recenter()
            else:
                gimbal.apply(command)
            bus.ack(
                gimbal_name,
                command.yaw_rate,
                command.pitch_rate,
                clear_recenter=do_recenter,
                capture=capture_kind,
                device=capture_name,
            )

            vis = draw_overlay(
                frame,
                tracks,
                command,
                locker.locked_id,
                fps,
                gimbal_name,
                settings.follow.deadzone,
                locked_name=next(
                    (t.person_name for t in tracks if t.id == locker.locked_id and t.person_name),
                    None,
                ),
                control_mode=control_mode,
                watch_line=" · ".join(
                    part
                    for part in (
                        capture_name,
                        hud_line(station) if station is not None else "",
                    )
                    if part
                ),
            )
            preview.publish(vis)
            if args.no_preview:
                if command.target_id is not None:
                    logger.debug(
                        "id=%s yaw=%.2f pitch=%.2f",
                        command.target_id,
                        command.yaw_rate,
                        command.pitch_rate,
                    )
                continue
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                locker.clear()
                follow.reset()
            if key == ord("n"):
                locker.cycle(tracks, 1)
            if key == ord("p"):
                locker.cycle(tracks, -1)
            if key == ord("r"):
                gimbal.recenter()
                follow.reset()
            if key == ord(" "):
                gimbal.set_velocity(0.0, 0.0)
            if key == ord("e") and recognizer is not None:
                target_track = next((t for t in tracks if t.id == locker.locked_id), None)
                if target_track is None and tracks:
                    target_track = locker.update(tracks, 0.0)
                if target_track is not None:
                    name = recognizer.enroll_track(frame, target_track)
                    if name:
                        logger.info("已登记 %s", name)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if gimbal is not None:
            gimbal.close()
        if capture is not None:
            capture.close()
        if client is not None:
            client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
