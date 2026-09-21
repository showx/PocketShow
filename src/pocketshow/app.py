from __future__ import annotations

import argparse
import logging
import sys
import threading
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
    looks_like_pocket,
    native_mosaic_size,
    open_capture,
    open_rtsp_many,
    pane_index,
    pane_source_xy,
    redact_rtsp_url,
    reopen_rtsp,
    should_reconnect,
    try_open_pocket,
)
from pocketshow.config import Settings, load_settings
from pocketshow.control import GimbalBus, mix_command
from pocketshow.detect_track import PersonTracker
from pocketshow.follow import FollowController
from pocketshow.gimbal.stub import StubGimbal
from pocketshow.geomap import SceneMapper
from pocketshow.mocap import MocapClient
from pocketshow.overlay import draw_overlay
from pocketshow.pocket3.udp import DjiUdpClient
from pocketshow.preview import PreviewHub
from pocketshow.recognize import PersonRecognizer
from pocketshow.scene import SceneNarrator
from pocketshow.seats import boxes_from_tracks
from pocketshow.target import TargetLock
from pocketshow.types import FollowCommand, FrameError, Track
from pocketshow.watch import StationWatch, hud_line

logger = logging.getLogger("pocketshow")
WINDOW = "PocketShow"
TRACK_ID_GAP = 100000


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
    pending: np.ndarray | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


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


def _push_frame(view: CameraView, frame: np.ndarray) -> None:
    with view.lock:
        view.frame = frame
        view.pending = frame
        view.miss = 0


def _display_snapshot(view: CameraView) -> tuple[np.ndarray | None, list[Track], int | None]:
    with view.lock:
        return view.frame, list(view.tracks), view.locker.locked_id


def _take_pending(view: CameraView) -> np.ndarray | None:
    with view.lock:
        frame = view.pending
        view.pending = None
        return frame


def _store_tracks(view: CameraView, tracks: list[Track], dt: float) -> None:
    with view.lock:
        view.tracks = tracks
        view.locker.update(tracks, dt)


def _analyze_view(
    view: CameraView,
    recognizer: PersonRecognizer | None,
    scene: SceneNarrator | None,
    mapper: SceneMapper | None,
    mocap: MocapClient | None,
    gate: threading.Lock,
    dt: float,
) -> bool:
    frame = _take_pending(view)
    if frame is None:
        return False
    tracks = view.tracker.track(
        frame,
        regions=recognizer.seats_for(view.cam_id) if recognizer is not None and recognizer.cfg.seats else None,
    )
    for track in tracks:
        track.id += view.offset
    if mapper is not None:
        mapper.annotate(tracks, view.cam_id)
    if mocap is not None:
        mocap.annotate(tracks, view.cam_id)
    if recognizer is not None:
        with gate:
            tracks = recognizer.apply(
                frame,
                tracks,
                tick_appear=False,
                camera_id=view.cam_id,
                camera_name=view.name,
            )
    _store_tracks(view, tracks, dt)
    if mapper is not None:
        mapper.offer(frame, tracks, camera_id=view.cam_id, camera_name=view.name)
    if mocap is not None:
        mocap.offer(frame, tracks, camera_id=view.cam_id, camera_name=view.name)
    if scene is not None:
        scene.offer(frame, tracks, camera_id=view.cam_id, camera_name=view.name)
    return True


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
    scene: SceneNarrator | None = None,
    mapper: SceneMapper | None = None,
    mocap: MocapClient | None = None,
) -> int:
    views = open_camera_views(store, settings, only_id)
    idle = _idle_command()
    latest: dict[str, list[Track]] = {"tracks": []}
    focus = 0
    layout = {"canvas": (1280, 720)}
    ctl: dict = {"views": views, "stop": threading.Event()}
    gate = threading.Lock()
    if not args.no_preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, _flags, _param) -> None:
            nonlocal focus
            live = ctl["views"]
            if event != cv2.EVENT_LBUTTONDOWN or not live:
                return
            canvas = layout["canvas"]
            index = pane_index(x, y, len(live), canvas)
            if index is None:
                return
            focus = index
            view = live[index]
            with view.lock:
                frame = view.frame
                tracks = list(view.tracks)
            if frame is None:
                return
            vh, vw = frame.shape[:2]
            hit = pane_source_xy(x, y, len(live), canvas, vw, vh)
            if hit is None:
                return
            with view.lock:
                locked = view.locker.lock_at(hit[1], hit[2], tracks)
            if locked is not None:
                name = next((t.person_name for t in tracks if t.id == locked), None)
                logger.info("锁定 %s ID %s %s", view.name, locked, name or "")

        cv2.setMouseCallback(WINDOW, on_mouse)

    def analyze_loop() -> None:
        prev_analyze = time.monotonic()
        while not ctl["stop"].is_set():
            batch: list[CameraView] = ctl["views"]
            now = time.monotonic()
            dt = max(1e-3, now - prev_analyze)
            prev_analyze = now
            groups: list[list[Track]] = []
            worked = False
            try:
                for view in batch:
                    if ctl["stop"].is_set():
                        return
                    try:
                        if _analyze_view(view, recognizer, scene, mapper, mocap, gate, dt):
                            worked = True
                    except Exception:
                        logger.exception("识别 %s 失败", view.name)
                    with view.lock:
                        groups.append(list(view.tracks))
                if recognizer is not None and worked:
                    with gate:
                        recognizer.flush_appear(groups)
            except Exception:
                logger.exception("识别线程异常")
                ctl["stop"].wait(0.2)
                continue
            if not worked:
                ctl["stop"].wait(0.01)

    def start_analyze() -> threading.Thread:
        ctl["stop"] = threading.Event()
        thread = threading.Thread(target=analyze_loop, daemon=True, name="rtsp-analyze")
        thread.start()
        return thread

    def stop_analyze(thread: threading.Thread | None) -> None:
        ctl["stop"].set()
        if thread is not None:
            thread.join(timeout=2.5)

    worker = start_analyze()
    prev = time.monotonic()
    fps = 0.0
    last_store_check = 0.0
    logger.info("多路监测已启动。预览和识别分开跑。q 退出。")
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
                        stop_analyze(worker)
                        close_camera_views(views)
                        views = refreshed
                        ctl["views"] = views
                        focus = 0
                        worker = start_analyze()
            panes: list[np.ndarray] = []
            groups: list[list[Track]] = []
            any_ok = False
            live = ctl["views"]
            for view in live:
                frame = view.capture.read()
                if frame is None:
                    view.miss += 1
                    if view.miss == 1 or view.miss % 80 == 0:
                        logger.warning("%s 中断，稍后重试", view.name)
                    if should_reconnect(view.miss):
                        try:
                            source = reopen_rtsp(store.rtsp, view.cam_id)
                        except Exception:
                            logger.exception("重连 %s 失败", view.name)
                        else:
                            view.capture.close()
                            view.capture = source
                            view.miss = 0
                            logger.info("已重连 %s", view.name)
                else:
                    _push_frame(view, frame)
                    any_ok = True
                shot, tracks, locked_id = _display_snapshot(view)
                if shot is not None and view.miss < 15:
                    groups.append(tracks)
                    panes.append(
                        draw_overlay(
                            shot,
                            tracks,
                            idle,
                            locked_id,
                            fps,
                            gimbal_name,
                            settings.follow.deadzone,
                            monitor=True,
                            title=view.name,
                            scene_line=scene.line_for(view.cam_id) if scene is not None else "",
                            map_line=mapper.line_for(view.cam_id) if mapper is not None else "",
                            mocap_line=mocap.line_for(view.cam_id) if mocap is not None else "",
                            seats=recognizer.seats_for(view.cam_id) if recognizer is not None else None,
                        )
                    )
                else:
                    groups.append([])
                    panes.append(np.zeros((180, 320, 3), dtype=np.uint8))
            merged = [track for group in groups for track in group]
            latest["tracks"] = merged
            if watch is not None:
                watch.tick(merged, camera_ok=any_ok)
            dt = now_wall - prev
            prev = now_wall
            fps = fps * 0.9 + (1.0 / max(dt, 1e-3)) * 0.1
            bus.ack(
                gimbal_name,
                0.0,
                0.0,
                capture="rtsp",
                device=" + ".join(view.name for view in live),
            )
            if not any_ok:
                time.sleep(0.02)
            layout["canvas"] = native_mosaic_size(panes)
            grid = compose_grid(panes, layout["canvas"])
            preview.publish(
                grid,
                cameras=[
                    {
                        "id": view.cam_id,
                        "name": view.name,
                        **(
                            {
                                "src_w": int(view.frame.shape[1]),
                                "src_h": int(view.frame.shape[0]),
                                "boxes": boxes_from_tracks(
                                    group,
                                    int(view.frame.shape[1]),
                                    int(view.frame.shape[0]),
                                ),
                            }
                            if view.frame is not None
                            else {}
                        ),
                    }
                    for view, group in zip(live, groups, strict=False)
                ],
                panes=[
                    {"id": view.cam_id, "name": view.name, "image": pane}
                    for view, pane in zip(live, panes, strict=False)
                ],
            )
            if args.no_preview:
                continue
            cv2.imshow(WINDOW, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("c") and live:
                with live[focus].lock:
                    live[focus].locker.clear()
            if key == ord("e") and recognizer is not None and live:
                view = live[focus]
                with view.lock:
                    frame = view.frame
                    tracks = list(view.tracks)
                    locked_id = view.locker.locked_id
                target = next((t for t in tracks if t.id == locked_id), None)
                if target is None and tracks:
                    with view.lock:
                        target = view.locker.update(tracks, 0.0)
                if target is not None and frame is not None:
                    with gate:
                        name = recognizer.enroll_track(frame, target)
                    if name:
                        logger.info("已登记 %s", name)
        return 0
    finally:
        stop_analyze(worker)
        close_camera_views(ctl["views"])


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
    scene: SceneNarrator | None = None
    mapper: SceneMapper | None = None
    mocap: MocapClient | None = None
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
        scene = SceneNarrator(settings.scene) if settings.scene.enabled else None
        if scene is not None:
            logger.info("场景理解旁路已开：%s %s（不进跟拍）", settings.scene.backend, settings.scene.model)
        mapper = SceneMapper(settings.geomap) if settings.geomap.enabled else None
        if mapper is not None:
            logger.info("三维重建旁路已开：LingBot-Map %s（不进跟拍）", settings.geomap.backend)
        mocap = MocapClient(settings.mocap) if settings.mocap.enabled else None
        if mocap is not None:
            logger.info("动作捕捉旁路已开：FreeMoCap %s（不进跟拍）", settings.mocap.backend)
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
                scene,
                mapper,
                mocap,
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
            if mapper is not None:
                mapper.annotate(tracks, capture_kind)
            if mocap is not None:
                mocap.annotate(tracks, capture_kind)
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
            if mapper is not None:
                mapper.offer(frame, tracks, camera_id=capture_kind, camera_name=capture_name)
            if mocap is not None:
                mocap.offer(frame, tracks, camera_id=capture_kind, camera_name=capture_name)
            if scene is not None:
                scene.offer(frame, tracks, camera_id=capture_kind, camera_name=capture_name)

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
                scene_line=scene.line_for(capture_kind) if scene is not None else "",
                map_line=mapper.line_for(capture_kind) if mapper is not None else "",
                mocap_line=mocap.line_for(capture_kind) if mocap is not None else "",
            )
            preview.publish(vis, cameras=[])
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
        if mocap is not None:
            mocap.close()
        if mapper is not None:
            mapper.close()
        if scene is not None:
            scene.close()
        if gimbal is not None:
            gimbal.close()
        if capture is not None:
            capture.close()
        if client is not None:
            client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
