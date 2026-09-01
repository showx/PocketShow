from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from pocketshow.capture import (
    FrameSource,
    capture_identity,
    looks_like_pocket,
    open_capture,
    try_open_pocket,
)
from pocketshow.config import Settings, load_settings
from pocketshow.control import GimbalBus, mix_command
from pocketshow.detect_track import PersonTracker
from pocketshow.follow import FollowController
from pocketshow.gimbal.stub import StubGimbal
from pocketshow.overlay import draw_overlay
from pocketshow.pocket3.udp import DjiUdpClient
from pocketshow.recognize import PersonRecognizer
from pocketshow.target import TargetLock
from pocketshow.types import FollowCommand, Track
from pocketshow.watch import StationWatch, hud_line

logger = logging.getLogger("pocketshow")
WINDOW = "PocketShow"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pocket 3 智能跟拍闭环")
    parser.add_argument("--config", default=str(Path("configs/default.yaml")))
    parser.add_argument("--source", choices=["auto", "usb", "camera", "file", "wifi"])
    parser.add_argument("--file")
    parser.add_argument("--device-index", type=int)
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
    config_path = Path(args.config)
    settings = load_settings(config_path if config_path.exists() else None)
    settings = apply_overrides(settings, args)

    need_link = settings.gimbal.backend == "wifi" or settings.capture.source == "wifi"
    client: DjiUdpClient | None = None
    if need_link:
        client = connect_pocket3(settings)

    capture: FrameSource | None = None
    gimbal = None
    try:
        capture = open_capture(settings.capture, settings.wifi, client)
        tracker = PersonTracker(settings.detect)
        recognizer: PersonRecognizer | None = None
        if settings.recognize.enabled:
            recognizer = PersonRecognizer(settings.recognize)
        locker = TargetLock(settings.follow.lost_timeout_s)
        follow = FollowController(settings.follow)
        gimbal, gimbal_name = build_gimbal(settings, client)
        bus = GimbalBus(settings.gimbal.command)
        capture_kind, capture_name = capture_identity(capture)
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
        logger.info("跟拍循环已启动。q 退出。")
        while True:
            frame = capture.read() if capture is not None else None
            if frame is None:
                miss += 1
                if watch is not None:
                    watch.tick([], camera_ok=False)
                if miss == 1 or miss % 60 == 0:
                    switched = try_open_pocket(settings.capture, capture_name, force=miss > 20)
                    if switched is not None:
                        if capture is not None:
                            capture.close()
                        capture = switched
                        capture_kind, capture_name = capture_identity(capture)
                        logger.info("已自动切到 %s", capture_name)
                        miss = 0
                time.sleep(0.02)
                continue
            if miss:
                miss = 0
            now_wall = time.monotonic()
            if (
                capture is not None
                and not looks_like_pocket(capture_name, capture_kind)
                and now_wall - last_pocket_probe > 3.0
            ):
                last_pocket_probe = now_wall
                switched = try_open_pocket(settings.capture, capture_name)
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

            if args.no_preview:
                if command.target_id is not None:
                    logger.debug(
                        "id=%s yaw=%.2f pitch=%.2f",
                        command.target_id,
                        command.yaw_rate,
                        command.pitch_rate,
                    )
                continue

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
                watch_line=hud_line(station) if station is not None else "",
            )
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
