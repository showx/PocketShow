from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import sys
import threading
import webbrowser
from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from pocketshow.capture import CaptureStore, looks_like_pocket, pane_from_norm, pocket3_usb_present
from pocketshow.config import Settings, load_settings
from pocketshow.control import GimbalBus
from pocketshow.gallery import FaceGallery, encode_jpeg
from pocketshow.preview import PreviewHub, placeholder_jpeg, safe_cam_id
from pocketshow.recognize import PersonRecognizer
from pocketshow.scene import SceneLog
from pocketshow.seats import box_pose, hit_box, occupied_seat_ids

logger = logging.getLogger("pocketshow.admin")


def _admin_page() -> str:
    local = Path(__file__).with_name("admin.html")
    if local.exists():
        return local.read_text(encoding="utf-8")
    return files("pocketshow").joinpath("admin.html").read_text(encoding="utf-8")


class PatchPerson(BaseModel):
    name: str | None = None
    note: str | None = None
    guest: bool | None = None


class CoverBody(BaseModel):
    filename: str


class MergeBody(BaseModel):
    source_id: str


class SeatPin(BaseModel):
    camera_id: str = ""
    camera_name: str = ""
    cx: float | None = None
    cy: float | None = None
    nx: float | None = None
    ny: float | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    rx: float | None = None
    ry: float | None = None


class SeatMark(SeatPin):
    name: str = ""


class GimbalBody(BaseModel):
    mode: str | None = None
    yaw: float | None = None
    pitch: float | None = None
    recenter: bool = False
    stop: bool = False


class WatchPatch(BaseModel):
    work_start: str | None = None
    work_end: str | None = None
    workdays: list[int] | None = None
    away_s: float | None = None


class CameraPatch(BaseModel):
    id: str | None = None
    name: str | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    channel: int | None = None
    stream: str | None = None
    url: str | None = None
    transport: str | None = None
    monitor: bool | None = None


class CapturePatch(BaseModel):
    source: str | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    channel: int | None = None
    stream: str | None = None
    url: str | None = None
    transport: str | None = None
    camera_id: str | None = None
    camera: CameraPatch | None = None
    remove_camera_id: str | None = None
    monitor_ids: list[str] | None = None


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "无法读取这张图片")
    return image


def jpeg_from_upload(image: np.ndarray) -> bytes:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > 1280:
        scale = 1280 / longest
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return encode_jpeg(image)


from pocketshow.watch import StationWatch, format_hhmm, normalize_workdays, on_duty_roster, person_away_today


def device_status(bus: GimbalBus, *, usb: bool | None = None, watch: StationWatch | None = None) -> dict:
    info = bus.public()
    follow = bool(info.get("connected"))
    if usb is None:
        usb = pocket3_usb_present()
    capture = str(info.get("capture") or "")
    device = str(info.get("device") or "")
    backend = str(info.get("backend") or "")
    pocket = bool(usb) or (follow and looks_like_pocket(device, capture))
    rtsp = capture == "rtsp" or device.lower().startswith("rtsp")
    parts: list[str] = []
    if rtsp and follow:
        parts.append(device or "局域网码流")
    elif usb:
        parts.append("USB 已插入")
    elif capture == "wifi" and follow:
        parts.append("WiFi 画面")
    if follow:
        parts.append("跟拍运行中")
    else:
        parts.append("跟拍未启动")
    if follow and backend == "wifi":
        parts.append("WiFi 云台")
    elif follow and not rtsp:
        parts.append("电机未接")
    if not pocket and not rtsp:
        parts = ["跟拍在跑，但不是 Pocket 3"] if follow else ["未检测到 Pocket 3"]
    return {
        "online": pocket or (rtsp and follow),
        "usb": bool(usb),
        "follow": follow,
        "gimbal": backend,
        "capture": capture,
        "device": device,
        "kind": "rtsp" if rtsp else ("pocket" if pocket else "none"),
        "detail": " · ".join(parts),
        **info,
        "watch": watch.public() if watch is not None else {},
    }


def create_app(settings: Settings) -> FastAPI:
    cfg = settings.recognize
    gallery = FaceGallery(cfg.gallery, cfg.photos)
    bus = GimbalBus(settings.gimbal.command)
    watch = StationWatch(
        settings.watch.status,
        settings.watch.away_s,
        settings_path=settings.watch.settings,
        log_path=settings.watch.log,
        work_start=settings.watch.work_start,
        work_end=settings.watch.work_end,
        workdays=settings.watch.workdays,
    )
    capture_store = CaptureStore(settings.capture, settings.rtsp)
    preview = PreviewHub(settings.preview)
    scene_log = SceneLog(settings.scene.status, settings.scene.log)
    map_log = SceneLog(settings.geomap.status, settings.geomap.log)
    mocap_log = SceneLog(settings.mocap.status, settings.mocap.log)
    blank_jpeg = placeholder_jpeg()
    recognizer: PersonRecognizer | None = None

    def get_recognizer() -> PersonRecognizer:
        nonlocal recognizer
        if recognizer is None:
            recognizer = PersonRecognizer(cfg, gallery=gallery)
        return recognizer

    def person_or_404(person_id: str) -> dict:
        gallery.maybe_reload()
        person = gallery.find(person_id)
        if person is None:
            raise HTTPException(404, "找不到这个人")
        return person

    app = FastAPI(title="PocketShow 人物库", docs_url=None, redoc_url=None)
    app.state.stopping = threading.Event()

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _admin_page()

    @app.get("/api/people")
    def list_people() -> list[dict]:
        gallery.maybe_reload()
        return gallery.public_all(dup_threshold=cfg.dup_threshold)

    @app.get("/api/people/{person_id}")
    def get_person(person_id: str) -> dict:
        return gallery.public(person_or_404(person_id))

    @app.patch("/api/people/{person_id}")
    def patch_person(person_id: str, body: PatchPerson) -> dict:
        person = person_or_404(person_id)
        name = body.name if body.name is not None else person["name"]
        try:
            person = gallery.rename(person_id, name, note=body.note, guest=body.guest)
        except KeyError as exc:
            raise HTTPException(404, "找不到这个人") from exc
        return gallery.public(person)

    @app.delete("/api/people/{person_id}/seats")
    def clear_seats(person_id: str, camera_id: str | None = None) -> dict:
        person_or_404(person_id)
        try:
            person = gallery.clear_seat(person_id, camera_id)
        except KeyError as exc:
            raise HTTPException(404, "找不到这个人") from exc
        return gallery.public(person)

    def _seat_cameras() -> list[dict]:
        cameras = preview.public().get("cameras")
        if cameras is not None:
            return [cam for cam in cameras if cam.get("id")]
        return [
            {"id": str(cam.get("id") or ""), "name": str(cam.get("name") or cam.get("id") or "")}
            for cam in capture_store.public().get("cameras") or []
            if cam.get("monitor") and cam.get("id")
        ]

    def _parse_box(body: SeatPin) -> dict | None:
        if None in (body.x1, body.y1, body.x2, body.y2):
            return None
        x1, x2 = float(body.x1), float(body.x2)
        y1, y2 = float(body.y1), float(body.y2)
        if x2 == x1 or y2 == y1:
            return None
        return {
            "x1": min(x1, x2),
            "y1": min(y1, y2),
            "x2": max(x1, x2),
            "y2": max(y1, y2),
        }

    def _boxes_for_camera(camera_id: str) -> list[dict]:
        for cam in preview.public().get("cameras") or []:
            if str(cam.get("id") or "") != camera_id:
                continue
            return [box for box in (cam.get("boxes") or []) if isinstance(box, dict)]
        return []

    def _seat_location(body: SeatPin) -> dict:
        camera_id = (body.camera_id or "").strip()
        camera_name = body.camera_name or ""
        cx, cy = body.cx, body.cy
        if body.nx is not None and body.ny is not None:
            cameras = _seat_cameras()
            layout = preview.public()
            hit = pane_from_norm(
                body.nx,
                body.ny,
                len(cameras),
                canvas=layout.get("canvas"),
                cameras=cameras,
            )
            if not cameras or hit is None:
                raise HTTPException(400, "当前不是固定镜头宫格，没法指定工位")
            index, local_x, local_y = hit
            camera_id = str(cameras[index].get("id") or "")
            camera_name = str(cameras[index].get("name") or camera_id)
            cx, cy = local_x, local_y
        if camera_id == "" or cx is None or cy is None:
            raise HTTPException(400, "需要镜头和画面位置")
        if not camera_name:
            for cam in _seat_cameras():
                if str(cam.get("id") or "") == camera_id:
                    camera_name = str(cam.get("name") or camera_id)
                    break
            camera_name = camera_name or camera_id
        loc = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "cx": float(cx),
            "cy": float(cy),
            "rx": float(body.rx) if body.rx is not None else None,
            "ry": float(body.ry) if body.ry is not None else None,
            "box": None,
        }
        box = _parse_box(body) or hit_box(_boxes_for_camera(camera_id), loc["cx"], loc["cy"])
        if box is not None:
            loc["cx"], loc["cy"], loc["rx"], loc["ry"] = box_pose(box)
            loc["box"] = box
        return loc

    def _cover_from_preview(
        person: dict,
        camera_id: str,
        cx: float,
        cy: float,
        *,
        rx: float | None = None,
        ry: float | None = None,
    ) -> None:
        data = preview.read_pane(camera_id) if camera_id else None
        if not data:
            cameras = _seat_cameras()
            if camera_id and len(cameras) > 1:
                return
            data = preview.read()
        if not data:
            return
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        kwargs: dict = {}
        if rx is not None and ry is not None:
            kwargs["rx"] = max(float(rx) * 0.92, 0.02)
            kwargs["ry"] = max(float(ry) * 0.92, 0.04)
        gallery.save_point_crop(person, image, cx, cy, **kwargs)

    def _pin_located(person_id: str, loc: dict) -> dict:
        try:
            return gallery.pin_seat(
                person_id,
                loc["camera_id"],
                loc["cx"],
                loc["cy"],
                camera_name=loc["camera_name"],
                rx=loc["rx"],
                ry=loc["ry"],
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "找不到这个人") from exc

    @app.post("/api/people/{person_id}/seats")
    def pin_seat(person_id: str, body: SeatPin) -> dict:
        person = person_or_404(person_id)
        loc = _seat_location(body)
        person = _pin_located(person_id, loc)
        if not person.get("photo"):
            _cover_from_preview(person, loc["camera_id"], loc["cx"], loc["cy"], rx=loc["rx"], ry=loc["ry"])
        return gallery.public(person)

    @app.post("/api/seats")
    def mark_seat(body: SeatMark) -> dict:
        loc = _seat_location(body)
        name = (body.name or "").strip()
        if not name and loc["box"] is None:
            raise HTTPException(400, "请填写姓名")
        person = gallery.enroll(None, name=name or None)
        try:
            person = _pin_located(person["id"], loc)
        except HTTPException:
            gallery.delete(person["id"])
            raise
        _cover_from_preview(person, loc["camera_id"], loc["cx"], loc["cy"], rx=loc["rx"], ry=loc["ry"])
        gallery.maybe_reload()
        person = gallery.find(person["id"]) or person
        return gallery.public(person)

    @app.delete("/api/people/{person_id}")
    def delete_person(person_id: str) -> dict:
        person_or_404(person_id)
        gallery.delete(person_id)
        return {"ok": True}

    @app.get("/api/duplicates")
    def list_duplicates() -> list[dict]:
        gallery.maybe_reload()
        return gallery.similar_pairs(threshold=cfg.dup_threshold)

    @app.get("/api/appear")
    def list_appearances(person_id: str | None = None, limit: int = 200) -> list[dict]:
        gallery.maybe_reload()
        return gallery.appear.visits(person_id=person_id, limit=min(max(limit, 1), 500))

    def scene_public() -> dict:
        data = scene_log.public()
        data["enabled"] = settings.scene.enabled
        data["backend"] = settings.scene.backend
        data["model"] = settings.scene.model
        data["events"] = scene_log.events(limit=40)
        return data

    def map_public() -> dict:
        data = map_log.public()
        data["enabled"] = settings.geomap.enabled
        data["backend"] = settings.geomap.backend
        data["events"] = map_log.events(limit=20)
        return data

    def mocap_public() -> dict:
        data = mocap_log.public()
        data["enabled"] = settings.mocap.enabled
        data["backend"] = settings.mocap.backend
        data["events"] = mocap_log.events(limit=20)
        return data

    @app.get("/api/present")
    def list_present() -> dict:
        return gallery.appear.present()

    @app.get("/api/scene")
    def get_scene() -> dict:
        return scene_public()

    @app.get("/api/map")
    def get_map() -> dict:
        return map_public()

    @app.get("/api/mocap")
    def get_mocap() -> dict:
        return mocap_public()

    @app.get("/api/status")
    def get_status() -> dict:
        capture_store.reload()
        data = device_status(bus, watch=watch)
        data["present"] = gallery.appear.present()
        data["capture_settings"] = capture_store.public()
        data["preview"] = preview.public()
        data["scene"] = scene_public()
        data["geomap"] = map_public()
        data["mocap"] = mocap_public()
        return data

    @app.get("/api/preview.jpg")
    def preview_jpeg() -> Response:
        data = preview.read()
        if data is None:
            raise HTTPException(404, "跟拍未启动，还没有监测画面")
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    async def _mjpeg(request: Request, reader):
        boundary = "frame"
        stopping = app.state.stopping

        async def frames():
            try:
                while not stopping.is_set():
                    if await request.is_disconnected():
                        break
                    data = reader() or blank_jpeg
                    yield (
                        b"--" + boundary.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                    await asyncio.sleep(0.15)
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Connection": "close",
            },
        )

    @app.get("/api/preview")
    async def preview_stream(request: Request) -> StreamingResponse:
        return await _mjpeg(request, preview.read)

    @app.get("/api/preview/{camera_id}")
    async def preview_camera_stream(request: Request, camera_id: str) -> StreamingResponse:
        safe = safe_cam_id(camera_id)
        return await _mjpeg(request, lambda: preview.read_pane(safe) or preview.read())

    @app.get("/api/capture")
    def get_capture() -> dict:
        capture_store.reload()
        return capture_store.public()

    @app.put("/api/capture")
    def put_capture(body: CapturePatch) -> dict:
        try:
            return capture_store.save(
                source=body.source,
                host=body.host,
                port=body.port,
                username=body.username,
                password=body.password,
                channel=body.channel,
                stream=body.stream,
                url=body.url,
                transport=body.transport,
                camera_id=body.camera_id,
                camera=body.camera.model_dump() if body.camera is not None else None,
                remove_camera_id=body.remove_camera_id,
                monitor_ids=body.monitor_ids,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def watch_report() -> dict:
        gallery.maybe_reload()
        snapshot = gallery.appear.present()
        present_ids = {str(p.get("person_id")) for p in snapshot.get("people") or []} if snapshot.get("fresh") else set()
        skip_ids = {str(p.get("id")) for p in gallery.people if p.get("guest")}
        preview_info = preview.public()
        cameras = preview_info.get("cameras") if preview_info.get("fresh") else []
        occupied_ids = occupied_seat_ids(gallery.people, cameras or [])
        present_ids.update(occupied_ids)
        report = watch.report()
        report["today"] = person_away_today(
            gallery.appear.raw_events(),
            work_start=watch.work_start,
            work_end=watch.work_end,
            workdays=watch.workdays,
            min_s=max(15.0, watch.away_limit_s),
            alarm_s=watch.away_limit_s,
            present_ids=present_ids,
            skip_ids=skip_ids,
        )
        report["present"] = snapshot
        report["roster"] = on_duty_roster(
            gallery.people,
            snapshot,
            report["today"],
            on_duty=bool((report.get("settings") or {}).get("on_duty", True)),
            occupied_ids=occupied_ids,
        )
        return report

    @app.get("/api/watch")
    def get_watch() -> dict:
        return watch_report()

    @app.put("/api/watch")
    def put_watch(body: WatchPatch) -> dict:
        try:
            start = format_hhmm(body.work_start) if body.work_start is not None else None
            end = format_hhmm(body.work_end) if body.work_end is not None else None
            days = normalize_workdays(body.workdays) if body.workdays is not None else None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        watch.save_settings(work_start=start, work_end=end, workdays=days, away_s=body.away_s)
        return watch_report()

    @app.get("/api/gimbal")
    def get_gimbal() -> dict:
        return bus.public()

    @app.post("/api/gimbal")
    def post_gimbal(body: GimbalBody) -> dict:
        if body.mode and body.mode not in ("follow", "manual"):
            raise HTTPException(400, "mode 只能是 follow 或 manual")
        bus.command(
            mode=body.mode,
            yaw=body.yaw,
            pitch=body.pitch,
            recenter=body.recenter,
            stop=body.stop,
        )
        return bus.public()

    @app.post("/api/people/{keep_id}/merge")
    def merge_person(keep_id: str, body: MergeBody) -> dict:
        person_or_404(keep_id)
        person_or_404(body.source_id)
        try:
            person = gallery.merge(keep_id, body.source_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "找不到这个人") from exc
        return gallery.public(person)

    @app.post("/api/people")
    async def create_person(file: UploadFile = File(...), name: str = Form("")) -> dict:
        data = await file.read()
        if not data:
            raise HTTPException(400, "空文件")
        image = decode_image(data)
        rec = get_recognizer()
        found = rec.embed_image(image)
        if found is None:
            raise HTTPException(400, "照片里没检测到人脸，换一张正面、光线清楚的试试")
        embedding, xyxy = found
        person = rec.enroll(embedding, name=name.strip() or None)
        rec.gallery.save_crop(person, image, xyxy, force_cover=True)
        return rec.gallery.public(person)

    @app.post("/api/people/{person_id}/photos")
    async def add_photo(
        person_id: str,
        file: UploadFile = File(...),
        as_cover: bool = Form(False),
    ) -> dict:
        person = person_or_404(person_id)
        data = await file.read()
        if not data:
            raise HTTPException(400, "空文件")
        image = decode_image(data)
        rec = get_recognizer()
        found = rec.embed_image(image)
        if found is not None:
            embedding, xyxy = found
            rec.update_embedding(person, embedding, force=True)
            rec.gallery.save_crop(person, image, xyxy, force_cover=as_cover)
        else:
            rec.gallery.add_image_bytes(person, jpeg_from_upload(image), as_cover=as_cover)
        gallery.maybe_reload()
        person = person_or_404(person_id)
        return gallery.public(person)

    @app.post("/api/people/{person_id}/cover")
    def set_cover(person_id: str, body: CoverBody) -> dict:
        person_or_404(person_id)
        try:
            gallery.set_cover(person_id, body.filename)
        except FileNotFoundError as exc:
            raise HTTPException(404, "找不到这张相片") from exc
        return gallery.public(person_or_404(person_id))

    @app.get("/media/{person_id}/{filename}")
    def media(person_id: str, filename: str) -> FileResponse:
        path = gallery.resolve_media(f"{person_id}/{filename}")
        if path is None:
            raise HTTPException(404, "找不到相片")
        return FileResponse(path, media_type="image/jpeg")

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PocketShow 人物库管理页")
    parser.add_argument("--config", default=str(Path("configs/default.yaml")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def _run_server(app: FastAPI, host: str, port: int) -> None:
    kwargs: dict = {
        "app": app,
        "host": host,
        "port": port,
        "log_level": "info",
        "timeout_keep_alive": 1,
    }
    if "timeout_graceful_shutdown" in inspect.signature(uvicorn.Config.__init__).parameters:
        kwargs["timeout_graceful_shutdown"] = 1
    server = uvicorn.Server(uvicorn.Config(**kwargs))
    inner = server.handle_exit

    def handle_exit(sig: int, frame) -> None:
        app.state.stopping.set()
        already = server.should_exit
        inner(sig, frame)
        if already:
            os._exit(0)

    server.handle_exit = handle_exit  # type: ignore[method-assign]
    server.run()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    app = create_app(settings)
    url = f"http://{args.host}:{args.port}"
    logger.info("人物库管理页 %s", url)
    if not args.no_browser:
        webbrowser.open(url)
    _run_server(app, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
