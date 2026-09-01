from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from pocketshow.capture import looks_like_pocket, pocket3_usb_present
from pocketshow.config import Settings, load_settings
from pocketshow.control import GimbalBus
from pocketshow.gallery import FaceGallery, encode_jpeg
from pocketshow.recognize import PersonRecognizer

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


from pocketshow.watch import StationWatch, format_hhmm, normalize_workdays, person_away_today


def device_status(bus: GimbalBus, *, usb: bool | None = None, watch: StationWatch | None = None) -> dict:
    info = bus.public()
    follow = bool(info.get("connected"))
    if usb is None:
        usb = pocket3_usb_present()
    capture = str(info.get("capture") or "")
    device = str(info.get("device") or "")
    backend = str(info.get("backend") or "")
    pocket = bool(usb) or (follow and looks_like_pocket(device, capture))
    parts: list[str] = []
    if usb:
        parts.append("USB 已插入")
    elif capture == "wifi" and follow:
        parts.append("WiFi 画面")
    if follow:
        parts.append("跟拍运行中")
    else:
        parts.append("跟拍未启动")
    if follow and backend == "wifi":
        parts.append("WiFi 云台")
    elif follow:
        parts.append("电机未接")
    if not pocket:
        parts = ["跟拍在跑，但不是 Pocket 3"] if follow else ["未检测到 Pocket 3"]
    return {
        "online": pocket,
        "usb": bool(usb),
        "follow": follow,
        "gimbal": backend,
        "capture": capture,
        "device": device,
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

    @app.get("/api/present")
    def list_present() -> dict:
        return gallery.appear.present()

    @app.get("/api/status")
    def get_status() -> dict:
        data = device_status(bus, watch=watch)
        data["present"] = gallery.appear.present()
        return data

    def watch_report() -> dict:
        gallery.maybe_reload()
        snapshot = gallery.appear.present()
        present_ids = {str(p.get("person_id")) for p in snapshot.get("people") or []} if snapshot.get("fresh") else set()
        skip_ids = {str(p.get("id")) for p in gallery.people if p.get("guest")}
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
            rec.update_embedding(person, embedding)
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = Path(args.config)
    settings = load_settings(config_path if config_path.exists() else None)
    app = create_app(settings)
    url = f"http://{args.host}:{args.port}"
    logger.info("人物库管理页 %s", url)
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
