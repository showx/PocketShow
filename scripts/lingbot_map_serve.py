#!/usr/bin/env python3
"""LingBot-Map 流式重建 HTTP 服务，给 PocketShow geomap 客户端用。

在 lingbot-map 的 conda/venv 里运行（需要 CUDA + 官方 checkpoint）：

    pip install -e .   # 在 lingbot-map 仓库
    python /path/to/PocketShow/scripts/lingbot_map_serve.py \\
        --model_path /path/to/lingbot-map.pt --port 8090
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

logger = logging.getLogger("lingbot_map_serve")


def _load_model(args, device):
    import torch
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        enable_depth=True,
        enable_point=True,
    )
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _preprocess(rgb: np.ndarray, image_size: int, patch_size: int):
    import torch
    from PIL import Image

    img = Image.fromarray(rgb)
    width, height = img.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    crop_y = 0
    if new_height > image_size:
        crop_y = (new_height - image_size) // 2
        img = img.crop((0, crop_y, new_width, crop_y + image_size))
        new_height = image_size
    tensor = torch.from_numpy(np.asarray(img).astype("float32") / 255.0).permute(2, 0, 1)
    meta = {
        "src_w": width,
        "src_h": height,
        "new_w": new_width,
        "new_h": new_height + crop_y * 2 if crop_y else new_height,
        "crop_y": crop_y,
        "out_w": tensor.shape[2],
        "out_h": tensor.shape[1],
    }
    return tensor, meta


def _map_uv(u: float, v: float, meta: dict) -> tuple[int, int] | None:
    px = float(u) * meta["src_w"] * (meta["new_w"] / max(meta["src_w"], 1))
    py = float(v) * meta["src_h"] * ((meta["new_h"]) / max(meta["src_h"], 1)) - meta["crop_y"]
    x = int(round(px))
    y = int(round(py))
    if x < 0 or y < 0 or x >= meta["out_w"] or y >= meta["out_h"]:
        return None
    return x, y


def _pose_from(output: dict, images, device):
    import torch
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    pose = output["pose_enc"]
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    h, w = images.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, (h, w))
    last_ext = extrinsic[0, -1]
    last_int = intrinsic[0, -1]
    mat = torch.zeros(4, 4, device=last_ext.device, dtype=last_ext.dtype)
    mat[:3, :4] = last_ext
    mat[3, 3] = 1.0
    c2w = closed_form_inverse_se3_general(mat.unsqueeze(0))[0]
    xyz = c2w[:3, 3]
    return (
        c2w[:3, :4].detach().float().cpu().numpy(),
        last_int.detach().float().cpu().numpy(),
        xyz.detach().float().cpu().numpy(),
    )


def _sample_points(output: dict, points: list[dict], meta: dict, extrinsic, intrinsic) -> list[dict]:
    world = output.get("world_points")
    conf = output.get("world_points_conf")
    depth = output.get("depth")
    sampled = []
    for item in points:
        mapped = _map_uv(float(item.get("u") or 0.5), float(item.get("v") or 0.5), meta)
        row = {
            "id": item.get("id") or "",
            "track_id": item.get("track_id"),
            "name": item.get("name") or "",
            "u": item.get("u"),
            "v": item.get("v"),
        }
        if mapped is None:
            sampled.append(row)
            continue
        x, y = mapped
        xyz = None
        score = 0.0
        z = None
        if world is not None:
            tensor = world[0, -1] if world.ndim == 5 else world[-1]
            xyz = tensor[y, x].detach().float().cpu().numpy()
            if conf is not None:
                c = conf[0, -1] if conf.ndim == 4 else conf[-1]
                score = float(c[y, x].detach().cpu())
        elif depth is not None:
            d = depth[0, -1] if depth.ndim == 5 else depth[-1]
            z = float(d[y, x].detach().cpu().reshape(-1)[0])
            fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
            cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
            cam = np.array([(x - cx) * z / max(fx, 1e-6), (y - cy) * z / max(fy, 1e-6), z], dtype=np.float32)
            xyz = extrinsic[:, :3] @ cam + extrinsic[:, 3]
        if xyz is not None:
            row["xyz"] = [round(float(v), 4) for v in xyz.tolist()]
            row["depth"] = round(float(z if z is not None else xyz[2]), 4)
            row["conf"] = round(score, 3)
        sampled.append(row)
    return sampled


class Mapper:
    def __init__(self, args) -> None:
        import torch

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("加载 LingBot-Map：%s (%s)", args.model_path, self.device)
        self.model = _load_model(args, self.device)
        self.lock = threading.Lock()
        self.camera_id = ""
        self.buffer: list = []
        self.frames = 0
        self.scale_done = False

    def reset(self, camera_id: str = "") -> None:
        with self.lock:
            self.model.clean_kv_cache()
            self.camera_id = camera_id
            self.buffer = []
            self.frames = 0
            self.scale_done = False

    def step(self, jpeg: bytes, camera_id: str, points: list[dict]) -> dict:
        import torch
        from PIL import Image

        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        tensor, meta = _preprocess(rgb, self.args.image_size, self.args.patch_size)
        with self.lock:
            if camera_id and self.camera_id and camera_id != self.camera_id:
                self.model.clean_kv_cache()
                self.buffer = []
                self.frames = 0
                self.scale_done = False
            if camera_id:
                self.camera_id = camera_id
            scale_n = max(1, int(self.args.num_scale_frames))
            if not self.scale_done:
                self.buffer.append(tensor)
                if len(self.buffer) < scale_n:
                    return {
                        "ready": False,
                        "warming": True,
                        "buffered": len(self.buffer),
                        "need": scale_n,
                        "frame_index": self.frames,
                        "points": [],
                    }
                images = torch.stack(self.buffer, dim=0).unsqueeze(0).to(self.device)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                    output = self.model.forward(
                        images,
                        num_frame_for_scale=scale_n,
                        num_frame_per_block=scale_n,
                        causal_inference=True,
                    )
                self.buffer = []
                self.scale_done = True
                self.frames = scale_n - 1
            else:
                frame = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                is_keyframe = (self.args.keyframe_interval <= 1) or (
                    (self.frames - scale_n + 1) % self.args.keyframe_interval == 0
                )
                if not is_keyframe:
                    self.model._set_skip_append(True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                    output = self.model.forward(
                        frame,
                        num_frame_for_scale=scale_n,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )
                if not is_keyframe:
                    self.model._set_skip_append(False)
                images = frame
            self.frames += 1 if self.scale_done else 0
            extrinsic, intrinsic, xyz = _pose_from(output, images, self.device)
            sampled = _sample_points(output, points, meta, extrinsic, intrinsic)
            return {
                "ready": True,
                "warming": False,
                "buffered": scale_n,
                "need": scale_n,
                "frame_index": self.frames,
                "keyframe": True,
                "extrinsic": extrinsic.tolist(),
                "intrinsic": intrinsic.tolist(),
                "camera_xyz": [round(float(v), 4) for v in xyz.tolist()],
                "points": sampled,
            }


def make_handler(mapper: Mapper):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] in {"/health", "/v1/map/health"}:
                self._send(200, {"ok": True, "camera_id": mapper.camera_id, "frames": mapper.frames})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json()
                if path == "/v1/map/reset":
                    mapper.reset(str(body.get("camera_id") or ""))
                    self._send(200, {"ok": True})
                    return
                if path == "/v1/map/frame":
                    b64 = body.get("jpeg_b64") or ""
                    jpeg = base64.b64decode(b64)
                    if not jpeg:
                        self._send(400, {"error": "缺少 jpeg_b64"})
                        return
                    result = mapper.step(
                        jpeg,
                        str(body.get("camera_id") or ""),
                        [item for item in (body.get("points") or []) if isinstance(item, dict)],
                    )
                    self._send(200, result)
                    return
            except Exception as exc:
                logger.exception("处理失败")
                self._send(500, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LingBot-Map HTTP 服务（PocketShow 旁路）")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=2)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--camera_num_iterations", type=int, default=1)
    parser.add_argument("--use_sdpa", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    mapper = Mapper(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(mapper))
    logger.info("LingBot-Map 服务 http://%s:%s/v1/map/frame", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
