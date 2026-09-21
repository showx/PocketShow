# PocketShow

本地闭环的 **DJI Osmo Pocket 3 智能跟拍**：从视频流里检出人、锁住身份，再把画面误差转成云台速度。

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS-000000?logo=apple&logoColor=white)](#环境要求)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#状态)

Pocket 3 没有官方第三方 SDK。PocketShow 把 **USB Webcam 取流**（画质好、延迟低）和 **WiFi DUML 云台控制**拼成一条可替换的管线；检测、识别、决策与传输层解耦。

> **非官方项目，与 DJI 无任何关联。** 云台协议来自社区对 DUML 的逆向，行为可能随固件变化。仅供个人研究与本地使用。

## 能力

- **取流**：USB UVC、相机 WiFi H.264（需 `ffmpeg`）、本地文件回放；`auto` 会优先识别 Pocket 3
- **检测与跟踪**：YOLO11n 只检 `person`，ByteTrack 维持 ID
- **身份**：YuNet 人脸 + SFace 特征；锁定后即使 ByteTrack ID 跳变也会接回同一个人
- **活体**：MiniFASNet，拦纸质照片和屏幕翻拍
- **跟拍**：死区 + PID + 速度前馈；短暂遮挡衰减最后速度，超时才重选目标
- **云台**：`stub` 只在预览 HUD 上画 yaw/pitch；`wifi` 经 UDP 9004 发 DUML
- **人物库**：本地管理页改名、合并重复档、上传底片、指定工位；可选工位离岗提醒
- **场景理解**（可选）：把画面旁路给 [MOSS-VL](https://github.com/OpenMOSS/MOSS-VL) 实时视频模型，HUD / 管理页显示「在干嘛」，不进跟拍
- **三维重建**（可选）：把画面旁路给 [LingBot-Map](https://github.com/Robbyant/lingbot-map)，得到相机位姿和人的 3D 坐标，辅助工位匹配
- **动作捕捉**（可选）：把画面旁路给 [FreeMoCap](https://github.com/freemocap/freemocap) / skellytracker，叠骨架并标坐着 / 站着 / 举手

## 管线

```mermaid
flowchart TD
  cam["Pocket 3 / 局域网相机"] -->|"USB UVC / WiFi H.264 / RTSP"| cap[Capture]
  cap --> det["YOLO11n + ByteTrack"]
  det --> rec["YuNet + SFace + ReID"]
  rec --> lock[Target lock]
  lock --> pid["PID + deadzone + feedforward"]
  pid --> port[GimbalPort]
  port -->|stub HUD / WiFi DUML| cam
  rec -.-> scene["MOSS-VL 实时理解"]
  rec -.-> geomap["LingBot-Map 流式重建"]
  rec -.-> mocap["FreeMoCap 姿态"]
  scene -.-> hud[HUD / 管理页]
  geomap -.-> hud
  mocap -.-> hud
```

`GimbalPort` 只暴露 `set_velocity` / `recenter` / `close`。换传输层不必动检测和决策。MOSS-VL、LingBot-Map 与 FreeMoCap 都是旁路：模型跑在独立环境里，本机只推帧、收文本 / 位姿 / 骨架。

Webcam 模式有可能与 WiFi 控制互斥：若云台无响应，把取流改成 `--source wifi`，决策层不用改。

## 环境要求

| 项 | 说明 |
|----|------|
| Python | 3.11+ |
| 系统 | macOS（Apple Silicon 走 MPS） |
| 相机 | Pocket 3 用数据线开 Webcam，或连相机 WiFi 热点 |
| 可选 | WiFi 视频回退需要本机 `ffmpeg` |

首次运行会下载 `yolo11n.pt`，以及人脸 / 活体模型到 `~/.pocketshow/models/`。macOS 需在 **系统设置 → 隐私与安全性 → 摄像头** 中允许终端或 Python。

## 安装

```bash
git clone https://github.com/showx/PocketShow.git
cd PocketShow
cp configs/default.example.yaml configs/default.yaml
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`configs/default.yaml` 只留在本机（含摄像机地址、密码等），不要提交。仓库里的模板是 [`configs/default.example.yaml`](configs/default.example.yaml)；若本地还没有 `default.yaml`，首次运行会自动复制一份。

## 启动

进入仓库后，**每个新开的终端都要先激活虚拟环境**，否则找不到 `pocketshow` 命令：

```bash
cd PocketShow
source .venv/bin/activate
```

日常用两路：一路网页后台，一路拉流监测。开两个终端，都先 `source .venv/bin/activate`。

**终端 1 — 管理页**

```bash
pocketshow-admin
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。`--no-browser` 可以不自动弹窗。

**终端 2 — 监测 / 跟拍**

局域网海康监控（管理页「离岗分析」监测画面上方点选要监测的通道）：

```bash
pocketshow --source rtsp --gimbal stub
```

笔记本摄像头或 Pocket 3 USB（云台只画在预览上，电机不转）：

```bash
pocketshow --gimbal stub
```

管理页「离岗分析」最上栏会同步桌面窗口那份画布。退出：监测窗口按 `q`，两个终端都可以 `Ctrl+C`。

若提示 `command not found: pocketshow`，多半是没激活 `.venv`，或还没做过上面的安装。

## 快速开始

文件回放：

```bash
pocketshow --source file --file /path/to/clip.mp4 --gimbal stub
```

接到 Pocket 3 云台（先在相机上打开 WiFi AP，或加 `--ble`）：

```bash
pocketshow --gimbal wifi --join-wifi --ssid OsmoPocket3-XXXX --password 'your-password'
```

USB 画面无响应时，改走 WiFi 720p 流：

```bash
pocketshow --source wifi --gimbal wifi --join-wifi --ssid OsmoPocket3-XXXX --password 'your-password'
```

### 预览快捷键

| 按键 | 作用 |
|------|------|
| 鼠标点人物框 | 锁定该人（有姓名后，跟踪 ID 变了也会接回） |
| `n` / `p` | 下一个 / 上一个目标 |
| `e` | 把当前锁定目标的人脸登记进 `data/faces.json` |
| `c` | 清除锁定，回到「画面里最大的人」 |
| `r` | 云台回中 |
| 空格 | 停转 |
| `q` / `Esc` | 退出 |

绿框是锁定目标，白框是其他人，黄框是人脸。自动登记会标成「人物A / 人物B…」，并裁一张脸部相片留底。关闭识别：把配置里 `recognize.enabled` 设为 `false`。

## 人物库与管理页

跟拍时会把人脸封面存到 `data/faces/`。本地打开管理页，可以改名、看底片、上传照片、删除：

```bash
pocketshow-admin
```

默认地址 [http://127.0.0.1:8765](http://127.0.0.1:8765)。跟拍进程不用关；网页里改的名字，预览里过几秒会跟上。

- 同一人被拆成两条时，打开卡片选 **并入** 即可合并
- 在「离岗分析」监测画面上点人框，直接用画面里这个人登记，并记下所在摄像头的位置；点空处仍可指定工位
- 右下角 **镜头** 十字键可手动左右 / 上下转：按住就转，松开就停，键盘方向键同样有效；**跟拍** 交回自动锁定
- `--gimbal stub` 时只在预览 HUD 上看到 yaw/pitch；电机真转需 `--gimbal wifi`

工位看护（`watch`）可在上班时段检测离岗，阈值与工作日在配置或管理页里改。管理页「在岗人员」会列出镜头里认到的人，以及指定了工位但人不在的空位。

## 接入 MOSS-VL、LingBot-Map 与 FreeMoCap

三套模型都不进 PocketShow 的 Python 环境，本机只当客户端。跟拍 PID 不变。

### MOSS-VL 实时场景理解

在 GPU 机器上按 [MOSS-VL 说明](https://github.com/OpenMOSS/MOSS-VL/blob/main/README_zh.md) 起实时服务：

```bash
CUDA_VISIBLE_DEVICES=0 python inference/realtime/run_online_inference.py \
  --serve --host 0.0.0.0 --port 8000
```

多路视频用 [SGLang-Omni](https://github.com/OpenMOSS/MOSS-VL/tree/main/third_party/sglang-omni)，WebSocket 地址改成 `ws://127.0.0.1:18500/v1/video/realtime`。

本机 `configs/default.yaml`：

```yaml
scene:
  enabled: true
  backend: moss-vl
  ws_url: ws://GPU主机:8000/v1/realtime
  model: OpenMOSS-Team/MOSS-VL-Realtime
  sample_fps: 1.0
```

也兼容原来的 OpenAI `chat/completions`（`backend: openai`，例如 Mage-VL / SGLang）。模型可主动沉默：回复「无事」或 `<|silence|>` 时 HUD 不刷。

### LingBot-Map 流式三维重建

在装好 [LingBot-Map](https://github.com/Robbyant/lingbot-map) 的 CUDA 环境里起 HTTP 服务：

```bash
python scripts/lingbot_map_serve.py --model_path /path/to/lingbot-map.pt --port 8090
```

本机：

```yaml
geomap:
  enabled: true
  backend: http
  base_url: http://GPU主机:8090
  camera_id: ""          # 空=第一路；多路填要重建的摄像机 id
```

一个 LingBot-Map 进程只维持一路 KV cache。多路就起多个 `--port`。位姿稳定后，人框会带上 3D 坐标，工位匹配优先用空间距离。自测可把 `backend` 设成 `stub`。

### FreeMoCap 姿态 / 骨架

[FreeMoCap](https://github.com/freemocap/freemocap) 自己的 HTTP 服务绑的是 SkellyCam 相机，不能直接收 PocketShow 的帧。在它的 `uv` 环境里起一层适配：

```bash
# 按官方 README 装好 skellytracker 后
python scripts/freemocap_serve.py --port 8006
# NVIDIA GPU 可改 --tracker rtmpose
```

本机：

```yaml
mocap:
  enabled: true
  backend: http
  base_url: http://127.0.0.1:8006
  camera_id: ""          # 空=所有路；多路只捕这一路 id
```

人框上会叠骨架，HUD / 管理页标「坐着 / 站着 / 举手」。自测可把 `backend` 设成 `stub`。

## 配置

主配置模板见 [`configs/default.example.yaml`](configs/default.example.yaml)，本地改 `configs/default.yaml`。跟拍不是把人死锁在画面中心：`deadzone` 内不推云台；ByteTrack 速度做前馈；短暂遮挡衰减最后速度，超时才重选。

| 段 | 关键项 |
|----|--------|
| `capture` | `source`（`auto` / `usb` / `file` / `wifi` / `rtsp`）、分辨率、帧率 |
| `rtsp` | 海康 NVR / 摄像机列表、码流、本机 `data/capture.json` |
| `detect` | YOLO 权重、`imgsz`（广角建议 1280）、`conf`、远处切片 `far_pass`、设备（`auto` / `mps` / `cpu`） |
| `follow` | `deadzone`、PID、`feedforward`、丢失保持 / 超时 |
| `recognize` | 匹配阈值、自动登记、活体开关、人物库路径 |
| `gimbal` | `stub` 或 `wifi` |
| `wifi` | 相机 IP / 端口、SSID、BLE 唤醒、视频分辨率 |
| `watch` | 上班时段、离岗秒数 |
| `scene` | MOSS-VL / OpenAI 兼容视觉模型；默认关 |
| `geomap` | LingBot-Map HTTP 服务；默认关 |
| `mocap` | FreeMoCap / skellytracker HTTP 服务；默认关 |

命令行覆盖配置，常用参数：

```text
pocketshow [--config PATH] [--source auto|usb|camera|file|wifi|rtsp]
           [--file PATH] [--device-index N]
           [--stream main|sub|third] [--rtsp-host HOST] [--rtsp-channel N]
           [--rtsp-camera ID]
           [--gimbal stub|wifi] [--ssid SSID] [--password PASS]
           [--ble] [--join-wifi] [--no-preview] [-v]
```

## 架构

| 模块 | 职责 |
|------|------|
| `capture` | USB / 文件 / WiFi 帧源 |
| `detect_track` | YOLO11n + ByteTrack，只要 person |
| `recognize` | YuNet 人脸 + SFace 特征，挂到人体框并给出稳定身份 |
| `liveness` | MiniFASNet 活体 |
| `gallery` / `admin` | 人物名册、相片留底、本地管理页 |
| `appear` / `watch` | 工位离岗（入镜流水只作内部去重 / 离岗统计，不再作为管理页） |
| `scene` | MOSS-VL WebSocket / OpenAI 视觉，旁路「在干嘛」 |
| `geomap` | LingBot-Map HTTP 客户端，旁路位姿与 3D 工位 |
| `mocap` | FreeMoCap / skellytracker 客户端，旁路骨架与姿态 |
| `target` | ID / 身份锁定与丢失恢复 |
| `follow` | 画面误差 + PID |
| `gimbal.stub` / `gimbal.wifi` | 可替换云台口 |
| `pocket3` | DUML、UDP 9004、可选 BLE / 加入热点 |

```
src/pocketshow/
├── app.py              # CLI 主循环
├── capture.py
├── detect_track.py
├── recognize.py
├── follow.py
├── target.py
├── scene.py
├── mossvl.py
├── geomap.py
├── mocap.py
├── seats.py
├── admin.py            # pocketshow-admin
├── gimbal/
│   ├── base.py         # GimbalPort
│   ├── stub.py
│   └── wifi.py
└── pocket3/            # DUML / UDP / BLE / 取流
```

## 状态

当前为 **0.1.0 experimental**：接口、协议字段和默认阈值都可能变。已在 macOS + Apple Silicon + Pocket 3 上自测；其他系统未作为目标平台。

已知约束：

- USB Webcam 与 WiFi 云台可能互斥，无响应时改 `--source wifi`
- 人脸库与日志默认写在仓库下的 `data/`，请勿提交个人相片
- 活体与识别都是启发式阈值，强光、侧脸、遮挡会误判；广角里后脑勺/口罩仍然认不出名字，但应先检出人框

## 开发

```bash
pytest
```

欢迎 Issue 与 Pull Request。改跟拍参数请附上场景说明（室内 / 逆光 / 多人）和 `configs/` 里对应项。

## License

尚未选定开源许可证。在补上 `LICENSE` 之前，请勿默认可以二次分发或商用。
