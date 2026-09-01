# PocketShow

Pocket 3 智能跟拍闭环：USB（或 WiFi）取流 → YOLO 人物检测 → ByteTrack ID → 跟拍决策 → 云台速度控制。

```
Pocket 3
   ↓  USB UVC / WiFi H.264
视频流
   ↓
YOLO person + ByteTrack
   ↓
人脸识别（YuNet + SFace）→ 稳定姓名
   ↓
锁定人物，计算画面误差
   ↓
PID 跟拍决策（死区 / 前馈 / 丢失衰减）
   ↓
Stub 或 WiFi DUML 云台
   ↓
Pocket 3
```

Pocket 3 没有官方第三方 SDK。USB Webcam 画质好、延迟低；云台走 WiFi UDP 上的 DUML。Webcam 模式有可能与 WiFi 控制互斥：若云台无响应，把取流改成 `--source wifi`，决策层不用改。

## 环境

- Python 3.11+
- macOS（Apple Silicon 走 MPS）
- Pocket 3 用 **数据线** 开 Webcam，或连相机 WiFi 热点
- WiFi 视频回退需要本机 `ffmpeg`

```bash
cd PocketShow
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

第一次跑会下载 `yolo11n.pt`，以及人脸模型到 `~/.pocketshow/models/`。macOS 需在「系统设置 → 隐私与安全性 → 摄像头」里允许终端/Python。

## 用法

只用笔记本摄像头或 Pocket 3 USB，云台打到预览上（不转电机）：

```bash
pocketshow --config configs/default.yaml --gimbal stub
```

文件回放：

```bash
pocketshow --source file --file /path/to/clip.mp4 --gimbal stub
```

接到 Pocket 3 云台（先在相机上打开 WiFi AP，或加 `--ble`）：

```bash
pocketshow --gimbal wifi --join-wifi --ssid OsmoPocket3-XXXX --password 'your-password'
```

USB 画面无响应、改走 WiFi 720p 流：

```bash
pocketshow --source wifi --gimbal wifi --join-wifi --ssid OsmoPocket3-XXXX --password 'your-password'
```

### 预览操作

- 鼠标点人物框：锁定该人（识别到姓名后，ByteTrack ID 变了也会接回）
- `n` / `p`：切换目标
- `e`：把当前锁定目标的人脸登记进 `data/faces.json`
- `c`：清除锁定，回到「最大的人」自动选
- `r`：云台回中
- 空格：停转
- `q`：退出

画面上绿框是锁定目标，白框是其他人，黄框是人脸。自动登记会标成「人物A / 人物B…」，并裁一张脸部相片留底。关闭识别：把配置里 `recognize.enabled` 设为 `false`。

## 人物库

跟拍时会把人脸封面存到 `data/faces/`。本地打开管理页，可以改名、看底片、上传照片、删除：

```bash
pocketshow-admin
```

默认地址 [http://127.0.0.1:8765](http://127.0.0.1:8765)。跟拍进程不用关；在网页里改的名字，预览里过几秒会跟上。同一人被拆成两条时，打开卡片选「并入」即可合并。点「入镜日志」可看每次入镜、离镜和停留时长。

右下角 **镜头** 十字键可手动左右/上下转：按住就转，松开就停，键盘方向键同样有效。「跟拍」交回自动锁定。当前若是 `--gimbal stub`，只在预览 HUD 上看到 yaw/pitch；要电机真转需 `--gimbal wifi`。

## 配置

见 [`configs/default.yaml`](configs/default.yaml)。跟拍不是把人死锁中心：`deadzone` 内不推云台；ByteTrack 速度做前馈；短暂遮挡衰减最后速度，超时才重选。

## 模块

| 模块 | 作用 |
|------|------|
| `capture` | USB / 文件 / WiFi 帧源 |
| `detect_track` | YOLO11n + ByteTrack，只要 person |
| `recognize` | YuNet 人脸 + SFace 特征，挂到人体框并给出稳定身份 |
| `gallery` / `admin` | 人物名册、相片留底、本地管理页 |
| `target` | ID / 人物身份锁定与丢失恢复 |
| `follow` | 画面误差 + PID |
| `gimbal.stub` / `gimbal.wifi` | 可替换云台口 |
| `pocket3` | DUML、UDP 9004、可选 BLE / 加入热点 |

`GimbalPort` 只有 `set_velocity` / `recenter` / `close`。换传输层不必动检测和决策。
