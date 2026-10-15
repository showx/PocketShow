use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub source: String,
    pub device_index: u32,
    pub file: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub imgsz: u32,
    pub conf: f32,
    pub iou: f32,
    pub deadzone: f64,
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub feedforward: f64,
    pub max_rate: f64,
    pub lost_hold_s: f64,
    pub lost_timeout_s: f64,
    pub target_area: f64,
    pub recognize: bool,
    pub match_threshold: f32,
    pub match_margin: f32,
    pub gimbal: String,
    pub camera_ip: String,
    pub camera_port: u16,
    pub send_hz: f64,
    pub ssid: String,
    pub wifi_password: String,
    pub join_wifi: bool,
    pub ble: bool,
    pub ble_timeout_s: f64,
    pub rtsp_url: String,
    pub rtsp_host: String,
    pub rtsp_port: u16,
    pub rtsp_user: String,
    pub rtsp_password: String,
    pub rtsp_channel: u32,
    pub rtsp_stream: String,
    pub rtsp_transport: String,
    pub camera_id: String,
    pub cameras: Vec<Camera>,
    pub gallery: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub id: String,
    pub name: String,
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub channel: u32,
    pub stream: String,
    pub url: String,
    pub transport: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            source: "usb".into(),
            device_index: 0,
            file: None,
            width: 1280,
            height: 720,
            fps: 30,
            imgsz: 640,
            conf: 0.22,
            iou: 0.5,
            deadzone: 0.08,
            kp: 1.35,
            ki: 0.05,
            kd: 0.18,
            feedforward: 0.35,
            max_rate: 0.55,
            lost_hold_s: 0.8,
            lost_timeout_s: 2.5,
            target_area: 0.12,
            recognize: true,
            match_threshold: 0.42,
            match_margin: 0.06,
            gimbal: "stub".into(),
            camera_ip: "192.168.2.1".into(),
            camera_port: 9004,
            send_hz: 30.0,
            ssid: String::new(),
            wifi_password: String::new(),
            join_wifi: false,
            ble: false,
            ble_timeout_s: 15.0,
            rtsp_url: String::new(),
            rtsp_host: String::new(),
            rtsp_port: 554,
            rtsp_user: "admin".into(),
            rtsp_password: String::new(),
            rtsp_channel: 1,
            rtsp_stream: "sub".into(),
            rtsp_transport: "tcp".into(),
            camera_id: String::new(),
            cameras: Vec::new(),
            gallery: PathBuf::from("data/faces.json"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct FileConfig {
    #[serde(default)]
    capture: CaptureFile,
    #[serde(default)]
    detect: DetectFile,
    #[serde(default)]
    follow: FollowFile,
    #[serde(default)]
    recognize: RecognizeFile,
    #[serde(default)]
    gimbal: GimbalFile,
    #[serde(default)]
    wifi: WifiFile,
    #[serde(default)]
    rtsp: RtspFile,
}

#[derive(Debug, Default, Deserialize)]
struct CaptureFile {
    source: Option<String>,
    device_index: Option<u32>,
    file: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct DetectFile {
    imgsz: Option<u32>,
    conf: Option<f32>,
    iou: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct FollowFile {
    deadzone: Option<f64>,
    kp: Option<f64>,
    ki: Option<f64>,
    kd: Option<f64>,
    feedforward: Option<f64>,
    max_rate: Option<f64>,
    lost_hold_s: Option<f64>,
    lost_timeout_s: Option<f64>,
    target_area: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct RecognizeFile {
    enabled: Option<bool>,
    match_threshold: Option<f32>,
    match_margin: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct GimbalFile {
    backend: Option<String>,
    send_hz: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct WifiFile {
    camera_ip: Option<String>,
    camera_port: Option<u16>,
    ssid: Option<String>,
    password: Option<String>,
    join: Option<bool>,
    ble: Option<bool>,
    ble_timeout_s: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct RtspFile {
    url: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    username: Option<String>,
    password: Option<String>,
    channel: Option<u32>,
    stream: Option<String>,
    transport: Option<String>,
}

impl AppConfig {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut cfg = Self::default();
        if let Some(path) = path.filter(|path| path.is_file()) {
            cfg.apply_yaml(path)?;
        }
        let capture = Path::new("data/capture.json");
        if capture.is_file() {
            cfg.apply_capture(capture)?;
        }
        if cfg.rtsp_password.is_empty() {
            if let Ok(value) = std::env::var("POCKETSHOW_RTSP_PASSWORD") {
                cfg.rtsp_password = value;
            }
        }
        if cfg.source == "rtsp" {
            let id = if cfg.cameras.iter().any(|cam| cam.id == cfg.camera_id) {
                cfg.camera_id.clone()
            } else {
                cfg.cameras.first().map(|cam| cam.id.clone()).unwrap_or_default()
            };
            if !id.is_empty() {
                cfg.use_camera(&id);
            }
        }
        Ok(cfg)
    }

    pub fn use_camera(&mut self, id: &str) -> bool {
        let Some(cam) = self.cameras.iter().find(|cam| cam.id == id).cloned() else {
            return false;
        };
        self.camera_id = cam.id;
        self.source = "rtsp".into();
        if !cam.host.is_empty() {
            self.rtsp_host = cam.host;
        }
        if cam.port > 0 {
            self.rtsp_port = cam.port;
        }
        if !cam.user.is_empty() {
            self.rtsp_user = cam.user;
        }
        if !cam.password.is_empty() {
            self.rtsp_password = cam.password;
        }
        self.rtsp_channel = cam.channel.max(1);
        if !cam.stream.is_empty() {
            self.rtsp_stream = cam.stream;
        }
        self.rtsp_url = cam.url;
        if !cam.transport.is_empty() {
            self.rtsp_transport = cam.transport;
        }
        true
    }

    pub fn active_name(&self) -> String {
        self.cameras
            .iter()
            .find(|cam| cam.id == self.camera_id)
            .map(|cam| if cam.name.is_empty() { cam.id.clone() } else { cam.name.clone() })
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| {
                if self.rtsp_host.is_empty() {
                    self.source.clone()
                } else {
                    format!("通道 {}", self.rtsp_channel)
                }
            })
    }

    fn apply_yaml(&mut self, path: &Path) -> Result<()> {
        let cfg = self;
        let text = fs::read_to_string(path).with_context(|| format!("读取 {}", path.display()))?;
        let file: FileConfig = serde_yaml::from_str(&text).with_context(|| format!("解析 {}", path.display()))?;
        if let Some(value) = file.capture.source {
            cfg.source = value;
        }
        if let Some(value) = file.capture.device_index {
            cfg.device_index = value;
        }
        if file.capture.file.as_deref().is_some_and(|value| !value.is_empty() && value != "null") {
            cfg.file = file.capture.file;
        }
        if let Some(value) = file.capture.width {
            cfg.width = value;
        }
        if let Some(value) = file.capture.height {
            cfg.height = value;
        }
        if let Some(value) = file.capture.fps {
            cfg.fps = value;
        }
        if let Some(value) = file.detect.imgsz {
            cfg.imgsz = value.clamp(320, 1280);
        }
        if let Some(value) = file.detect.conf {
            cfg.conf = value;
        }
        if let Some(value) = file.detect.iou {
            cfg.iou = value;
        }
        if let Some(value) = file.follow.deadzone {
            cfg.deadzone = value;
        }
        if let Some(value) = file.follow.kp {
            cfg.kp = value;
        }
        if let Some(value) = file.follow.ki {
            cfg.ki = value;
        }
        if let Some(value) = file.follow.kd {
            cfg.kd = value;
        }
        if let Some(value) = file.follow.feedforward {
            cfg.feedforward = value;
        }
        if let Some(value) = file.follow.max_rate {
            cfg.max_rate = value;
        }
        if let Some(value) = file.follow.lost_hold_s {
            cfg.lost_hold_s = value;
        }
        if let Some(value) = file.follow.lost_timeout_s {
            cfg.lost_timeout_s = value;
        }
        if let Some(value) = file.follow.target_area {
            cfg.target_area = value;
        }
        if let Some(value) = file.recognize.enabled {
            cfg.recognize = value;
        }
        if let Some(value) = file.recognize.match_threshold {
            cfg.match_threshold = value;
        }
        if let Some(value) = file.recognize.match_margin {
            cfg.match_margin = value;
        }
        if let Some(value) = file.gimbal.backend {
            cfg.gimbal = value;
        }
        if let Some(value) = file.gimbal.send_hz {
            cfg.send_hz = value;
        }
        if let Some(value) = file.wifi.camera_ip {
            cfg.camera_ip = value;
        }
        if let Some(value) = file.wifi.camera_port {
            cfg.camera_port = value;
        }
        if let Some(value) = file.wifi.ssid {
            cfg.ssid = value;
        }
        if let Some(value) = file.wifi.password {
            cfg.wifi_password = value;
        }
        if let Some(value) = file.wifi.join {
            cfg.join_wifi = value;
        }
        if let Some(value) = file.wifi.ble {
            cfg.ble = value;
        }
        if let Some(value) = file.wifi.ble_timeout_s {
            cfg.ble_timeout_s = value;
        }
        if let Some(value) = file.rtsp.url {
            cfg.rtsp_url = value;
        }
        if let Some(value) = file.rtsp.host {
            cfg.rtsp_host = value;
        }
        if let Some(value) = file.rtsp.port {
            cfg.rtsp_port = value;
        }
        if let Some(value) = file.rtsp.username {
            cfg.rtsp_user = value;
        }
        if let Some(value) = file.rtsp.password {
            cfg.rtsp_password = value;
        }
        if let Some(value) = file.rtsp.channel {
            cfg.rtsp_channel = value;
        }
        if let Some(value) = file.rtsp.stream {
            cfg.rtsp_stream = value;
        }
        if let Some(value) = file.rtsp.transport {
            cfg.rtsp_transport = value;
        }
        Ok(())
    }

    fn apply_capture(&mut self, path: &Path) -> Result<()> {
        let text = fs::read_to_string(path).with_context(|| format!("读取 {}", path.display()))?;
        let file: CaptureJson = serde_json::from_str(&text).with_context(|| format!("解析 {}", path.display()))?;
        if let Some(value) = file.source.filter(|value| !value.is_empty()) {
            self.source = value;
        }
        if let Some(value) = file.camera_id {
            self.camera_id = value;
        }
        if let Some(value) = file.host.filter(|value| !value.is_empty()) {
            self.rtsp_host = value;
        }
        if let Some(value) = file.port {
            self.rtsp_port = value;
        }
        if let Some(value) = file.username.filter(|value| !value.is_empty()) {
            self.rtsp_user = value;
        }
        if let Some(value) = file.password.filter(|value| !value.is_empty()) {
            self.rtsp_password = value;
        }
        if let Some(value) = file.channel {
            self.rtsp_channel = value;
        }
        if let Some(value) = file.stream.filter(|value| !value.is_empty()) {
            self.rtsp_stream = value;
        }
        if let Some(value) = file.url {
            self.rtsp_url = value;
        }
        if let Some(value) = file.transport.filter(|value| !value.is_empty()) {
            self.rtsp_transport = value;
        }
        let shared_password = self.rtsp_password.clone();
        let shared_user = self.rtsp_user.clone();
        let shared_host = self.rtsp_host.clone();
        if let Some(cameras) = file.cameras {
            self.cameras = cameras
                .into_iter()
                .enumerate()
                .map(|(index, cam)| {
                    let host = cam.host.filter(|value| !value.is_empty()).unwrap_or_else(|| shared_host.clone());
                    let user = cam.username.filter(|value| !value.is_empty()).unwrap_or_else(|| shared_user.clone());
                    let password = cam
                        .password
                        .filter(|value| !value.is_empty())
                        .unwrap_or_else(|| shared_password.clone());
                    Camera {
                        id: cam.id.filter(|value| !value.is_empty()).unwrap_or_else(|| format!("cam{}", index + 1)),
                        name: cam.name.unwrap_or_default(),
                        host,
                        port: cam.port.unwrap_or(self.rtsp_port),
                        user,
                        password,
                        channel: cam.channel.unwrap_or(1),
                        stream: cam.stream.filter(|value| !value.is_empty()).unwrap_or_else(|| self.rtsp_stream.clone()),
                        url: cam.url.unwrap_or_default(),
                        transport: cam
                            .transport
                            .filter(|value| !value.is_empty())
                            .unwrap_or_else(|| self.rtsp_transport.clone()),
                    }
                })
                .collect();
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct CaptureJson {
    source: Option<String>,
    camera_id: Option<String>,
    cameras: Option<Vec<CameraJson>>,
    host: Option<String>,
    port: Option<u16>,
    username: Option<String>,
    password: Option<String>,
    channel: Option<u32>,
    stream: Option<String>,
    url: Option<String>,
    transport: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CameraJson {
    id: Option<String>,
    name: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    username: Option<String>,
    password: Option<String>,
    channel: Option<u32>,
    stream: Option<String>,
    url: Option<String>,
    transport: Option<String>,
}

pub fn rtsp_url(cfg: &AppConfig) -> Result<String> {
    if !cfg.rtsp_url.trim().is_empty() {
        return Ok(cfg.rtsp_url.trim().to_string());
    }
    if cfg.rtsp_host.trim().is_empty() {
        return Err(anyhow::anyhow!("RTSP 还没填摄像机地址。用 --rtsp-host，或在配置里写 rtsp.host / rtsp.url"));
    }
    if cfg.rtsp_password.is_empty() {
        return Err(anyhow::anyhow!("缺少 RTSP 密码。写到 rtsp.password，或设置 POCKETSHOW_RTSP_PASSWORD"));
    }
    let code = match cfg.rtsp_stream.as_str() {
        "main" => 1,
        "third" => 3,
        _ => 2,
    };
    let stream_id = cfg.rtsp_channel.max(1) * 100 + code;
    Ok(format!(
        "rtsp://{}:{}@{}:{}/Streaming/Channels/{stream_id}",
        url_encode(&cfg.rtsp_user),
        url_encode(&cfg.rtsp_password),
        cfg.rtsp_host.trim(),
        cfg.rtsp_port
    ))
}

fn url_encode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

pub fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".pocketshow").join("models")
}
