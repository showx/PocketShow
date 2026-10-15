use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use pocketshow_core::duml::Packet;
use pocketshow_core::session::Events;

use crate::capture::Frame;

pub struct LiveFrames {
    width: u32,
    height: u32,
    latest: Arc<Mutex<Option<Vec<u8>>>>,
    stop: Arc<AtomicBool>,
    child: Child,
    joins: Vec<JoinHandle<()>>,
    tx: Option<SyncSender<Vec<u8>>>,
}

struct H264Events {
    tx: SyncSender<Vec<u8>>,
}

impl Events for H264Events {
    fn on_video(&self, h264: &[u8]) {
        match self.tx.try_send(h264.to_vec()) {
            Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
        }
    }

    fn on_duml(&self, _packets: &[Packet]) {}
}

impl LiveFrames {
    pub fn h264(width: u32, height: u32) -> Result<(Self, Arc<dyn Events>)> {
        let (tx, rx) = sync_channel(24);
        let live = Self::spawn(width, height, h264_cmd(width, height), Some(rx))?;
        let events = Arc::new(H264Events { tx: tx.clone() });
        let mut live = live;
        live.tx = Some(tx);
        Ok((live, events))
    }

    pub fn rtsp(url: &str, width: u32, height: u32, transport: &str) -> Result<Self> {
        Self::spawn(width, height, rtsp_cmd(url, width, height, transport), None)
    }

    pub fn wait_frame(&self) -> Result<Frame> {
        let start = Instant::now();
        loop {
            if let Some(rgb) = self.latest.lock().expect("画面锁").clone() {
                return Ok(Frame {
                    width: self.width,
                    height: self.height,
                    rgb,
                });
            }
            if start.elapsed() > Duration::from_secs(8) {
                return Err(anyhow!("8 秒内没有画面。检查视频源、ffmpeg，以及相机是不是已经在推流"));
            }
            thread::sleep(Duration::from_millis(15));
        }
    }

    fn spawn(width: u32, height: u32, cmd: Vec<String>, rx: Option<Receiver<Vec<u8>>>) -> Result<Self> {
        let mut command = Command::new("ffmpeg");
        command.args(&cmd).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = command.spawn().context("启动 ffmpeg 失败。WiFi 视频和 RTSP 都要本机有 ffmpeg")?;
        let stdout = child.stdout.take().context("ffmpeg 没有输出")?;
        let stdin = child.stdin.take();
        let latest = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let mut joins = Vec::new();
        if let Some(rx) = rx {
            let stop_write = Arc::clone(&stop);
            let stdin = stdin.context("ffmpeg 没有输入")?;
            joins.push(
                thread::Builder::new()
                    .name("h264-in".into())
                    .spawn(move || write_h264(rx, stdin, stop_write))
                    .expect("h264-in"),
            );
        }
        let stop_read = Arc::clone(&stop);
        let latest_read = Arc::clone(&latest);
        let frame_bytes = (width * height * 3) as usize;
        joins.push(
            thread::Builder::new()
                .name("frame-out".into())
                .spawn(move || read_rgb(stdout, frame_bytes, latest_read, stop_read))
                .expect("frame-out"),
        );
        Ok(Self {
            width,
            height,
            latest,
            stop,
            child,
            joins,
            tx: None,
        })
    }
}

impl Drop for LiveFrames {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.tx.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        for handle in self.joins.drain(..) {
            let _ = handle.join();
        }
    }
}

fn write_h264(rx: Receiver<Vec<u8>>, mut stdin: impl Write, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        let chunk = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(chunk) => chunk,
            Err(_) => continue,
        };
        if stdin.write_all(&chunk).is_err() {
            break;
        }
    }
}

fn read_rgb(mut stdout: impl Read, frame_bytes: usize, latest: Arc<Mutex<Option<Vec<u8>>>>, stop: Arc<AtomicBool>) {
    let mut buf = vec![0u8; frame_bytes];
    while !stop.load(Ordering::Relaxed) {
        if stdout.read_exact(&mut buf).is_err() {
            break;
        }
        *latest.lock().expect("画面锁") = Some(buf.clone());
    }
}

fn h264_cmd(width: u32, height: u32) -> Vec<String> {
    vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-fflags".into(),
        "nobuffer".into(),
        "-flags".into(),
        "low_delay".into(),
        "-f".into(),
        "h264".into(),
        "-i".into(),
        "pipe:0".into(),
        "-an".into(),
        "-sn".into(),
        "-vf".into(),
        format!("scale={width}:{height}"),
        "-pix_fmt".into(),
        "rgb24".into(),
        "-f".into(),
        "rawvideo".into(),
        "pipe:1".into(),
    ]
}

fn rtsp_cmd(url: &str, width: u32, height: u32, transport: &str) -> Vec<String> {
    vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-rtsp_transport".into(),
        transport.into(),
        "-fflags".into(),
        "+genpts+discardcorrupt".into(),
        "-flags".into(),
        "low_delay".into(),
        "-i".into(),
        url.into(),
        "-an".into(),
        "-sn".into(),
        "-vf".into(),
        format!("scale={width}:{height}"),
        "-pix_fmt".into(),
        "rgb24".into(),
        "-f".into(),
        "rawvideo".into(),
        "pipe:1".into(),
    ]
}
