use std::io::Read;
use std::process::{Child, Command, Stdio};

use anyhow::{anyhow, Context, Result};

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

pub enum Source {
    Test { tick: u64, width: u32, height: u32 },
    Still(Frame),
    Live(super::video::LiveFrames),
    Usb(nokhwa::Camera),
    File(FfmpegFile),
}

pub struct FfmpegFile {
    child: Child,
    width: u32,
    height: u32,
    path: String,
}

impl Source {
    pub fn open(kind: &str, device: u32, file: Option<&str>, width: u32, height: u32, fps: u32) -> Result<Self> {
        match kind {
            "test" => Ok(Self::Test { tick: 0, width, height }),
            "file" => {
                let path = file.ok_or_else(|| anyhow!("--source file 需要 --file"))?;
                if is_image(path) {
                    Ok(Self::Still(load_image(path, width, height)?))
                } else {
                    Ok(Self::File(FfmpegFile::open(path, width, height)?))
                }
            }
            _ => {
                nokhwa::nokhwa_initialize(|granted| {
                    if !granted {
                        eprintln!("摄像头权限未授予");
                    }
                });
                let requested = nokhwa::utils::RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
                    nokhwa::utils::RequestedFormatType::Closest(nokhwa::utils::CameraFormat::new(
                        nokhwa::utils::Resolution::new(width, height),
                        nokhwa::utils::FrameFormat::MJPEG,
                        fps,
                    )),
                );
                let mut camera = nokhwa::Camera::new(nokhwa::utils::CameraIndex::Index(device), requested)
                    .context("打开摄像头失败。到系统设置里允许终端使用摄像头。")?;
                camera.open_stream().context("摄像头开流失败")?;
                Ok(Self::Usb(camera))
            }
        }
    }

    pub fn read(&mut self) -> Result<Frame> {
        match self {
            Self::Test { tick, width, height } => {
                *tick += 1;
                Ok(test_frame(*width, *height, *tick))
            }
            Self::Still(frame) => Ok(Frame {
                width: frame.width,
                height: frame.height,
                rgb: frame.rgb.clone(),
            }),
            Self::Live(live) => live.wait_frame(),
            Self::Usb(camera) => {
                let buffer = camera.frame().context("读取摄像头失败")?;
                let image = buffer.decode_image::<nokhwa::pixel_format::RgbFormat>()?;
                Ok(Frame {
                    width: image.width(),
                    height: image.height(),
                    rgb: image.into_raw(),
                })
            }
            Self::File(file) => file.read(),
        }
    }
}

fn is_image(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [".jpg", ".jpeg", ".png", ".webp"]
        .iter()
        .any(|suffix| lower.ends_with(suffix))
}

fn load_image(path: &str, width: u32, height: u32) -> Result<Frame> {
    let image = image::open(path).with_context(|| format!("打不开图片 {path}"))?;
    let image = image.resize_exact(width, height, image::imageops::FilterType::Triangle);
    let rgb = image.to_rgb8();
    Ok(Frame {
        width: rgb.width(),
        height: rgb.height(),
        rgb: rgb.into_raw(),
    })
}

fn test_frame(width: u32, height: u32, tick: u64) -> Frame {
    let mut rgb = vec![28u8; (width * height * 3) as usize];
    let cx = (width as f32 * (0.5 + 0.25 * (tick as f32 / 40.0).sin())) as i32;
    let cy = height as i32 / 2;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let dx = (x - cx) as f32 / 40.0;
            let dy = (y - cy) as f32 / 90.0;
            if dx * dx + dy * dy < 1.0 {
                let pixel = ((y as u32 * width + x as u32) * 3) as usize;
                rgb[pixel] = 220;
                rgb[pixel + 1] = 190;
                rgb[pixel + 2] = 160;
            }
        }
    }
    Frame { width, height, rgb }
}

impl FfmpegFile {
    fn open(path: &str, width: u32, height: u32) -> Result<Self> {
        let child = Command::new("ffmpeg")
            .args([
                "-loglevel",
                "error",
                "-stream_loop",
                "-1",
                "-i",
                path,
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vf",
                &format!("scale={width}:{height}"),
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("启动 ffmpeg 失败")?;
        Ok(Self {
            child,
            width,
            height,
            path: path.to_string(),
        })
    }

    fn read(&mut self) -> Result<Frame> {
        let bytes = (self.width * self.height * 3) as usize;
        let mut rgb = vec![0u8; bytes];
        let stdout = self.child.stdout.as_mut().context("ffmpeg 没有输出")?;
        stdout.read_exact(&mut rgb).context("文件读完或 ffmpeg 中断")?;
        Ok(Frame {
            width: self.width,
            height: self.height,
            rgb,
        })
    }
}

impl Drop for FfmpegFile {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.path;
    }
}
