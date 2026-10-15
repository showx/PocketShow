mod capture;
mod config;
mod console;
mod draw;
mod face;
mod gallery;
mod link;
mod lock;
mod reid;
mod track;
mod video;
mod watch;
mod yolo;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use pocketshow_core::duml::CMD_TYPE_REQ;
use pocketshow_core::follow::{Controller, Gains, Target};
use pocketshow_core::gimbal::{recenter_payload, CMD_ABS};
use pocketshow_core::session::{NopEvents, Session, DEV_APP, DEV_GIMBAL};

use crate::capture::Source;
use crate::config::{model_dir, AppConfig};
use crate::face::Faces;
use crate::gallery::Gallery;
use crate::lock::TargetLock;
use crate::track::Tracker;
use crate::yolo::Yolo;

struct Cli {
    config: Option<std::path::PathBuf>,
    source: Option<String>,
    file: Option<String>,
    gimbal: Option<String>,
    device: Option<u32>,
    frames: Option<u64>,
    ssid: Option<String>,
    password: Option<String>,
    join_wifi: bool,
    ble: bool,
    rtsp_host: Option<String>,
    rtsp_channel: Option<u32>,
    rtsp_stream: Option<String>,
    rtsp_camera: Option<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("PocketShow 退出：{err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = parse_cli()?;
    let mut cfg = AppConfig::load(cli.config.as_deref())?;
    if let Some(source) = cli.source {
        cfg.source = source;
    }
    if let Some(file) = cli.file {
        cfg.file = Some(file);
    }
    if let Some(gimbal) = cli.gimbal {
        cfg.gimbal = gimbal;
    }
    if let Some(device) = cli.device {
        cfg.device_index = device;
    }
    if let Some(ssid) = cli.ssid {
        cfg.ssid = ssid;
    }
    if let Some(password) = cli.password {
        cfg.wifi_password = password;
    }
    if cli.join_wifi {
        cfg.join_wifi = true;
    }
    if cli.ble {
        cfg.ble = true;
    }
    if let Some(host) = cli.rtsp_host {
        cfg.rtsp_host = host;
        if cfg.source == "usb" || cfg.source == "auto" {
            cfg.source = "rtsp".into();
        }
    }
    if let Some(channel) = cli.rtsp_channel {
        cfg.rtsp_channel = channel;
    }
    if let Some(stream) = cli.rtsp_stream {
        cfg.rtsp_stream = stream;
    }
    if let Some(camera) = &cli.rtsp_camera {
        cfg.use_camera(camera);
    }
    if cfg.source == "camera" || cfg.source == "auto" {
        cfg.source = "usb".into();
    }
    if cfg.ble {
        link::wake_wifi(cfg.ble_timeout_s)?;
        std::thread::sleep(std::time::Duration::from_secs(18));
    }
    if cfg.join_wifi {
        link::join_wifi(&cfg.ssid, &cfg.wifi_password, &cfg.camera_ip)?;
        link::wait_camera(&cfg.camera_ip)?;
    }

    let hub = console::Hub::new();
    if cli.frames.is_none() {
        let cfg_worker = cfg.clone();
        let hub_worker = hub.clone();
        std::thread::spawn(move || {
            if let Err(err) = drive(cfg_worker, hub_worker.clone(), None) {
                hub_worker.fail(format!("{err:#}"));
            } else {
                hub_worker.finish();
            }
        });
        return hub.desktop();
    }
    drive(cfg, hub, cli.frames)
}

fn drive(cfg: AppConfig, hub: console::Hub, frames: Option<u64>) -> Result<()> {
    let models = model_dir();
    let yolo_path = models.join("yolo11n.onnx");
    yolo::ensure_model(&yolo_path)?;
    let yunet = models.join("face_detection_yunet_2023mar.onnx");
    let sface = models.join("face_recognition_sface_2021dec.onnx");
    if cfg.recognize && (!yunet.is_file() || !sface.is_file()) {
        return Err(anyhow!(
            "缺少人脸模型 {} 或 {}",
            yunet.display(),
            sface.display()
        ));
    }

    let ort = ort::init()
        .with_name("pocketshow")
        .build()
        .context("初始化 ONNX Runtime 失败")?;
    let mut detector = Yolo::load(&ort, &yolo_path, cfg.imgsz, cfg.conf, cfg.iou)?;
    let mut faces = if cfg.recognize {
        Some(Faces::load(&ort, &yunet, &sface)?)
    } else {
        None
    };
    let mut bodies = if cfg.recognize {
        reid::Body::load(&ort, &models.join("person-reidentification-retail-0288.onnx")).ok()
    } else {
        None
    };
    let mut gallery = Gallery::load(&cfg.gallery)?;
    let away_log = cfg.gallery.parent().unwrap_or(std::path::Path::new("data")).join("away.jsonl");
    let mut attendance = watch::Watch::open(&away_log);
    if let Some(bodies) = bodies.as_mut() {
        bodies.seed(&mut gallery)?;
    }
    let mut tracker = Tracker::new();
    let mut target_lock = TargetLock::new(cfg.lost_timeout_s);
    let mut follow = Controller::new(Gains {
        deadzone: cfg.deadzone,
        kp: cfg.kp,
        ki: cfg.ki,
        kd: cfg.kd,
        feedforward: cfg.feedforward,
        max_rate: cfg.max_rate,
        lost_hold_s: cfg.lost_hold_s,
        lost_timeout_s: cfg.lost_timeout_s,
        target_area: cfg.target_area,
    });

    let mut cfg = cfg;
    let mut paused = false;
    let mut shown = 0u64;
    loop {
        if hub.quit() {
            break;
        }
        let cameras = camera_snaps(&cfg);
        hub.show_cameras(cameras, cfg.camera_id.clone(), cfg.active_name());
        hub.note(format!("正在连接 {}", cfg.active_name()));
        let (session, mut source) = match open_source(&cfg) {
            Ok(opened) => opened,
            Err(err) => {
                if wait_camera(&hub, &mut cfg, &err)? {
                    continue;
                }
                break;
            }
        };
        let mut frame = match source.read() {
            Ok(frame) => frame,
            Err(err) => {
                drop(source);
                if let Some(session) = &session {
                    session.shutdown();
                }
                if wait_camera(&hub, &mut cfg, &err)? {
                    continue;
                }
                break;
            }
        };
        let mut previous = Instant::now();
        let mut next_camera = None;
        loop {
        if hub.quit() {
            break;
        }
        if shown > 0 {
            frame = source.read()?;
        }
        shown += 1;
        let dt = previous.elapsed().as_secs_f64().clamp(1e-3, 0.2);
        previous = Instant::now();

        let seats = gallery.search_windows(&cfg.camera_id);
        let mut people = tracker.update(&detector.detect(&frame.rgb, frame.width, frame.height, &seats)?);
        if let Some(recognizer) = faces.as_mut() {
            recognizer.annotate(&frame.rgb, frame.width, frame.height, &mut people)?;
        }
        if let Some(bodies) = bodies.as_mut() {
            bodies.annotate(&frame.rgb, frame.width, frame.height, &mut people, shown)?;
        }
        if cfg.recognize {
            gallery.apply(
                &mut people,
                &cfg.camera_id,
                frame.width,
                frame.height,
                cfg.match_threshold,
                cfg.match_margin,
            );
        }
        for order in hub.drain() {
            match order {
                console::Order::LockAt(x, y) => {
                    target_lock.lock_at(x, y, &people);
                }
                console::Order::LockId(id) => {
                    if let Some(person) = people.iter().find(|person| person.id == id) {
                        target_lock.lock(person);
                    }
                }
                console::Order::Cycle(step) => {
                    target_lock.cycle(&people, step);
                }
                console::Order::Clear => target_lock.clear(),
                console::Order::Pause => {
                    paused = !paused;
                    follow.reset();
                    if let Some(session) = &session {
                        session.set_gimbal_velocity(0.0, 0.0);
                    }
                }
                console::Order::Recenter => {
                    follow.reset();
                    if let Some(session) = &session {
                        session.set_gimbal_velocity(0.0, 0.0);
                        session.send_duml(DEV_APP, 0, DEV_GIMBAL, 0, 4, CMD_ABS, &recenter_payload(), CMD_TYPE_REQ);
                    }
                }
                console::Order::Rename(id, name) => {
                    gallery.rename(&id, &name);
                }
                console::Order::Delete(id) => {
                    gallery.remove(&id);
                }
                console::Order::Camera(id) => {
                    if id != cfg.camera_id {
                        next_camera = Some(id);
                    }
                }
                console::Order::Pin { track, person_id, name } => {
                    if let Some(person) = people.iter().find(|person| person.id == track).cloned() {
                        let width = frame.width.max(1) as f32;
                        let height = frame.height.max(1) as f32;
                        let cx = (person.x1 + person.x2) * 0.5 / width;
                        let cy = (person.y1 + person.y2) * 0.5 / height;
                        let rx = (person.x2 - person.x1) / width * 0.5;
                        let ry = (person.y2 - person.y1) / height * 0.5;
                        match gallery.pin(
                            &cfg.camera_id,
                            &cfg.active_name(),
                            &person_id,
                            &name,
                            cx,
                            cy,
                            rx,
                            ry,
                            person.embedding.clone(),
                            person.body.clone(),
                        ) {
                            Ok((id, saved)) => {
                                let _ = gallery.save_cover(&id, &frame.rgb, frame.width, frame.height, person.x1, person.y1, person.x2, person.y2);
                                if let Some(shown) = people.iter_mut().find(|item| item.id == track) {
                                    shown.person_id = Some(id);
                                    shown.name = Some(saved);
                                }
                            }
                            Err(err) => hub.note(err.to_string()),
                        }
                    }
                }
            }
        }
        if next_camera.is_some() {
            break;
        }
        let locked = target_lock.update(&people, dt);
        let command = if paused {
            follow.reset();
            if let Some(session) = &session {
                session.set_gimbal_velocity(0.0, 0.0);
            }
            pocketshow_core::follow::Command {
                yaw_rate: 0.0,
                pitch_rate: 0.0,
                lost: true,
                ex: 0.0,
                ey: 0.0,
                size_ratio: 0.0,
                target_id: locked.map(|person| person.id),
            }
        } else if let Some((yaw, pitch)) = hub.stick() {
            follow.reset();
            if let Some(session) = &session {
                session.set_gimbal_velocity(yaw, pitch);
            }
            pocketshow_core::follow::Command {
                yaw_rate: yaw,
                pitch_rate: pitch,
                lost: false,
                ex: 0.0,
                ey: 0.0,
                size_ratio: 0.0,
                target_id: locked.map(|person| person.id),
            }
        } else {
            let target = locked.map(|person| Target {
                id: person.id,
                x1: f64::from(person.x1),
                y1: f64::from(person.y1),
                x2: f64::from(person.x2),
                y2: f64::from(person.y2),
                vx: f64::from(person.vx),
                vy: f64::from(person.vy),
            });
            let command = follow.update(target, frame.width as i32, frame.height as i32, dt, None);
            if let Some(session) = &session {
                session.set_gimbal_velocity(command.yaw_rate, command.pitch_rate);
            }
            command
        };

        let present: Vec<String> = people.iter().filter_map(|person| person.person_id.clone()).collect();
        let away = attendance.tick(&present, &gallery.seated(&cfg.camera_id), dt as f32, watch::Watch::on_duty_now());
        let pixels = draw::frame_buffer(&frame.rgb, frame.width, frame.height, &people, target_lock.locked_id());
        let label = hud(&people, target_lock.locked_id(), command.yaw_rate, command.pitch_rate, paused, gallery.len());
        hub.publish(
            &pixels,
            frame.width,
            frame.height,
            console::View {
                title: label.clone(),
                source: cfg.active_name(),
                camera_id: cfg.camera_id.clone(),
                cameras: camera_snaps(&cfg),
                paused,
                width: frame.width,
                height: frame.height,
                yaw: command.yaw_rate,
                pitch: command.pitch_rate,
                people: people
                    .iter()
                    .map(|person| console::PersonSnap {
                        id: person.id,
                        name: person.name.clone().unwrap_or_else(|| format!("#{}", person.id)),
                        locked: Some(person.id) == target_lock.locked_id(),
                        face: person.face.is_some(),
                    })
                    .collect(),
                gallery: gallery
                    .cards()
                    .into_iter()
                    .map(|(id, name, templates)| console::CardSnap { id, name, templates })
                    .collect(),
                away: away
                    .into_iter()
                    .map(|row| console::AwaySnap {
                        name: row.name,
                        detail: row.detail,
                        alarm: row.alarm,
                    })
                    .collect(),
                message: String::new(),
            },
        );
        if frames.is_some() {
            let names: Vec<String> = people
                .iter()
                .map(|person| person.name.clone().unwrap_or_else(|| format!("#{}", person.id)))
                .collect();
            let faces = people.iter().filter(|person| person.face.is_some()).count();
            eprintln!("帧 {shown} 检出 {} 人 {}，其中 {faces} 张脸", people.len(), names.join(" "));
        }
        if frames.is_some_and(|limit| shown >= limit) || next_camera.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        }
        if let Some(session) = &session {
            session.stop_gimbal();
            session.shutdown();
        }
        drop(source);
        let Some(id) = next_camera else {
            break;
        };
        if frames.is_some() {
            break;
        }
        cfg.use_camera(&id);
        detector.reset();
        if let Some(bodies) = bodies.as_mut() {
            bodies.clear();
        }
        tracker = Tracker::new();
        target_lock = TargetLock::new(cfg.lost_timeout_s);
        follow.reset();
    }
    Ok(())
}

fn camera_snaps(cfg: &AppConfig) -> Vec<console::CameraSnap> {
    cfg.cameras
        .iter()
        .map(|camera| console::CameraSnap {
            id: camera.id.clone(),
            name: if camera.name.is_empty() { camera.id.clone() } else { camera.name.clone() },
        })
        .collect()
}

fn wait_camera(hub: &console::Hub, cfg: &mut AppConfig, err: &anyhow::Error) -> Result<bool> {
    hub.note(format!("{} 连不上：{err:#}", cfg.active_name()));
    loop {
        if hub.quit() {
            return Ok(false);
        }
        for order in hub.drain() {
            if let console::Order::Camera(id) = order {
                if cfg.use_camera(&id) {
                    return Ok(true);
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn open_source(cfg: &AppConfig) -> Result<(Option<Session>, Source)> {
    if cfg.source == "wifi" {
        let (live, events) = video::LiveFrames::h264(cfg.width, cfg.height)?;
        let session = connect_pocket(cfg, events, cfg.gimbal == "wifi")?;
        session.start_video();
        return Ok((Some(session), Source::Live(live)));
    }
    let session = if cfg.gimbal == "wifi" {
        Some(connect_pocket(cfg, Arc::new(NopEvents), true)?)
    } else {
        None
    };
    let source = if cfg.source == "rtsp" {
        let url = config::rtsp_url(cfg)?;
        Source::Live(video::LiveFrames::rtsp(&url, cfg.width, cfg.height, &cfg.rtsp_transport)?)
    } else {
        Source::open(
            &cfg.source,
            cfg.device_index,
            cfg.file.as_deref(),
            cfg.width,
            cfg.height,
            cfg.fps,
        )?
    };
    Ok((session, source))
}

fn connect_pocket(cfg: &AppConfig, events: Arc<dyn pocketshow_core::session::Events>, gimbal: bool) -> Result<Session> {
    let session = Session::new(&cfg.camera_ip, cfg.camera_port, events).map_err(|err| anyhow!(err))?;
    if !session.connect(4.0) {
        return Err(anyhow!("连不上相机 {}：{}", cfg.camera_ip, session.last_error()));
    }
    if !session.start() {
        return Err(anyhow!("相机会话起不来：{}", session.last_error()));
    }
    if gimbal {
        session.start_gimbal(cfg.send_hz);
    }
    Ok(session)
}

fn hud(people: &[track::Person], locked: Option<i64>, yaw: f64, pitch: f64, paused: bool, gallery: usize) -> String {
    let person = locked.and_then(|id| people.iter().find(|person| person.id == id));
    let who = person
        .and_then(|person| person.name.clone())
        .unwrap_or_else(|| person.map(|person| format!("#{}", person.id)).unwrap_or_else(|| "无人".into()));
    let state = if paused { "停" } else { "跟" };
    format!("PocketShow  {state}  {who}  yaw {yaw:.2}  pitch {pitch:.2}  库 {gallery}")
}

fn parse_cli() -> Result<Cli> {
    let mut cli = Cli {
        config: None,
        source: None,
        file: None,
        gimbal: None,
        device: None,
        frames: None,
        ssid: None,
        password: None,
        join_wifi: false,
        ble: false,
        rtsp_host: None,
        rtsp_channel: None,
        rtsp_stream: None,
        rtsp_camera: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => cli.config = Some(need(&mut args, "--config")?.into()),
            "--source" => cli.source = Some(need(&mut args, "--source")?),
            "--file" => cli.file = Some(need(&mut args, "--file")?),
            "--gimbal" => cli.gimbal = Some(need(&mut args, "--gimbal")?),
            "--device" | "--device-index" => {
                cli.device = Some(need(&mut args, "--device")?.parse().context("--device 要是数字")?);
            }
            "--frames" => {
                cli.frames = Some(need(&mut args, "--frames")?.parse().context("--frames 要是数字")?);
            }
            "--ssid" => cli.ssid = Some(need(&mut args, "--ssid")?),
            "--password" => cli.password = Some(need(&mut args, "--password")?),
            "--join-wifi" => cli.join_wifi = true,
            "--ble" => cli.ble = true,
            "--rtsp-host" => cli.rtsp_host = Some(need(&mut args, "--rtsp-host")?),
            "--rtsp-channel" => {
                cli.rtsp_channel = Some(need(&mut args, "--rtsp-channel")?.parse().context("--rtsp-channel 要是数字")?);
            }
            "--stream" => cli.rtsp_stream = Some(need(&mut args, "--stream")?),
            "--rtsp-camera" => cli.rtsp_camera = Some(need(&mut args, "--rtsp-camera")?),
            "--no-preview" | "-v" | "--verbose" => {}
            "-h" | "--help" => {
                println!(
                    "pocketshow [--config configs/default.yaml] [--source usb|file|test|wifi|rtsp] [--file 图片或视频] [--gimbal stub|wifi] [--device 0] [--ssid SSID] [--password 密码] [--join-wifi] [--ble] [--rtsp-host IP] [--stream main|sub|third]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("不认识的参数 {other}。加 --help 看用法")),
        }
    }
    Ok(cli)
}

fn need(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next().ok_or_else(|| anyhow!("{flag} 后面要跟值"))
}
