//! Pocket 3 UDP 会话。
//!
//! 套接字只有发送线程在写；序号都在一把锁里。接收线程遇到 ICMP 不可达
//! （macOS 上 UDP `recv` 会返回 `ConnectionRefused`）只丢这一拍，不把会话拆掉。

use std::fs::File;
use std::io::{ErrorKind, Read};
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::process::Command;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use socket2::{Domain, Socket, Type};

use crate::duml::{self, Packet};
use crate::frame;
use crate::gimbal;

pub const DEV_APP: u8 = 2;
pub const DEV_GIMBAL: u8 = 4;
pub const DEV_DM368: u8 = 8;

const VIDEO_START: [u8; 14] = [
    0x17, 0x00, 0x46, 0x23, 0x73, 0x41, 0x50, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
];

pub trait Events: Send + Sync {
    fn on_video(&self, h264: &[u8]);
    fn on_duml(&self, packets: &[Packet]);
}

pub struct NopEvents;

impl Events for NopEvents {
    fn on_video(&self, _h264: &[u8]) {}
    fn on_duml(&self, _packets: &[Packet]) {}
}

enum Msg {
    Data(Vec<u8>),
    Stop,
}

struct State {
    session_id: u16,
    seed: u16,
    cmd_seq: u16,
    msg_seq: u16,
    video_rx_seq: u16,
    acktelem_rx_seq: u16,
    cmd_rx_seq: u16,
    running: bool,
    started: bool,
    video_hb: bool,
    video_hb_counter: u8,
    hb_tick: u32,
    gimbal_on: bool,
    gimbal_hz: f64,
    gimbal_yaw: f64,
    gimbal_pitch: f64,
}

struct Shared {
    camera: SocketAddr,
    events: Arc<dyn Events>,
    sock: Mutex<Option<Arc<UdpSocket>>>,
    tx: Mutex<Option<Sender<Msg>>>,
    state: Mutex<State>,
    joins: Mutex<Vec<JoinHandle<()>>>,
    last_error: Mutex<String>,
}

pub struct Session {
    shared: Arc<Shared>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

fn random_u16() -> u16 {
    let mut buf = [0u8; 2];
    if let Ok(mut file) = File::open("/dev/urandom") {
        if file.read_exact(&mut buf).is_ok() {
            return u16::from_le_bytes(buf);
        }
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u16)
        .unwrap_or(0x1234)
}

fn camera_iface() -> Option<Ipv4Addr> {
    for iface in ["en0", "en1"] {
        let Ok(output) = Command::new("ipconfig").args(["getifaddr", iface]).output() else {
            continue;
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let ip = text.trim();
        if let Ok(addr) = ip.parse::<Ipv4Addr>() {
            if ip.starts_with("192.168.") {
                return Some(addr);
            }
        }
    }
    None
}

fn local_bind(camera: Ipv4Addr) -> SocketAddr {
    if camera.is_loopback() {
        return SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    }
    if let Some(ip) = camera_iface() {
        return SocketAddr::from((ip, 0));
    }
    SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))
}

fn open_socket(bind_to: SocketAddr, timeout: Duration) -> Result<UdpSocket, String> {
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, None).map_err(|err| err.to_string())?;
    let _ = socket.set_recv_buffer_size(1024 * 1024);
    socket
        .set_read_timeout(Some(timeout))
        .map_err(|err| err.to_string())?;
    socket
        .bind(&bind_to.into())
        .map_err(|err| err.to_string())?;
    Ok(socket.into())
}

impl Session {
    pub fn new(camera_ip: &str, camera_port: u16, events: Arc<dyn Events>) -> Result<Self, String> {
        let ip: Ipv4Addr = camera_ip
            .parse()
            .map_err(|_| format!("无效的相机地址 {camera_ip}"))?;
        let session_id = random_u16();
        Ok(Self {
            shared: Arc::new(Shared {
                camera: SocketAddr::from((ip, camera_port)),
                events,
                sock: Mutex::new(None),
                tx: Mutex::new(None),
                state: Mutex::new(State {
                    session_id,
                    seed: 0,
                    cmd_seq: 0,
                    msg_seq: 1,
                    video_rx_seq: 0,
                    acktelem_rx_seq: 0,
                    cmd_rx_seq: 0,
                    running: false,
                    started: false,
                    video_hb: false,
                    video_hb_counter: 0,
                    hb_tick: 0,
                    gimbal_on: false,
                    gimbal_hz: 30.0,
                    gimbal_yaw: 0.0,
                    gimbal_pitch: 0.0,
                }),
                joins: Mutex::new(Vec::new()),
                last_error: Mutex::new(String::new()),
            }),
        })
    }

    pub fn connect(&self, timeout: f64) -> bool {
        let timeout = Duration::from_secs_f64(timeout.max(0.2));
        let bind_to = local_bind(match self.shared.camera.ip() {
            std::net::IpAddr::V4(ip) => ip,
            _ => Ipv4Addr::UNSPECIFIED,
        });
        let socket = match open_socket(bind_to, timeout) {
            Ok(socket) => socket,
            Err(err) => {
                self.fail(format!("创建套接字失败：{err}"));
                return false;
            }
        };
        let seed = random_u16() & 0xFFF8;
        let session_id = lock(&self.shared.state).session_id;
        let packet = frame::handshake_packet(session_id, seed);
        if let Err(err) = socket.send_to(&packet, self.shared.camera) {
            self.fail(format!("握手发送失败：{err}"));
            return false;
        }
        let mut buf = [0u8; 4096];
        let n = match socket.recv_from(&mut buf) {
            Ok((n, _)) => n,
            Err(err)
                if err.kind() == ErrorKind::TimedOut || err.kind() == ErrorKind::WouldBlock =>
            {
                self.fail("握手超时".to_string());
                return false;
            }
            Err(err) => {
                self.fail(format!("握手失败：{err}"));
                return false;
            }
        };
        let Some(header) = frame::parse_header(&buf[..n]) else {
            self.fail("握手响应异常".to_string());
            return false;
        };
        if header.pkt_type != frame::TYPE_HANDSHAKE {
            self.fail("握手响应异常".to_string());
            return false;
        }
        let _ = socket.set_read_timeout(Some(Duration::from_millis(200)));
        {
            let mut state = lock(&self.shared.state);
            state.session_id = header.session_id;
            state.seed = seed;
            state.video_rx_seq = seed;
            state.acktelem_rx_seq = seed;
            state.cmd_seq = seed;
            state.msg_seq = 1;
            state.cmd_rx_seq = 0;
        }
        *lock(&self.shared.sock) = Some(Arc::new(socket));
        *lock(&self.shared.last_error) = String::new();
        true
    }

    pub fn start(&self) -> bool {
        let sock = {
            let guard = lock(&self.shared.sock);
            let Some(sock) = guard.as_ref() else {
                self.fail("先调用 connect()".to_string());
                return false;
            };
            Arc::clone(sock)
        };
        {
            let mut state = lock(&self.shared.state);
            if state.started {
                return true;
            }
            state.running = true;
            state.started = true;
        }
        let (tx, rx) = mpsc::channel();
        *lock(&self.shared.tx) = Some(tx);
        let mut joins = Vec::new();
        let camera = self.shared.camera;
        joins.push(
            thread::Builder::new()
                .name("pocket3-tx".into())
                .spawn(move || sender_loop(sock, camera, rx))
                .expect("pocket3-tx"),
        );
        let shared = Arc::clone(&self.shared);
        joins.push(
            thread::Builder::new()
                .name("pocket3-rx".into())
                .spawn(move || shared.rx_loop())
                .expect("pocket3-rx"),
        );
        let shared = Arc::clone(&self.shared);
        joins.push(
            thread::Builder::new()
                .name("pocket3-tick".into())
                .spawn(move || shared.timer_loop())
                .expect("pocket3-tick"),
        );
        *lock(&self.shared.joins) = joins;
        true
    }

    pub fn shutdown(&self) {
        self.shared.signal_stop();
        let handles = std::mem::take(&mut *lock(&self.shared.joins));
        for handle in handles {
            let _ = handle.join();
        }
        *lock(&self.shared.sock) = None;
    }

    pub fn session_id(&self) -> u16 {
        lock(&self.shared.state).session_id
    }

    pub fn seed(&self) -> u16 {
        lock(&self.shared.state).seed
    }

    pub fn last_error(&self) -> String {
        lock(&self.shared.last_error).clone()
    }

    pub fn send_duml(
        &self,
        sender_type: u8,
        sender_id: u8,
        receiver_type: u8,
        receiver_id: u8,
        cmd_set: u8,
        cmd_id: u8,
        payload: &[u8],
        cmd_type: u8,
    ) -> u16 {
        self.shared.send_duml(
            sender_type,
            sender_id,
            receiver_type,
            receiver_id,
            cmd_set,
            cmd_id,
            payload,
            cmd_type,
        )
    }

    pub fn start_video(&self) {
        self.shared.send_duml(
            DEV_APP,
            0,
            DEV_DM368,
            1,
            0x00,
            0x88,
            &VIDEO_START,
            duml::CMD_TYPE_PUSH,
        );
        self.shared.send_dm368_register();
        let mut state = lock(&self.shared.state);
        state.video_hb = true;
        state.video_hb_counter = 0;
        state.hb_tick = 0;
    }

    pub fn start_gimbal(&self, hz: f64) {
        let mut state = lock(&self.shared.state);
        state.gimbal_hz = hz.max(5.0);
        state.gimbal_on = true;
    }

    pub fn set_gimbal_velocity(&self, yaw: f64, pitch: f64) {
        let mut state = lock(&self.shared.state);
        state.gimbal_yaw = gimbal::clamp_axis(yaw);
        state.gimbal_pitch = gimbal::clamp_axis(pitch);
    }

    pub fn stop_gimbal(&self) {
        let mut state = lock(&self.shared.state);
        state.gimbal_on = false;
        state.gimbal_yaw = 0.0;
        state.gimbal_pitch = 0.0;
    }

    fn fail(&self, message: String) {
        *lock(&self.shared.last_error) = message;
        *lock(&self.shared.sock) = None;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shared.signal_stop();
    }
}

fn sender_loop(sock: Arc<UdpSocket>, camera: SocketAddr, rx: mpsc::Receiver<Msg>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Stop => break,
            Msg::Data(packet) => {
                let _ = sock.send_to(&packet, camera);
            }
        }
    }
}

impl Shared {
    fn signal_stop(&self) {
        {
            let mut state = lock(&self.state);
            state.running = false;
            state.started = false;
            state.gimbal_on = false;
            state.video_hb = false;
        }
        if let Some(tx) = lock(&self.tx).take() {
            let _ = tx.send(Msg::Stop);
        }
    }

    fn is_running(&self) -> bool {
        lock(&self.state).running
    }

    fn transmit(&self, packet: Vec<u8>) {
        let sender = lock(&self.tx).clone();
        if let Some(sender) = sender {
            match sender.send(Msg::Data(packet)) {
                Ok(()) => return,
                Err(mpsc::SendError(Msg::Data(packet))) => self.direct(packet),
                Err(mpsc::SendError(Msg::Stop)) => {}
            }
            return;
        }
        self.direct(packet);
    }

    fn direct(&self, packet: Vec<u8>) {
        if let Some(sock) = lock(&self.sock).as_ref() {
            let _ = sock.send_to(&packet, self.camera);
        }
    }

    fn send_duml(
        &self,
        sender_type: u8,
        sender_id: u8,
        receiver_type: u8,
        receiver_id: u8,
        cmd_set: u8,
        cmd_id: u8,
        payload: &[u8],
        cmd_type: u8,
    ) -> u16 {
        let (seq, msg_seq, session_id) = {
            let mut state = lock(&self.state);
            let seq = state.cmd_seq;
            state.cmd_seq = state.cmd_seq.wrapping_add(8);
            let msg_seq = state.msg_seq;
            state.msg_seq = state.msg_seq.wrapping_add(1);
            (seq, msg_seq, state.session_id)
        };
        let duml_pkt = duml::build(
            sender_type,
            sender_id,
            receiver_type,
            receiver_id,
            cmd_set,
            cmd_id,
            payload,
            seq,
            cmd_type,
            0,
            1,
        );
        self.transmit(frame::command_packet(session_id, seq, msg_seq, &duml_pkt));
        seq
    }

    fn send_dm368_register(&self) {
        let mut app_payload = vec![0u8; 64];
        app_payload[1] = b'A';
        app_payload[2] = b'P';
        app_payload[3] = b'P';
        self.send_duml(
            DEV_APP,
            0,
            DEV_DM368,
            2,
            0x00,
            0x81,
            &app_payload,
            duml::CMD_TYPE_WRITE,
        );
        self.send_duml(
            DEV_APP,
            0,
            DEV_DM368,
            2,
            0x00,
            0x82,
            &[0x00],
            duml::CMD_TYPE_WRITE,
        );
    }

    fn send_ack(&self) {
        let (session_id, video, telem, cmd) = {
            let state = lock(&self.state);
            (
                state.session_id,
                state.video_rx_seq,
                state.acktelem_rx_seq,
                state.cmd_seq,
            )
        };
        self.transmit(frame::ack_packet(session_id, video, telem, cmd));
    }

    fn send_heartbeat(&self) {
        let (counter, tick) = {
            let mut state = lock(&self.state);
            let tick = state.hb_tick;
            let counter = state.video_hb_counter;
            if tick % 2 == 1 {
                state.video_hb_counter = state.video_hb_counter.wrapping_add(1);
            }
            state.hb_tick = tick.wrapping_add(1);
            (counter, tick)
        };
        let mut hb = [0u8; 10];
        hb[0] = 0x01;
        hb[2] = counter;
        hb[6] = 0xFF;
        hb[7] = 0xFF;
        hb[8] = 0xFF;
        hb[9] = 0xFF;
        self.send_duml(
            DEV_APP,
            0,
            DEV_DM368,
            2,
            0x00,
            0x4F,
            &hb,
            duml::CMD_TYPE_PUSH,
        );
        if tick % 10 == 0 && tick > 0 {
            self.send_dm368_register();
        }
    }

    fn send_gimbal(&self, yaw: f64, pitch: f64) {
        let (pwm, speed) = gimbal::packets(yaw, pitch);
        self.send_duml(
            DEV_APP,
            0,
            DEV_GIMBAL,
            0,
            4,
            gimbal::CMD_PWM,
            &pwm,
            duml::CMD_TYPE_REQ,
        );
        self.send_duml(
            DEV_APP,
            0,
            DEV_GIMBAL,
            0,
            4,
            gimbal::CMD_SPEED,
            &speed,
            duml::CMD_TYPE_REQ,
        );
    }

    fn rx_loop(&self) {
        let sock = {
            let guard = lock(&self.sock);
            let Some(sock) = guard.as_ref() else {
                return;
            };
            Arc::clone(sock)
        };
        let mut buf = [0u8; 65536];
        while self.is_running() {
            match sock.recv_from(&mut buf) {
                Ok((n, _)) => self.handle_datagram(&buf[..n]),
                Err(err)
                    if matches!(
                        err.kind(),
                        ErrorKind::TimedOut
                            | ErrorKind::WouldBlock
                            | ErrorKind::Interrupted
                            | ErrorKind::ConnectionRefused
                    ) => {}
                Err(_) if self.is_running() => {}
                Err(_) => break,
            }
        }
    }

    fn handle_datagram(&self, data: &[u8]) {
        let Some(header) = frame::parse_header(data) else {
            return;
        };
        let mut video = None;
        let mut duml_blob = None;
        {
            let mut state = lock(&self.state);
            match header.pkt_type {
                frame::TYPE_VIDEO => {
                    if frame::seq_ahead(header.seq, state.video_rx_seq) {
                        state.video_rx_seq = header.seq;
                    }
                    if header.data.len() > 12 {
                        video = Some(header.data[12..].to_vec());
                    }
                }
                frame::TYPE_TELEMETRY => duml_blob = Some(header.data),
                frame::TYPE_ACK_TELEMETRY => {
                    if frame::seq_ahead(header.seq, state.acktelem_rx_seq) {
                        state.acktelem_rx_seq = header.seq;
                    }
                    duml_blob = Some(header.data);
                }
                frame::TYPE_COMMAND => {
                    if frame::seq_ahead(header.seq, state.cmd_rx_seq) {
                        state.cmd_rx_seq = header.seq;
                    }
                    duml_blob = Some(header.data);
                }
                _ => {}
            }
        }
        if let Some(h264) = video {
            self.events.on_video(&h264);
        }
        if let Some(blob) = duml_blob {
            let packets = duml::parse(&blob);
            if !packets.is_empty() {
                self.events.on_duml(&packets);
            }
        }
    }

    fn timer_loop(&self) {
        let mut next_ack = Instant::now();
        let mut next_hb = Instant::now();
        let mut next_gimbal = Instant::now();
        loop {
            if !self.is_running() {
                break;
            }
            let now = Instant::now();
            let (gimbal_on, gimbal_hz, video_hb, yaw, pitch) = {
                let state = lock(&self.state);
                (
                    state.gimbal_on,
                    state.gimbal_hz,
                    state.video_hb,
                    state.gimbal_yaw,
                    state.gimbal_pitch,
                )
            };
            if now >= next_ack {
                self.send_ack();
                next_ack = now + Duration::from_millis(20);
            }
            if video_hb && now >= next_hb {
                self.send_heartbeat();
                next_hb = now + Duration::from_millis(200);
            }
            if gimbal_on && now >= next_gimbal {
                self.send_gimbal(yaw, pitch);
                next_gimbal = now + Duration::from_secs_f64(1.0 / gimbal_hz.max(5.0));
            }
            let mut wake = next_ack;
            if video_hb {
                wake = wake.min(next_hb);
            }
            if gimbal_on {
                wake = wake.min(next_gimbal);
            }
            let pause = wake
                .saturating_duration_since(Instant::now())
                .clamp(Duration::from_millis(2), Duration::from_millis(50));
            thread::sleep(pause);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    struct VideoSink(StdMutex<Vec<Vec<u8>>>);

    impl Events for VideoSink {
        fn on_video(&self, h264: &[u8]) {
            self.0.lock().unwrap().push(h264.to_vec());
        }
        fn on_duml(&self, _packets: &[Packet]) {}
    }

    #[test]
    fn handshake_command_and_video() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let port = server.local_addr().unwrap().port();
        let sink = Arc::new(VideoSink(StdMutex::new(Vec::new())));
        let session = Session::new("127.0.0.1", port, sink.clone()).unwrap();

        let server = thread::spawn(move || {
            let mut buf = [0u8; 4096];
            let (n, src) = server.recv_from(&mut buf).expect("handshake");
            let header = frame::parse_header(&buf[..n]).unwrap();
            assert_eq!(header.pkt_type, frame::TYPE_HANDSHAKE);
            assert!(header.xor_ok);
            let reply = frame::build_header(8, 0x1234, 0, frame::TYPE_HANDSHAKE);
            server.send_to(&reply, src).unwrap();

            let mut payload = vec![0u8; 12];
            payload.extend_from_slice(b"nal");
            let mut video =
                frame::build_header(8 + payload.len(), 0x1234, 10, frame::TYPE_VIDEO).to_vec();
            video.extend(payload);
            server.send_to(&video, src).unwrap();

            let deadline = Instant::now() + Duration::from_secs(2);
            let mut saw = false;
            while Instant::now() < deadline {
                match server.recv_from(&mut buf) {
                    Ok((n, _)) => {
                        let Some(header) = frame::parse_header(&buf[..n]) else {
                            continue;
                        };
                        if header.pkt_type != frame::TYPE_COMMAND || header.data.len() < 12 {
                            continue;
                        }
                        let packets = duml::parse(&header.data[12..]);
                        if packets
                            .iter()
                            .any(|pkt| pkt.cmd_id == 0x0C && pkt.payload == [1, 2, 3])
                        {
                            saw = true;
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            saw
        });

        assert!(session.connect(2.0), "{}", session.last_error());
        assert_eq!(session.session_id(), 0x1234);
        assert_eq!(session.seed() & 0x7, 0);
        assert!(session.start());
        let seq = session.send_duml(
            DEV_APP,
            0,
            DEV_GIMBAL,
            0,
            4,
            0x0C,
            &[1, 2, 3],
            duml::CMD_TYPE_REQ,
        );
        assert_eq!(seq, session.seed());
        let started = Instant::now();
        for _ in 0..50 {
            if !sink.0.lock().unwrap().is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            sink.0.lock().unwrap().first().map(Vec::as_slice),
            Some(&b"nal"[..])
        );
        session.shutdown();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(server.join().unwrap());
    }

    #[test]
    fn handshake_timeout_does_not_panic() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        let session = Session::new("127.0.0.1", port, Arc::new(NopEvents)).unwrap();
        assert!(!session.connect(0.3));
        assert!(session.last_error().contains("超时"));
        drop(server);
    }

    #[test]
    fn gimbal_loop_emits_pwm() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let port = server.local_addr().unwrap().port();
        let session = Session::new("127.0.0.1", port, Arc::new(NopEvents)).unwrap();
        let ready = thread::spawn(move || {
            let mut buf = [0u8; 2048];
            let (n, src) = server.recv_from(&mut buf).unwrap();
            assert_eq!(
                frame::parse_header(&buf[..n]).unwrap().pkt_type,
                frame::TYPE_HANDSHAKE
            );
            let reply = frame::build_header(8, 7, 0, frame::TYPE_HANDSHAKE);
            server.send_to(&reply, src).unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                let Ok((n, _)) = server.recv_from(&mut buf) else {
                    continue;
                };
                let Some(header) = frame::parse_header(&buf[..n]) else {
                    continue;
                };
                if header.pkt_type != frame::TYPE_COMMAND || header.data.len() < 12 {
                    continue;
                }
                if duml::parse(&header.data[12..])
                    .iter()
                    .any(|pkt| pkt.cmd_id == gimbal::CMD_PWM)
                {
                    return true;
                }
            }
            false
        });
        assert!(session.connect(2.0), "{}", session.last_error());
        assert!(session.start());
        session.set_gimbal_velocity(0.5, 0.0);
        session.start_gimbal(20.0);
        assert!(ready.join().unwrap());
        session.shutdown();
    }
}
