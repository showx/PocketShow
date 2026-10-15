//! 给现有 Python 管线用的扩展。检测、识别、管理页仍在 Python。

use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::duml::{self, Packet};
use crate::follow::{self, Gains, Target};
use crate::gimbal;
use crate::session::{self, Events, Session};

struct PyEvents {
    video: Mutex<Option<Py<PyAny>>>,
    duml: Mutex<Vec<(u8, u8, Py<PyAny>)>>,
}

impl Events for PyEvents {
    fn on_video(&self, h264: &[u8]) {
        Python::attach(|py| {
            let callback = {
                let guard = self.video.lock().unwrap_or_else(|err| err.into_inner());
                guard.as_ref().map(|callback| callback.clone_ref(py))
            };
            if let Some(callback) = callback {
                if let Err(err) = callback.call1(py, (PyBytes::new(py, h264),)) {
                    err.print(py);
                }
            }
        });
    }

    fn on_duml(&self, packets: &[Packet]) {
        Python::attach(|py| {
            let callbacks: Vec<(u8, u8, Py<PyAny>)> = {
                let guard = self.duml.lock().unwrap_or_else(|err| err.into_inner());
                guard
                    .iter()
                    .map(|(cmd_set, cmd_id, callback)| (*cmd_set, *cmd_id, callback.clone_ref(py)))
                    .collect()
            };
            for packet in packets {
                for (cmd_set, cmd_id, callback) in &callbacks {
                    if *cmd_set == packet.cmd_set && *cmd_id == packet.cmd_id {
                        if let Err(err) = callback.call1(py, (packet_dict(py, packet),)) {
                            err.print(py);
                        }
                    }
                }
            }
        });
    }
}

fn packet_dict<'py>(py: Python<'py>, packet: &Packet) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("length", packet.length).unwrap();
    dict.set_item("version", packet.version).unwrap();
    dict.set_item("sender_type", packet.sender_type).unwrap();
    dict.set_item("sender_id", packet.sender_id).unwrap();
    dict.set_item("receiver_type", packet.receiver_type)
        .unwrap();
    dict.set_item("receiver_id", packet.receiver_id).unwrap();
    dict.set_item("seq", packet.seq).unwrap();
    dict.set_item("cmd_type", packet.cmd_type).unwrap();
    dict.set_item("encrypt", packet.encrypt).unwrap();
    dict.set_item("cmd_set", packet.cmd_set).unwrap();
    dict.set_item("cmd_id", packet.cmd_id).unwrap();
    dict.set_item("payload", PyBytes::new(py, &packet.payload))
        .unwrap();
    dict
}

#[pyfunction]
#[pyo3(signature = (data, init=0x77))]
fn crc8(data: &[u8], init: u8) -> u8 {
    duml::crc8(data, init)
}

#[pyfunction]
#[pyo3(signature = (data, init=0x3692))]
fn crc16(data: &[u8], init: u16) -> u16 {
    duml::crc16(data, init)
}

#[pyfunction]
#[pyo3(signature = (
    sender_type,
    sender_id,
    receiver_type,
    receiver_id,
    cmd_set,
    cmd_id,
    payload=None,
    seq=0,
    cmd_type=0,
    encrypt=0,
    version=1
))]
fn build_duml(
    py: Python<'_>,
    sender_type: u8,
    sender_id: u8,
    receiver_type: u8,
    receiver_id: u8,
    cmd_set: u8,
    cmd_id: u8,
    payload: Option<&[u8]>,
    seq: u16,
    cmd_type: u8,
    encrypt: u8,
    version: u8,
) -> Py<PyBytes> {
    let bytes = duml::build(
        sender_type,
        sender_id,
        receiver_type,
        receiver_id,
        cmd_set,
        cmd_id,
        payload.unwrap_or(b""),
        seq,
        cmd_type,
        encrypt,
        version,
    );
    PyBytes::new(py, &bytes).into()
}

#[pyfunction]
fn parse_duml(py: Python<'_>, data: &[u8]) -> Vec<Py<PyDict>> {
    duml::parse(data)
        .iter()
        .map(|packet| packet_dict(py, packet).unbind())
        .collect()
}

#[pyfunction]
fn compute_frame_error(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    frame_w: i32,
    frame_h: i32,
    target_area: f64,
) -> (f64, f64, f64) {
    let err = follow::frame_error(
        Target {
            id: 0,
            x1,
            y1,
            x2,
            y2,
            vx: 0.0,
            vy: 0.0,
        },
        f64::from(frame_w),
        f64::from(frame_h),
        target_area,
    );
    (err.ex, err.ey, err.size_ratio)
}

#[pyfunction]
fn gimbal_packets(py: Python<'_>, yaw: f64, pitch: f64) -> (Py<PyBytes>, Py<PyBytes>) {
    let (pwm, speed) = gimbal::packets(yaw, pitch);
    (
        PyBytes::new(py, &pwm).into(),
        PyBytes::new(py, &speed).into(),
    )
}

#[pyfunction]
fn gimbal_recenter(py: Python<'_>) -> Py<PyBytes> {
    PyBytes::new(py, &gimbal::recenter_payload()).into()
}

#[pyclass]
struct FollowController {
    inner: follow::Controller,
}

#[pymethods]
impl FollowController {
    #[new]
    fn new(
        deadzone: f64,
        kp: f64,
        ki: f64,
        kd: f64,
        feedforward: f64,
        max_rate: f64,
        lost_hold_s: f64,
        lost_timeout_s: f64,
        target_area: f64,
    ) -> Self {
        Self {
            inner: follow::Controller::new(Gains {
                deadzone,
                kp,
                ki,
                kd,
                feedforward,
                max_rate,
                lost_hold_s,
                lost_timeout_s,
                target_area,
            }),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[pyo3(signature = (target=None, frame_w=0, frame_h=0, dt=0.033, now=None))]
    fn update(
        &mut self,
        target: Option<(i64, f64, f64, f64, f64, f64, f64)>,
        frame_w: i32,
        frame_h: i32,
        dt: f64,
        now: Option<f64>,
    ) -> (f64, f64, bool, f64, f64, f64, i64) {
        let mapped = target.map(|(id, x1, y1, x2, y2, vx, vy)| Target {
            id,
            x1,
            y1,
            x2,
            y2,
            vx,
            vy,
        });
        let command = self.inner.update(mapped, frame_w, frame_h, dt, now);
        (
            command.yaw_rate,
            command.pitch_rate,
            command.lost,
            command.ex,
            command.ey,
            command.size_ratio,
            command.target_id.unwrap_or(-1),
        )
    }
}

#[pyclass]
struct UdpClient {
    inner: Session,
    events: Arc<PyEvents>,
}

#[pymethods]
impl UdpClient {
    #[new]
    fn new(camera_ip: &str, camera_port: u16) -> PyResult<Self> {
        let events = Arc::new(PyEvents {
            video: Mutex::new(None),
            duml: Mutex::new(Vec::new()),
        });
        let inner = Session::new(camera_ip, camera_port, events.clone())
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner, events })
    }

    fn connect(&self, timeout: f64) -> bool {
        self.inner.connect(timeout)
    }

    fn start(&self) -> PyResult<()> {
        if self.inner.start() {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(self.inner.last_error()))
        }
    }

    fn stop(&self, py: Python<'_>) {
        py.detach(|| self.inner.shutdown());
    }

    fn session_id(&self) -> u16 {
        self.inner.session_id()
    }

    fn seed(&self) -> u16 {
        self.inner.seed()
    }

    fn last_error(&self) -> String {
        self.inner.last_error()
    }

    fn set_video_callback(&self, callback: Py<PyAny>) {
        *self
            .events
            .video
            .lock()
            .unwrap_or_else(|err| err.into_inner()) = Some(callback);
    }

    fn register_duml_callback(&self, cmd_set: u8, cmd_id: u8, callback: Py<PyAny>) {
        self.events
            .duml
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .push((cmd_set, cmd_id, callback));
    }

    #[pyo3(signature = (
        sender_type,
        sender_id,
        receiver_type,
        receiver_id,
        cmd_set,
        cmd_id,
        payload=None,
        cmd_type=0
    ))]
    fn send_duml(
        &self,
        sender_type: u8,
        sender_id: u8,
        receiver_type: u8,
        receiver_id: u8,
        cmd_set: u8,
        cmd_id: u8,
        payload: Option<&[u8]>,
        cmd_type: u8,
    ) -> u16 {
        self.inner.send_duml(
            sender_type,
            sender_id,
            receiver_type,
            receiver_id,
            cmd_set,
            cmd_id,
            payload.unwrap_or(b""),
            cmd_type,
        )
    }

    #[pyo3(signature = (receiver_type, receiver_id, cmd_set, cmd_id, payload=None))]
    fn send_duml_req(
        &self,
        receiver_type: u8,
        receiver_id: u8,
        cmd_set: u8,
        cmd_id: u8,
        payload: Option<&[u8]>,
    ) -> u16 {
        self.send_duml(
            session::DEV_APP,
            0,
            receiver_type,
            receiver_id,
            cmd_set,
            cmd_id,
            payload,
            duml::CMD_TYPE_REQ,
        )
    }

    #[pyo3(signature = (receiver_type, receiver_id, cmd_set, cmd_id, payload=None))]
    fn send_duml_push(
        &self,
        receiver_type: u8,
        receiver_id: u8,
        cmd_set: u8,
        cmd_id: u8,
        payload: Option<&[u8]>,
    ) -> u16 {
        self.send_duml(
            session::DEV_APP,
            0,
            receiver_type,
            receiver_id,
            cmd_set,
            cmd_id,
            payload,
            duml::CMD_TYPE_PUSH,
        )
    }

    fn start_video(&self) {
        self.inner.start_video();
    }

    fn start_gimbal(&self, hz: f64) {
        self.inner.start_gimbal(hz);
    }

    fn set_gimbal_velocity(&self, yaw: f64, pitch: f64) {
        self.inner.set_gimbal_velocity(yaw, pitch);
    }

    fn stop_gimbal(&self) {
        self.inner.stop_gimbal();
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crc8, m)?)?;
    m.add_function(wrap_pyfunction!(crc16, m)?)?;
    m.add_function(wrap_pyfunction!(build_duml, m)?)?;
    m.add_function(wrap_pyfunction!(parse_duml, m)?)?;
    m.add_function(wrap_pyfunction!(compute_frame_error, m)?)?;
    m.add_function(wrap_pyfunction!(gimbal_packets, m)?)?;
    m.add_function(wrap_pyfunction!(gimbal_recenter, m)?)?;
    m.add_class::<FollowController>()?;
    m.add_class::<UdpClient>()?;
    Ok(())
}
