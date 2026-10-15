//! 画面误差 → 云台速度。死区、PID、前馈、丢失衰减与原先 Python 一致。

use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Gains {
    pub deadzone: f64,
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub feedforward: f64,
    pub max_rate: f64,
    pub lost_hold_s: f64,
    pub lost_timeout_s: f64,
    pub target_area: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub id: i64,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub vx: f64,
    pub vy: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Command {
    pub yaw_rate: f64,
    pub pitch_rate: f64,
    pub lost: bool,
    pub ex: f64,
    pub ey: f64,
    pub size_ratio: f64,
    pub target_id: Option<i64>,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameError {
    pub ex: f64,
    pub ey: f64,
    pub size_ratio: f64,
}

pub fn frame_error(target: Target, frame_w: f64, frame_h: f64, target_area: f64) -> FrameError {
    let cx = (target.x1 + target.x2) * 0.5;
    let cy = (target.y1 + target.y2) * 0.5;
    let area = (target.x2 - target.x1).max(0.0) * (target.y2 - target.y1).max(0.0);
    let norm = area / (frame_w * frame_h).max(1.0);
    FrameError {
        ex: cx / frame_w - 0.5,
        ey: cy / frame_h - 0.5,
        size_ratio: norm / target_area.max(1e-6),
    }
}

struct Axis {
    kp: f64,
    ki: f64,
    kd: f64,
    i: f64,
    prev: Option<f64>,
}

impl Axis {
    fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            i: 0.0,
            prev: None,
        }
    }

    fn reset(&mut self) {
        self.i = 0.0;
        self.prev = None;
    }

    fn update(&mut self, error: f64, dt: f64, deadzone: bool) -> f64 {
        if deadzone {
            self.i *= 0.5;
            self.prev = Some(0.0);
            return 0.0;
        }
        self.i = (self.i + error * dt).clamp(-1.0, 1.0);
        let deriv = match self.prev {
            None if dt <= 1e-6 => 0.0,
            None => 0.0,
            Some(_) if dt <= 1e-6 => 0.0,
            Some(prev) => (error - prev) / dt,
        };
        self.prev = Some(error);
        self.kp * error + self.ki * self.i + self.kd * deriv
    }
}

pub struct Controller {
    cfg: Gains,
    yaw: Axis,
    pitch: Axis,
    last_cmd: (f64, f64),
    lost_since: Option<f64>,
    clock: Instant,
}

impl Controller {
    pub fn new(cfg: Gains) -> Self {
        Self {
            yaw: Axis::new(cfg.kp, cfg.ki, cfg.kd),
            pitch: Axis::new(cfg.kp, cfg.ki, cfg.kd),
            cfg,
            last_cmd: (0.0, 0.0),
            lost_since: None,
            clock: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.yaw.reset();
        self.pitch.reset();
        self.last_cmd = (0.0, 0.0);
        self.lost_since = None;
    }

    pub fn update(
        &mut self,
        target: Option<Target>,
        frame_w: i32,
        frame_h: i32,
        dt: f64,
        now: Option<f64>,
    ) -> Command {
        let now = now.unwrap_or_else(|| self.clock.elapsed().as_secs_f64());
        let dt = dt.max(1e-3);
        let Some(target) = target else {
            return self.on_lost(now, dt);
        };
        self.lost_since = None;
        let err = frame_error(
            target,
            f64::from(frame_w),
            f64::from(frame_h),
            self.cfg.target_area,
        );
        let limit = self.cfg.max_rate;
        let yaw = axis_output(
            &self.cfg,
            &mut self.yaw,
            err.ex,
            target.vx / f64::from(frame_w).max(1.0) / dt,
            dt,
        )
        .clamp(-limit, limit);
        // 人物在画面下方 (ey>0) 时下俯，pitch 取反。
        let pitch = (-axis_output(
            &self.cfg,
            &mut self.pitch,
            err.ey,
            target.vy / f64::from(frame_h).max(1.0) / dt,
            dt,
        ))
        .clamp(-limit, limit);
        self.last_cmd = (yaw, pitch);
        Command {
            yaw_rate: yaw,
            pitch_rate: pitch,
            lost: false,
            ex: err.ex,
            ey: err.ey,
            size_ratio: err.size_ratio,
            target_id: Some(target.id),
        }
    }

    fn on_lost(&mut self, now: f64, dt: f64) -> Command {
        if self.lost_since.is_none() {
            self.lost_since = Some(now);
        }
        let elapsed = now - self.lost_since.unwrap_or(now);
        if elapsed >= self.cfg.lost_timeout_s {
            self.yaw.reset();
            self.pitch.reset();
            self.last_cmd = (0.0, 0.0);
            return Command {
                yaw_rate: 0.0,
                pitch_rate: 0.0,
                lost: true,
                ex: 0.0,
                ey: 0.0,
                size_ratio: 0.0,
                target_id: None,
            };
        }
        let mut decay = 0.85_f64.powf((elapsed / dt.max(1e-3) * 0.25).max(1.0));
        if elapsed > self.cfg.lost_hold_s {
            decay *= 0.4;
        }
        let yaw = self.last_cmd.0 * decay;
        let pitch = self.last_cmd.1 * decay;
        self.last_cmd = (yaw, pitch);
        Command {
            yaw_rate: yaw,
            pitch_rate: pitch,
            lost: true,
            ex: 0.0,
            ey: 0.0,
            size_ratio: 0.0,
            target_id: None,
        }
    }
}

fn axis_output(cfg: &Gains, pid: &mut Axis, error: f64, vel_norm: f64, dt: f64) -> f64 {
    let in_deadzone = error.abs() < cfg.deadzone;
    let shaped = if in_deadzone { 0.0 } else { error };
    let pid_out = pid.update(shaped, dt, in_deadzone);
    if in_deadzone {
        return 0.0;
    }
    pid_out + cfg.feedforward * vel_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gains() -> Gains {
        Gains {
            deadzone: 0.02,
            kp: 1.35,
            ki: 0.0,
            kd: 0.0,
            feedforward: 0.0,
            max_rate: 1.0,
            lost_hold_s: 0.8,
            lost_timeout_s: 2.5,
            target_area: 0.12,
        }
    }

    fn box_target(x1: f64, y1: f64, x2: f64, y2: f64) -> Target {
        Target {
            id: 1,
            x1,
            y1,
            x2,
            y2,
            vx: 0.0,
            vy: 0.0,
        }
    }

    #[test]
    fn right_side_positive_yaw() {
        let mut ctrl = Controller::new(gains());
        let cmd = ctrl.update(
            Some(box_target(700.0, 200.0, 900.0, 500.0)),
            1000,
            1000,
            0.05,
            Some(1.0),
        );
        assert!(cmd.yaw_rate > 0.0);
        assert!((cmd.yaw_rate - 0.4050000000000001).abs() < 1e-12);
        assert!(!cmd.lost);
    }

    #[test]
    fn below_center_negative_pitch() {
        let mut ctrl = Controller::new(gains());
        let cmd = ctrl.update(
            Some(box_target(400.0, 600.0, 600.0, 900.0)),
            1000,
            1000,
            0.05,
            Some(1.0),
        );
        assert!(cmd.pitch_rate < 0.0);
    }

    #[test]
    fn deadzone_holds_still() {
        let mut cfg = gains();
        cfg.deadzone = 0.1;
        let mut ctrl = Controller::new(cfg);
        let cmd = ctrl.update(
            Some(box_target(480.0, 480.0, 520.0, 520.0)),
            1000,
            1000,
            0.05,
            Some(1.0),
        );
        assert_eq!(cmd.yaw_rate, 0.0);
        assert_eq!(cmd.pitch_rate, 0.0);
    }

    #[test]
    fn max_rate_clamp() {
        let mut cfg = gains();
        cfg.deadzone = 0.0;
        cfg.kp = 20.0;
        cfg.max_rate = 0.4;
        let mut ctrl = Controller::new(cfg);
        let cmd = ctrl.update(
            Some(box_target(900.0, 100.0, 990.0, 400.0)),
            1000,
            1000,
            0.05,
            Some(1.0),
        );
        assert!(cmd.yaw_rate.abs() <= 0.4 + 1e-9);
    }

    #[test]
    fn lost_decays_then_stops() {
        let mut cfg = gains();
        cfg.ki = 0.05;
        cfg.kd = 0.18;
        cfg.lost_hold_s = 0.2;
        cfg.lost_timeout_s = 0.5;
        let mut ctrl = Controller::new(cfg);
        let first = ctrl.update(
            Some(box_target(700.0, 200.0, 900.0, 500.0)),
            1000,
            1000,
            0.05,
            Some(1.0),
        );
        assert!((first.yaw_rate - 0.40575000000000006).abs() < 1e-12);
        let holding = ctrl.update(None, 1000, 1000, 0.05, Some(1.1));
        assert!(holding.lost);
        assert!((holding.yaw_rate - 0.3448875).abs() < 1e-12);
        let stopped = ctrl.update(None, 1000, 1000, 0.05, Some(2.0));
        assert!(stopped.lost);
        assert_eq!(stopped.yaw_rate, 0.0);
        assert_eq!(stopped.pitch_rate, 0.0);
    }

    #[test]
    fn center_error() {
        let err = frame_error(box_target(250.0, 250.0, 750.0, 750.0), 1000.0, 1000.0, 0.25);
        assert!(err.ex.abs() < 1e-6);
        assert!(err.ey.abs() < 1e-6);
        assert!((err.size_ratio - 1.0).abs() < 1e-6);
    }
}
