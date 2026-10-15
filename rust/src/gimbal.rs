//! 云台速度包。字节布局与原先 Python `WifiGimbal._send` 一致。

pub const SPEED_CENTER: i32 = 1024;
pub const SPEED_SPAN: i32 = 1024;
pub const CONTROL_FLAGS: u16 = 0x8000;
pub const CONTROL_EXTRA: u16 = 0x0042;

pub const CMD_PWM: u8 = 0x01;
pub const CMD_SPEED: u8 = 0x0C;
pub const CMD_ABS: u8 = 0x14;

fn trunc_toward_zero(value: f64) -> i32 {
    value as i32
}

pub fn clamp_axis(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

/// 返回 (PWM 包, 角速度包)。
pub fn packets(yaw: f64, pitch: f64) -> (Vec<u8>, Vec<u8>) {
    let mut yaw_pwm = trunc_toward_zero(f64::from(SPEED_CENTER) + yaw * f64::from(SPEED_SPAN));
    let mut pitch_pwm = trunc_toward_zero(f64::from(SPEED_CENTER) + pitch * f64::from(SPEED_SPAN));
    yaw_pwm = yaw_pwm.clamp(0, 2048);
    pitch_pwm = pitch_pwm.clamp(0, 2048);
    let mut pwm = Vec::with_capacity(10);
    for value in [
        yaw_pwm as u16,
        SPEED_CENTER as u16,
        pitch_pwm as u16,
        CONTROL_FLAGS,
        CONTROL_EXTRA,
    ] {
        pwm.extend_from_slice(&value.to_le_bytes());
    }

    let pitch_speed = trunc_toward_zero(pitch * 30.0 * 10.0).clamp(-1800, 1800) as i16;
    let yaw_speed = trunc_toward_zero(yaw * 30.0 * 10.0).clamp(-1800, 1800) as i16;
    let mut speed = Vec::with_capacity(7);
    speed.extend_from_slice(&pitch_speed.to_le_bytes());
    speed.extend_from_slice(&0i16.to_le_bytes());
    speed.extend_from_slice(&yaw_speed.to_le_bytes());
    speed.push(0x01);
    (pwm, speed)
}

pub fn recenter_payload() -> [u8; 8] {
    let mut buf = [0u8; 8];
    buf[6] = 0x07;
    buf[7] = 30;
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(data: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(data.len() * 2);
        for byte in data {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0F) as usize] as char);
        }
        out
    }

    #[test]
    fn zero_is_pwm_center() {
        let (pwm, _) = packets(0.0, 0.0);
        assert_eq!(u16::from_le_bytes([pwm[0], pwm[1]]), SPEED_CENTER as u16);
        assert_eq!(u16::from_le_bytes([pwm[2], pwm[3]]), SPEED_CENTER as u16);
        assert_eq!(u16::from_le_bytes([pwm[4], pwm[5]]), SPEED_CENTER as u16);
    }

    #[test]
    fn matches_python_golden() {
        let (pwm, speed) = packets(0.5, -0.25);
        assert_eq!(hex(&pwm), "00060004000300804200");
        assert_eq!(hex(&speed), "b5ff0000960001");
    }

    #[test]
    fn positive_yaw_and_negative_pitch() {
        let (pwm, _) = packets(0.5, 0.0);
        let yaw = u16::from_le_bytes([pwm[0], pwm[1]]);
        let pitch = u16::from_le_bytes([pwm[4], pwm[5]]);
        assert!(yaw > SPEED_CENTER as u16);
        assert_eq!(pitch, SPEED_CENTER as u16);

        let (_, speed) = packets(1.0, -1.0);
        let pitch_s = i16::from_le_bytes([speed[0], speed[1]]);
        let roll_s = i16::from_le_bytes([speed[2], speed[3]]);
        let yaw_s = i16::from_le_bytes([speed[4], speed[5]]);
        assert!(yaw_s > 0);
        assert!(pitch_s < 0);
        assert_eq!(roll_s, 0);
        assert_eq!(speed[6], 0x01);
    }
}
