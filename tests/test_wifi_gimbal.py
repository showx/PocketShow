import struct

from pocketshow.gimbal.wifi import SPEED_CENTER, WifiGimbal


class FakeUdp:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    def register_duml_callback(self, cmd_set, cmd_id, callback) -> None:
        return None

    def send_duml_req(self, receiver_type, receiver_id, cmd_set, cmd_id, payload=b""):
        self.sent.append((receiver_type, cmd_set, cmd_id, payload))
        return 1


def test_zero_velocity_is_pwm_center():
    client = FakeUdp()
    gimbal = WifiGimbal(client, send_hz=10)
    gimbal._send(0.0, 0.0)
    pwm = [item for item in client.sent if item[2] == 0x01]
    assert pwm
    yaw, roll, pitch, _flags, _extra = struct.unpack("<HHHHH", pwm[0][3])
    assert yaw == SPEED_CENTER
    assert roll == SPEED_CENTER
    assert pitch == SPEED_CENTER


def test_positive_yaw_increases_pwm():
    client = FakeUdp()
    gimbal = WifiGimbal(client, send_hz=10)
    gimbal._send(0.5, 0.0)
    pwm = [item for item in client.sent if item[2] == 0x01][0]
    yaw, _roll, pitch, _flags, _extra = struct.unpack("<HHHHH", pwm[3])
    assert yaw > SPEED_CENTER
    assert pitch == SPEED_CENTER


def test_speed_cmd_present():
    client = FakeUdp()
    gimbal = WifiGimbal(client, send_hz=10)
    gimbal._send(1.0, -1.0)
    speed = [item for item in client.sent if item[2] == 0x0C]
    assert speed
    pitch_s, roll_s, yaw_s, flag = struct.unpack("<hhhB", speed[0][3])
    assert flag == 0x01
    assert yaw_s > 0
    assert pitch_s < 0
    assert roll_s == 0
