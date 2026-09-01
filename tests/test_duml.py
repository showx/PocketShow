from pocketshow.pocket3.duml import build_duml, crc8, parse_duml


def test_duml_roundtrip():
    payload = b"\x01\x02\x03"
    pkt = build_duml(
        sender_type=2,
        sender_id=0,
        receiver_type=4,
        receiver_id=0,
        cmd_set=4,
        cmd_id=0x0C,
        payload=payload,
        seq=42,
    )
    assert pkt[0] == 0x55
    parsed = parse_duml(pkt)
    assert len(parsed) == 1
    msg = parsed[0]
    assert msg["cmd_set"] == 4
    assert msg["cmd_id"] == 0x0C
    assert msg["seq"] == 42
    assert msg["payload"] == payload
    assert msg["sender_type"] == 2
    assert msg["receiver_type"] == 4


def test_crc8_header():
    header = bytes([0x55, 0x0D, 0x04])
    assert crc8(header) == crc8(header)
    pkt = build_duml(2, 0, 4, 0, 4, 1, b"", seq=1)
    assert crc8(pkt[:3]) == pkt[3]


def test_parse_skips_garbage():
    pkt = build_duml(2, 0, 4, 0, 4, 1, b"\x00\x01")
    parsed = parse_duml(b"\x00\x11junk" + pkt + b"\xff")
    assert len(parsed) == 1
    assert parsed[0]["cmd_id"] == 1
