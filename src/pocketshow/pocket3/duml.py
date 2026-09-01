from __future__ import annotations

DUML_SOF = 0x55
CMD_TYPE_REQ = 0
CMD_TYPE_ACK = 1
CMD_TYPE_PUSH = 2
CMD_TYPE_WRITE = 4


def _crc8_table() -> list[int]:
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
        table.append(crc)
    return table


def _crc16_table() -> list[int]:
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC8_TABLE = _crc8_table()
_CRC16_TABLE = _crc16_table()


def crc8(data: bytes, init: int = 0x77) -> int:
    crc = init
    for byte in data:
        crc = _CRC8_TABLE[(crc ^ byte) & 0xFF]
    return crc


def crc16(data: bytes, init: int = 0x3692) -> int:
    crc = init
    for byte in data:
        crc = (crc >> 8) ^ _CRC16_TABLE[(crc ^ byte) & 0xFF]
    return crc & 0xFFFF


def build_duml(
    sender_type: int,
    sender_id: int,
    receiver_type: int,
    receiver_id: int,
    cmd_set: int,
    cmd_id: int,
    payload: bytes = b"",
    seq: int = 0,
    cmd_type: int = CMD_TYPE_REQ,
    encrypt: int = 0,
    version: int = 1,
) -> bytes:
    pkt_len = 11 + len(payload) + 2
    len_ver = (pkt_len & 0x03FF) | ((version & 0x3F) << 10)
    buf = bytearray()
    buf.append(DUML_SOF)
    buf.append(len_ver & 0xFF)
    buf.append((len_ver >> 8) & 0xFF)
    buf.append(crc8(bytes(buf)))
    buf.append((sender_id << 5) | (sender_type & 0x1F))
    buf.append((receiver_id << 5) | (receiver_type & 0x1F))
    buf.append(seq & 0xFF)
    buf.append((seq >> 8) & 0xFF)
    buf.append(((cmd_type & 0x07) << 5) | (encrypt & 0x07))
    buf.append(cmd_set & 0xFF)
    buf.append(cmd_id & 0xFF)
    buf.extend(payload)
    crc = crc16(bytes(buf))
    buf.append(crc & 0xFF)
    buf.append((crc >> 8) & 0xFF)
    return bytes(buf)


def parse_duml(data: bytes) -> list[dict]:
    packets: list[dict] = []
    i = 0
    while i < len(data) - 4:
        if data[i] != DUML_SOF:
            i += 1
            continue
        if i + 3 > len(data):
            break
        len_ver = data[i + 1] | (data[i + 2] << 8)
        pkt_len = len_ver & 0x3FF
        if pkt_len < 13 or pkt_len > 2048 or i + pkt_len > len(data):
            i += 1
            continue
        pkt = data[i : i + pkt_len]
        if crc8(pkt[:3]) != pkt[3]:
            i += 1
            continue
        got = pkt[pkt_len - 2] | (pkt[pkt_len - 1] << 8)
        if crc16(pkt[: pkt_len - 2]) != got:
            i += 1
            continue
        sender = pkt[4]
        receiver = pkt[5]
        cmd_byte = pkt[8]
        packets.append(
            {
                "length": pkt_len,
                "version": (len_ver >> 10) & 0x3F,
                "sender_type": sender & 0x1F,
                "sender_id": (sender >> 5) & 0x07,
                "receiver_type": receiver & 0x1F,
                "receiver_id": (receiver >> 5) & 0x07,
                "seq": pkt[6] | (pkt[7] << 8),
                "cmd_type": (cmd_byte >> 5) & 0x07,
                "encrypt": cmd_byte & 0x07,
                "cmd_set": pkt[9],
                "cmd_id": pkt[10],
                "payload": pkt[11 : pkt_len - 2],
            }
        )
        i += pkt_len
    return packets
