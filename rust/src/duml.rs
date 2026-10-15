//! DJI DUML：CRC、组包、从字节流里切出完整帧。

pub const SOF: u8 = 0x55;
pub const CMD_TYPE_REQ: u8 = 0;
pub const CMD_TYPE_ACK: u8 = 1;
pub const CMD_TYPE_PUSH: u8 = 2;
pub const CMD_TYPE_WRITE: u8 = 4;

fn crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for i in 0..256 {
        let mut crc = i as u8;
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0x8C
            } else {
                crc >> 1
            };
        }
        table[i] = crc;
    }
    table
}

fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    for i in 0..256 {
        let mut crc = i as u16;
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0x8408
            } else {
                crc >> 1
            };
        }
        table[i] = crc;
    }
    table
}

pub fn crc8(data: &[u8], init: u8) -> u8 {
    let table = crc8_table();
    let mut crc = init;
    for byte in data {
        crc = table[(crc ^ byte) as usize];
    }
    crc
}

pub fn crc16(data: &[u8], init: u16) -> u16 {
    let table = crc16_table();
    let mut crc = init;
    for byte in data {
        crc = (crc >> 8) ^ table[((crc ^ (*byte as u16)) & 0xFF) as usize];
    }
    crc
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub length: usize,
    pub version: u8,
    pub sender_type: u8,
    pub sender_id: u8,
    pub receiver_type: u8,
    pub receiver_id: u8,
    pub seq: u16,
    pub cmd_type: u8,
    pub encrypt: u8,
    pub cmd_set: u8,
    pub cmd_id: u8,
    pub payload: Vec<u8>,
}

pub fn build(
    sender_type: u8,
    sender_id: u8,
    receiver_type: u8,
    receiver_id: u8,
    cmd_set: u8,
    cmd_id: u8,
    payload: &[u8],
    seq: u16,
    cmd_type: u8,
    encrypt: u8,
    version: u8,
) -> Vec<u8> {
    let pkt_len = 13 + payload.len();
    let len_ver = ((pkt_len as u16) & 0x03FF) | ((version as u16 & 0x3F) << 10);
    let mut buf = Vec::with_capacity(pkt_len);
    buf.push(SOF);
    buf.push((len_ver & 0xFF) as u8);
    buf.push((len_ver >> 8) as u8);
    let header_crc = crc8(&buf, 0x77);
    buf.push(header_crc);
    buf.push((sender_id << 5) | (sender_type & 0x1F));
    buf.push((receiver_id << 5) | (receiver_type & 0x1F));
    buf.push((seq & 0xFF) as u8);
    buf.push((seq >> 8) as u8);
    buf.push(((cmd_type & 0x07) << 5) | (encrypt & 0x07));
    buf.push(cmd_set);
    buf.push(cmd_id);
    buf.extend_from_slice(payload);
    let crc = crc16(&buf, 0x3692);
    buf.push((crc & 0xFF) as u8);
    buf.push((crc >> 8) as u8);
    buf
}

pub fn parse(data: &[u8]) -> Vec<Packet> {
    let mut packets = Vec::new();
    let mut i = 0;
    while i + 4 < data.len() {
        if data[i] != SOF {
            i += 1;
            continue;
        }
        if i + 3 > data.len() {
            break;
        }
        let len_ver = u16::from(data[i + 1]) | (u16::from(data[i + 2]) << 8);
        let pkt_len = (len_ver & 0x3FF) as usize;
        if pkt_len < 13 || pkt_len > 2048 || i + pkt_len > data.len() {
            i += 1;
            continue;
        }
        let pkt = &data[i..i + pkt_len];
        if crc8(&pkt[..3], 0x77) != pkt[3] {
            i += 1;
            continue;
        }
        let got = u16::from(pkt[pkt_len - 2]) | (u16::from(pkt[pkt_len - 1]) << 8);
        if crc16(&pkt[..pkt_len - 2], 0x3692) != got {
            i += 1;
            continue;
        }
        let sender = pkt[4];
        let receiver = pkt[5];
        let cmd_byte = pkt[8];
        packets.push(Packet {
            length: pkt_len,
            version: ((len_ver >> 10) & 0x3F) as u8,
            sender_type: sender & 0x1F,
            sender_id: (sender >> 5) & 0x07,
            receiver_type: receiver & 0x1F,
            receiver_id: (receiver >> 5) & 0x07,
            seq: u16::from(pkt[6]) | (u16::from(pkt[7]) << 8),
            cmd_type: (cmd_byte >> 5) & 0x07,
            encrypt: cmd_byte & 0x07,
            cmd_set: pkt[9],
            cmd_id: pkt[10],
            payload: pkt[11..pkt_len - 2].to_vec(),
        });
        i += pkt_len;
    }
    packets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_matches_python_golden() {
        let pkt = build(
            2,
            0,
            4,
            0,
            4,
            0x0C,
            &[0x01, 0x02, 0x03],
            42,
            CMD_TYPE_REQ,
            0,
            1,
        );
        assert_eq!(hex::encode_like(&pkt), "5510045602042a0000040c0102036247");
        let parsed = parse(&pkt);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].cmd_set, 4);
        assert_eq!(parsed[0].cmd_id, 0x0C);
        assert_eq!(parsed[0].seq, 42);
        assert_eq!(parsed[0].payload, vec![1, 2, 3]);
        assert_eq!(parsed[0].sender_type, 2);
        assert_eq!(parsed[0].receiver_type, 4);
    }

    #[test]
    fn crc8_header_and_empty_packet() {
        assert_eq!(crc8(&[0x55, 0x0D, 0x04], 0x77), 51);
        let pkt = build(2, 0, 4, 0, 4, 1, b"", 1, CMD_TYPE_REQ, 0, 1);
        assert_eq!(crc8(&pkt[..3], 0x77), pkt[3]);
        assert_eq!(hex::encode_like(&pkt), "550d043302040100000401589e");
        assert_eq!(crc16(b"abc", 0x3692), 34394);
    }

    #[test]
    fn parse_skips_garbage() {
        let pkt = build(2, 0, 4, 0, 4, 1, &[0x00, 0x01], 0, CMD_TYPE_REQ, 0, 1);
        let mut blob = vec![0x00, 0x11];
        blob.extend_from_slice(b"junk");
        blob.extend_from_slice(&pkt);
        blob.push(0xFF);
        let parsed = parse(&blob);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].cmd_id, 1);
    }
}

#[cfg(test)]
mod hex {
    pub fn encode_like(data: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(data.len() * 2);
        for byte in data {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0F) as usize] as char);
        }
        out
    }
}
