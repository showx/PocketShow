//! Pocket 3 UDP 9004 外层帧：8 字节头 + 负载。

pub const TYPE_HANDSHAKE: u8 = 0x00;
pub const TYPE_TELEMETRY: u8 = 0x01;
pub const TYPE_VIDEO: u8 = 0x02;
pub const TYPE_ACK_TELEMETRY: u8 = 0x03;
pub const TYPE_ACK: u8 = 0x04;
pub const TYPE_COMMAND: u8 = 0x05;

pub const HANDSHAKE_TAIL: [u8; 38] = [
    0x64, 0x00, 0x64, 0x00, 0xC0, 0x05, 0x14, 0x00, 0x00, 0x64, 0x00, 0x00, 0x01, 0x90, 0x01, 0xC0,
    0x05, 0x14, 0x00, 0x00, 0x64, 0x00, 0x14, 0x00, 0x64, 0x00, 0xC0, 0x05, 0x14, 0x00, 0x00, 0x64,
    0x00, 0x01, 0x01, 0x04, 0x01, 0x02,
];

#[derive(Clone, Debug)]
pub struct Header {
    pub length: usize,
    pub session_id: u16,
    pub seq: u16,
    pub pkt_type: u8,
    pub xor_ok: bool,
    pub data: Vec<u8>,
}

pub fn seq_ahead(new_seq: u16, old_seq: u16) -> bool {
    let diff = new_seq.wrapping_sub(old_seq);
    diff > 0 && diff < 0x8000
}

pub fn build_header(pkt_len: usize, session_id: u16, seq: u16, pkt_type: u8) -> [u8; 8] {
    let mut buf = [0u8; 8];
    let encoded = (pkt_len as u16) | 0x8000;
    buf[0] = (encoded & 0xFF) as u8;
    buf[1] = (encoded >> 8) as u8;
    buf[2] = (session_id & 0xFF) as u8;
    buf[3] = (session_id >> 8) as u8;
    buf[4] = (seq & 0xFF) as u8;
    buf[5] = (seq >> 8) as u8;
    buf[6] = pkt_type;
    let mut xor = 0u8;
    for item in buf.iter().take(7) {
        xor ^= item;
    }
    buf[7] = xor;
    buf
}

pub fn parse_header(data: &[u8]) -> Option<Header> {
    if data.len() < 8 {
        return None;
    }
    let pkt_len = (u16::from(data[0]) | (u16::from(data[1]) << 8)) & 0x7FFF;
    let mut xor = 0u8;
    for item in data.iter().take(7) {
        xor ^= item;
    }
    Some(Header {
        length: pkt_len as usize,
        session_id: u16::from(data[2]) | (u16::from(data[3]) << 8),
        seq: u16::from(data[4]) | (u16::from(data[5]) << 8),
        pkt_type: data[6],
        xor_ok: xor == data[7],
        data: data[8..].to_vec(),
    })
}

pub fn handshake_packet(session_id: u16, seed: u16) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + HANDSHAKE_TAIL.len());
    payload.push((seed & 0xFF) as u8);
    payload.push((seed >> 8) as u8);
    payload.extend_from_slice(&HANDSHAKE_TAIL);
    let mut packet = build_header(8 + payload.len(), session_id, 0, TYPE_HANDSHAKE).to_vec();
    packet.extend(payload);
    packet
}

pub fn ack_packet(session_id: u16, video_rx: u16, acktelem_rx: u16, cmd_seq: u16) -> Vec<u8> {
    let mut payload = Vec::with_capacity(26);
    for seq in [video_rx, acktelem_rx, cmd_seq] {
        payload.extend_from_slice(&seq.to_le_bytes());
        payload.extend_from_slice(&seq.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
    }
    payload.extend_from_slice(&0u16.to_le_bytes());
    let mut packet = build_header(8 + payload.len(), session_id, 0, TYPE_ACK).to_vec();
    packet.extend(payload);
    packet
}

pub fn command_packet(session_id: u16, seq: u16, msg_seq: u16, duml: &[u8]) -> Vec<u8> {
    let win_start = seq.wrapping_sub(32);
    let mut body = Vec::with_capacity(12 + duml.len());
    body.extend_from_slice(&win_start.to_le_bytes());
    body.extend_from_slice(&seq.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes());
    body.push((msg_seq & 0xFF) as u8);
    body.push(0x01);
    body.extend_from_slice(&0x0060u16.to_le_bytes());
    body.extend_from_slice(duml);
    let mut packet = build_header(8 + body.len(), session_id, seq, TYPE_COMMAND).to_vec();
    packet.extend(body);
    packet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duml;

    #[test]
    fn header_xor_roundtrip() {
        let header = build_header(21, 0x1234, 0x00A0, TYPE_COMMAND);
        let parsed = parse_header(&header).unwrap();
        assert!(parsed.xor_ok);
        assert_eq!(parsed.session_id, 0x1234);
        assert_eq!(parsed.seq, 0x00A0);
        assert_eq!(parsed.pkt_type, TYPE_COMMAND);
        assert_eq!(parsed.length, 21);
    }

    #[test]
    fn seq_window() {
        assert!(seq_ahead(2, 1));
        assert!(!seq_ahead(1, 1));
        assert!(!seq_ahead(1, 2));
        assert!(seq_ahead(0, 0xFFFF));
    }

    #[test]
    fn command_wraps_duml() {
        let duml_pkt = duml::build(2, 0, 4, 0, 4, 1, &[0x0A], 8, duml::CMD_TYPE_REQ, 0, 1);
        let packet = command_packet(0x10, 8, 3, &duml_pkt);
        let header = parse_header(&packet).unwrap();
        assert!(header.xor_ok);
        assert_eq!(header.pkt_type, TYPE_COMMAND);
        assert_eq!(header.seq, 8);
        let parsed = duml::parse(&header.data[12..]);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].payload, vec![0x0A]);
        assert_eq!(parsed[0].seq, 8);
    }

    #[test]
    fn ack_is_34_bytes() {
        let packet = ack_packet(0x55, 8, 8, 16);
        assert_eq!(packet.len(), 34);
        let header = parse_header(&packet).unwrap();
        assert_eq!(header.pkt_type, TYPE_ACK);
        assert!(header.xor_ok);
    }
}
