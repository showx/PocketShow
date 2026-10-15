use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use pocketshow_core::duml;

pub fn join_wifi(ssid: &str, password: &str, camera_ip: &str) -> Result<()> {
    if ssid.is_empty() {
        return Err(anyhow!("加入相机 WiFi 需要 ssid。用 --ssid，或写到配置 wifi.ssid"));
    }
    if current_ssid().as_deref() == Some(ssid) {
        eprintln!("已在 {ssid} 上");
        return Ok(());
    }
    let iface = wifi_interface();
    eprintln!("加入 WiFi {ssid}");
    let status = Command::new("networksetup")
        .args(["-setairportnetwork", &iface, ssid, password])
        .status()
        .context("调用 networksetup 失败")?;
    if !status.success() {
        return Err(anyhow!("加入 {ssid} 失败"));
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if current_ssid().as_deref() == Some(ssid) || camera_reachable(camera_ip) {
            eprintln!("已连接 {ssid}");
            return Ok(());
        }
        thread::sleep(Duration::from_secs(1));
    }
    Err(anyhow!("加入 {ssid} 超时"))
}

pub fn wait_camera(ip: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if camera_reachable(ip) {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(1));
    }
    Err(anyhow!("相机 {ip} 还 ping 不通"))
}

pub fn wake_wifi(timeout_s: f64) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("蓝牙运行时起不来")?;
    runtime.block_on(wake_wifi_async(timeout_s))
}

fn wifi_interface() -> String {
    let output = Command::new("networksetup")
        .args(["-listallhardwareports"])
        .output();
    let Ok(output) = output else {
        return "en0".into();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = text.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if line.contains("Wi-Fi") || line.contains("AirPort") {
            for extra in lines.iter().skip(index + 1).take(3) {
                if let Some(device) = extra.strip_prefix("Device:") {
                    return device.trim().to_string();
                }
            }
        }
    }
    "en0".into()
}

fn current_ssid() -> Option<String> {
    let output = Command::new("networksetup")
        .args(["-getairportnetwork", &wifi_interface()])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_once(':').map(|(_, ssid)| ssid.trim().to_string())
}

fn camera_reachable(ip: &str) -> bool {
    Command::new("ping")
        .args(["-c", "1", "-t", "2", ip])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

const PAIR_HASH: &[u8] = &[
    0x20, 0x32, 0x38, 0x34, 0x61, 0x65, 0x35, 0x62, 0x38, 0x64, 0x37, 0x36, 0x62, 0x33, 0x33, 0x37, 0x35, 0x61, 0x30,
    0x34, 0x61, 0x36, 0x34, 0x31, 0x37, 0x61, 0x64, 0x37, 0x31, 0x62, 0x65, 0x61, 0x33,
];

fn ble_message() -> Vec<u8> {
    let pin = b"mbln";
    let mut payload = PAIR_HASH.to_vec();
    payload.push(pin.len() as u8);
    payload.extend_from_slice(pin);
    let total_len = 13 + payload.len();
    let mut buf = vec![0x55, (total_len & 0xFF) as u8, 0x04];
    buf.push(duml::crc8(&buf, 0x77));
    buf.extend_from_slice(&0x0702u16.to_le_bytes());
    buf.extend_from_slice(&0x8092u16.to_le_bytes());
    buf.extend_from_slice(&0x450740u32.to_le_bytes()[..3]);
    buf.extend_from_slice(&payload);
    buf.extend_from_slice(&duml::crc16(&buf, 0x3692).to_le_bytes());
    buf
}

async fn wake_wifi_async(timeout_s: f64) -> Result<()> {
    use btleplug::api::{Central, Manager as _, Peripheral as _, ScanFilter, WriteType};
    use btleplug::platform::Manager;

    eprintln!("扫描 Pocket 3 蓝牙，用来打开相机 WiFi");
    let manager = Manager::new().await.context("打不开蓝牙")?;
    let adapters = manager.adapters().await.context("列举蓝牙适配器失败")?;
    let central = adapters.into_iter().next().ok_or_else(|| anyhow!("这台电脑没有蓝牙适配器"))?;
    central.start_scan(ScanFilter::default()).await.context("蓝牙扫描失败")?;
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(3.0));
    let mut chosen = None;
    while Instant::now() < deadline {
        for peripheral in central.peripherals().await.context("读取蓝牙设备失败")? {
            if let Some(props) = peripheral.properties().await.context("读取蓝牙广播失败")? {
                let name = props.local_name.unwrap_or_default();
                let looks = name.contains("Osmo") || name.contains("Pocket") || name.contains("DJI") || looks_dji(&props.manufacturer_data);
                if looks {
                    eprintln!("发现 {name}");
                    chosen = Some(peripheral);
                    break;
                }
            }
        }
        if chosen.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    let _ = central.stop_scan().await;
    let peripheral = chosen.ok_or_else(|| anyhow!("没扫到 Pocket 3。可以在相机上手动打开 WiFi，然后去掉 --ble"))?;
    peripheral.connect().await.context("蓝牙连不上相机")?;
    peripheral.discover_services().await.context("读不到蓝牙服务")?;
    let write_uuid = uuid::Uuid::parse_str("0000fff5-0000-1000-8000-00805f9b34fb").expect("uuid");
    let characteristic = peripheral
        .characteristics()
        .into_iter()
        .find(|item| item.uuid == write_uuid)
        .ok_or_else(|| anyhow!("相机没有 WiFi 配对通道"))?;
    peripheral
        .write(&characteristic, &ble_message(), WriteType::WithoutResponse)
        .await
        .context("蓝牙配对命令没写进去")?;
    eprintln!("已发送蓝牙配对，等相机 WiFi 起来");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let _ = peripheral.disconnect().await;
    Ok(())
}

fn looks_dji(data: &std::collections::HashMap<u16, Vec<u8>>) -> bool {
    for (company, payload) in data {
        let mut blob = company.to_le_bytes().to_vec();
        blob.extend_from_slice(payload);
        if blob.len() >= 4 && (blob[..2] == [0xAA, 0x08] || blob[..2] == [0xAA, 0xF7]) && blob[2..4] == [0x20, 0x00] {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ble_pair_message_has_crc_tail() {
        let msg = ble_message();
        assert_eq!(msg[0], 0x55);
        assert_eq!(msg.len(), msg[1] as usize);
        assert!(msg.len() > 40);
    }
}
