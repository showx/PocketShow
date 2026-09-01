from __future__ import annotations

import asyncio
import logging
import struct

from pocketshow.pocket3.duml import crc16, crc8

logger = logging.getLogger(__name__)

# BLE GATT：DJI 相机常用 fff4 notify / fff5 write（Moblin 公开文档）。
FFF4_UUID = "0000fff4-0000-1000-8000-00805f9b34fb"
FFF5_UUID = "0000fff5-0000-1000-8000-00805f9b34fb"

PAIR_ID = 0x8092
PAIR_TARGET = 0x0702
PAIR_TYPE = 0x450740
# Moblin 公开的配对哈希；也可在相机上手动打开 WiFi AP，跳过 BLE。
PAIR_HASH = bytes(
    [
        0x20, 0x32, 0x38, 0x34, 0x61, 0x65, 0x35, 0x62,
        0x38, 0x64, 0x37, 0x36, 0x62, 0x33, 0x33, 0x37,
        0x35, 0x61, 0x30, 0x34, 0x61, 0x36, 0x34, 0x31,
        0x37, 0x61, 0x64, 0x37, 0x31, 0x62, 0x65, 0x61,
        0x33,
    ]
)
PAIR_PIN = "mbln"


def _pack_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return bytes([len(data)]) + data


def build_ble_message(target: int, msg_id: int, msg_type: int, payload: bytes) -> bytes:
    total_len = 13 + len(payload)
    buf = bytearray()
    buf.append(0x55)
    buf.append(total_len & 0xFF)
    buf.append(0x04)
    buf.append(crc8(bytes(buf[:3])))
    buf += struct.pack("<H", target)
    buf += struct.pack("<H", msg_id)
    buf.append(msg_type & 0xFF)
    buf.append((msg_type >> 8) & 0xFF)
    buf.append((msg_type >> 16) & 0xFF)
    buf += payload
    buf += struct.pack("<H", crc16(bytes(buf)))
    return bytes(buf)


async def activate_wifi_ap(timeout: float = 15.0) -> bool:
    """扫描 Pocket 3，BLE 配对以唤醒相机 WiFi AP（约 20s 后可加入）。"""
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        logger.error("未安装 bleak，无法 BLE 配对")
        return False

    found: dict = {}
    ready = asyncio.Event()

    def _on_adv(device, adv) -> None:
        name = (device.name or "") + " " + (adv.local_name or "")
        mfr = adv.manufacturer_data or {}
        looks_dji = "Osmo" in name or "Pocket" in name or "DJI" in name
        if not looks_dji:
            for company_id, payload in mfr.items():
                blob = struct.pack("<H", company_id) + payload
                if blob[:2] in (b"\xaa\x08", b"\xaa\xf7") and blob[2:4] == b"\x20\x00":
                    looks_dji = True
                    break
        if looks_dji and device.address not in found:
            logger.info("发现 BLE 设备 %s (%s)", device.name, device.address)
            found[device.address] = device
            ready.set()

    logger.info("扫描 Pocket 3 BLE（%.0fs）...", timeout)
    scanner = BleakScanner(detection_callback=_on_adv)
    await scanner.start()
    try:
        await asyncio.wait_for(ready.wait(), timeout=timeout)
        await asyncio.sleep(0.4)
    except asyncio.TimeoutError:
        logger.error("未扫描到 Pocket 3")
        await scanner.stop()
        return False
    await scanner.stop()

    device = next(iter(found.values()))
    client = BleakClient(device)
    await client.connect()
    try:
        await client.start_notify(FFF4_UUID, lambda *_args: None)
        payload = PAIR_HASH + _pack_string(PAIR_PIN)
        msg = build_ble_message(PAIR_TARGET, PAIR_ID, PAIR_TYPE, payload)
        await client.write_gatt_char(FFF5_UUID, msg, response=False)
        logger.info("已发送 BLE 配对，等待相机 WiFi AP 起来（约 20s）")
        await asyncio.sleep(2.0)
        return True
    finally:
        await client.disconnect()


def activate_wifi_ap_sync(timeout: float = 15.0) -> bool:
    return asyncio.run(activate_wifi_ap(timeout))
