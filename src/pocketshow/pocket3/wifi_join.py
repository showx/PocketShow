from __future__ import annotations

import logging
import platform
import socket
import subprocess
import time

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def wifi_interface() -> str:
    if platform.system() != "Darwin":
        return "wlan0"
    result = _run(["networksetup", "-listallhardwareports"])
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "Wi-Fi" in line or "AirPort" in line:
            for extra in lines[i + 1 : i + 4]:
                if extra.startswith("Device:"):
                    return extra.split(":", 1)[1].strip()
    return "en0"


def current_ssid() -> str | None:
    if platform.system() == "Darwin":
        result = _run(["networksetup", "-getairportnetwork", wifi_interface()])
        if "Current Wi-Fi Network:" in result.stdout:
            return result.stdout.split(":", 1)[1].strip()
    elif platform.system() == "Linux":
        result = _run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"])
        for line in result.stdout.splitlines():
            if line.startswith("yes:"):
                return line.split(":", 1)[1]
    return None


def camera_reachable(ip: str = "192.168.2.1", timeout_s: float = 1.0) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        # 相机管理口不一定开 TCP；用 UDP 探测 + ping 兜底
        sock.close()
    except OSError:
        pass
    result = _run(["ping", "-c", "1", "-t", "2", ip] if platform.system() == "Darwin" else ["ping", "-c", "1", "-W", "2", ip])
    return result.returncode == 0


def join_wifi(ssid: str, password: str, timeout: float = 30.0) -> bool:
    if not ssid:
        logger.error("未配置相机 WiFi SSID")
        return False
    existing = current_ssid()
    if existing == ssid:
        logger.info("已在 %s 上", ssid)
        return True
    logger.info("加入 WiFi %s", ssid)
    system = platform.system()
    if system == "Darwin":
        result = _run(["networksetup", "-setairportnetwork", wifi_interface(), ssid, password])
        if result.returncode != 0:
            logger.error("networksetup 失败: %s", result.stderr.strip())
            return False
    elif system == "Linux":
        result = _run(["nmcli", "dev", "wifi", "connect", ssid, "password", password])
        if result.returncode != 0:
            logger.error("nmcli 失败: %s", result.stderr.strip())
            return False
    else:
        logger.error("不支持的系统: %s", system)
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1)
        if current_ssid() == ssid or camera_reachable():
            logger.info("已连接 %s", ssid)
            return True
    logger.error("加入 %s 超时", ssid)
    return False


def wait_for_camera(ip: str = "192.168.2.1", timeout: float = 30.0) -> bool:
    logger.info("等待相机 %s", ip)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if camera_reachable(ip):
            return True
        time.sleep(1)
    logger.error("相机不可达: %s", ip)
    return False
