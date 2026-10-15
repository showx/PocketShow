"""加载 Rust 控制核心。检测和网页仍走 Python。"""


def core():
    try:
        from pocketshow import _core
    except ImportError as exc:
        raise ImportError(
            "未找到 pocketshow._core。跟拍控制已改为 Rust，请先安装 Rust 工具链，再执行 pip install -e ."
        ) from exc
    return _core
