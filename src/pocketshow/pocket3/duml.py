"""DJI DUML。CRC 和组包在 Rust 里，常量仍从这里导入。"""

from pocketshow.native import core

_core = core()

DUML_SOF = 0x55
CMD_TYPE_REQ = 0
CMD_TYPE_ACK = 1
CMD_TYPE_PUSH = 2
CMD_TYPE_WRITE = 4

crc8 = _core.crc8
crc16 = _core.crc16
build_duml = _core.build_duml
parse_duml = _core.parse_duml
