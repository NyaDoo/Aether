"""
WebSocket \u96a7\u9053\u4e8c\u8fdb\u5236\u5e27\u534f\u8bae

\u5e27\u683c\u5f0f:
| stream_id (4B) | msg_type (1B) | flags (1B) | payload_len (4B) | payload (NB) |

\u7528\u4e8e Aether \u4e0e aether-proxy \u4e4b\u95f4\u7684 WebSocket \u96a7\u9053\u591a\u8def\u590d\u7528\u901a\u4fe1\u3002
"""

import struct
from enum import IntEnum
from typing import Self

_U64_MAX = (1 << 64) - 1

HEADER_SIZE = 10  # 4 + 1 + 1 + 4 bytes


class MsgType(IntEnum):
    """\u6d88\u606f\u7c7b\u578b"""

    REQUEST_HEADERS = 0x01  # Aether -> Proxy: \u8bf7\u6c42\u5143\u6570\u636e (JSON)
    REQUEST_BODY = 0x02  # Aether -> Proxy: \u8bf7\u6c42\u4f53
    RESPONSE_HEADERS = 0x03  # Proxy -> Aether: \u54cd\u5e94\u72b6\u6001\u7801 + headers (JSON)
    RESPONSE_BODY = 0x04  # Proxy -> Aether: \u54cd\u5e94\u4f53\uff08\u6d41\u5f0f\u5206\u5757\uff09
    STREAM_END = 0x05  # \u53cc\u5411: \u6d41\u7ed3\u675f
    STREAM_ERROR = 0x06  # \u53cc\u5411: \u6d41\u9519\u8bef

    PING = 0x10  # \u53cc\u5411: \u5fc3\u8df3 (stream_id=0)
    PONG = 0x11  # \u53cc\u5411: \u5fc3\u8df3\u54cd\u5e94 (stream_id=0)
    GOAWAY = 0x12  # \u53cc\u5411: \u4f18\u96c5\u5173\u95ed (stream_id=0)
    HEARTBEAT_DATA = 0x13  # Proxy -> Aether: \u6307\u6807\u4e0a\u62a5
    HEARTBEAT_ACK = 0x14  # Aether -> Proxy: \u5fc3\u8df3\u786e\u8ba4 + \u8fdc\u7a0b\u914d\u7f6e
    NODE_STATUS = 0x15  # Tunnel relay -> Worker: \u8282\u70b9\u8fde\u63a5\u72b6\u6001\u5e7f\u64ad


class FrameFlags:
    """\u5e27\u6807\u5fd7\u4f4d"""

    END_STREAM = 0x01
    GZIP_COMPRESSED = 0x02


class Frame:
    """WebSocket \u96a7\u9053\u5e27"""

    __slots__ = ("stream_id", "msg_type", "flags", "payload")

    def __init__(
        self,
        stream_id: int,
        msg_type: MsgType,
        flags: int = 0,
        payload: bytes = b"",
    ) -> None:
        self.stream_id = stream_id
        self.msg_type = msg_type
        self.flags = flags
        self.payload = payload

    def encode(self) -> bytes:
        header = struct.pack(
            "!IBBI",
            self.stream_id,
            self.msg_type,
            self.flags,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def decode(cls, data: bytes) -> Self:
        if len(data) < HEADER_SIZE:
            raise ValueError(
                f"\u5e27\u6570\u636e\u592a\u77ed: \u9700\u8981 {HEADER_SIZE} \u5b57\u8282, \u5b9e\u9645 {len(data)}"
            )
        stream_id, msg_type_raw, flags, payload_len = struct.unpack("!IBBI", data[:HEADER_SIZE])
        expected_total = HEADER_SIZE + payload_len
        if len(data) < expected_total:
            raise ValueError(
                f"\u5e27\u6570\u636e\u4e0d\u5b8c\u6574: \u9700\u8981 {expected_total} \u5b57\u8282, \u5b9e\u9645 {len(data)}"
            )
        try:
            msg_type = MsgType(msg_type_raw)
        except ValueError:
            raise ValueError(f"\u672a\u77e5\u6d88\u606f\u7c7b\u578b: 0x{msg_type_raw:02x}")
        payload = data[HEADER_SIZE:expected_total]
        return cls(stream_id, msg_type, flags, payload)

    @property
    def is_end_stream(self) -> bool:
        return bool(self.flags & FrameFlags.END_STREAM)

    @property
    def is_gzip(self) -> bool:
        return bool(self.flags & FrameFlags.GZIP_COMPRESSED)

    def __repr__(self) -> str:
        return (
            f"Frame(stream={self.stream_id}, type={self.msg_type.name}, "
            f"flags=0x{self.flags:02x}, payload_len={len(self.payload)})"
        )


def normalize_heartbeat_id(value: object) -> int | None:
    """Normalize heartbeat_id to u64-compatible int for Rust ACK parsing."""
    if isinstance(value, bool) or value is None:
        return None
    parsed: int | None = None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if value.is_integer():
            parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            try:
                parsed = int(stripped)
            except ValueError:
                parsed = None
    if parsed is None or parsed < 0 or parsed > _U64_MAX:
        return None
    return parsed
