"""Shared wire protocol between the P30 app and the PC collector.

Kept in one place so the Kotlin uploader and this collector agree.
All timestamps are Unix seconds (float) in the sender clock domain;
the collector also stores its own receive timestamp.
"""

HELLO_PATH = "/api/v1/hello"
TELEMETRY_PATH = "/api/v1/telemetry"
FRAME_PATH = "/api/v1/frame"
SESSION_PATH = "/api/v1/session"

PHONE_CSV_HEADER = [
    "phone_ts",
    "recv_ts",
    "device_id",
    "state",
    "confidence",
    "offset_x",
    "area_ratio",
    "inference_ms",
    "analysis_fps",
    "inference_fps",
    "backend",
    "battery_pct",
    "disk_free_mb",
]

ROBOT_CSV_HEADER = [
    "recv_ts",
    "velocity",
    "rotation",
    "source",
    "rtt_ms",
    "failures",
]

# Phone -> collector heartbeat freshness budget (seconds).
PHONE_HEARTBEAT_GRACE_S = 15.0
PHONE_HEARTBEAT_FATAL_S = 60.0

# Cap for stored session frames on PC disk (bytes). Oldest evicted first.
FRAME_DIR_CAP_BYTES = 5 * 1024 * 1024 * 1024

# Zero-touch discovery: phone broadcasts, collector replies with its port.
DISCOVERY_PORT = 8766
DISCOVER_MSG = b"catstalker-discover-v1"
REPLY_PREFIX = b"catstalker-collector-v1 "
