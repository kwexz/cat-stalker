"""Hardware-free regression checks for safety-critical plumbing.

These tests use only the Python standard library. They must never import
Ultralytics, open a camera, or create a DreameVacuum instance.
"""

import os
import tempfile
import unittest
from io import StringIO
from unittest import mock

from session import auto_session, follow, protocol
from session.collector import Collector
from session.robot_link import LatestCommand


class ArgumentSafetyTests(unittest.TestCase):
    def test_observe_is_default_and_no_robot_is_explicit(self):
        args = auto_session.parse_args([])
        self.assertEqual(args.mode, "observe")
        self.assertFalse(args.no_robot)

        args = auto_session.parse_args(["--mode", "observe", "--no-robot"])
        self.assertTrue(args.no_robot)

    def test_no_robot_takes_precedence_over_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(
                os.environ,
                {"DREAME_TOKEN": "not-a-real-token", "DREAME_IP": "192.0.2.10"},
                clear=False,
            ):
                with mock.patch.object(
                    auto_session,
                    "connect_with_wake_retries",
                    side_effect=AssertionError("robot connection attempted"),
                ):
                    with mock.patch.object(auto_session, "DiscoveryResponder") as discovery:
                        with mock.patch.object(auto_session.Collector, "serve"):
                            with mock.patch("sys.stdout", new=StringIO()):
                                # A zero-duration session exits immediately after startup.
                                rc = auto_session.main(
                                    [
                                        "--mode",
                                        "observe",
                                        "--no-robot",
                                        "--duration-min",
                                        "0",
                                        "--session-root",
                                        root,
                                    ]
                                )
        self.assertEqual(rc, 0)
        discovery.assert_called_once()


class LatestCommandTests(unittest.TestCase):
    def test_nonzero_command_expires_to_stop(self):
        command = LatestCommand()
        command.set(12, -7, "TEST")
        command.updated_at -= 10.0
        self.assertEqual(command.get_for_send(), (0, 0, "WATCHDOG"))

    def test_zero_command_does_not_become_movement(self):
        command = LatestCommand()
        command.set(0, 0, "OBSERVE")
        command.updated_at -= 10.0
        self.assertEqual(command.get_for_send(), (0, 0, "OBSERVE"))


class ControllerTests(unittest.TestCase):
    def test_target_lock_requires_multiple_hits(self):
        lock = follow.TargetLock()
        now = 100.0
        self.assertEqual(lock.update(True, 0.2, 0.04, 0.8, now)["state"], "SEARCHING")
        self.assertEqual(
            lock.update(True, 0.2, 0.04, 0.8, now + 0.1)["state"], "SEARCHING"
        )
        self.assertEqual(
            lock.update(True, 0.2, 0.04, 0.8, now + 0.2)["state"], "TRACKING"
        )

    def test_forward_controller_stops_for_stale_or_off_center_target(self):
        lock = follow.TargetLock()
        target = None
        for index in range(follow.LOCK_REQUIRED_HITS):
            target = lock.update(True, 0.0, 0.04, 0.8, 100.0 + index * 0.1)

        steering = follow.SteeringController()
        controller = follow.ForwardController()
        self.assertEqual(
            follow.decide(target, steering, controller, follow.MAX_FRAME_AGE_S + 0.01),
            (0, 0, "STALE_STOP"),
        )

        off_center = dict(target, offset_x=follow.FORWARD_MAX_OFFSET + 0.01)
        self.assertEqual(controller.update(off_center, follow.MAX_AUTO_ROTATION), 0)


class ProtocolTests(unittest.TestCase):
    def test_telemetry_schema_contains_controller_inputs(self):
        required = {
            "state",
            "confidence",
            "offset_x",
            "area_ratio",
            "inference_ms",
            "analysis_fps",
            "backend",
        }
        self.assertTrue(required.issubset(protocol.PHONE_CSV_HEADER))


class CollectorTests(unittest.TestCase):
    def test_telemetry_rows_are_persisted_without_devices(self):
        with tempfile.TemporaryDirectory() as root:
            collector = Collector(root)
            row = {
                "ts": 123.0,
                "state": "DETECTED",
                "confidence": 0.75,
                "offset_x": -0.2,
                "area_ratio": 0.04,
                "inference_ms": 110.0,
                "analysis_fps": 8.2,
                "inference_fps": 8.1,
                "backend": "TEST",
                "battery_pct": 90,
                "disk_free_mb": 1000,
            }
            collector.append_telemetry(
                {"device_id": "test-phone", "session_id": "test", "rows": [row]}
            )
            collector._phone_csv.close()

            with open(
                os.path.join(root, "phone_telemetry.csv"), encoding="utf-8"
            ) as handle:
                contents = handle.read()

        self.assertEqual(collector.rows_received, 1)
        self.assertEqual(collector.last_row["state"], "DETECTED")
        self.assertIn("state,confidence,offset_x,area_ratio", contents)
        self.assertIn("DETECTED,0.75,-0.2,0.04", contents)


if __name__ == "__main__":
    unittest.main()
