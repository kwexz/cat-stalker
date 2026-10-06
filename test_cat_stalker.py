import unittest

from cat_stalker import COMMAND_TIMEOUT_SECONDS, ForwardController, SharedCommand


class SafetyTests(unittest.TestCase):
    def test_stale_movement_becomes_stop(self):
        command = SharedCommand()
        command.set(10, 5, "TEST")

        self.assertEqual(command.get_for_send(), (10, 5, "TEST"))
        self.assertEqual(
            command.get_for_send(command.updated_at + COMMAND_TIMEOUT_SECONDS + 0.01),
            (0, 0, "WATCHDOG"),
        )

    def test_forward_gates_and_hysteresis(self):
        controller = ForwardController()

        def target(area, offset=0.0, state="TRACKING"):
            return {
                "state": state,
                "offset_x": offset,
                "area_ratio": area,
                "confidence": 0.9,
            }

        self.assertGreater(controller.update(target(0.05), 0), 0)
        self.assertGreater(controller.update(target(0.11), 0), 0)
        self.assertEqual(controller.update(target(0.14), 0), 0)
        self.assertEqual(controller.update(target(0.05, offset=0.5), 0), 0)
        self.assertEqual(controller.update(target(0.05, state="TEMP_LOST"), 0), 0)


if __name__ == "__main__":
    unittest.main()
