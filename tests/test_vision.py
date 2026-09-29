import unittest

import cv2
import numpy as np

from liveplay.config import VisionConfig
from liveplay.points import InteractionPoint, PointTracker
from liveplay.soccer import BALL, GOAL_LEFT, GOAL_RIGHT, PUCK, PUCK_CORE
from liveplay.vision import BlobVision


def _vision(**overrides: object) -> VisionConfig:
    values: dict[str, object] = {
        "method": "diff+skin",
        "diff_threshold": 28,
        "min_area": 400,
        "max_area_fraction": 0.4,
        "max_blobs": 6,
        "track_dark_blobs": False,
        "dark_delta": 35,
        "morph_kernel": 3,
        "scale": 1.0,
        "smoothing": 1.0,
        "match_distance": 300,
    }
    values.update(overrides)
    return VisionConfig(**values)  # type: ignore[arg-type]


def _scene() -> tuple[np.ndarray, np.ndarray]:
    background = np.full((180, 320, 3), 90, dtype=np.uint8)
    frame = background.copy()
    # Skin-toned hand near the center. BGR chosen to land in the HSV gate.
    cv2.ellipse(frame, (160, 90), (28, 20), 0, 0, 360, (100, 140, 200), -1)
    # Bright cyan disc. Difference says yes, skin says no.
    cv2.circle(frame, (40, 40), 16, (255, 255, 40), -1)
    return background, frame


class VisionTest(unittest.TestCase):
    def test_diff_and_skin_tracks_the_hand_not_the_spark(self) -> None:
        background, frame = _scene()
        backend = BlobVision(_vision(), background)
        points = backend.detect(frame)
        self.assertEqual(len(points), 1, backend.last_warning)
        self.assertLess(abs(points[0].x - 160), 20)
        self.assertLess(abs(points[0].y - 90), 20)
        self.assertIsNone(backend.last_warning)

    def test_diff_alone_also_sees_the_bright_spark(self) -> None:
        background, frame = _scene()
        backend = BlobVision(_vision(method="diff"), background)
        points = backend.detect(frame)
        self.assertGreaterEqual(len(points), 2)

    def test_skin_works_without_a_calibration_frame(self) -> None:
        _background, frame = _scene()
        backend = BlobVision(_vision(method="skin"), None)
        points = backend.detect(frame)
        self.assertEqual(len(points), 1)
        self.assertFalse(backend.needs_calibration)

    def test_dark_toy_needs_the_flag(self) -> None:
        background = np.full((180, 320, 3), 90, dtype=np.uint8)
        frame = background.copy()
        cv2.circle(frame, (200, 100), 22, (15, 15, 15), -1)
        quiet = BlobVision(_vision(track_dark_blobs=False), background)
        self.assertEqual(quiet.detect(frame), [])
        toys = BlobVision(_vision(track_dark_blobs=True), background)
        points = toys.detect(frame)
        self.assertEqual(len(points), 1)
        self.assertLess(abs(points[0].x - 200), 20)

    def test_missing_calibration_is_a_warning_not_a_hang(self) -> None:
        frame = np.full((80, 80, 3), 90, dtype=np.uint8)
        backend = BlobVision(_vision(method="diff"), None)
        self.assertEqual(backend.detect(frame), [])
        self.assertIsNotNone(backend.last_warning)
        self.assertIn("calibration", (backend.last_warning or "").lower())

    def test_huge_foreground_is_rejected(self) -> None:
        background = np.full((80, 80, 3), 90, dtype=np.uint8)
        frame = np.full((80, 80, 3), 220, dtype=np.uint8)
        backend = BlobVision(_vision(method="diff", max_area_fraction=0.2), background)
        self.assertEqual(backend.detect(frame), [])
        self.assertIn("too much", (backend.last_warning or "").lower())

    def test_half_resolution_stays_in_screen_coordinates(self) -> None:
        background, frame = _scene()
        backend = BlobVision(_vision(scale=0.5, min_area=500), background)
        points = backend.detect(frame)
        self.assertEqual(len(points), 1, backend.last_warning)
        self.assertLess(abs(points[0].x - 160), 25)
        self.assertLess(abs(points[0].y - 90), 25)

    def test_soccer_colors_fail_the_skin_gate(self) -> None:
        # The camera sees the goals, the ball, and the puck. Those drawings
        # have to miss the skin bands or the color tracker chases the game.
        frame = np.full((80, 320, 3), 90, dtype=np.uint8)
        for index, rgb in enumerate((GOAL_LEFT, GOAL_RIGHT, BALL, PUCK, PUCK_CORE)):
            bgr = (rgb[2], rgb[1], rgb[0])
            x = 8 + index * 62
            cv2.rectangle(frame, (x, 12), (x + 48, 68), bgr, thickness=-1)
        backend = BlobVision(_vision(method="skin", min_area=80), None)
        self.assertEqual(backend.detect(frame), [])


class TrackerTest(unittest.TestCase):
    def test_velocity_is_pixels_per_second(self) -> None:
        tracker = PointTracker(smoothing=1.0, hold_frames=0)
        first = tracker.update([InteractionPoint(0, 0, 10)], dt=0.5)
        self.assertTrue(first[0].is_new)
        second = tracker.update([InteractionPoint(10, 0, 10)], dt=0.5)
        self.assertFalse(second[0].is_new)
        self.assertAlmostEqual(second[0].vx, 20.0)
        self.assertAlmostEqual(second[0].vy, 0.0)

    def test_dropped_frames_coast_briefly(self) -> None:
        tracker = PointTracker(smoothing=1.0, hold_frames=2)
        tracker.update([InteractionPoint(5, 5, 10)], dt=0.1)
        self.assertEqual(len(tracker.update([], dt=0.1)), 1)
        self.assertEqual(len(tracker.update([], dt=0.1)), 1)
        self.assertEqual(tracker.update([], dt=0.1), [])


if __name__ == "__main__":
    unittest.main()
