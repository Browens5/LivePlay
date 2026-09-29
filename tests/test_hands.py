"""Palm-center math and the MediaPipe backend, without a webcam."""

import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from liveplay.config import VisionConfig
from liveplay.errors import LivePlayError
from liveplay.hands import HandVision, hand_center, open_landmarker, pack_frame, uses_metal
from liveplay.vision import BlobVision, make_backend


def _cfg(**overrides: object) -> VisionConfig:
    values: dict[str, object] = {
        "method": "mediapipe",
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
        "hands": 2,
        "min_confidence": 0.5,
        "model": Path("missing-hand.task"),
    }
    values.update(overrides)
    return VisionConfig(**values)  # type: ignore[arg-type]


def _hand(palm: tuple[float, float], tip: tuple[float, float] = (0.1, 0.1)) -> list[tuple[float, float]]:
    marks = [tip] * 21
    for index in (0, 5, 9, 13, 17):
        marks[index] = palm
    return marks


class _Fake:
    def __init__(self, hands: list) -> None:
        self.hands = hands
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list:
        self.calls.append((rgb.shape, int(timestamp_ms)))
        return self.hands

    def close(self) -> None:
        return


class MetalDelegateTest(unittest.TestCase):
    def test_only_macos_starts_the_metal_service(self) -> None:
        self.assertTrue(uses_metal("darwin"))
        self.assertFalse(uses_metal("linux"))
        self.assertFalse(uses_metal("win32"))

    def test_metal_frames_include_an_alpha_channel(self) -> None:
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.assertEqual(pack_frame(frame, metal=True).shape, (4, 5, 4))
        self.assertEqual(pack_frame(frame, metal=False).shape, (4, 5, 3))


class PalmCenterTest(unittest.TestCase):
    def test_palm_is_the_knuckle_average_in_screen_pixels(self) -> None:
        point = hand_center(_hand((0.5, 0.4), tip=(0.2, 0.1)), 200, 100)
        assert point is not None
        self.assertAlmostEqual(point.x, 100.0)
        self.assertAlmostEqual(point.y, 40.0)
        self.assertGreater(point.size, 8.0)

    def test_short_landmark_list_is_dropped(self) -> None:
        self.assertIsNone(hand_center([(0.5, 0.5)] * 5, 100, 100))


class HandVisionTest(unittest.TestCase):
    def test_two_hands_become_two_centers(self) -> None:
        fake = _Fake([_hand((0.25, 0.5)), _hand((0.75, 0.5))])
        backend = HandVision(_cfg(), fake)
        points = backend.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].x, 50.0)
        self.assertAlmostEqual(points[1].x, 150.0)
        self.assertFalse(backend.needs_calibration)
        self.assertEqual(backend.last_poses, [])

    def test_scale_shrinks_the_image_and_keeps_screen_pixels(self) -> None:
        fake = _Fake([_hand((0.5, 0.5))])
        backend = HandVision(_cfg(scale=0.5), fake)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        points = backend.detect(frame)
        self.assertEqual(fake.calls[0][0], (40, 50, 3))
        self.assertAlmostEqual(points[0].x, 50.0)
        self.assertAlmostEqual(points[0].y, 40.0)
        backend.detect(frame)
        self.assertGreater(fake.calls[1][1], fake.calls[0][1])

    def test_tracker_failure_is_a_warning(self) -> None:
        class Boom:
            def detect(self, rgb, timestamp_ms):
                raise RuntimeError("landmarker stopped")

            def close(self):
                return

        backend = HandVision(_cfg(), Boom())
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        self.assertEqual(backend.detect(frame), [])
        self.assertIn("landmarker stopped", backend.last_warning or "")

    def test_missing_model_does_not_try_to_download(self) -> None:
        cfg = _cfg(model=Path("/tmp/liveplay-no-such-hand.task"))
        with self.assertRaises(LivePlayError) as caught:
            open_landmarker(cfg)
        text = str(caught.exception)
        self.assertIn("does not download", text)
        self.assertIn("diff+skin", text)

    def test_blob_backend_is_unchanged(self) -> None:
        backend = make_backend(_cfg(method="diff+skin"), None)
        self.assertIsInstance(backend, BlobVision)

    def test_real_model_sees_no_hand_on_flat_gray(self) -> None:
        model = ROOT / "models" / "hand_landmarker.task"
        if not model.is_file():
            self.skipTest("hand model is not in the tree")
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            self.skipTest("mediapipe is not installed")
        backend = HandVision(_cfg(model=model, scale=0.5))
        try:
            frame = np.full((180, 320, 3), 90, dtype=np.uint8)
            self.assertEqual(backend.detect(frame), [])
            self.assertIsNone(backend.last_warning)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
