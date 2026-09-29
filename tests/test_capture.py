import unittest

import numpy as np

from liveplay.capture import build_undistort_maps, clamp_roi, map_frame, nudge_roi
from liveplay.errors import ConfigError


class CaptureMappingTest(unittest.TestCase):
    def test_roi_crop_keeps_the_right_half(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, 100:] = (255, 0, 0)
        mapped = map_frame(image, (50, 40), roi=(100, 0, 100, 100))
        self.assertEqual(mapped.shape, (40, 50, 3))
        self.assertGreater(mapped[:, :, 0].mean(), 250)
        self.assertLess(mapped[:, :, 1].mean(), 5)

    def test_full_frame_resize_shape(self) -> None:
        image = np.full((30, 40, 3), 90, dtype=np.uint8)
        mapped = map_frame(image, (20, 10), roi=(0, 0, 0, 0))
        self.assertEqual(mapped.shape, (10, 20, 3))
        self.assertGreater(mapped.mean(), 80)

    def test_perspective_output_size(self) -> None:
        image = np.full((100, 80, 3), 40, dtype=np.uint8)
        points = [[0, 0], [79, 0], [79, 99], [0, 99]]
        mapped = map_frame(image, (40, 30), perspective=points)
        self.assertEqual(mapped.shape, (30, 40, 3))

    def test_nudge_shrinks_a_full_frame_roi(self) -> None:
        roi = nudge_roi((0, 0, 0, 0), 100, 80, dx=0, dy=0, dw=-1, dh=-1, step=8)
        self.assertEqual(roi, (0, 0, 92, 72))

    def test_nudge_clamps_inside_the_frame(self) -> None:
        roi = nudge_roi((0, 0, 20, 20), 100, 80, dx=-1, dy=-1, dw=0, dh=0, step=8)
        self.assertEqual(roi[:2], (0, 0))
        self.assertEqual(clamp_roi((-10, -10, 1000, 1000), 100, 80), (0, 0, 100, 80))

    def test_undistort_maps_reject_a_bad_matrix(self) -> None:
        with self.assertRaises(ConfigError):
            build_undistort_maps([[1, 0, 0]], [0, 0, 0, 0, 0], (20, 20))

    def test_undistort_maps_can_be_applied(self) -> None:
        matrix = [[80.0, 0.0, 20.0], [0.0, 80.0, 15.0], [0.0, 0.0, 1.0]]
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]
        maps = build_undistort_maps(matrix, dist, (40, 30))
        image = np.full((30, 40, 3), 90, dtype=np.uint8)
        mapped = map_frame(image, (40, 30), undistort_maps=maps)
        self.assertEqual(mapped.shape, (30, 40, 3))


if __name__ == "__main__":
    unittest.main()
