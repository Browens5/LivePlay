"""Webcam measurement of the framebuffer rectangle the TV actually shows."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np

from liveplay.display import fill_visible
from liveplay.errors import CalibrationError
from liveplay.geometry import (
    DisplayScan,
    load_geometry,
    render_pattern,
    save_geometry,
    solve_geometry,
)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _intersection(a, b, c, d) -> np.ndarray:
    """Where segment ab crosses segment cd."""
    ab = b - a
    cd = d - c
    denom = ab[0] * cd[1] - ab[1] * cd[0]
    t = ((c[0] - a[0]) * cd[1] - (c[1] - a[1]) * cd[0]) / denom
    return a + t * ab


def _table():
    """A webcam photo of a cropped, slightly skewed panel."""
    frame_size = (640, 360)
    crop = (80, 50, 480, 260)
    camera_size = (400, 280)
    cx, cy, cw, ch = crop
    src = np.float32(
        [
            [cx, cy],
            [cx + cw - 1, cy],
            [cx + cw - 1, cy + ch - 1],
            [cx, cy + ch - 1],
        ]
    )
    quad = np.float32(
        [
            [36, 28],
            [350, 40],
            [340, 240],
            [28, 220],
        ]
    )
    frame_to_cam = cv2.getPerspectiveTransform(src, quad)

    def shoot(pattern: np.ndarray) -> np.ndarray:
        photo = cv2.warpPerspective(
            pattern,
            frame_to_cam,
            camera_size,
            flags=cv2.INTER_NEAREST,
            borderValue=(0, 0, 0),
        )
        mask = np.zeros(photo.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        photo[mask == 0] = 0
        return photo

    return frame_size, crop, src, quad, shoot


def _solve_table(inset: int = 0):
    frame_size, crop, src, quad, shoot = _table()
    width, height = frame_size
    nbits = 7
    dark = _gray(shoot(render_pattern(width, height, "dark")))
    bright = _gray(shoot(render_pattern(width, height, "bright")))
    x_pairs = []
    y_pairs = []
    for bit in range(nbits - 1, -1, -1):
        x_pairs.append(
            (
                _gray(shoot(render_pattern(width, height, "x", bit, nbits, False))),
                _gray(shoot(render_pattern(width, height, "x", bit, nbits, True))),
            )
        )
        y_pairs.append(
            (
                _gray(shoot(render_pattern(width, height, "y", bit, nbits, False))),
                _gray(shoot(render_pattern(width, height, "y", bit, nbits, True))),
            )
        )
    geometry = solve_geometry(dark, bright, x_pairs, y_pairs, frame_size, nbits, inset)
    return geometry, crop, src, quad, shoot


class DisplayGeometryTest(unittest.TestCase):
    def test_scan_recovers_the_cropped_panel(self) -> None:
        geometry, crop, src, quad, shoot = _solve_table(inset=0)
        x, y, width, height = geometry.visible
        cx, cy, cw, ch = crop
        self.assertLess(abs(x - cx), 24, geometry.visible)
        self.assertLess(abs(y - cy), 24, geometry.visible)
        self.assertLess(abs((x + width) - (cx + cw)), 24, geometry.visible)
        self.assertLess(abs((y + height) - (cy + ch)), 24, geometry.visible)

        cam_center = _intersection(quad[0], quad[2], quad[1], quad[3])
        frame_center = _intersection(src[0], src[2], src[1], src[3])
        mapped = geometry.map_xy([cam_center])[0]
        self.assertLess(abs(float(mapped[0]) - float(frame_center[0])), 8)
        self.assertLess(abs(float(mapped[1]) - float(frame_center[1])), 8)

        photo = shoot(render_pattern(640, 360, "bright"))
        warped = geometry.warp(photo, border_gray=90)
        sample = warped[int(frame_center[1]), int(frame_center[0])]
        self.assertGreater(int(sample.reshape(-1)[0]), 180)
        self.assertTrue(np.all(warped[0, 0] == 90))

    def test_inset_pulls_content_off_the_edge(self) -> None:
        loose, *_ = _solve_table(inset=0)
        tight, *_ = _solve_table(inset=18)
        self.assertGreater(tight.visible[0], loose.visible[0])
        self.assertGreater(tight.visible[1], loose.visible[1])
        self.assertLess(
            tight.visible[0] + tight.visible[2],
            loose.visible[0] + loose.visible[2],
        )
        self.assertLess(
            tight.visible[1] + tight.visible[3],
            loose.visible[1] + loose.visible[3],
        )

    def test_display_scan_accepts_a_settled_frame(self) -> None:
        frame_size, _crop, _src, _quad, shoot = _table()
        scan = DisplayScan(frame_size, bits=7, settle=0.0, inset=0)
        shown = None
        geometry = None
        blank = np.zeros((280, 400, 3), np.uint8)
        now = 1.0
        for _ in range(scan.total + 2):
            incoming = blank if shown is None else shoot(shown)
            shown, geometry = scan.tick(incoming, now)
            if scan.error:
                self.fail(scan.error)
            if geometry is not None:
                break
        self.assertIsNotNone(geometry)
        assert geometry is not None
        _x, _y, width, height = geometry.visible
        self.assertGreater(width, 400)
        self.assertGreater(height, 200)
        mapped = geometry.map_xy(np.array([[40.0, 30.0]], dtype=np.float32))
        self.assertGreater(float(mapped[0, 0]), 70)

    def test_unreadable_stripes_are_an_error(self) -> None:
        scan = DisplayScan((200, 120), bits=4, settle=0.0, inset=0)
        blank = np.zeros((60, 80, 3), np.uint8)
        shown = None
        for _ in range(scan.total + 2):
            shown, _geometry = scan.tick(blank, 0.0)
            if scan.error:
                break
        self.assertIsNotNone(scan.error)
        self.assertIn("webcam", scan.error or "")

    def test_save_and_load_roundtrip(self) -> None:
        geometry, *_ = _solve_table(inset=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib" / "display_geometry.json"
            save_geometry(path, geometry)
            loaded = load_geometry(path, geometry.frame_size)
            assert loaded is not None
            self.assertEqual(loaded.visible, geometry.visible)
            self.assertEqual(loaded.camera_size, geometry.camera_size)
            self.assertTrue(np.allclose(loaded.homography, geometry.homography))
            self.assertIsNone(load_geometry(path, (100, 100)))
            path.write_text("{")
            with self.assertRaises(CalibrationError):
                load_geometry(path, geometry.frame_size)

    def test_playfield_stays_inside_the_lit_rectangle(self) -> None:
        import pygame

        pygame.display.init()
        try:
            surface = pygame.display.set_mode((640, 360))
            visible = (80, 50, 480, 260)
            fill_visible(surface, visible, 90)
            self.assertEqual(surface.get_at((4, 4))[:3], (0, 0, 0))
            self.assertEqual(surface.get_at((639, 359))[:3], (0, 0, 0))
            self.assertEqual(surface.get_at((300, 180))[:3], (90, 90, 90))
            self.assertEqual(surface.get_at((80, 50))[:3], (90, 90, 90))
        finally:
            pygame.display.quit()


if __name__ == "__main__":
    unittest.main()
