"""Webcam capture and the mapping onto the TV rectangle.

The overhead camera does not see a perfect 1920x1080 image. `map_frame`
crops (or perspective-warps) the camera view so one pixel of output lines
up with one pixel of the table. Passthrough mode shows that mapping so
you can fix the crop before trusting hand positions.

Latency: request a short camera buffer and a modest resolution (default
1280x720). The display pipeline adds its own delay; the TV's motion
smoothing often costs more than this process. See the README.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from liveplay.config import AppConfig
from liveplay.errors import CaptureError, ConfigError

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


class FrameSource:
    def read_raw(self) -> np.ndarray | None:
        """Camera image before it is stretched onto the TV."""
        return self.read()

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def map_to_screen(self, raw: np.ndarray) -> np.ndarray:
        return raw

    def close(self) -> None:
        return None


def clamp_roi(
    roi: tuple[int, int, int, int], frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    if w <= 0:
        w = frame_w
    if h <= 0:
        h = frame_h
    w = max(16, min(int(w), frame_w))
    h = max(16, min(int(h), frame_h))
    x = max(0, min(int(x), frame_w - w))
    y = max(0, min(int(y), frame_h - h))
    return x, y, w, h


def nudge_roi(
    roi: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    dx: int,
    dy: int,
    dw: int,
    dh: int,
    step: int = 8,
) -> tuple[int, int, int, int]:
    """Move or resize the crop. A 0 width/height means the full frame."""
    current = clamp_roi(roi, frame_w, frame_h)
    x, y, w, h = current
    return clamp_roi((x + dx * step, y + dy * step, w + dw * step, h + dh * step), frame_w, frame_h)


def map_frame(
    frame: np.ndarray,
    output_size: tuple[int, int],
    roi: tuple[int, int, int, int] = (0, 0, 0, 0),
    perspective: list | None = None,
    undistort_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return a BGR image of `output_size` (width, height) in screen space."""
    image = frame
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if undistort_maps is not None:
        image = cv2.remap(image, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)
    out_w, out_h = output_size
    if perspective is not None:
        src = np.array(perspective, dtype=np.float32)
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (out_w, out_h))
    height, width = image.shape[:2]
    x, y, w, h = clamp_roi(roi, width, height)
    cropped = image[y : y + h, x : x + w]
    if cropped.size == 0:
        raise CaptureError("ROI crop is empty. Check roi in config.json.")
    if cropped.shape[1] == out_w and cropped.shape[0] == out_h:
        return cropped
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def build_undistort_maps(
    camera_matrix: list, dist_coeffs: list, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(camera_matrix, dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ConfigError("undistort.camera_matrix must be 3x3.")
    return cv2.initUndistortRectifyMap(matrix, dist, None, matrix, size, cv2.CV_16SC2)


class CameraCapture(FrameSource):
    def __init__(self, cfg: AppConfig) -> None:
        self.output_size = (cfg.display.width, cfg.display.height)
        self.roi = cfg.roi
        self.perspective = cfg.perspective
        self.undistort_enabled = cfg.undistort.enabled
        self.camera_matrix = cfg.undistort.camera_matrix
        self.dist_coeffs = cfg.undistort.dist_coeffs
        self.raw_size: tuple[int, int] | None = None
        self._maps: tuple[np.ndarray, np.ndarray] | None = None
        self._map_size: tuple[int, int] | None = None
        index = cfg.camera.index
        print(f"[liveplay] opening camera {index}...")
        self.cap = _open_device(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
        # One buffered frame. A deep queue is how you miss the 100 ms budget
        # before a single pixel is drawn.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _try_lock_exposure(self.cap)
        self._warmup()
        ok, sample = self.cap.read()
        if not ok or sample is None:
            self.close()
            raise CaptureError(_camera_error(index))
        self.raw_size = (int(sample.shape[1]), int(sample.shape[0]))
        print(
            f"[liveplay] camera {index} delivering {self.raw_size[0]}x{self.raw_size[1]}"
        )

    def read_raw(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        self.raw_size = (int(frame.shape[1]), int(frame.shape[0]))
        self._last_raw = frame
        return frame

    def map_to_screen(self, raw: np.ndarray) -> np.ndarray:
        maps = self._maps_for((int(raw.shape[1]), int(raw.shape[0])))
        try:
            return map_frame(
                raw,
                self.output_size,
                roi=self.roi,
                perspective=self.perspective,
                undistort_maps=maps,
            )
        except cv2.error as exc:
            raise CaptureError(f"Camera mapping failed: {exc}") from exc

    def read(self) -> np.ndarray | None:
        raw = self.read_raw()
        if raw is None:
            return None
        return self.map_to_screen(raw)

    def close(self) -> None:
        cap = getattr(self, "cap", None)
        if cap is not None:
            cap.release()

    def _maps_for(self, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.undistort_enabled:
            return None
        if self._maps is None or self._map_size != size:
            assert self.camera_matrix is not None and self.dist_coeffs is not None
            self._maps = build_undistort_maps(self.camera_matrix, self.dist_coeffs, size)
            self._map_size = size
        return self._maps

    def _warmup(self) -> None:
        # Some UVC cameras return empty frames until exposure settles.
        for _ in range(20):
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size:
                return
            time.sleep(0.05)


class FakeCamera(FrameSource):
    """Gray playfield plus a moving skin-colored blob. Development only."""

    def __init__(self, width: int, height: int, gray: int) -> None:
        self.width = width
        self.height = height
        self.gray = int(gray)
        self._t = 0.0
        print("[liveplay] using a fake camera (no webcam).")

    def read_raw(self) -> np.ndarray | None:
        return self.read()

    def read(self) -> np.ndarray | None:
        frame = np.full((self.height, self.width, 3), self.gray, dtype=np.uint8)
        cx = int(self.width * (0.5 + 0.28 * np.sin(self._t)))
        cy = int(self.height * (0.5 + 0.18 * np.cos(self._t * 0.7)))
        # BGR skin-ish tone. Kept inside the default HSV gate in vision.py.
        cv2.ellipse(frame, (cx, cy), (90, 60), 0, 0, 360, (100, 140, 200), -1)
        self._t += 0.12
        return frame

    def close(self) -> None:
        return None


def open_frame_source(cfg: AppConfig) -> FrameSource:
    if cfg.fake_camera:
        return FakeCamera(cfg.display.width, cfg.display.height, cfg.calibration.gray)
    return CameraCapture(cfg)


def _open_device(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        # AVFoundation can reject a valid index if the backend flag is wrong
        # for a given OpenCV build. Retry the default backend once.
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise CaptureError(_camera_error(index))
    return cap


def _try_lock_exposure(cap: cv2.VideoCapture) -> None:
    """Best-effort manual exposure.

    Auto exposure chases the picture on the TV. Particles get bright, the
    camera darkens, the empty-table snapshot no longer matches, and the
    whole glass becomes "foreground." Many webcams ignore these properties.
    If blobs drift after a few minutes, lock exposure in the camera's own
    tool and recalibrate. See vision.py.
    """
    # UVC convention: 0.25 (or 1) is manual, 0.75 (or 3) is auto. Try both.
    for value in (0.25, 1):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)


def _camera_error(index: int) -> str:
    return (
        f"No camera at index {index}. Plug in the USB webcam, quit other apps "
        f"using it, and on macOS allow Camera access for Terminal (System "
        f"Settings → Privacy & Security → Camera). Then set camera.index or "
        f"pass --camera."
    )
