"""Palm centers from MediaPipe Hand Landmarker.

The overhead camera sees hands, not whole people, so this is a hand model
and not a body-pose model. Each hand becomes one interaction point at the
palm: the wrist plus the four knuckle bases. That point is what the puck
follows. There is no mask and no skeleton on the glass.

The task file is local (`vision.model`, default `models/hand_landmarker.task`).
This module never downloads it. Official MediaPipe wheels are published for
Python 3.12 and the 1.x package is a py3 wheel, so 3.14 may work; if
`import mediapipe` fails, the error says so and `--vision diff+skin` still runs.

The landmarker runs on the CPU. `vision.scale` (default 0.5) shrinks the
frame first. Landmarks come back in 0..1, then are multiplied by the
full frame size, so the point stays in screen pixels.

Video mode keeps a hand identity across frames. The timestamp passed in
must increase every call. A cyan puck is a disc, not a hand, so the model
should not track the drawing. The color blobs in vision.py are the fallback
when this model is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from liveplay.config import VisionConfig
from liveplay.errors import LivePlayError
from liveplay.points import InteractionPoint

# Google's float16 hand landmarker. Shipped in the repo. Not fetched at runtime.
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Wrist, index MCP, middle MCP, ring MCP, pinky MCP.
_PALM = (0, 5, 9, 13, 17)


class HandDetector(Protocol):
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list:
        """Each item is one hand: 21 landmarks with x and y in 0..1."""

    def close(self) -> None:
        ...


class HandVision:
    """One BGR frame in, palm centers out. No empty-table snapshot."""

    def __init__(self, cfg: VisionConfig, detector: HandDetector | None = None) -> None:
        self.cfg = cfg
        self.last_warning: str | None = None
        self.last_poses: list = []
        self._stamp = 0
        self._detector = detector if detector is not None else open_landmarker(cfg)

    @property
    def needs_calibration(self) -> bool:
        return False

    @property
    def has_background(self) -> bool:
        return False

    def set_background(self, frame_bgr: np.ndarray | None) -> None:
        return

    def close(self) -> None:
        closer = getattr(self._detector, "close", None)
        if closer is not None:
            closer()

    def detect(self, frame_bgr: np.ndarray) -> list[InteractionPoint]:
        self.last_warning = None
        self.last_poses = []
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            self.last_warning = "Camera frame was not a color image."
            return []
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        image = frame_bgr
        scale = self.cfg.scale
        if scale != 1.0:
            small_w = max(1, int(round(width * scale)))
            small_h = max(1, int(round(height * scale)))
            image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Video mode rejects a repeated timestamp. 33 ms is one 30 fps step.
        self._stamp += 33
        try:
            hands = self._detector.detect(rgb, self._stamp)
        except Exception as exc:
            self.last_warning = f"Hand tracker failed: {exc}"
            return []
        points: list[InteractionPoint] = []
        for landmarks in hands:
            point = hand_center(landmarks, width, height)
            if point is not None:
                points.append(point)
        return points[: max(1, self.cfg.hands)]


def hand_center(landmarks, width: int, height: int) -> InteractionPoint | None:
    """Palm center in pixels. `landmarks` are normalized, or (x, y) pairs."""
    if len(landmarks) <= max(_PALM):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for index in _PALM:
        x, y = _xy(landmarks[index])
        xs.append(x)
        ys.append(y)
    cx = (sum(xs) / len(xs)) * width
    cy = (sum(ys) / len(ys)) * height
    all_xy = [_xy(landmark) for landmark in landmarks]
    span_x = (max(p[0] for p in all_xy) - min(p[0] for p in all_xy)) * width
    span_y = (max(p[1] for p in all_xy) - min(p[1] for p in all_xy)) * height
    size = max(8.0, 0.5 * max(span_x, span_y))
    return InteractionPoint(x=cx, y=cy, size=size)


def open_landmarker(cfg: VisionConfig) -> HandDetector:
    """Load the local task file. Raises LivePlayError if it cannot be used."""
    path = Path(cfg.model)
    if not path.is_file():
        raise LivePlayError(
            f"Hand model not found at {path}. "
            "Live Play does not download models. Restore models/hand_landmarker.task "
            f"from the project, or download it once from {HAND_MODEL_URL} . "
            "Until then, --vision diff+skin still tracks color blobs."
        )
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise LivePlayError(
            "Hand tracking needs MediaPipe (pip install mediapipe). "
            f"This interpreter is Python {version}. "
            "If that install fails, use a Python 3.12 virtualenv, "
            "or run with --vision diff+skin."
        ) from exc
    base = mp_python.BaseOptions(
        model_asset_path=str(path),
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(cfg.hands),
        min_hand_detection_confidence=float(cfg.min_confidence),
        min_hand_presence_confidence=float(cfg.min_confidence),
        min_tracking_confidence=float(cfg.min_confidence),
    )
    try:
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        raise LivePlayError(f"Could not open the hand model at {path}: {exc}") from exc
    return _MediaPipeDetector(mp, landmarker)


class _MediaPipeDetector:
    def __init__(self, mp, landmarker) -> None:
        self._mp = mp
        self._landmarker = landmarker

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        landmarks = result.hand_landmarks
        if not landmarks:
            return []
        return list(landmarks)

    def close(self) -> None:
        self._landmarker.close()


def _xy(landmark) -> tuple[float, float]:
    if isinstance(landmark, (tuple, list)):
        return float(landmark[0]), float(landmark[1])
    return float(landmark.x), float(landmark.y)
