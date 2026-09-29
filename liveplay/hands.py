"""Palm centers from MediaPipe Hand Landmarker.

The overhead camera sees hands, not whole people, so this is a hand model
and not a body-pose model. Each hand becomes one interaction point at the
palm: the wrist plus the four knuckle bases. That point is what the puck
follows. There is no mask and no skeleton on the glass.

The task file is local (`vision.model`, default `models/hand_landmarker.task`).
This module never downloads it. Official MediaPipe wheels are published for
Python 3.12 and the 1.x package is a py3 wheel, so 3.14 may work; if
`import mediapipe` fails, the error says so and `--vision diff+skin` still runs.

Linux uses the CPU delegate. macOS wheels still open a Metal helper
inside the detector, and that helper abort()s unless the graph was
started with the GPU delegate. On a Mac the frames are therefore RGBA,
which is the only format that helper accepts. `vision.scale` (default
0.5) shrinks the frame first. Landmarks come back in 0..1, then are
multiplied by the full frame size, so the point stays in screen pixels.

The Mac GPU path also leaks about three BGRA surfaces per frame
(mediapipe#5267). After a few minutes `CVPixelBufferCreate` fails with
-6662 and the check abort()s the process. Rebuilding the landmarker
drops that texture cache. The long side of a Mac frame is capped first,
because the leaked surfaces are the size of the input and the model
scales the picture down internally anyway.

Video mode keeps a hand identity across frames. The timestamp passed in
must increase every call. A cyan puck is a disc, not a hand, so the model
should not track the drawing. The color blobs in vision.py are the fallback
when this model is missing.
"""

from __future__ import annotations

import os
import sys
import time
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

# Three BGRA buffers per Mac GPU frame. Measured against a 1280×720
# hand landmarker: about 10.6 MB, which is 12 bytes of footprint per
# input pixel. 512 MiB is well short of the allocation failure.
_METAL_LEAK_BYTES_PER_PIXEL = 12
_METAL_LEAK_BUDGET = 512 * 1024 * 1024
_METAL_LONG_SIDE = 640
_METAL_REFRESH_S = 10.0


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
        # Video mode rejects a repeated timestamp. 33 ms is one 30 fps step.
        self._stamp += 33
        try:
            hands = self._detector.detect(image, self._stamp)
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
    metal = uses_metal(sys.platform)
    model = path.read_bytes()
    try:
        landmarker = _open_hand_landmarker(mp_python, vision, model, cfg, metal)
    except Exception as exc:
        raise LivePlayError(f"Could not open the hand model at {path}: {exc}") from exc
    return _MediaPipeDetector(
        mp,
        landmarker,
        metal=metal,
        reopen=lambda: _open_hand_landmarker(mp_python, vision, model, cfg, metal),
    )


def _open_hand_landmarker(mp_python, vision, model: bytes, cfg: VisionConfig, metal: bool):
    delegate = (
        mp_python.BaseOptions.Delegate.GPU
        if metal
        else mp_python.BaseOptions.Delegate.CPU
    )
    base = mp_python.BaseOptions(model_asset_buffer=model, delegate=delegate)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(cfg.hands),
        min_hand_detection_confidence=float(cfg.min_confidence),
        min_hand_presence_confidence=float(cfg.min_confidence),
        min_tracking_confidence=float(cfg.min_confidence),
    )
    return vision.HandLandmarker.create_from_options(options)


def metal_frame_budget(width: int, height: int) -> int:
    """How many Mac GPU frames to run before rebuilding the landmarker."""
    leaked = max(1, int(width) * int(height) * _METAL_LEAK_BYTES_PER_PIXEL)
    frames = _METAL_LEAK_BUDGET // leaked
    return max(12, min(int(frames), 300))


def limit_metal_frame(frame_bgr: np.ndarray, long_side: int = _METAL_LONG_SIDE) -> np.ndarray:
    """Shrink a Mac frame so the leaked surfaces stay small.

    Landmarks are normalized, so a smaller input does not change the
    pixel the games follow. The model resizes again internally.
    """
    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    longest = max(height, width)
    if longest <= long_side or longest <= 0:
        return frame_bgr
    scale = long_side / longest
    sized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(frame_bgr, sized, interpolation=cv2.INTER_AREA)


def uses_metal(platform: str) -> bool:
    """True when this OS build abort()s unless the Metal service is on.

    `TensorsToDetectionsCalculator` on the macOS wheel always constructs
    `DrishtiMetalHelper`. The helper checks for the graph's GPU service
    and abort()s if it is missing. The CPU delegate never registers that
    service. The GPU delegate does.
    """
    return platform == "darwin"


def pack_frame(frame_bgr: np.ndarray, metal: bool) -> np.ndarray:
    """BGR frame to the layout MediaPipe will accept.

    Metal rejects RGB (`unsupported ImageFrame format`) and wants an
    alpha channel. The CPU delegate accepts RGB.
    """
    if metal:
        converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    else:
        converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(converted)


class _MediaPipeDetector:
    def __init__(
        self,
        mp,
        landmarker,
        metal: bool,
        reopen,
        frame_budget=metal_frame_budget,
        refresh_s: float = _METAL_REFRESH_S,
    ) -> None:
        self._mp = mp
        self._landmarker = landmarker
        self._metal = metal
        self._reopen = reopen
        self._frame_budget = frame_budget
        self._refresh_s = refresh_s
        self._served = 0
        self._opened_at = time.monotonic()
        self._warned = False

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list:
        frame = limit_metal_frame(frame_bgr) if self._metal else frame_bgr
        self._maybe_refresh(frame)
        pixels = pack_frame(frame, self._metal)
        image_format = self._mp.ImageFormat.SRGBA if self._metal else self._mp.ImageFormat.SRGB
        image = self._mp.Image(image_format=image_format, data=pixels)
        result = None
        try:
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
            landmarks = getattr(result, "hand_landmarks", None)
            return _copy_hands(landmarks)
        finally:
            # Drop the packet before the next frame. On a Mac the texture
            # cache still keeps the surface until `_maybe_refresh` rebuilds
            # the landmarker.
            del result
            del image
            self._served += 1

    def close(self) -> None:
        self._landmarker.close()

    def _maybe_refresh(self, frame: np.ndarray) -> None:
        if not self._metal or self._served <= 0:
            return
        height, width = int(frame.shape[0]), int(frame.shape[1])
        budget = self._frame_budget(width, height)
        aged = (time.monotonic() - self._opened_at) >= self._refresh_s
        if self._served < budget and not aged:
            return
        try:
            fresh = self._reopen()
        except Exception as exc:
            # Keep the current graph. The next frame tries again. One
            # failed rebuild must not turn into a frame with no tracker.
            if not self._warned:
                print(f"[liveplay] could not refresh the hand tracker ({exc}).")
                self._warned = True
            return
        old = self._landmarker
        self._landmarker = fresh
        self._served = 0
        self._opened_at = time.monotonic()
        self._warned = False
        try:
            old.close()
        except Exception:
            pass


def _copy_hands(landmarks) -> list:
    """Plain coordinates, so the MediaPipe packet can be released now."""
    if not landmarks:
        return []
    hands = []
    for hand in landmarks:
        copied = []
        for landmark in hand:
            x, y = _xy(landmark)
            copied.append((x, y))
        hands.append(copied)
    return hands


def _xy(landmark) -> tuple[float, float]:
    if isinstance(landmark, (tuple, list)):
        return float(landmark[0]), float(landmark[1])
    return float(landmark.x), float(landmark.y)
