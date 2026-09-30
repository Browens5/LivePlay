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
-6662 and the check abort()s the process. On a Mac the landmarker
therefore runs in its own process. A spare process is loaded beside it
and takes over before that cache is full, so the game thread never
waits on the reload and a crash in the tracker does not abort the game.
The long side of a Mac frame is capped first, because the leaked
surfaces are the size of the input and the model scales the picture
down internally anyway.

Video mode keeps a hand identity across frames. The timestamp passed in
must increase every call. A cyan puck is a disc, not a hand, so the model
should not track the drawing. The color blobs in vision.py are the fallback
when this model is missing.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
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
# input pixel. The real failure arrives after a few minutes, so the
# spare takes over on a short cycle, and this byte ceiling is only a
# backstop if frames are larger than expected.
_METAL_LEAK_BYTES_PER_PIXEL = 12
_METAL_LEAK_BUDGET = 3 * 1024 * 1024 * 1024
_METAL_LONG_SIDE = 640
_METAL_CYCLE_S = 15.0
_METAL_SPARE_LEAD_S = 8.0


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
    if metal:
        # A second process owns the Metal cache. Reloading it happens off
        # the game thread, and an abort inside MediaPipe stays in the child.
        def opener() -> _ProcessSlot:
            return _ProcessSlot(
                target=_landmarker_process,
                args=(model, int(cfg.hands), float(cfg.min_confidence)),
            )

        try:
            slot = opener()
        except Exception as exc:
            raise LivePlayError(f"Could not open the hand model at {path}: {exc}") from exc
        return _MediaPipeDetector(mp, slot, metal=True, reopen=opener)
    try:
        landmarker = _open_hand_landmarker(mp_python, vision, model, cfg, metal)
    except Exception as exc:
        raise LivePlayError(f"Could not open the hand model at {path}: {exc}") from exc
    return _MediaPipeDetector(
        mp,
        landmarker,
        metal=False,
        reopen=lambda: _open_hand_landmarker(mp_python, vision, model, cfg, False),
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


class _LandmarkerSettings:
    """The fields `_open_hand_landmarker` reads. Safe to build in a child."""

    def __init__(self, hands: int, min_confidence: float) -> None:
        self.hands = hands
        self.min_confidence = min_confidence


def metal_frame_budget(width: int, height: int) -> int:
    """How many Mac GPU frames fit under the leak ceiling."""
    leaked = max(1, int(width) * int(height) * _METAL_LEAK_BYTES_PER_PIXEL)
    frames = _METAL_LEAK_BUDGET // leaked
    return max(12, int(frames))


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


def _infer_in_process(mp, landmarker, frame_bgr: np.ndarray, timestamp_ms: int, metal: bool) -> list:
    """One in-process detect. The packet is dropped before the next frame."""
    pixels = pack_frame(frame_bgr, metal)
    image_format = mp.ImageFormat.SRGBA if metal else mp.ImageFormat.SRGB
    image = mp.Image(image_format=image_format, data=pixels)
    result = None
    try:
        result = landmarker.detect_for_video(image, int(timestamp_ms))
        return _copy_hands(getattr(result, "hand_landmarks", None))
    finally:
        del result
        del image


def _landmarker_process(conn, model: bytes, hands: int, confidence: float) -> None:
    """MediaPipe in a child process so its Metal cache cannot abort the game."""
    landmarker = None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        settings = _LandmarkerSettings(hands, confidence)
        landmarker = _open_hand_landmarker(mp_python, vision, model, settings, True)
        conn.send(("ready", None))
        while True:
            message = conn.recv()
            if message is None:
                break
            frame, timestamp_ms = message
            try:
                found = _infer_in_process(mp, landmarker, frame, int(timestamp_ms), True)
            except Exception as exc:
                conn.send(("err", str(exc)))
                continue
            conn.send(("ok", found))
    except Exception as exc:
        try:
            conn.send(("err", str(exc)))
        except Exception:
            pass
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass


def _close_slot(slot) -> None:
    if slot is None:
        return
    try:
        slot.close()
    except Exception:
        pass


class _ProcessSlot:
    """One landmarker process. `submit`/`poll` never block the game thread."""

    def __init__(self, target, args: tuple) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._proc = context.Process(target=target, args=(child, *args), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent
        self._waiting = False
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self._conn.poll(0.1):
                kind, payload = self._conn.recv()
                if kind != "ready":
                    self.close()
                    raise RuntimeError(payload or "hand tracker did not start")
                return
            if not self._proc.is_alive():
                self.close()
                raise RuntimeError("hand tracker stopped while starting")
        self.close()
        raise RuntimeError("hand tracker did not start")

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        if self._waiting:
            return
        if not self._proc.is_alive():
            raise RuntimeError("hand tracker stopped")
        self._conn.send((np.ascontiguousarray(frame_bgr), int(timestamp_ms)))
        self._waiting = True

    def poll(self):
        """The latest hands, or None if this frame is still running."""
        if not self._waiting:
            return None
        if not self._conn.poll(0):
            if not self._proc.is_alive():
                self._waiting = False
                raise RuntimeError("hand tracker stopped")
            return None
        self._waiting = False
        kind, payload = self._conn.recv()
        if kind != "ok":
            raise RuntimeError(payload or "hand tracker failed")
        return payload

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list:
        """Block until this frame is done. Used to warm a spare, not to play."""
        self.submit(frame_bgr, timestamp_ms)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            found = self.poll()
            if found is not None:
                return found
            time.sleep(0.01)
        raise TimeoutError("hand tracker timed out")

    def close(self) -> None:
        try:
            if self._proc.is_alive():
                self._conn.send(None)
        except Exception:
            pass
        self._proc.join(0.2)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(0.2)
        try:
            self._conn.close()
        except Exception:
            pass


class _MediaPipeDetector:
    def __init__(
        self,
        mp,
        landmarker,
        metal: bool,
        reopen,
        frame_budget=metal_frame_budget,
        cycle_s: float = _METAL_CYCLE_S,
        spare_lead_s: float = _METAL_SPARE_LEAD_S,
    ) -> None:
        self._mp = mp
        self._landmarker = landmarker
        self._metal = metal
        self._reopen = reopen
        self._frame_budget = frame_budget
        self._cycle_s = cycle_s
        self._spare_lead_s = spare_lead_s
        self._served = 0
        self._opened_at = time.monotonic()
        self._warned = False
        self._last: list = []
        self._got_result = False
        self._closed = False
        self._starting = False
        self._ready = None
        self._lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list:
        frame = limit_metal_frame(frame_bgr) if self._metal else frame_bgr
        if self._metal:
            self._install_spare(frame, force=self._landmarker is None)
        if isinstance(self._landmarker, _ProcessSlot):
            return self._detect_async(frame, timestamp_ms)
        hands = self._infer(self._landmarker, frame, timestamp_ms)
        self._remember(hands)
        if self._metal:
            self._ensure_spare(frame)
        return hands

    def close(self) -> None:
        with self._lock:
            self._closed = True
            ready = self._ready
            self._ready = None
        _close_slot(ready)
        current = self._landmarker
        self._landmarker = None
        _close_slot(current)

    def _detect_async(self, frame: np.ndarray, timestamp_ms: int) -> list:
        slot = self._landmarker
        if not isinstance(slot, _ProcessSlot):
            self._ensure_spare(frame, force=True)
            return list(self._last)
        try:
            ready = slot.poll()
            if ready is not None:
                self._remember(ready)
            slot.submit(frame, timestamp_ms)
        except Exception as exc:
            self._warn(exc)
            self._drop_current()
            self._ensure_spare(frame, force=True)
            return list(self._last)
        if not self._got_result:
            self._wait_for_first(slot)
        self._ensure_spare(frame)
        return list(self._last)

    def _wait_for_first(self, slot: _ProcessSlot) -> None:
        deadline = time.monotonic() + 2.0
        while not self._got_result and time.monotonic() < deadline:
            try:
                ready = slot.poll()
            except Exception as exc:
                self._warn(exc)
                self._drop_current()
                return
            if ready is not None:
                self._remember(ready)
                return
            time.sleep(0.005)

    def _infer(self, landmarker, frame: np.ndarray, timestamp_ms: int) -> list:
        if landmarker is None:
            return list(self._last)
        if hasattr(landmarker, "detect_for_video"):
            return _infer_in_process(self._mp, landmarker, frame, timestamp_ms, self._metal)
        return landmarker.detect(frame, int(timestamp_ms))

    def _remember(self, hands: list) -> None:
        self._last = hands
        self._served += 1
        self._got_result = True

    def _install_spare(self, frame: np.ndarray, force: bool = False) -> None:
        with self._lock:
            if self._ready is None:
                return
            # The spare is loaded early so the switch itself does not wait.
            # Leave it idle until this tracker is actually due.
            if not force and not self._swap_due(frame):
                return
            spare = self._ready
            self._ready = None
        old = self._landmarker
        self._landmarker = spare
        self._served = 0
        self._opened_at = time.monotonic()
        self._warned = False
        if old is not None:
            threading.Thread(target=_close_slot, args=(old,), daemon=True).start()

    def _drop_current(self) -> None:
        old = self._landmarker
        self._landmarker = None
        if old is not None:
            threading.Thread(target=_close_slot, args=(old,), daemon=True).start()

    def _ensure_spare(self, frame: np.ndarray, force: bool = False) -> None:
        if not self._metal or self._closed:
            return
        with self._lock:
            if self._starting or self._ready is not None:
                return
            if not force and not self._spare_due(frame):
                return
            self._starting = True
        warmup = np.ascontiguousarray(frame)

        def build() -> None:
            orphan = None
            try:
                slot = self._reopen()
                try:
                    if isinstance(slot, _ProcessSlot) or (
                        hasattr(slot, "detect") and not hasattr(slot, "detect_for_video")
                    ):
                        slot.detect(warmup, 0)
                except Exception:
                    _close_slot(slot)
                    raise
            except Exception as exc:
                self._warn(exc)
                with self._lock:
                    self._starting = False
                return
            with self._lock:
                if self._closed:
                    orphan = slot
                else:
                    self._ready = slot
                self._starting = False
            if orphan is not None:
                _close_slot(orphan)

        threading.Thread(target=build, name="liveplay-hand-spare", daemon=True).start()

    def _swap_due(self, frame: np.ndarray) -> bool:
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._cycle_s:
            return True
        height, width = int(frame.shape[0]), int(frame.shape[1])
        return self._served >= self._frame_budget(width, height)

    def _spare_due(self, frame: np.ndarray) -> bool:
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= max(0.0, self._cycle_s - self._spare_lead_s):
            return True
        if elapsed < 0.25 or self._served <= 0:
            return False
        height, width = int(frame.shape[0]), int(frame.shape[1])
        budget = self._frame_budget(width, height)
        rate = self._served / elapsed
        return self._served + rate * self._spare_lead_s >= budget

    def _warn(self, exc: Exception) -> None:
        if self._warned:
            return
        print(f"[liveplay] could not refresh the hand tracker ({exc}).")
        self._warned = True


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
