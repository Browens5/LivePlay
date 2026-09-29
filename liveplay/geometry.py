"""Measure which part of the framebuffer the TV actually shows.

Cheap panels overscan. The Mac draws 1920×1080, and the glass only lights
up a smaller rectangle of that image, so particles drawn at the edge never
appear. The webcam is looking at the glass, so it can see the difference.

The scan paints Gray-code stripes (and their inverse) and reads them back
through the camera. Each camera pixel that lands on the panel gets the
framebuffer coordinate it is looking at. From that:

* `visible` is the framebuffer rectangle that reaches the glass. It is
  the full span of those coordinates. `inset` can pull it in; the
  default is 0, so the playfield uses the whole lit area.
* `homography` maps camera pixels onto that framebuffer. Hand contours
  go through it, so a pose drawn on the TV sits under the real hand.

A full-white frame cannot measure overscan: the panel is lit either way.
The codes are what make a camera pixel say "I am seeing framebuffer x=140",
not "I am seeing something bright".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from liveplay.errors import CalibrationError

_DARK = 16
_BRIGHT = 240
_BIT_MARGIN = 12


@dataclass
class HandPose:
    """A hand in one coordinate space: camera pixels, or framebuffer pixels."""

    x: float
    y: float
    size: float
    contour: list[tuple[float, float]]
    fingertips: list[tuple[float, float]]


@dataclass
class DisplayGeometry:
    """Camera-to-framebuffer map plus the rectangle the TV actually shows."""

    visible: tuple[int, int, int, int]
    homography: np.ndarray
    frame_size: tuple[int, int]
    camera_size: tuple[int, int]
    # True when the camera image is already the framebuffer (fake camera).
    direct_screen: bool = False

    @classmethod
    def full_frame(cls, width: int, height: int) -> DisplayGeometry:
        """Identity map. Used when the camera image is already screen-sized."""
        return cls(
            visible=(0, 0, int(width), int(height)),
            homography=np.eye(3, dtype=np.float64),
            frame_size=(int(width), int(height)),
            camera_size=(int(width), int(height)),
            direct_screen=True,
        )

    def map_xy(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, self.homography.astype(np.float64))
        return mapped.reshape(-1, 2)

    def warp(self, frame_bgr: np.ndarray, border_gray: int = 90) -> np.ndarray:
        """Camera image onto the framebuffer.

        `homography` maps camera pixels to framebuffer pixels, the same
        map as `map_xy`. Pixels outside `visible` stay flat gray so the
        room around the TV, and the cropped margin, cannot become a
        tracked blob.
        """
        gray = int(border_gray)
        warped = cv2.warpPerspective(
            frame_bgr,
            self.homography.astype(np.float64),
            self.frame_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(gray, gray, gray),
        )
        x, y, width, height = self.visible
        masked = np.full(warped.shape, gray, dtype=warped.dtype)
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(int(warped.shape[1]), int(x) + int(width))
        y1 = min(int(warped.shape[0]), int(y) + int(height))
        if x1 > x0 and y1 > y0:
            masked[y0:y1, x0:x1] = warped[y0:y1, x0:x1]
        return masked

    def apply_pose(self, pose: HandPose) -> HandPose:
        palm = self.map_xy(np.array([[pose.x, pose.y], [pose.x + pose.size, pose.y]]))
        size = float(np.hypot(palm[1, 0] - palm[0, 0], palm[1, 1] - palm[0, 1]))
        contour = _pairs(self.map_xy(np.array(pose.contour, dtype=np.float32))) if pose.contour else []
        tips = _pairs(self.map_xy(np.array(pose.fingertips, dtype=np.float32))) if pose.fingertips else []
        return HandPose(
            x=float(palm[0, 0]),
            y=float(palm[0, 1]),
            size=max(1.0, size),
            contour=contour,
            fingertips=tips,
        )


def render_pattern(
    width: int,
    height: int,
    kind: str,
    bit: int = 0,
    nbits: int = 7,
    inverse: bool = False,
) -> np.ndarray:
    """BGR pattern the size of the framebuffer.

    kind is "dark", "bright", "x" (vertical stripes, encodes x), or "y".
    """
    if kind == "dark":
        value = _DARK
        plane = None
    elif kind == "bright":
        value = _BRIGHT
        plane = None
    elif kind == "x":
        plane = _gray_bits(width, bit, nbits, inverse)
        value = None
    elif kind == "y":
        plane = _gray_bits(height, bit, nbits, inverse)
        value = None
    else:
        raise ValueError(f"Unknown pattern {kind!r}")
    image = np.empty((height, width, 3), dtype=np.uint8)
    if plane is None:
        image[:] = value
        return image
    if kind == "x":
        stripe = plane.reshape(1, width)
        image[:, :, 0] = stripe
        image[:, :, 1] = stripe
        image[:, :, 2] = stripe
    else:
        stripe = plane.reshape(height, 1)
        image[:, :, 0] = stripe
        image[:, :, 1] = stripe
        image[:, :, 2] = stripe
    return image


def solve_geometry(
    dark: np.ndarray,
    bright: np.ndarray,
    x_pairs: list[tuple[np.ndarray, np.ndarray]],
    y_pairs: list[tuple[np.ndarray, np.ndarray]],
    frame_size: tuple[int, int],
    nbits: int,
    inset: int,
) -> DisplayGeometry:
    """Build a geometry from grayscale captures of the patterns.

    `x_pairs` / `y_pairs` are (positive, inverse) images, MSB first.
    """
    if len(x_pairs) != nbits or len(y_pairs) != nbits:
        raise CalibrationError("Display scan was incomplete. Press SPACE to try again.")
    mask = panel_mask(dark, bright)
    area = int(cv2.countNonZero(mask))
    if area < max(400, int(0.02 * mask.size)):
        raise CalibrationError(
            "The webcam cannot see a bright TV. Point it at the whole glass, "
            "turn off glare, and press SPACE to measure again."
        )
    valid = mask > 0
    x_map, x_valid = _decode_axis(x_pairs, frame_size[0], nbits)
    y_map, y_valid = _decode_axis(y_pairs, frame_size[1], nbits)
    valid = valid & x_valid & y_valid
    if int(np.count_nonzero(valid)) < 400:
        raise CalibrationError(
            "The stripe pattern was unreadable. The TV may be slow to update. "
            "Raise geometry.settle in config.json and press SPACE to try again."
        )
    visible = _bounds(x_map, y_map, valid, inset, frame_size, nbits)
    homography = _homography(x_map, y_map, valid)
    camera_size = (int(dark.shape[1]), int(dark.shape[0]))
    return DisplayGeometry(
        visible=visible,
        homography=homography,
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        camera_size=camera_size,
        direct_screen=False,
    )


def panel_mask(dark: np.ndarray, bright: np.ndarray) -> np.ndarray:
    """Largest bright region that appears when the framebuffer goes white."""
    diff = cv2.absdiff(bright, dark)
    if diff.ndim == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(mask.shape, dtype=np.uint8)
    biggest = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [biggest], -1, 255, thickness=-1)
    return filled


def save_geometry(path: Path, geometry: DisplayGeometry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_width": geometry.frame_size[0],
        "frame_height": geometry.frame_size[1],
        "camera_width": geometry.camera_size[0],
        "camera_height": geometry.camera_size[1],
        "visible": list(geometry.visible),
        "homography": geometry.homography.tolist(),
        "direct_screen": geometry.direct_screen,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    x, y, w, h = geometry.visible
    print(
        f"[liveplay] saved display geometry {path}  "
        f"visible x={x} y={y} w={w} h={h}"
    )


def load_geometry(path: Path, frame_size: tuple[int, int]) -> DisplayGeometry | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        homography = np.array(payload["homography"], dtype=np.float64)
        visible = tuple(int(v) for v in payload["visible"])
        if len(visible) != 4 or homography.shape != (3, 3):
            raise ValueError("bad shape")
        stored = (int(payload["frame_width"]), int(payload["frame_height"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            f"Could not read display geometry at {path} ({exc}). "
            f"Press G to measure the TV again."
        ) from exc
    if stored != frame_size:
        print(
            f"[liveplay] display geometry {path} is for {stored[0]}x{stored[1]}, "
            f"not {frame_size[0]}x{frame_size[1]}. Measuring again."
        )
        return None
    return DisplayGeometry(
        visible=(visible[0], visible[1], visible[2], visible[3]),
        homography=homography,
        frame_size=stored,
        camera_size=(int(payload["camera_width"]), int(payload["camera_height"])),
        direct_screen=bool(payload.get("direct_screen", False)),
    )


class DisplayScan:
    """Show one pattern at a time and accept the camera frame that saw it.

    The frame passed into `tick` was captured at the start of the loop, so
    it shows whatever was on the TV during the previous draw. `settle` is
    how long that pattern must stay up before the frame counts. Cheap TVs
    smear a new image for a few frames; 0.3s is the default.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        bits: int = 7,
        settle: float = 0.32,
        inset: int = 12,
    ) -> None:
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.nbits = int(bits)
        self.settle = float(settle)
        self.inset = int(inset)
        self._plan = _plan(self.nbits)
        self._index = 0
        self._showing: tuple | None = None
        self._accept_at: float | None = None
        self._captured: list[tuple[tuple, np.ndarray]] = []
        self.error: str | None = None

    @property
    def step(self) -> int:
        return self._index

    @property
    def total(self) -> int:
        return len(self._plan)

    def tick(
        self, frame_bgr: np.ndarray | None, now: float
    ) -> tuple[np.ndarray | None, DisplayGeometry | None]:
        if self.error:
            return None, None
        if (
            self._showing is not None
            and self._accept_at is not None
            and now >= self._accept_at
            and frame_bgr is not None
        ):
            gray = frame_bgr
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            self._captured.append((self._showing, gray))
            print(f"[liveplay] display scan {self._index + 1}/{len(self._plan)}")
            self._index += 1
            self._showing = None
            if self._index >= len(self._plan):
                try:
                    return None, self._solve()
                except CalibrationError as exc:
                    self.error = str(exc)
                    return None, None
        if self._showing is None and self._index < len(self._plan):
            self._showing = self._plan[self._index]
            self._accept_at = now + self.settle
        if self._showing is None:
            return None, None
        return _pattern_image(self.frame_size, self.nbits, self._showing), None

    def _solve(self) -> DisplayGeometry:
        dark = bright = None
        x_pos: dict[int, np.ndarray] = {}
        x_neg: dict[int, np.ndarray] = {}
        y_pos: dict[int, np.ndarray] = {}
        y_neg: dict[int, np.ndarray] = {}
        for spec, gray in self._captured:
            kind = spec[0]
            if kind == "dark":
                dark = gray
            elif kind == "bright":
                bright = gray
            elif kind == "x":
                (x_neg if spec[2] else x_pos)[int(spec[1])] = gray
            elif kind == "y":
                (y_neg if spec[2] else y_pos)[int(spec[1])] = gray
        if dark is None or bright is None:
            raise CalibrationError("Display scan lost the black/white frames.")
        order = list(range(self.nbits - 1, -1, -1))
        x_pairs = [(x_pos[bit], x_neg[bit]) for bit in order]
        y_pairs = [(y_pos[bit], y_neg[bit]) for bit in order]
        return solve_geometry(
            dark, bright, x_pairs, y_pairs, self.frame_size, self.nbits, self.inset
        )


def _plan(nbits: int) -> list[tuple]:
    steps: list[tuple] = [("dark",), ("bright",)]
    for bit in range(nbits - 1, -1, -1):
        steps.append(("x", bit, False))
        steps.append(("x", bit, True))
    for bit in range(nbits - 1, -1, -1):
        steps.append(("y", bit, False))
        steps.append(("y", bit, True))
    return steps


def _pattern_image(frame_size: tuple[int, int], nbits: int, spec: tuple) -> np.ndarray:
    width, height = frame_size
    kind = spec[0]
    if kind in ("dark", "bright"):
        return render_pattern(width, height, kind)
    return render_pattern(
        width,
        height,
        kind,
        bit=int(spec[1]),
        nbits=nbits,
        inverse=bool(spec[2]),
    )


def _gray_bits(length: int, bit: int, nbits: int, inverse: bool) -> np.ndarray:
    levels = 1 << nbits
    index = np.arange(length, dtype=np.int32)
    bucket = np.minimum((index * levels) // max(length, 1), levels - 1)
    gray = bucket ^ (bucket >> 1)
    plane = ((gray >> int(bit)) & 1).astype(np.uint8)
    if inverse:
        plane = 1 - plane
    return np.where(plane == 1, _BRIGHT, _DARK).astype(np.uint8)


def _decode_axis(
    pairs: list[tuple[np.ndarray, np.ndarray]], length: int, nbits: int
) -> tuple[np.ndarray, np.ndarray]:
    shape = pairs[0][0].shape
    code = np.zeros(shape, dtype=np.int32)
    valid = np.ones(shape, dtype=bool)
    for positive, negative in pairs:
        delta = positive.astype(np.int16) - negative.astype(np.int16)
        valid &= np.abs(delta) >= _BIT_MARGIN
        bit = delta > 0
        code = (code << 1) | bit.astype(np.int32)
    binary = _gray_to_binary(code)
    levels = float(1 << nbits)
    position = (binary.astype(np.float32) + 0.5) * (float(length) / levels)
    return position, valid


def _gray_to_binary(gray: np.ndarray) -> np.ndarray:
    binary = gray.copy()
    shifted = gray.copy()
    while True:
        shifted = shifted >> 1
        if not np.any(shifted):
            break
        binary = binary ^ shifted
    return binary


def _bounds(
    x_map: np.ndarray,
    y_map: np.ndarray,
    valid: np.ndarray,
    inset: int,
    frame_size: tuple[int, int],
    nbits: int,
) -> tuple[int, int, int, int]:
    """Full span of the framebuffer coordinates the camera actually saw.

    Each code is the center of a stripe. Extending by half a stripe
    includes that whole stripe. `inset` is an optional extra pull-in;
    0 keeps the entire lit area.
    """
    xs = x_map[valid]
    ys = y_map[valid]
    width, height = frame_size
    levels = float(1 << int(nbits))
    half_x = 0.5 * width / levels
    half_y = 0.5 * height / levels
    left = float(np.min(xs)) - half_x
    right = float(np.max(xs)) + half_x
    top = float(np.min(ys)) - half_y
    bottom = float(np.max(ys)) + half_y
    x0 = int(np.floor(left)) + int(inset)
    y0 = int(np.floor(top)) + int(inset)
    x1 = int(np.ceil(right)) - int(inset)
    y1 = int(np.ceil(bottom)) - int(inset)
    x0 = max(0, min(x0, width - 2))
    y0 = max(0, min(y0, height - 2))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    if (x1 - x0) < width * 0.3 or (y1 - y0) < height * 0.3:
        raise CalibrationError(
            "The measured TV area is too small. Point the webcam at the whole "
            "panel and press SPACE to measure again."
        )
    return (x0, y0, x1 - x0, y1 - y0)


def _homography(x_map: np.ndarray, y_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(valid)
    step = max(1, len(cols) // 1500)
    cols = cols[::step]
    rows = rows[::step]
    camera = np.stack([cols, rows], axis=1).astype(np.float32)
    frame = np.stack([x_map[rows, cols], y_map[rows, cols]], axis=1).astype(np.float32)
    homography, mask = cv2.findHomography(camera, frame, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < 30:
        raise CalibrationError(
            "Could not fit the camera to the TV. Keep the webcam still and "
            "press SPACE to measure again."
        )
    return homography.astype(np.float64)


def _pairs(points: np.ndarray) -> list[tuple[float, float]]:
    if points.size == 0:
        return []
    return [(float(x), float(y)) for x, y in points.reshape(-1, 2)]
