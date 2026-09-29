"""Local vision. One mapped camera frame in, interaction points out.

Feedback loop
-------------
The webcam looks at the TV, so the picture we draw is part of the scene.

* Calibration is a flat gray screen with the table empty. The playfield
  stays that same gray. A gradient, photo, or bright scene would differ
  from the snapshot everywhere and look like a giant hand.
* Hands are detected by difference from that snapshot, then gated with a
  skin-color mask (`diff+skin`, the default). Bright particles fail the
  skin test, so the game is less likely to chase its own sparks.
* Small blobs are dropped. Particle specks are smaller than a hand.
* `track_dark_blobs` is off by default. Turn it on for dark toys. It will
  also see some shadows and heavy glare edges.
* Auto exposure still breaks this: if the camera re-meters when particles
  appear, the gray field no longer matches the snapshot. capture.py tries
  to pin exposure; many webcams ignore it.
* Ceiling lights reflecting on the plexiglass show up as bright blobs.
  Skin gating rejects most of them. A large washed-out reflection can
  still win. Dim the room or matte the plexi. See the README.

This module does not know about particles or pygame. A MediaPipe hands
backend, or anything that emits palm/fingertip centers, should implement
`VisionBackend` and be constructed in `make_backend` without changes to
capture or the renderer.
"""

from __future__ import annotations

import os
from typing import Protocol

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from liveplay.config import VisionConfig
from liveplay.points import InteractionPoint

# OpenCV HSV hue is 0..180. These bands cover typical skin under indoor
# light and wrap around red. Yellow playfield graphics sit higher (around
# 25..35) and are rejected. Tune if your lighting is very warm or very blue.
_SKIN_LOW_1 = (0, 40, 50)
_SKIN_HIGH_1 = (20, 180, 255)
_SKIN_LOW_2 = (170, 40, 50)
_SKIN_HIGH_2 = (180, 180, 255)


class VisionBackend(Protocol):
    """Turn one BGR screen-space frame into interaction points.

    `last_warning` is a short operator message, or None. The app prints it
    once when it changes. Detection must not block on the network.
    """

    last_warning: str | None

    def detect(self, frame_bgr: np.ndarray) -> list[InteractionPoint]:
        ...

    def set_background(self, frame_bgr: np.ndarray | None) -> None:
        ...


class BlobVision:
    """Background subtraction, optional skin gate, optional dark objects."""

    def __init__(self, cfg: VisionConfig, background: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.background: np.ndarray | None = None
        self._bg_ready: np.ndarray | None = None
        self.last_warning: str | None = None
        kernel = cfg.morph_kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        if background is not None:
            self.set_background(background)

    @property
    def needs_calibration(self) -> bool:
        return self.cfg.method != "skin"

    @property
    def has_background(self) -> bool:
        return self._bg_ready is not None

    def set_background(self, frame_bgr: np.ndarray | None) -> None:
        if frame_bgr is None:
            self.background = None
            self._bg_ready = None
            return
        self.background = frame_bgr.copy()
        self._bg_ready = _prepare(self.background, self.cfg.scale)

    def detect(self, frame_bgr: np.ndarray) -> list[InteractionPoint]:
        self.last_warning = None
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            self.last_warning = "Camera frame was not a color image."
            return []
        if self.needs_calibration and self.background is None:
            self.last_warning = (
                "No empty-table calibration. Press C, clear the table, then SPACE."
            )
            return []
        if self.background is not None and self.background.shape != frame_bgr.shape:
            self.last_warning = (
                "Calibration size does not match the screen. Press C to snapshot again."
            )
            return []

        small = _prepare(frame_bgr, self.cfg.scale)
        mask = self._mask(small)
        if mask is None:
            return []
        fraction = float(cv2.countNonZero(mask)) / float(mask.size)
        if fraction > self.cfg.max_area_fraction:
            self.last_warning = (
                "Foreground covers too much of the table. Recalibrate on a clear "
                "table, reduce glare, or raise vision.diff_threshold."
            )
            return []
        return self._contours(mask)

    def _mask(self, small: np.ndarray) -> np.ndarray | None:
        method = self.cfg.method
        diff = None
        if method in ("diff", "diff+skin"):
            if self._bg_ready is None or self._bg_ready.shape[:2] != small.shape[:2]:
                self.last_warning = (
                    "Calibration is missing or the wrong size. Press C to snapshot."
                )
                return None
            delta = cv2.absdiff(small, self._bg_ready)
            gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
            _, diff = cv2.threshold(gray, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        skin = _skin_mask(small) if method in ("skin", "diff+skin") else None
        if method == "diff":
            assert diff is not None
            mask = diff
        elif method == "skin":
            assert skin is not None
            mask = skin
        else:
            assert diff is not None and skin is not None
            mask = cv2.bitwise_and(diff, skin)
            if self.cfg.track_dark_blobs and self._bg_ready is not None:
                dark = _dark_mask(small, self._bg_ready, self.cfg.dark_delta)
                mask = cv2.bitwise_or(mask, cv2.bitwise_and(dark, diff))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def _contours(self, mask: np.ndarray) -> list[InteractionPoint]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = self.cfg.scale
        area_scale = scale * scale
        min_area = self.cfg.min_area * area_scale
        found: list[tuple[float, InteractionPoint]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = float(moments["m10"] / moments["m00"]) / scale
            cy = float(moments["m01"] / moments["m00"]) / scale
            radius = (area / np.pi) ** 0.5 / scale
            found.append((area, InteractionPoint(x=cx, y=cy, size=radius)))
        found.sort(key=lambda item: item[0], reverse=True)
        return [point for _, point in found[: self.cfg.max_blobs]]


def make_backend(cfg: VisionConfig, background: np.ndarray | None) -> BlobVision:
    """Build the v0 backend.

    Swap this function's body to select another `VisionBackend` later
    (for example MediaPipe hand centers). Callers only need `detect`.
    """
    return BlobVision(cfg, background)


def _prepare(frame: np.ndarray, scale: float) -> np.ndarray:
    image = frame
    if scale != 1.0:
        width = max(1, int(round(frame.shape[1] * scale)))
        height = max(1, int(round(frame.shape[0] * scale)))
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, _SKIN_LOW_1, _SKIN_HIGH_1)
    high = cv2.inRange(hsv, _SKIN_LOW_2, _SKIN_HIGH_2)
    return cv2.bitwise_or(low, high)


def _dark_mask(frame_bgr: np.ndarray, background_bgr: np.ndarray, delta: int) -> np.ndarray:
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    darker = cv2.subtract(bg_gray, frame_gray)
    _, mask = cv2.threshold(darker, delta, 255, cv2.THRESH_BINARY)
    return mask
