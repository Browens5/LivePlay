"""Empty-table snapshot for the color tracker.

MediaPipe does not use this photo. `diff` and `diff+skin` do. The
overhead camera sees the TV, so the snapshot is a flat gray screen with
nothing on the glass: the field, static glare, and the bezel in the
crop. If the gray level, the crop, or the room lights change, snapshot
again. A new display scan throws the old photo away, because the warp
changed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from liveplay.errors import CalibrationError


def load_calibration(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CalibrationError(
            f"Could not read the calibration image at {path}. "
            f"Delete it and press C to snapshot the empty table again."
        )
    return image


def save_calibration(path: Path, frame_bgr: np.ndarray) -> None:
    if frame_bgr is None or frame_bgr.size == 0:
        raise CalibrationError("The camera frame was empty, so nothing was saved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise CalibrationError(f"Could not write the calibration image to {path}.")
