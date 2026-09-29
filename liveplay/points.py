"""Interaction points shared by vision and the games.

Coordinates are screen pixels on the table display (origin top-left).
Velocity is pixels per second. Soccer and the forest both consume this
list. They do not read the camera.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InteractionPoint:
    x: float
    y: float
    size: float
    vx: float = 0.0
    vy: float = 0.0
    is_new: bool = False


class PointTracker:
    """Nearest-neighbor match across frames so points get velocity.

    Crossing hands can swap identities. Smoothing damps webcam jitter
    before a puck or the dinosaur follows the point.
    """

    def __init__(
        self,
        match_distance: float = 300.0,
        smoothing: float = 0.55,
        hold_frames: int = 2,
    ) -> None:
        if match_distance <= 0:
            raise ValueError("match_distance must be positive")
        self.match_distance = match_distance
        self.smoothing = min(1.0, max(0.0, smoothing))
        # Keep a point briefly if a frame drops it, so one missed frame
        # does not look like the hand left and came back.
        self.hold_frames = max(0, hold_frames)
        self._prev: list[tuple[InteractionPoint, int]] = []

    def reset(self) -> None:
        self._prev = []

    def update(self, points: list[InteractionPoint], dt: float) -> list[InteractionPoint]:
        dt = max(float(dt), 1e-3)
        used: set[int] = set()
        updated: list[InteractionPoint] = []
        limit = self.match_distance
        previous = [point for point, _missed in self._prev]
        for point in points:
            best_i = -1
            best_d = limit
            for i, prev in enumerate(previous):
                if i in used:
                    continue
                dx = point.x - prev.x
                dy = point.y - prev.y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i >= 0:
                used.add(best_i)
                prev = previous[best_i]
                raw_x, raw_y = point.x, point.y
                point.vx = (raw_x - prev.x) / dt
                point.vy = (raw_y - prev.y) / dt
                blend = self.smoothing
                point.x = blend * raw_x + (1.0 - blend) * prev.x
                point.y = blend * raw_y + (1.0 - blend) * prev.y
                point.is_new = False
            else:
                point.vx = 0.0
                point.vy = 0.0
                point.is_new = True
            # A bad dt spike should not fling a puck across the table.
            point.vx = _clamp(point.vx, -4000.0, 4000.0)
            point.vy = _clamp(point.vy, -4000.0, 4000.0)
            updated.append(point)
        state: list[tuple[InteractionPoint, int]] = [(point, 0) for point in updated]
        for i, (prev, missed) in enumerate(self._prev):
            if i in used:
                continue
            missed += 1
            if missed <= self.hold_frames:
                ghost = InteractionPoint(prev.x, prev.y, prev.size, prev.vx, prev.vy, False)
                updated.append(ghost)
                state.append((ghost, missed))
        self._prev = state
        return updated


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
