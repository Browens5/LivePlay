"""A forest the dinosaur walks through.

One dinosaur follows the nearest hand. It does not jump to the palm: it
walks, with a little side-to-side wander, and plays when the hand stops
over a place it knows.

  waterfall   stands in the spray and looks up
  lake        drinks at the shore
  rock        hops onto the top
  mountain    roars at the foot of the slope
  flowers     sniffs the patch

The camera is looking at this painting. MediaPipe is looking for a hand
shape, and the trees, water, and dinosaur are not hands. `--vision diff+skin`
will not work here: the whole picture differs from the empty gray snapshot,
so the color tracker sees a table full of hands. Use `mediapipe` for this mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from liveplay import sdl_env  # noqa: F401  # before pygame
import pygame

from liveplay.points import InteractionPoint

# RGB.
SKY_TOP = (126, 186, 224)
SKY_BOTTOM = (198, 224, 206)
GRASS = (78, 146, 68)
GRASS_LIGHT = (108, 170, 78)
MOUNTAIN = (118, 132, 146)
MOUNTAIN_DARK = (86, 98, 112)
SNOW = (236, 242, 246)
FALLS = (186, 224, 236)
FALLS_BRIGHT = (244, 252, 255)
LAKE = (42, 122, 176)
LAKE_DEEP = (28, 86, 140)
ROCK = (132, 128, 120)
ROCK_DARK = (88, 84, 78)
MOSS = (70, 120, 58)
TRUNK = (92, 64, 40)
LEAF = (36, 110, 48)
LEAF_LIGHT = (64, 150, 62)
LEAF_DARK = (24, 78, 36)
FLOWER = (214, 96, 122)
FLOWER_CENTER = (236, 196, 72)
DINO = (96, 168, 78)
DINO_DARK = (52, 110, 48)
DINO_BELLY = (214, 206, 150)
BIRD = (48, 52, 58)

_SPEED = 0.25
_ARRIVE = 0.05
_POSE_TIME = {
    "splash": 1.7,
    "drink": 1.8,
    "hop": 1.6,
    "roar": 1.4,
    "sniff": 1.3,
}

# Foot of each tree, in field fractions, and a size multiplier.
TREES = (
    (0.06, 0.56, 1.05),
    (0.11, 0.84, 0.85),
    (0.20, 0.64, 0.72),
    (0.45, 0.48, 0.78),
    (0.49, 0.92, 1.05),
    (0.66, 0.46, 0.7),
    (0.86, 0.50, 1.15),
    (0.93, 0.78, 0.88),
    (0.72, 0.90, 0.82),
    (0.36, 0.88, 0.66),
    (0.08, 0.42, 0.55),
    (0.58, 0.36, 0.5),
)


@dataclass(frozen=True)
class Spot:
    """Where the dinosaur stands, and how close a hand must be to choose it."""

    name: str
    pose: str
    x: float
    y: float
    reach: float


# The hand can be anywhere inside `reach` of the stand point. The lake and
# the falls are drawn a little above the stand point, on the near shore.
SPOTS = (
    Spot("waterfall", "splash", 0.30, 0.50, 0.12),
    Spot("lake", "drink", 0.34, 0.66, 0.14),
    Spot("rock", "hop", 0.78, 0.64, 0.11),
    Spot("mountain", "roar", 0.18, 0.44, 0.12),
    Spot("flowers", "sniff", 0.60, 0.82, 0.09),
)


def waterfall_bright(y: int, time: float) -> bool:
    """True on the bright bands of the falls. They scroll downward."""
    return (int(y + time * 70) // 6) % 2 == 0


class ForestGame:
    """One dinosaur, a fixed forest. No camera and no window of its own."""

    def __init__(self) -> None:
        self.field: tuple[int, int, int, int] | None = None
        self.nx = 0.56
        self.ny = 0.74
        self.facing = 1
        self.phase = 0.0
        self.time = 0.0
        self.pose = "idle"
        self.pose_time = 0.0
        self.cooldown = 0.0
        self.spot: Spot | None = None
        self.speed = 0.0
        self._drops = [(0.285 + (i % 5) * 0.008, 0.26 + (i * 0.03) % 0.22) for i in range(8)]

    def update(
        self,
        points: list[InteractionPoint],
        field: tuple[int, int, int, int],
        dt: float,
    ) -> None:
        dt = max(0.0, min(float(dt), 0.05))
        self.field = (int(field[0]), int(field[1]), int(field[2]), int(field[3]))
        self.time += dt
        self._fall(dt)
        hand = self._nearest_hand(points)
        if hand is None:
            self.pose = "idle"
            self.pose_time = 0.0
            self.spot = None
            self.speed = 0.0
            self.phase += dt * 2.0
            self.facing = 1 if math.sin(self.time * 0.35) >= 0 else -1
            return
        if self._hold_pose(hand, dt):
            return
        if self.cooldown > 0.0:
            self.cooldown = max(0.0, self.cooldown - dt)
            self._step_toward(hand[0], hand[1], dt, wander=True)
            self.pose = "walk" if self.speed > 0.02 else "idle"
            return
        spot = _spot_under(hand)
        if spot is not None:
            dist = math.hypot(self.nx - spot.x, self.ny - spot.y)
            if dist <= _ARRIVE:
                self.nx, self.ny = spot.x, spot.y
                self.pose = spot.pose
                self.pose_time = _POSE_TIME[spot.pose]
                self.spot = spot
                self.speed = 0.0
                return
            self._step_toward(spot.x, spot.y, dt, wander=True)
        else:
            self._step_toward(hand[0], hand[1], dt, wander=True)
        self.pose = "walk" if self.speed > 0.02 else "idle"
        self._clamp()

    def draw(self, surface: pygame.Surface) -> None:
        if self.field is None:
            return
        x, y, width, height = self.field
        if width < 8 or height < 8:
            return
        pygame.draw.rect(surface, GRASS, (x, y, width, height))
        _draw_sky(surface, self.field)
        _draw_clearing(surface, self.field)
        _draw_mountain(surface, self.field)
        _draw_lake(surface, self.field, self.time)
        _draw_waterfall(surface, self.field, self.time)
        _draw_flowers(surface, self.field, self.time)
        layers: list[tuple[float, str, int]] = []
        for index, tree in enumerate(TREES):
            layers.append((tree[1], "tree", index))
        layers.append((0.64, "rock", 0))
        layers.append((self.ny, "dino", 0))
        layers.sort(key=lambda item: item[0])
        for _foot, kind, index in layers:
            if kind == "tree":
                _draw_tree(surface, self.field, TREES[index], index, self.time)
            elif kind == "rock":
                _draw_rock(surface, self.field)
            else:
                self._draw_dino(surface)
        _draw_mist(surface, self.field, self._drops)
        _draw_butterflies(surface, self.field, self.time)
        _draw_bird(surface, self.field, self.time)

    def pixel(self) -> tuple[int, int]:
        """Dinosaur feet, in screen pixels."""
        assert self.field is not None
        return _px(self.field, self.nx, self.ny)

    def _nearest_hand(self, points: list[InteractionPoint]) -> tuple[float, float] | None:
        if self.field is None or not points:
            return None
        best = None
        best_d = 1e9
        for point in points:
            nx, ny = _to_norm(self.field, point.x, point.y)
            dist = (nx - self.nx) ** 2 + (ny - self.ny) ** 2
            if dist < best_d:
                best_d = dist
                best = (nx, ny)
        return best

    def _hold_pose(self, hand: tuple[float, float], dt: float) -> bool:
        if self.pose_time <= 0.0 or self.spot is None:
            return False
        dist = math.hypot(hand[0] - self.spot.x, hand[1] - self.spot.y)
        if dist > self.spot.reach + 0.02:
            self.pose_time = 0.0
            self.spot = None
            return False
        self.pose_time -= dt
        self.phase += dt * (7.0 if self.pose == "hop" else 2.2)
        self.nx, self.ny = self.spot.x, self.spot.y
        if self.pose_time <= 0.0:
            self.pose = "idle"
            self.spot = None
            self.cooldown = 0.4
        return True

    def _step_toward(self, tx: float, ty: float, dt: float, wander: bool) -> None:
        dx = tx - self.nx
        dy = ty - self.ny
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            self.speed = 0.0
            return
        if wander and dist > 0.08:
            side = math.sin(self.time * 2.4) * 0.04 * min(1.0, (dist - 0.08) / 0.22)
            # Perpendicular to the path, so the walk weaves between the trees.
            px, py = -dy / dist, dx / dist
            dx += px * side
            dy += py * side
            dist = math.hypot(dx, dy)
        step = min(dist, _SPEED * dt)
        self.nx += dx / dist * step
        self.ny += dy / dist * step
        self.facing = 1 if dx >= 0.0 else -1
        self.speed = step / max(dt, 1e-4)
        self.phase += step * 16.0
        self._avoid_trunks()
        self._clamp()

    def _avoid_trunks(self) -> None:
        for tree_x, tree_y, _size in TREES:
            dx = self.nx - tree_x
            dy = self.ny - tree_y
            dist = math.hypot(dx, dy)
            if 1e-4 < dist < 0.04:
                push = 0.04 - dist
                self.nx += dx / dist * push
                self.ny += dy / dist * push

    def _clamp(self) -> None:
        self.nx = min(0.96, max(0.04, self.nx))
        self.ny = min(0.94, max(0.36, self.ny))

    def _fall(self, dt: float) -> None:
        drops = []
        for index, (dx, dy) in enumerate(self._drops):
            dy += (0.18 + (index % 3) * 0.05) * dt
            if dy > 0.56:
                dy = 0.24
                dx = 0.285 + (index % 5) * 0.008
            drops.append((dx, dy))
        self._drops = drops

    def _draw_dino(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        scale = max(16.0, 0.11 * min(self.field[2], self.field[3]))
        cx, feet = self.pixel()
        _paint_dino(surface, cx, feet, scale, self.facing, self.phase, self.pose)


def _spot_under(hand: tuple[float, float]) -> Spot | None:
    best = None
    best_d = 1e9
    for spot in SPOTS:
        dist = math.hypot(hand[0] - spot.x, hand[1] - spot.y)
        if dist <= spot.reach and dist < best_d:
            best = spot
            best_d = dist
    return best


def _to_norm(field: tuple[int, int, int, int], px: float, py: float) -> tuple[float, float]:
    x, y, width, height = field
    width = max(1, width)
    height = max(1, height)
    return (px - x) / width, (py - y) / height


def _px(field: tuple[int, int, int, int], nx: float, ny: float) -> tuple[int, int]:
    x, y, width, height = field
    return int(x + nx * width), int(y + ny * height)


def _draw_sky(surface: pygame.Surface, field: tuple[int, int, int, int]) -> None:
    x, y, width, height = field
    bands = 10
    sky_h = int(height * 0.38)
    for band in range(bands):
        blend = band / max(1, bands - 1)
        color = tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * blend) for i in range(3))
        top = y + int(sky_h * band / bands)
        bot = y + int(sky_h * (band + 1) / bands)
        pygame.draw.rect(surface, color, (x, top, width, max(1, bot - top)))


def _draw_clearing(surface: pygame.Surface, field: tuple[int, int, int, int]) -> None:
    x, y, width, height = field
    rect = (
        x + int(width * 0.28),
        y + int(height * 0.62),
        int(width * 0.46),
        int(height * 0.28),
    )
    pygame.draw.ellipse(surface, GRASS_LIGHT, rect)


def _draw_mountain(surface: pygame.Surface, field: tuple[int, int, int, int]) -> None:
    # A far ridge, then the nearer peak with snow. Drawn back to front.
    pygame.draw.polygon(
        surface,
        MOUNTAIN_DARK,
        (_px(field, 0.22, 0.46), _px(field, 0.40, 0.16), _px(field, 0.58, 0.46)),
    )
    peak = _px(field, 0.16, 0.10)
    pygame.draw.polygon(
        surface,
        MOUNTAIN,
        (_px(field, 0.00, 0.46), peak, _px(field, 0.36, 0.46)),
    )
    pygame.draw.polygon(
        surface,
        SNOW,
        (peak, _px(field, 0.11, 0.22), _px(field, 0.22, 0.22)),
    )


def _draw_waterfall(surface: pygame.Surface, field: tuple[int, int, int, int], time: float) -> None:
    x0, y0 = _px(field, 0.275, 0.22)
    x1, y1 = _px(field, 0.325, 0.52)
    left, right = min(x0, x1), max(x0, x1)
    top, bot = min(y0, y1), max(y0, y1)
    width = max(2, right - left)
    for py in range(top, bot):
        color = FALLS_BRIGHT if waterfall_bright(py, time) else FALLS
        pygame.draw.line(surface, color, (left, py), (left + width, py))


def _draw_lake(surface: pygame.Surface, field: tuple[int, int, int, int], time: float) -> None:
    x, y, width, height = field
    rect = (
        x + int(width * 0.18),
        y + int(height * 0.48),
        int(width * 0.32),
        int(height * 0.16),
    )
    pygame.draw.ellipse(surface, LAKE_DEEP, rect)
    inner = (rect[0] + 8, rect[1] + 6, max(4, rect[2] - 16), max(4, rect[3] - 12))
    pygame.draw.ellipse(surface, LAKE, inner)
    center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
    for ring in range(3):
        grow = (time * 18 + ring * 10) % 28
        pygame.draw.ellipse(
            surface,
            FALLS,
            (center[0] - int(grow), center[1] - int(grow * 0.4), int(grow * 2), int(grow * 0.8)),
            1,
        )


def _draw_rock(surface: pygame.Surface, field: tuple[int, int, int, int]) -> None:
    cx, cy = _px(field, 0.78, 0.62)
    scale = max(20, int(0.09 * min(field[2], field[3])))
    pygame.draw.ellipse(surface, ROCK_DARK, (cx - scale, cy - int(scale * 0.35), scale * 2, int(scale * 0.9)))
    pygame.draw.ellipse(surface, ROCK, (cx - int(scale * 0.8), cy - int(scale * 0.7), int(scale * 1.6), int(scale * 0.85)))
    pygame.draw.circle(surface, MOSS, (cx - scale // 5, cy - scale // 3), max(4, scale // 5))


def _draw_flowers(surface: pygame.Surface, field: tuple[int, int, int, int], time: float) -> None:
    sway = int(math.sin(time * 1.6) * 2)
    for index in range(7):
        nx = 0.55 + (index % 4) * 0.025
        ny = 0.80 + (index // 4) * 0.03
        cx, cy = _px(field, nx, ny)
        cx += sway
        pygame.draw.line(surface, LEAF_DARK, (cx, cy), (cx, cy - 8), 2)
        pygame.draw.circle(surface, FLOWER, (cx, cy - 10), 4)
        pygame.draw.circle(surface, FLOWER_CENTER, (cx, cy - 10), 2)


def _draw_tree(
    surface: pygame.Surface,
    field: tuple[int, int, int, int],
    tree: tuple[float, float, float],
    index: int,
    time: float,
) -> None:
    nx, ny, size = tree
    foot_x, foot_y = _px(field, nx, ny)
    height = int(min(field[2], field[3]) * 0.20 * size)
    height = max(18, height)
    sway = int(math.sin(time * 1.2 + index * 0.7) * max(2, height * 0.03))
    trunk_w = max(4, height // 10)
    trunk_h = int(height * 0.42)
    pygame.draw.rect(surface, TRUNK, (foot_x - trunk_w // 2, foot_y - trunk_h, trunk_w, trunk_h))
    crown_y = foot_y - height + sway // 2
    radius = int(height * 0.30)
    color = LEAF_DARK if index % 3 == 0 else LEAF
    pygame.draw.circle(surface, color, (foot_x + sway, crown_y), radius)
    pygame.draw.circle(surface, LEAF_LIGHT, (foot_x + sway - radius // 3, crown_y + radius // 4), int(radius * 0.72))
    pygame.draw.circle(surface, color, (foot_x + sway + radius // 2, crown_y + radius // 6), int(radius * 0.7))


def _draw_mist(
    surface: pygame.Surface,
    field: tuple[int, int, int, int],
    drops: list[tuple[float, float]],
) -> None:
    for dx, dy in drops:
        pygame.draw.circle(surface, FALLS_BRIGHT, _px(field, dx, dy), 2)


def _draw_butterflies(surface: pygame.Surface, field: tuple[int, int, int, int], time: float) -> None:
    for index in range(3):
        nx = 0.58 + 0.08 * math.sin(time * 0.9 + index * 2.0)
        ny = 0.74 + 0.04 * math.cos(time * 1.3 + index)
        cx, cy = _px(field, nx, ny)
        wing = int(3 + abs(math.sin(time * 9 + index)) * 5)
        pygame.draw.line(surface, FLOWER, (cx, cy), (cx - wing, cy - wing // 2), 2)
        pygame.draw.line(surface, FLOWER_CENTER, (cx, cy), (cx + wing, cy - wing // 2), 2)


def _draw_bird(surface: pygame.Surface, field: tuple[int, int, int, int], time: float) -> None:
    nx = (time * 0.045) % 1.25 - 0.12
    ny = 0.10 + 0.015 * math.sin(time * 2.0)
    cx, cy = _px(field, nx, ny)
    flap = int(4 + abs(math.sin(time * 6)) * 6)
    pygame.draw.line(surface, BIRD, (cx - flap, cy - flap // 2), (cx, cy), 2)
    pygame.draw.line(surface, BIRD, (cx, cy), (cx + flap, cy - flap // 2), 2)


def _paint_dino(
    surface: pygame.Surface,
    cx: int,
    feet: int,
    scale: float,
    facing: int,
    phase: float,
    pose: str,
) -> None:
    s = max(12, int(scale))
    direction = 1 if facing >= 0 else -1
    bob = 0
    if pose == "walk":
        bob = int(math.sin(phase * 2.0) * s * 0.06)
    elif pose == "hop":
        bob = -int(abs(math.sin(phase * 3.0)) * s * 0.55)
    feet = feet + bob
    body_w = int(s * 1.2)
    body_h = int(s * 0.62)
    body_cy = feet - int(s * 0.58)
    tail_y = body_cy + int(math.sin(phase) * s * 0.08)
    if pose == "sniff":
        tail_y -= int(s * 0.2)
    tail_from = (cx - direction * int(body_w * 0.42), body_cy + body_h // 6)
    tail_to = (tail_from[0] - direction * int(s * 0.72), tail_y)
    pygame.draw.line(surface, DINO_DARK, tail_from, tail_to, max(3, s // 7))
    body = (cx - body_w // 2, body_cy - body_h // 2, body_w, body_h)
    pygame.draw.ellipse(surface, DINO, body)
    belly = (
        cx - body_w // 4,
        body_cy,
        body_w // 2,
        max(4, body_h // 3),
    )
    pygame.draw.ellipse(surface, DINO_BELLY, belly)
    step = math.sin(phase * 2.0) if pose == "walk" else 0.0
    for index, swing in enumerate((step, -step)):
        hip = cx + direction * int((index - 0.5) * s * 0.22)
        knee_x = hip + int(swing * s * 0.16)
        top = body_cy + body_h // 3
        foot = (knee_x, feet)
        pygame.draw.line(surface, DINO_DARK, (hip, top), foot, max(2, s // 9))
        pygame.draw.circle(surface, DINO_DARK, foot, max(2, s // 11))
    head_dx = direction * int(s * 0.58)
    head_dy = -int(s * 0.46)
    if pose in ("drink", "sniff"):
        head_dx = direction * int(s * 0.72)
        head_dy = int(s * 0.02)
    elif pose in ("roar", "splash"):
        head_dy = -int(s * 0.7)
    neck_from = (cx + direction * body_w // 3, body_cy - body_h // 6)
    head = (neck_from[0] + head_dx, neck_from[1] + head_dy)
    pygame.draw.line(surface, DINO, neck_from, head, max(4, s // 5))
    radius = max(5, s // 4)
    pygame.draw.circle(surface, DINO, head, radius)
    snout = (head[0] + direction * int(s * 0.22), head[1] + s // 14)
    pygame.draw.circle(surface, DINO, snout, max(3, s // 7))
    if pose == "roar":
        pygame.draw.circle(surface, (176, 64, 64), (snout[0], snout[1] + s // 10), max(2, s // 9))
    eye = (head[0] + direction * s // 14, head[1] - s // 12)
    blink = pose == "idle" and int(phase * 2) % 11 == 0
    if blink:
        pygame.draw.line(surface, DINO_DARK, (eye[0] - 3, eye[1]), (eye[0] + 3, eye[1]), 2)
    else:
        pygame.draw.circle(surface, (248, 248, 242), eye, max(2, s // 11))
        pygame.draw.circle(surface, (28, 28, 28), (eye[0] + direction, eye[1]), max(1, s // 16))
