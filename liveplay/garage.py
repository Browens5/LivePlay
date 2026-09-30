"""A monster truck with no tires, and the parts to put them on.

The truck is a side view, so it has two axles. Hover a tire in the pile
for a second, then hover an open axle to fit it. Do that twice. Hover
the bolt bucket, then a wheel, to set each lug nut. Six nuts. Hover the
wrench, then a wheel, and each second there tightens one loose nut.
Hover LET'S RACE when it looks ready.

Both tires on and every nut tight: the truck jumps the track and wins.
Anything left off or left loose: the truck crashes. After the finish,
hover RESET to start the bay over.

The camera is looking at this painting. MediaPipe is looking for a hand
shape, so the picture never draws a hand. `--vision diff+skin` will not
work here. The whole shop differs from the empty gray snapshot. Use
`mediapipe` for this mode.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from liveplay import sdl_env  # noqa: F401  # before pygame
import pygame

from liveplay.points import InteractionPoint

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "garage"

# Measured from the cropped truck sprite. Index 0 is the rear axle.
TRUCK_ASPECT = 978 / 520
AXLES = ((0.168, 0.911), (0.821, 0.912))
# Lug holes in the tire sprite, fractions of that sprite from its center.
# Top, then lower left, then lower right.
LUG_OFFSETS = ((0.0146, -0.1402), (-0.1287, 0.0662), (0.1241, 0.0775))

WHEEL_COUNT = 2
LUGS_PER_WHEEL = 3

GRAB_S = 1.0
PLACE_S = 1.0
TIGHTEN_S = 1.0
RACE_S = 1.5
RESET_S = 1.5
# The finish has to be on screen before RESET will take a hover.
RESET_AFTER_S = 4.0
GRACE_S = 0.18
POP_S = 0.28

# Truck sprite center, and width as a fraction of the field width.
TRUCK_CX = 0.63
TRUCK_CY = 0.49
TRUCK_W = 0.50
# Tire diameter as a fraction of the truck sprite's height.
TIRE_OF_TRUCK = 0.58
LUG_OF_TIRE = 0.145

# Tool stations: center x, center y, radius. x is a field fraction.
# Radius is a fraction of the field height, so the hotspot is a circle.
PILE = (0.155, 0.28, 0.115)
BUCKET = (0.155, 0.52, 0.108)
WRENCH_SPOT = (0.155, 0.745, 0.100)
AXLE_R = 0.16

RACE_RECT = (0.695, 0.028, 0.275, 0.108)
RESET_RECT = (0.028, 0.028, 0.20, 0.108)

HOLD = {
    "pile": GRAB_S,
    "bucket": GRAB_S,
    "wrench": GRAB_S,
    "race": RACE_S,
    "reset": RESET_S,
}

# RGB. These are the controls we paint, not the colors inside the art.
INK = (16, 20, 28)
PAPER = (255, 248, 236)
GOLD = (255, 204, 64)
BLUE = (22, 104, 176)
BLUE_EDGE = (186, 220, 255)
GREEN = (24, 150, 72)
GREEN_EDGE = (210, 255, 214)
RED = (168, 42, 36)
RED_EDGE = (255, 210, 196)
AMBER = (255, 176, 48)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SPRITES: dict[str, pygame.Surface] = {}
_SCALED: dict[tuple[str, int, int], pygame.Surface] = {}


@dataclass
class Wheel:
    mounted: bool = False
    # 0 empty, 1 sitting loose, 2 tightened.
    lugs: list[int] = field(default_factory=lambda: [0, 0, 0])


@dataclass(frozen=True)
class Spot:
    id: str
    kind: str
    cx: float
    cy: float
    radius: float
    rect: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Pose:
    """Where the truck sits during the finish. `ny` is the sprite center."""

    nx: float
    ny: float
    angle: float
    spin: float
    detach: int | None
    detach_nx: float
    detach_ny: float
    detach_spin: float
    title: str
    subtitle: str


@dataclass
class _Hand:
    nx: float
    ny: float
    px: float
    py: float
    size: float


@dataclass
class _Bit:
    nx: float
    ny: float
    vx: float
    vy: float
    spin: float
    size: float
    color: tuple[int, int, int]


def truck_pixels(field: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """Pixel rect of the truck sprite inside `field` (left, top, width, height)."""
    x, y, width, height = field
    tw = TRUCK_W * width
    th = tw / TRUCK_ASPECT
    left = x + TRUCK_CX * width - tw * 0.5
    top = y + TRUCK_CY * height - th * 0.5
    return left, top, tw, th


def axle_center(field: tuple[int, int, int, int], index: int) -> tuple[float, float]:
    """Pixel center of one axle. Index 0 is the rear wheel."""
    left, top, tw, th = truck_pixels(field)
    ax, ay = AXLES[index]
    return left + ax * tw, top + ay * th


def spots_for(field: tuple[int, int, int, int]) -> dict[str, Spot]:
    """Every hover target, in pixels, for this field."""
    x, y, width, height = field
    spots: dict[str, Spot] = {}
    for name, (nx, ny, radius) in (
        ("pile", PILE),
        ("bucket", BUCKET),
        ("wrench", WRENCH_SPOT),
    ):
        spots[name] = Spot(name, "circle", x + nx * width, y + ny * height, radius * height)
    for index in range(WHEEL_COUNT):
        cx, cy = axle_center(field, index)
        spots[f"axle-{index}"] = Spot(f"axle-{index}", "circle", cx, cy, AXLE_R * height)
    for name, rect in (("race", RACE_RECT), ("reset", RESET_RECT)):
        rx, ry, rw, rh = rect
        box = (x + rx * width, y + ry * height, rw * width, rh * height)
        spots[name] = Spot(
            name,
            "rect",
            box[0] + box[2] * 0.5,
            box[1] + box[3] * 0.5,
            0.0,
            box,
        )
    return spots


def cutscene_pose(
    t: float,
    outcome: str,
    fault: str | None,
    problem: int = 1,
) -> Pose:
    """Truck pose at `t` seconds into a win or a crash.

    `problem` is the wheel that comes off when the nuts were the mistake.
    """
    t = max(0.0, float(t))
    if outcome == "win":
        return _win_pose(t)
    return _crash_pose(t, fault, problem)


def fault_of(wheels: list[Wheel]) -> str | None:
    """Why the truck is not ready, or None when it can win."""
    if any(not wheel.mounted for wheel in wheels):
        return "tire"
    if any(value == 0 for wheel in wheels for value in wheel.lugs):
        return "lug"
    if any(value == 1 for wheel in wheels for value in wheel.lugs):
        return "loose"
    return None


def _win_pose(t: float) -> Pose:
    # Sprite center. The axles sit near the bottom of the art, so this
    # keeps the tires on the dirt instead of under the frame.
    body = 0.56
    if t < 1.35:
        nx = -0.18 + 0.46 * (t / 1.35)
        ny = body
        angle = 0.0
    elif t < 3.15:
        p = (t - 1.35) / 1.8
        nx = 0.28 + 0.26 * p
        ny = body - math.sin(math.pi * p) * 0.28
        angle = math.sin(math.pi * p) * 16.0
    else:
        p = min((t - 3.15) / 1.15, 1.0)
        nx = 0.54 + 0.10 * p
        ny = body
        angle = 0.0
    title = "WINNER" if t >= 4.0 else ""
    subtitle = "Every nut is tight" if t >= 4.0 else ""
    return Pose(nx, ny, angle, t * 220.0, None, 0.0, 0.0, 0.0, title, subtitle)


def _crash_pose(t: float, fault: str | None, problem: int) -> Pose:
    body = 0.56
    drive = min(t, 1.7) / 1.7
    nx = -0.16 + 0.52 * drive
    ny = body
    angle = 0.0
    detach = None
    dx = dy = dspin = 0.0
    if fault != "tire" and t >= 1.15:
        age = t - 1.15
        detach = problem
        dx = nx + 0.10 + min(age, 1.6) * 0.28
        dy = body - math.sin(min(age, 1.1) * 5.0) * 0.06 + max(0.0, age - 0.8) * 0.05
        dspin = age * 280.0
    if t >= 1.55:
        p = min((t - 1.55) / 1.7, 1.0)
        angle = -(p ** 1.15) * 100.0
        ny = body + p * 0.05
        nx += p * 0.06
    title = "CRASH" if t >= 3.15 else ""
    if fault == "lug":
        subtitle = "Lug nuts were missing"
    elif fault == "loose":
        subtitle = "The lug nuts were loose"
    else:
        subtitle = "A tire was left off"
    if t < 3.15:
        subtitle = ""
    return Pose(nx, ny, angle, min(t, 1.7) * 220.0, detach, dx, dy, dspin, title, subtitle)


class GarageGame:
    """One truck, one pair of hands. No camera and no window of its own."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.field: tuple[int, int, int, int] | None = None
        self.time = 0.0
        self._last_hands: list[_Hand] = []
        self.reset()

    def reset(self) -> None:
        """Empty bay. Tires off, nuts off, nothing in hand."""
        self.wheels = [Wheel() for _ in range(WHEEL_COUNT)]
        self.held: str | None = None
        self.carry: tuple[float, float] | None = None
        self.phase = "repair"
        self.outcome: str | None = None
        self.fault: str | None = None
        self.cut_t = 0.0
        self.hover_id: str | None = None
        self.hover_t = 0.0
        self.grace = 0.0
        self.suppress_id: str | None = None
        self.pops: dict[str, float] = {}
        self.confetti: list[_Bit] = []
        self._confetti_on = False
        self.problem = 1

    @property
    def ready(self) -> bool:
        return fault_of(self.wheels) is None

    @property
    def hint(self) -> str:
        if self.phase != "repair":
            if self.cut_t < RESET_AFTER_S:
                return ""
            if self.hover_id == "reset":
                return "Hold to play again"
            return "Hover RESET to try again"
        if self.hover_id == "race":
            if self.ready:
                return "Hold still to race"
            return "Hold to race. Missing parts crash"
        if self.held == "tire":
            if any(not wheel.mounted for wheel in self.wheels):
                return "Hover an open axle"
            return "Hover the tire pile to put it back"
        if self.held == "bolt":
            if any(wheel.mounted and 0 in wheel.lugs for wheel in self.wheels):
                return "Hover a wheel to add the lug nut"
            return "Hover the bucket to put it back"
        if self.held == "wrench":
            if any(1 in wheel.lugs for wheel in self.wheels):
                return "Hover a wheel to tighten the nuts"
            return "Hover the wrench to put it back"
        if any(not wheel.mounted for wheel in self.wheels):
            return "Hover a tire for one second"
        if any(wheel.mounted and 0 in wheel.lugs for wheel in self.wheels):
            return "Hover the bolt bucket"
        if any(1 in wheel.lugs for wheel in self.wheels):
            return "Hover the wrench"
        return "Hover LET'S RACE"

    def counts(self) -> tuple[int, int, int]:
        """Tires mounted, nuts placed, nuts tightened."""
        tires = sum(1 for wheel in self.wheels if wheel.mounted)
        placed = sum(1 for wheel in self.wheels for value in wheel.lugs if value > 0)
        tight = sum(1 for wheel in self.wheels for value in wheel.lugs if value == 2)
        return tires, placed, tight

    def update(
        self,
        points: list[InteractionPoint],
        field: tuple[int, int, int, int],
        dt: float,
    ) -> None:
        dt = max(0.0, min(float(dt), 0.05))
        self.field = (int(field[0]), int(field[1]), int(field[2]), int(field[3]))
        self.time += dt
        self._decay_pops(dt)
        hands = self._hands(points)
        self._last_hands = hands
        if self.phase == "cutscene":
            self.cut_t += dt
            self._maybe_confetti()
            self._tick_confetti(dt)
        else:
            self._follow(hands, dt)
        self._hover(self._acting(hands), dt)

    def draw(self, surface: pygame.Surface) -> None:
        if self.field is None:
            return
        _x, _y, width, height = self.field
        if width < 8 or height < 8:
            return
        if self.phase == "cutscene":
            self._draw_cutscene(surface)
        else:
            self._draw_repair(surface)

    def _hands(self, points: list[InteractionPoint]) -> list[_Hand]:
        assert self.field is not None
        x, y, width, height = self.field
        width = max(1, width)
        height = max(1, height)
        hands = []
        for point in points:
            nx = (point.x - x) / width
            ny = (point.y - y) / height
            hands.append(_Hand(nx, ny, point.x, point.y, float(point.size)))
        return hands

    def _spots(self) -> dict[str, Spot]:
        assert self.field is not None
        return spots_for(self.field)

    def _actionable(self, spots: dict[str, Spot]) -> dict[str, Spot]:
        if self.phase != "repair":
            if self.cut_t >= RESET_AFTER_S and "reset" in spots:
                return {"reset": spots["reset"]}
            return {}
        chosen = {"race": spots["race"]}
        open_axle = [
            f"axle-{index}"
            for index, wheel in enumerate(self.wheels)
            if not wheel.mounted
        ]
        bolt_axle = [
            f"axle-{index}"
            for index, wheel in enumerate(self.wheels)
            if wheel.mounted and 0 in wheel.lugs
        ]
        wrench_axle = [
            f"axle-{index}"
            for index, wheel in enumerate(self.wheels)
            if wheel.mounted and 1 in wheel.lugs
        ]
        if self.held is None:
            # A part can be picked up only when the truck still needs it.
            # That keeps a full axle, a full wheel, or a tight truck from
            # eating a hover that was meant for the next step.
            if any(not wheel.mounted for wheel in self.wheels):
                chosen["pile"] = spots["pile"]
            if any(wheel.mounted and 0 in wheel.lugs for wheel in self.wheels):
                chosen["bucket"] = spots["bucket"]
            if any(1 in wheel.lugs for wheel in self.wheels):
                chosen["wrench"] = spots["wrench"]
        elif self.held == "tire":
            chosen["pile"] = spots["pile"]
            for name in open_axle:
                chosen[name] = spots[name]
        elif self.held == "bolt":
            chosen["bucket"] = spots["bucket"]
            for name in bolt_axle:
                chosen[name] = spots[name]
        elif self.held == "wrench":
            chosen["wrench"] = spots["wrench"]
            for name in wrench_axle:
                chosen[name] = spots[name]
        return chosen

    def _suggested(self) -> set[str]:
        if self.phase != "repair":
            if self.cut_t >= RESET_AFTER_S:
                return {"reset"}
            return set()
        if self.held == "tire":
            names = {f"axle-{i}" for i, wheel in enumerate(self.wheels) if not wheel.mounted}
            return names or {"pile"}
        if self.held == "bolt":
            names = {
                f"axle-{i}"
                for i, wheel in enumerate(self.wheels)
                if wheel.mounted and 0 in wheel.lugs
            }
            return names or {"bucket"}
        if self.held == "wrench":
            names = {f"axle-{i}" for i, wheel in enumerate(self.wheels) if 1 in wheel.lugs}
            return names or {"wrench"}
        if any(not wheel.mounted for wheel in self.wheels):
            return {"pile"}
        if any(wheel.mounted and 0 in wheel.lugs for wheel in self.wheels):
            return {"bucket"}
        if any(1 in wheel.lugs for wheel in self.wheels):
            return {"wrench"}
        return {"race"}

    def _acting(self, hands: list[_Hand]) -> list[_Hand]:
        """While a part is carried, only the hand on that part can drop it."""
        if self.held is None or self.carry is None or len(hands) < 2:
            return hands
        cx, cy = self.carry
        return [min(hands, key=lambda hand: (hand.nx - cx) ** 2 + (hand.ny - cy) ** 2)]

    def _hover(self, hands: list[_Hand], dt: float) -> None:
        spots = self._spots()
        if self.suppress_id is not None and self.suppress_id in spots:
            if not any(_hits(hand, spots[self.suppress_id]) for hand in hands):
                self.suppress_id = None
        actionable = self._actionable(spots)
        actionable.pop(self.suppress_id, None)
        hit, hand = _pick(hands, actionable, self.hover_id)
        if hit is not None and hit == self.hover_id:
            self.grace = 0.0
            self.hover_t += dt
            if self.hover_t + 1e-6 >= self._hold_needed(hit):
                self.hover_t = 0.0
                self._fire(hit, hand)
        elif hit is not None:
            self.hover_id = hit
            self.hover_t = 0.0
            self.grace = 0.0
        else:
            self.grace += dt
            if self.grace >= GRACE_S:
                self.hover_id = None
                self.hover_t = 0.0
                self.grace = 0.0

    def _hold_needed(self, target: str) -> float:
        if target.startswith("axle-"):
            return TIGHTEN_S if self.held == "wrench" else PLACE_S
        return HOLD.get(target, PLACE_S)

    def _fire(self, target: str, hand: _Hand | None) -> None:
        if target == "pile":
            self._toggle("tire", hand, "pile")
        elif target == "bucket":
            self._toggle("bolt", hand, "bucket")
        elif target == "wrench":
            self._toggle("wrench", hand, "wrench")
        elif target.startswith("axle-"):
            self._on_axle(int(target.rsplit("-", 1)[1]))
        elif target == "race" and self.phase == "repair":
            self._start_race()
        elif target == "reset" and self.phase == "cutscene":
            self.reset()

    def _toggle(self, item: str, hand: _Hand | None, spot: str) -> None:
        if self.held is None:
            self.held = item
            if hand is not None:
                self.carry = (hand.nx, hand.ny)
            self.suppress_id = spot
        elif self.held == item:
            self.held = None
            self.carry = None
            self.suppress_id = spot

    def _on_axle(self, index: int) -> None:
        wheel = self.wheels[index]
        spot = f"axle-{index}"
        if self.held == "tire" and not wheel.mounted:
            wheel.mounted = True
            self.held = None
            self.carry = None
            self.suppress_id = spot
            self.pops[f"tire-{index}"] = POP_S
            return
        if self.held == "bolt" and wheel.mounted:
            for lug, value in enumerate(wheel.lugs):
                if value == 0:
                    wheel.lugs[lug] = 1
                    self.held = None
                    self.carry = None
                    self.suppress_id = spot
                    self.pops[f"lug-{index}-{lug}"] = POP_S
                    return
        if self.held == "wrench" and wheel.mounted:
            for lug, value in enumerate(wheel.lugs):
                if value == 1:
                    wheel.lugs[lug] = 2
                    self.pops[f"lug-{index}-{lug}"] = POP_S
                    return

    def _start_race(self) -> None:
        self.fault = fault_of(self.wheels)
        self.problem = _problem_wheel(self.wheels)
        self.outcome = "win" if self.fault is None else "crash"
        self.phase = "cutscene"
        self.cut_t = 0.0
        self.held = None
        self.carry = None
        self.hover_id = None
        self.hover_t = 0.0
        self.suppress_id = None
        self.confetti = []
        self._confetti_on = False

    def _follow(self, hands: list[_Hand], dt: float) -> None:
        if self.held is None or not hands or self.carry is None:
            return
        carry_x, carry_y = self.carry
        hand = min(hands, key=lambda item: (item.nx - carry_x) ** 2 + (item.ny - carry_y) ** 2)
        blend = min(1.0, dt * 30.0)
        self.carry = (
            carry_x + (hand.nx - carry_x) * blend,
            carry_y + (hand.ny - carry_y) * blend,
        )

    def _decay_pops(self, dt: float) -> None:
        dead = []
        for key, left in self.pops.items():
            left -= dt
            if left <= 0.0:
                dead.append(key)
            else:
                self.pops[key] = left
        for key in dead:
            del self.pops[key]

    def _maybe_confetti(self) -> None:
        if self.outcome != "win" or self._confetti_on or self.cut_t < 4.0:
            return
        self._confetti_on = True
        colors = (
            (255, 208, 64),
            (255, 255, 255),
            (64, 196, 255),
            (255, 96, 72),
            (80, 220, 120),
        )
        for _ in range(90):
            angle = float(self.rng.random()) * math.tau
            speed = 0.08 + float(self.rng.random()) * 0.35
            self.confetti.append(
                _Bit(
                    nx=0.15 + float(self.rng.random()) * 0.7,
                    ny=0.05 + float(self.rng.random()) * 0.25,
                    vx=math.cos(angle) * speed,
                    vy=-0.05 + float(self.rng.random()) * 0.15,
                    spin=float(self.rng.random()) * math.tau,
                    size=0.012 + float(self.rng.random()) * 0.02,
                    color=colors[self.rng.randrange(len(colors))],
                )
            )

    def _tick_confetti(self, dt: float) -> None:
        for bit in self.confetti:
            bit.vy += 0.55 * dt
            bit.nx += bit.vx * dt
            bit.ny += bit.vy * dt
            bit.spin += dt * 4.0

    def _draw_repair(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        _blit_cover(surface, "shop", pygame.Rect(x, y, width, height))
        spots = self._spots()
        suggested = self._suggested()
        self._draw_station(surface, spots["pile"], "tire", self.held != "tire")
        self._draw_station(surface, spots["bucket"], "bucket", True)
        self._draw_station(surface, spots["wrench"], "wrench", self.held != "wrench")
        self._draw_truck(surface, truck_pixels(self.field), 0.0, 0.0, None, repair=True)
        self._draw_marks(surface, spots, suggested)
        self._draw_carry(surface)
        self._draw_hands(surface)
        self._draw_race(surface, spots["race"])
        self._draw_status(surface)

    def _draw_cutscene(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        pose = cutscene_pose(self.cut_t, self.outcome or "crash", self.fault, self.problem)
        _blit_cover(surface, "track", pygame.Rect(x, y, width, height))
        tw = TRUCK_W * width * 0.92
        th = tw / TRUCK_ASPECT
        left = x + pose.nx * width - tw * 0.5
        top = y + pose.ny * height - th * 0.5
        self._draw_truck(
            surface,
            (left, top, tw, th),
            pose.angle,
            pose.spin,
            pose.detach,
            repair=False,
        )
        if pose.detach is not None:
            self._draw_loose_wheel(surface, pose, th)
        self._draw_confetti(surface)
        if pose.title:
            self._draw_title(surface, pose)
        if self.cut_t >= RESET_AFTER_S:
            spots = self._spots()
            self._draw_reset(surface, spots["reset"])
            self._draw_status(surface)

    def _draw_station(
        self,
        surface: pygame.Surface,
        spot: Spot,
        art: str,
        show_art: bool,
    ) -> None:
        assert self.field is not None
        height = self.field[3]
        pad = int(spot.radius * 2.15)
        _disc(surface, spot.cx, spot.cy, spot.radius * 1.05, (10, 14, 22, 120))
        if art == "tire" and show_art:
            offsets = ((-0.22, 0.16, -12.0), (0.24, 0.10, 10.0), (0.0, -0.16, 0.0))
            diameter = spot.radius * 1.15
            for ox, oy, angle in offsets:
                _sprite_at(
                    surface,
                    "tire",
                    spot.cx + ox * spot.radius,
                    spot.cy + oy * spot.radius,
                    diameter,
                    diameter,
                    angle,
                )
        elif art == "bucket" and show_art:
            _sprite_at(surface, "bucket", spot.cx, spot.cy + spot.radius * 0.02, pad * 0.78, pad * 0.95, 0.0)
        elif art == "wrench" and show_art:
            _sprite_at(surface, "wrench", spot.cx, spot.cy, pad * 0.78, pad * 0.88, -18.0)
        elif not show_art:
            pygame.draw.circle(
                surface,
                GOLD,
                (int(spot.cx), int(spot.cy)),
                max(8, int(spot.radius * 0.72)),
                max(2, int(height / 280)),
            )
        label = {"tire": "TIRES", "bucket": "BOLTS", "wrench": "WRENCH"}[art]
        scale = max(0.45, height / 1100.0)
        _center_text(
            surface,
            label,
            pygame.Rect(
                int(spot.cx - spot.radius),
                int(spot.cy + spot.radius * 0.62),
                int(spot.radius * 2),
                int(height * 0.045),
            ),
            PAPER,
            scale,
            INK,
        )

    def _draw_truck(
        self,
        surface: pygame.Surface,
        rect: tuple[float, float, float, float],
        angle: float,
        spin: float,
        detach: int | None,
        *,
        repair: bool,
    ) -> None:
        left, top, tw, th = rect
        cx = left + tw * 0.5
        cy = top + th * 0.5
        _shadow(surface, cx, top + th * 0.94, tw * 0.42, th * 0.08)
        # The wheel wells are painted solid, so the tires go on after the
        # body or the arches hide them.
        _sprite_at(surface, "truck_body", cx, cy, tw, th, angle)
        for index in range(WHEEL_COUNT):
            if detach == index or not self.wheels[index].mounted:
                continue
            self._draw_wheel_on(surface, rect, index, angle, spin, repair)

    def _draw_wheel_on(
        self,
        surface: pygame.Surface,
        rect: tuple[float, float, float, float],
        index: int,
        angle: float,
        spin: float,
        repair: bool,
    ) -> None:
        left, top, tw, th = rect
        ax, ay = AXLES[index]
        px = left + ax * tw
        py = top + ay * th
        cx = left + tw * 0.5
        cy = top + th * 0.5
        px, py = _spin_point(px, py, cx, cy, angle)
        diameter = th * TIRE_OF_TRUCK
        pop = self.pops.get(f"tire-{index}", 0.0)
        diameter *= 1.0 + 0.18 * (pop / POP_S if POP_S else 0.0)
        wheel_angle = angle + (0.0 if repair else spin)
        _sprite_at(surface, "tire", px, py, diameter, diameter, wheel_angle)
        wheel = self.wheels[index]
        lug_d = diameter * LUG_OF_TIRE
        for lug, (ox, oy) in enumerate(LUG_OFFSETS):
            value = wheel.lugs[lug]
            # A loose nut sits out from the hole and crooked. A tight nut
            # sits in the hole, square to the rim.
            loose = value == 1
            reach = 1.22 if loose else 1.0
            lx = px + ox * diameter * reach
            ly = py + oy * diameter * reach
            lx, ly = _spin_point(lx, ly, px, py, wheel_angle)
            if value == 0:
                if self.held == "bolt" and wheel.mounted and wheel.lugs.index(0) == lug:
                    pulse = 0.5 + 0.5 * math.sin(self.time * 7.0)
                    pygame.draw.circle(
                        surface,
                        GOLD,
                        (int(lx), int(ly)),
                        max(4, int(lug_d * (0.28 + 0.08 * pulse))),
                        max(2, int(lug_d * 0.08)),
                    )
                continue
            lug_angle = wheel_angle + (38.0 if loose else 0.0)
            scale = 1.0
            pop_lug = self.pops.get(f"lug-{index}-{lug}", 0.0)
            if pop_lug > 0.0:
                scale += 0.22 * (pop_lug / POP_S)
            if (
                self.held == "wrench"
                and loose
                and wheel.lugs.index(1) == lug
            ):
                scale += 0.08 * (0.5 + 0.5 * math.sin(self.time * 8.0))
            _sprite_at(surface, "lug", lx, ly, lug_d * scale, lug_d * scale * (914 / 761), lug_angle)

    def _draw_loose_wheel(self, surface: pygame.Surface, pose: Pose, th: float) -> None:
        assert self.field is not None and pose.detach is not None
        x, y, width, height = self.field
        diameter = th * TIRE_OF_TRUCK
        px = x + pose.detach_nx * width
        py = y + pose.detach_ny * height
        _sprite_at(surface, "tire", px, py, diameter, diameter, pose.detach_spin)
        wheel = self.wheels[pose.detach]
        lug_d = diameter * LUG_OF_TIRE
        for lug, (ox, oy) in enumerate(LUG_OFFSETS):
            if wheel.lugs[lug] == 0:
                continue
            loose = wheel.lugs[lug] == 1
            reach = 1.22 if loose else 1.0
            lx = px + ox * diameter * reach
            ly = py + oy * diameter * reach
            lx, ly = _spin_point(lx, ly, px, py, pose.detach_spin)
            _sprite_at(
                surface,
                "lug",
                lx,
                ly,
                lug_d,
                lug_d * (914 / 761),
                pose.detach_spin + (38.0 if loose else 0.0),
            )

    def _draw_marks(self, surface: pygame.Surface, spots: dict[str, Spot], suggested: set[str]) -> None:
        progress = 0.0
        if self.hover_id in spots and self.hover_t > 0.0:
            progress = min(1.0, self.hover_t / self._hold_needed(self.hover_id))
        for name in suggested:
            spot = spots.get(name)
            if spot is None or name in ("race", "reset"):
                continue
            self._pulse(surface, spot)
        if self.hover_id is None or progress <= 0.0:
            return
        spot = spots.get(self.hover_id)
        if spot is None or spot.kind != "circle":
            return
        _arc(surface, spot.cx, spot.cy, spot.radius * 1.08, progress, GOLD, max(4, int(spot.radius * 0.08)))

    def _pulse(self, surface: pygame.Surface, spot: Spot) -> None:
        if spot.kind != "circle":
            return
        pulse = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(self.time * 5.0))
        color = tuple(
            max(0, min(255, int(channel)))
            for channel in (
                GOLD[0] * pulse + 30,
                GOLD[1] * pulse + 20,
                GOLD[2] * pulse,
            )
        )
        width = max(2, int(spot.radius * 0.045))
        pygame.draw.circle(surface, color, (int(spot.cx), int(spot.cy)), int(spot.radius * 1.02), width)

    def _draw_carry(self, surface: pygame.Surface) -> None:
        if self.held is None or self.carry is None or self.field is None:
            return
        x, y, width, height = self.field
        # Above and aside the palm, so the axle under the hand stays visible.
        px = x + (self.carry[0] - 0.045) * width
        py = y + (self.carry[1] - 0.13) * height
        if self.held == "tire":
            size = height * 0.16
            _sprite_at(surface, "tire", px, py, size, size, -8.0)
        elif self.held == "bolt":
            size = height * 0.11
            _sprite_at(surface, "lug", px, py, size, size * (914 / 761), 8.0)
        else:
            size = height * 0.15
            _sprite_at(surface, "wrench", px, py, size * 0.9, size, 24.0)

    def _draw_hands(self, surface: pygame.Surface) -> None:
        # A ring, not a picture of a hand. The real fingers stay visible
        # around it for the tracker.
        if self.field is None:
            return
        for hand in self._last_hands:
            radius = int(max(16.0, min(hand.size * 0.34, self.field[3] * 0.035)))
            pygame.draw.circle(surface, INK, (int(hand.px), int(hand.py)), radius + 3, 5)
            pygame.draw.circle(surface, GOLD, (int(hand.px), int(hand.py)), radius, 3)

    def _draw_race(self, surface: pygame.Surface, spot: Spot) -> None:
        assert spot.rect is not None
        rect = pygame.Rect(int(spot.rect[0]), int(spot.rect[1]), int(spot.rect[2]), int(spot.rect[3]))
        ready = self.ready
        fill = GREEN if ready else BLUE
        edge = GREEN_EDGE if ready else BLUE_EDGE
        if ready:
            pulse = 0.5 + 0.5 * math.sin(self.time * 4.0)
            fill = (
                int(GREEN[0] + 30 * pulse),
                int(GREEN[1]),
                int(GREEN[2] + 20 * pulse),
            )
        _button(surface, rect, "LET'S RACE", fill, edge, self._progress("race"))

    def _draw_reset(self, surface: pygame.Surface, spot: Spot) -> None:
        assert spot.rect is not None
        rect = pygame.Rect(int(spot.rect[0]), int(spot.rect[1]), int(spot.rect[2]), int(spot.rect[3]))
        _button(surface, rect, "RESET", RED, RED_EDGE, self._progress("reset"))

    def _progress(self, target: str) -> float:
        if self.hover_id != target or self.hover_t <= 0.0:
            return 0.0
        return min(1.0, self.hover_t / self._hold_needed(target))

    def _draw_status(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        bar_h = max(36, int(height * 0.09))
        rect = pygame.Rect(x, y + height - bar_h, width, bar_h)
        _panel(surface, rect, (*INK, 210))
        tires, placed, tight = self.counts()
        scale = max(0.5, height / 1000.0)
        _center_text(
            surface,
            f"TIRES {tires}/2    NUTS {placed}/6    TIGHT {tight}/6",
            pygame.Rect(rect.x + 12, rect.y + 4, int(rect.w * 0.46), rect.h // 2),
            GOLD,
            scale * 0.72,
            INK,
        )
        hint = self.hint
        if hint:
            _center_text(
                surface,
                hint,
                pygame.Rect(rect.x + int(rect.w * 0.34), rect.y + rect.h // 2 - 4, int(rect.w * 0.64), rect.h // 2),
                PAPER,
                scale * 0.8,
                INK,
            )

    def _draw_title(self, surface: pygame.Surface, pose: Pose) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        scale = max(1.1, height / 420.0)
        plate = pygame.Rect(int(x + width * 0.28), int(y + height * 0.08), int(width * 0.44), int(height * 0.20))
        fill = (12, 70, 36, 220) if self.outcome == "win" else (110, 24, 20, 220)
        _panel(surface, plate, fill)
        _center_text(surface, pose.title, pygame.Rect(plate.x, plate.y + 8, plate.w, int(plate.h * 0.58)), PAPER, scale, fill[:3])
        if pose.subtitle:
            _center_text(
                surface,
                pose.subtitle,
                pygame.Rect(plate.x, plate.y + int(plate.h * 0.55), plate.w, int(plate.h * 0.38)),
                GOLD,
                scale * 0.42,
                fill[:3],
            )

    def _draw_confetti(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        for bit in self.confetti:
            cx = int(x + bit.nx * width)
            cy = int(y + bit.ny * height)
            length = max(6, int(bit.size * height))
            end = (
                int(cx + math.cos(bit.spin) * length),
                int(cy + math.sin(bit.spin) * length * 0.6),
            )
            pygame.draw.line(surface, bit.color, (cx, cy), end, max(2, length // 4))


def _problem_wheel(wheels: list[Wheel]) -> int:
    for index, wheel in enumerate(wheels):
        if not wheel.mounted or any(value != 2 for value in wheel.lugs):
            return index
    return 1


def _hits(hand: _Hand, spot: Spot) -> bool:
    if spot.kind == "circle":
        dx = hand.px - spot.cx
        dy = hand.py - spot.cy
        return dx * dx + dy * dy <= spot.radius * spot.radius
    assert spot.rect is not None
    x, y, width, height = spot.rect
    return x <= hand.px <= x + width and y <= hand.py <= y + height


def _dist(hand: _Hand, spot: Spot) -> float:
    return math.hypot(hand.px - spot.cx, hand.py - spot.cy)


def _pick(
    hands: list[_Hand],
    spots: dict[str, Spot],
    sticky: str | None,
) -> tuple[str | None, _Hand | None]:
    if sticky in spots:
        for hand in hands:
            if _hits(hand, spots[sticky]):
                return sticky, hand
    best_id = None
    best_hand = None
    best_d = 1e9
    for hand in hands:
        for spot in spots.values():
            if not _hits(hand, spot):
                continue
            distance = _dist(hand, spot)
            if distance < best_d:
                best_d = distance
                best_id = spot.id
                best_hand = hand
    return best_id, best_hand


def _spin_point(
    px: float,
    py: float,
    cx: float,
    cy: float,
    degrees: float,
) -> tuple[float, float]:
    """Rotate a point the same way pygame rotates a sprite (positive is counter-clockwise)."""
    if abs(degrees) < 0.01:
        return px, py
    angle = math.radians(degrees)
    cos = math.cos(angle)
    sin = math.sin(angle)
    dx = px - cx
    dy = py - cy
    return cx + dx * cos + dy * sin, cy - dx * sin + dy * cos


def _sprite(name: str) -> pygame.Surface:
    cached = _SPRITES.get(name)
    if cached is not None:
        return cached
    path = ASSET_DIR / f"{name}.png"
    image = pygame.image.load(str(path)).convert_alpha()
    _SPRITES[name] = image
    return image


def _scaled(name: str, width: int, height: int) -> pygame.Surface:
    width = max(1, int(width))
    height = max(1, int(height))
    key = (name, width, height)
    cached = _SCALED.get(key)
    if cached is not None:
        return cached
    image = pygame.transform.smoothscale(_sprite(name), (width, height))
    _SCALED[key] = image
    return image


def _sprite_at(
    surface: pygame.Surface,
    name: str,
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
) -> None:
    image = _scaled(name, int(width), int(height))
    if abs(angle) > 0.4:
        image = pygame.transform.rotate(image, angle)
    rect = image.get_rect(center=(int(cx), int(cy)))
    surface.blit(image, rect)


def _blit_cover(surface: pygame.Surface, name: str, rect: pygame.Rect) -> None:
    image = _scaled(name, rect.w, rect.h)
    surface.blit(image, rect.topleft)


def _disc(surface: pygame.Surface, cx: float, cy: float, radius: float, color: tuple[int, int, int, int]) -> None:
    diameter = max(2, int(radius * 2))
    pad = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(pad, color, (diameter // 2, diameter // 2), diameter // 2)
    surface.blit(pad, (int(cx - radius), int(cy - radius)))


def _shadow(surface: pygame.Surface, cx: float, cy: float, rx: float, ry: float) -> None:
    width = max(2, int(rx * 2))
    height = max(2, int(ry * 2))
    pad = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.ellipse(pad, (0, 0, 0, 90), pad.get_rect())
    surface.blit(pad, (int(cx - rx), int(cy - ry)))


def _panel(surface: pygame.Surface, rect: pygame.Rect, color: tuple[int, int, int, int]) -> None:
    pad = pygame.Surface((max(1, rect.w), max(1, rect.h)), pygame.SRCALPHA)
    pygame.draw.rect(pad, color, pad.get_rect(), border_radius=min(18, rect.h // 3))
    surface.blit(pad, rect.topleft)


def _button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    progress: float,
) -> None:
    radius = max(8, min(rect.h // 3, 22))
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, edge, rect, width=max(3, rect.h // 16), border_radius=radius)
    if progress > 0.0:
        inset = pygame.Rect(rect.x + 8, rect.bottom - max(8, rect.h // 7), rect.w - 16, max(6, rect.h // 9))
        pygame.draw.rect(surface, (0, 0, 0), inset, border_radius=4)
        done = inset.copy()
        done.w = max(1, int(inset.w * progress))
        pygame.draw.rect(surface, GOLD, done, border_radius=4)
    scale = max(0.55, rect.h / 78.0)
    _center_text(surface, text, rect, PAPER, scale, fill)


def _arc(
    surface: pygame.Surface,
    cx: float,
    cy: float,
    radius: float,
    fraction: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    box = pygame.Rect(0, 0, int(radius * 2), int(radius * 2))
    box.center = (int(cx), int(cy))
    start = math.pi / 2
    # pygame's arc runs counter-clockwise from the positive x axis.
    # Starting at the top and sweeping with a negative direction fills clockwise.
    end = start - max(0.05, fraction) * math.tau
    pygame.draw.arc(surface, color, box, end, start, width)


def _center_text(
    surface: pygame.Surface,
    text: str,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    scale: float,
    bg: tuple[int, int, int],
) -> None:
    thickness = 2 if scale >= 0.85 else 1
    (width, height), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x = rect.x + max(0, (rect.w - width) // 2)
    y = rect.y + max(0, (rect.h - height) // 2)
    _blit_text(surface, text, (x, y), color, scale, bg, thickness, height, baseline)


def _blit_text(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    bg: tuple[int, int, int],
    thickness: int,
    height: int,
    baseline: int,
) -> None:
    """OpenCV text. pygame.font crashes on the source build for Python 3.14."""
    (width, _height), _base = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 2
    image = np.full(
        (height + baseline + pad * 2, width + pad * 2, 3),
        (bg[2], bg[1], bg[0]),
        dtype=np.uint8,
    )
    cv2.putText(
        image,
        text,
        (pad, pad + height),
        _FONT,
        scale,
        (color[2], color[1], color[0]),
        thickness,
        cv2.LINE_AA,
    )
    rgb = np.ascontiguousarray(np.transpose(image[:, :, ::-1], (1, 0, 2)))
    surface.blit(pygame.surfarray.make_surface(rgb), pos)


__all__ = [
    "ASSET_DIR",
    "AXLE_R",
    "GRAB_S",
    "GRACE_S",
    "LUGS_PER_WHEEL",
    "PLACE_S",
    "RACE_S",
    "RESET_AFTER_S",
    "RESET_S",
    "TIGHTEN_S",
    "WHEEL_COUNT",
    "GarageGame",
    "Pose",
    "Wheel",
    "axle_center",
    "cutscene_pose",
    "fault_of",
    "spots_for",
    "truck_pixels",
]
