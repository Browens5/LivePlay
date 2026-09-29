"""Two pterodactyls fly a course of tree limbs.

Each hand is one player. The pterodactyl stays on that palm, so moving
the hand swoops through the gap between a limb that hangs from the top
and a limb that rises from the bottom. The limbs drift left. They start
slow and speed up as the leading score climbs toward 100. Passing an
opening is one point. The first player to 100 gets confetti and the
course stops.

A touch on either limb freezes that pterodactyl for five seconds. It
stays where it hit, it does not follow the hand, and it cannot score
until the freeze ends. The other player keeps flying.

Hover either hand on the reset control at the top left for three
seconds to start the course over.

The camera is looking at this painting. MediaPipe is looking for a hand
shape, so the picture stays in cool blues and greens and never draws a
hand. The pterodactyl is smaller than the hand it sits on, and its wing
is one membrane on one spar rather than a fan of fingers, so the real
fingertips stay visible around it. `--vision diff+skin` will not work
here: the whole sky differs from the empty gray snapshot. Use
`mediapipe` for this mode.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

from liveplay import sdl_env  # noqa: F401  # before pygame
import pygame

from liveplay.display import draw_score
from liveplay.points import InteractionPoint

# Painted sky. Cool blues and teals only, so a skin-color tracker finds
# nothing in the art. The file is local; the game does not download it.
SKY_PATH = Path(__file__).resolve().parent / "assets" / "flight_sky.png"

WIN_SCORE = 100
FREEZE_S = 5.0
RESET_HOLD_S = 3.0

# Field-widths per second. The course eases from the first value toward
# the second as the leading score approaches 100.
SPEED_START = 0.10
SPEED_END = 0.38

# Opening height, as a fraction of the field. It shrinks a little as the
# course speeds up, and each gate still jitters around that height.
GAP_TALL = 0.42
GAP_SHORT = 0.30

# Top-left reset control, in fractions of the field (x, y, w, h).
RESET_RECT = (0.016, 0.018, 0.118, 0.086)

# How far a hand can jump, in field fractions, and still be the same player.
HAND_MATCH = 0.5

# RGB. Hue sits outside the skin bands (OpenCV hue 0–20 and 170–180).
TEAL = (38, 186, 198)
TEAL_DARK = (14, 110, 130)
TEAL_WING = (20, 150, 168)
TEAL_BELLY = (214, 244, 236)
VIOLET = (148, 102, 224)
VIOLET_DARK = (84, 48, 168)
VIOLET_WING = (124, 78, 206)
VIOLET_BELLY = (232, 226, 252)
BEAK = (186, 230, 210)
EYE = (248, 252, 255)
PUPIL = (16, 24, 32)
LEAF = (36, 140, 78)
LEAF_DARK = (16, 78, 52)
LEAF_LIGHT = (72, 176, 104)
BARK = (28, 74, 62)
ICE = (186, 224, 236)
ICE_RING = (210, 236, 244)
RESET_FILL = (16, 58, 140)
RESET_EDGE = (110, 200, 245)
RESET_TEXT = (236, 248, 255)
CONFETTI_COLORS = (
    (64, 196, 255),
    (70, 220, 160),
    (170, 130, 255),
    (255, 255, 255),
    (90, 150, 255),
    (40, 200, 190),
)

# Every flat color the mode paints, including the score digits. The skin
# gate in vision.py has to reject all of them.
FLIGHT_COLORS = (
    TEAL,
    TEAL_DARK,
    TEAL_WING,
    TEAL_BELLY,
    VIOLET,
    VIOLET_DARK,
    VIOLET_WING,
    VIOLET_BELLY,
    BEAK,
    EYE,
    PUPIL,
    LEAF,
    LEAF_DARK,
    LEAF_LIGHT,
    BARK,
    ICE,
    ICE_RING,
    RESET_FILL,
    RESET_EDGE,
    RESET_TEXT,
    *CONFETTI_COLORS,
)

PLAYER_COLORS = (
    (TEAL, TEAL_DARK, TEAL_WING, TEAL_BELLY),
    (VIOLET, VIOLET_DARK, VIOLET_WING, VIOLET_BELLY),
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SKY_CACHE: dict[tuple[int, int], tuple[pygame.Surface, pygame.Surface]] = {}


@dataclass
class Player:
    score: int = 0
    nx: float = 0.32
    ny: float = 0.5
    # Where the hand was last seen. The body stays put while frozen, so
    # identity follows this point instead of the frozen sprite.
    hand_nx: float = 0.32
    hand_ny: float = 0.5
    active: bool = False
    seen: bool = False
    freeze: float = 0.0
    flap: float = 0.0
    hand_size: float = 0.0


@dataclass
class Gate:
    """One top limb, one bottom limb, and the opening between them."""

    x: float
    gap_y: float
    gap_h: float
    width: float
    seed: int
    settled: set[int] = field(default_factory=set)


@dataclass
class Confetti:
    nx: float
    ny: float
    vx: float
    vy: float
    spin: float
    size: float
    color: tuple[int, int, int]


def speed_for_score(score: int) -> float:
    """How fast the limbs move, in field-widths per second."""
    climbed = min(max(int(score), 0), WIN_SCORE - 1) / (WIN_SCORE - 1)
    return SPEED_START + (SPEED_END - SPEED_START) * climbed


def reset_bounds(field: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """Pixel rectangle of the reset control inside `field`."""
    x, y, width, height = field
    rx, ry, rw, rh = RESET_RECT
    return (x + rx * width, y + ry * height, rw * width, rh * height)


class FlightGame:
    """Two flyers, one shared course. No camera and no window of its own."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.field: tuple[int, int, int, int] | None = None
        self.time = 0.0
        self.parallax = 0.0
        self.reset()

    def reset(self) -> None:
        """Clear the course and both scores. The sky keeps its place."""
        self.players = [Player(ny=0.38), Player(ny=0.62)]
        self.gates = []
        self.confetti = []
        self.won = False
        self.winner = None
        self.reset_hold = 0.0
        self._next_gap = self.rng.uniform(0.42, 0.56)
        self._gate_seed = 1

    @property
    def obstacle_speed(self) -> float:
        lead = max(player.score for player in self.players)
        return speed_for_score(lead)

    def update(
        self,
        points: list[InteractionPoint],
        field: tuple[int, int, int, int],
        dt: float,
    ) -> None:
        dt = max(0.0, min(float(dt), 0.05))
        self.field = (int(field[0]), int(field[1]), int(field[2]), int(field[3]))
        self.time += dt
        if self._hand_over_reset(points):
            self.reset_hold += dt
            if self.reset_hold >= RESET_HOLD_S:
                self.reset()
        else:
            self.reset_hold = 0.0
        self._assign(self._hands(points), dt)
        if not self.won:
            self._move(dt)
            self._spawn()
            self._collide()
            self._award()
        self.parallax += self.obstacle_speed * self.field[2] * 0.22 * dt
        self._tick_confetti(dt)

    def draw(self, surface: pygame.Surface) -> None:
        if self.field is None:
            return
        _x, _y, width, height = self.field
        if width < 8 or height < 8:
            return
        self._draw_sky(surface)
        for gate in self.gates:
            _draw_gate(surface, self.field, gate)
        for index, player in enumerate(self.players):
            if not player.active:
                continue
            self._draw_player(surface, player, index)
        self._draw_confetti(surface)
        draw_score(
            surface,
            self.players[0].score,
            self.players[1].score,
            self.field,
            TEAL,
            VIOLET,
            28,
        )
        self._draw_reset(surface)

    def _hands(self, points: list[InteractionPoint]) -> list[tuple[float, float, float]]:
        assert self.field is not None
        x, y, width, height = self.field
        width = max(1, width)
        height = max(1, height)
        hands = []
        for point in points:
            hands.append(((point.x - x) / width, (point.y - y) / height, float(point.size)))
        return hands

    def _assign(self, hands: list[tuple[float, float, float]], dt: float) -> None:
        match = _stable_match(self.players, hands)
        used = set(match.values())
        assigned: list[tuple[float, float, float] | None] = [None, None]
        for player_index, hand_index in match.items():
            assigned[player_index] = hands[hand_index]
        leftovers = [hands[index] for index in range(len(hands)) if index not in used]
        open_slots = [index for index in range(2) if assigned[index] is None]
        if len(open_slots) == 2 and leftovers:
            ordered = sorted(leftovers, key=lambda hand: hand[0])
            if len(ordered) == 1:
                slot = 0 if ordered[0][0] <= 0.5 else 1
                assigned[slot] = ordered[0]
            else:
                assigned[0] = ordered[0]
                assigned[1] = ordered[1]
        else:
            for slot, hand in zip(open_slots, leftovers):
                assigned[slot] = hand
        for index, player in enumerate(self.players):
            if player.freeze > 0.0:
                player.freeze = max(0.0, player.freeze - dt)
            hand = assigned[index]
            if hand is None:
                player.active = False
                continue
            first = not player.seen
            player.active = True
            player.hand_size = hand[2]
            player.hand_nx, player.hand_ny = _clamp_norm(hand[0], hand[1])
            if player.freeze <= 0.0:
                player.nx, player.ny = player.hand_nx, player.hand_ny
                player.flap += dt
            if first:
                # A hand that appears behind limbs already on the glass
                # does not collect those openings.
                for gate in self.gates:
                    if player.nx > gate.x + gate.width:
                        gate.settled.add(index)
            player.seen = True

    def _move(self, dt: float) -> None:
        step = self.obstacle_speed * dt
        for gate in self.gates:
            gate.x -= step
        self.gates = [gate for gate in self.gates if gate.x + gate.width > -0.06]

    def _spawn(self) -> None:
        if not self.gates:
            self.gates.append(self._make_gate(1.28))
            self._next_gap = self.rng.uniform(0.40, 0.58)
            return
        last = self.gates[-1]
        if last.x < 1.0 - self._next_gap:
            self.gates.append(self._make_gate(1.02))
            self._next_gap = self.rng.uniform(0.36, 0.56)

    def _make_gate(self, x: float) -> Gate:
        lead = max(player.score for player in self.players)
        climbed = min(lead, WIN_SCORE - 1) / (WIN_SCORE - 1)
        base = GAP_TALL + (GAP_SHORT - GAP_TALL) * climbed
        gap_h = min(0.52, max(0.24, base * self.rng.uniform(0.90, 1.10)))
        half = gap_h * 0.5
        margin = 0.07
        gap_y = self.rng.uniform(half + margin, 1.0 - half - margin)
        width = self.rng.uniform(0.062, 0.10)
        self._gate_seed += 1
        return Gate(x=x, gap_y=gap_y, gap_h=gap_h, width=width, seed=self._gate_seed)

    def _collide(self) -> None:
        assert self.field is not None
        for index, player in enumerate(self.players):
            if not player.active or player.freeze > 0.0:
                continue
            for gate in self.gates:
                if _body_hits(player, gate, self.field):
                    player.freeze = FREEZE_S
                    gate.settled.add(index)

    def _award(self) -> None:
        if self.won:
            return
        for gate in self.gates:
            right = gate.x + gate.width
            for index, player in enumerate(self.players):
                if index in gate.settled or player.nx <= right:
                    continue
                gate.settled.add(index)
                if not player.active or player.freeze > 0.0:
                    continue
                if self.field is not None and _body_hits(player, gate, self.field):
                    continue
                player.score += 1
                if player.score >= WIN_SCORE:
                    player.score = WIN_SCORE
                    self._win(index)
                    return

    def _win(self, index: int) -> None:
        if self.won:
            return
        self.won = True
        self.winner = index
        for _ in range(120):
            angle = self.rng.random() * math.tau
            speed = self.rng.uniform(0.08, 0.42)
            self.confetti.append(
                Confetti(
                    nx=self.rng.uniform(0.12, 0.88),
                    ny=self.rng.uniform(0.0, 0.35),
                    vx=math.cos(angle) * speed,
                    vy=self.rng.uniform(-0.05, 0.2),
                    spin=self.rng.random() * math.tau,
                    size=self.rng.uniform(0.012, 0.028),
                    color=CONFETTI_COLORS[self._gate_seed % len(CONFETTI_COLORS)],
                )
            )
            self._gate_seed += 1

    def _tick_confetti(self, dt: float) -> None:
        for bit in self.confetti:
            bit.vy += 0.45 * dt
            bit.nx += bit.vx * dt
            bit.ny += bit.vy * dt
            bit.spin += dt * 5.0
            if bit.ny > 1.08:
                bit.ny = self.rng.uniform(-0.08, -0.01)
                bit.nx = self.rng.uniform(0.05, 0.95)
                bit.vy = self.rng.uniform(0.05, 0.25)
                bit.vx = self.rng.uniform(-0.15, 0.15)

    def _hand_over_reset(self, points: list[InteractionPoint]) -> bool:
        if self.field is None:
            return False
        left, top, width, height = reset_bounds(self.field)
        right = left + width
        bottom = top + height
        for point in points:
            if left <= point.x <= right and top <= point.y <= bottom:
                return True
        return False

    def _draw_sky(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        sky, mirror = _sky_tiles(width, height)
        span = width * 2
        offset = int(self.parallax) % span
        base = x - offset
        surface.blit(sky, (base, y))
        surface.blit(mirror, (base + width, y))
        surface.blit(sky, (base + span, y))

    def _draw_player(self, surface: pygame.Surface, player: Player, index: int) -> None:
        assert self.field is not None
        cx, cy, rx, _ry = _body_radii(player, self.field)
        wing = math.sin(player.flap * 8.0)
        frozen = player.freeze > 0.0
        _paint_pterodactyl(surface, cx, cy, rx, wing, PLAYER_COLORS[index], frozen)
        if frozen:
            seconds = max(1, int(math.ceil(player.freeze)))
            _blit_label(
                surface,
                str(seconds),
                (int(cx - rx * 0.2), int(cy - rx * 1.35)),
                ICE_RING,
                scale=max(0.55, self.field[3] / 700.0),
                bg=(12, 40, 78),
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
                int(cy + math.sin(bit.spin) * length),
            )
            pygame.draw.line(surface, bit.color, (cx, cy), end, max(3, length // 3))
            pygame.draw.circle(surface, bit.color, (cx, cy), max(3, length // 4))

    def _draw_reset(self, surface: pygame.Surface) -> None:
        assert self.field is not None
        left, top, width, height = reset_bounds(self.field)
        rect = pygame.Rect(int(left), int(top), max(1, int(width)), max(1, int(height)))
        curve = min(14, rect.width // 2, rect.height // 2)
        pygame.draw.rect(surface, RESET_FILL, rect, border_radius=curve)
        pygame.draw.rect(surface, RESET_EDGE, rect, width=3, border_radius=curve)
        _draw_arrow(surface, rect)
        scale = max(0.45, min(1.05, rect.height / 78.0))
        _blit_label(
            surface,
            "RESET",
            (rect.x + int(rect.height * 0.72), rect.y + int(rect.height * 0.22)),
            RESET_TEXT,
            scale=scale,
            bg=RESET_FILL,
        )
        if self.reset_hold > 0.0:
            frac = min(1.0, self.reset_hold / RESET_HOLD_S)
            bar_h = max(8, rect.height // 4)
            bar = pygame.Rect(
                rect.x + 8,
                rect.bottom - bar_h - 6,
                max(1, int((rect.width - 16) * frac)),
                bar_h,
            )
            pygame.draw.rect(surface, RESET_EDGE, bar, border_radius=4)


def _stable_match(
    players: list[Player],
    hands: list[tuple[float, float, float]],
) -> dict[int, int]:
    """Pair each hand with the flyer it was already on, when it still is."""
    seen = [index for index, player in enumerate(players) if player.seen]
    if not seen or not hands:
        return {}
    best: dict[int, int] = {}
    best_cost: float | None = None
    limit = min(len(seen), len(hands))
    # Two players and two hands: a handful of pairings, so try them all
    # and keep the one that moves each flyer the least.
    for count in range(limit, 0, -1):
        found = False
        for player_ids in permutations(seen, count):
            for hand_ids in permutations(range(len(hands)), count):
                cost = 0.0
                ok = True
                pairing: dict[int, int] = {}
                for player_index, hand_index in zip(player_ids, hand_ids):
                    hand = hands[hand_index]
                    player = players[player_index]
                    dist = math.hypot(hand[0] - player.hand_nx, hand[1] - player.hand_ny)
                    if dist > HAND_MATCH:
                        ok = False
                        break
                    cost += dist
                    pairing[player_index] = hand_index
                if not ok:
                    continue
                found = True
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = pairing
        if found:
            return best
    return {}


def _clamp_norm(nx: float, ny: float) -> tuple[float, float]:
    return min(0.96, max(0.04, nx)), min(0.94, max(0.06, ny))


def _body_radii(
    player: Player,
    field: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Center and radii of the body that can touch a limb.

    The wing is outside this ellipse on purpose. A wingtip tagging a
    branch would punish the camera jitter, and a wing as wide as the
    hand would hide the fingers MediaPipe needs.
    """
    x, y, width, height = field
    rx = max(12.0, height * 0.046)
    if player.hand_size > 0.0:
        rx = min(rx, max(12.0, player.hand_size * 0.42))
    ry = rx * 0.58
    cx = x + player.nx * width
    cy = y + player.ny * height
    return cx, cy, rx, ry


def _body_hits(player: Player, gate: Gate, field: tuple[int, int, int, int]) -> bool:
    cx, cy, rx, ry = _body_radii(player, field)
    for rect in _gate_rects(gate, field):
        if _ellipse_hits_rect(cx, cy, rx, ry, rect):
            return True
    return False


def _gate_rects(
    gate: Gate,
    field: tuple[int, int, int, int],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    x, y, width, height = field
    left = x + gate.x * width
    limb_w = gate.width * width
    gap_top = y + (gate.gap_y - gate.gap_h * 0.5) * height
    gap_bot = y + (gate.gap_y + gate.gap_h * 0.5) * height
    top = (left, float(y), limb_w, max(0.0, gap_top - y))
    bottom = (left, gap_bot, limb_w, max(0.0, y + height - gap_bot))
    return top, bottom


def _ellipse_hits_rect(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rect: tuple[float, float, float, float],
) -> bool:
    if rx <= 0.0 or ry <= 0.0:
        return False
    left, top, width, height = rect
    if width <= 0.0 or height <= 0.0:
        return False
    scale = rx / ry
    nearest_x = min(max(cx, left), left + width)
    nearest_y = min(max(cy * scale, top * scale), (top + height) * scale)
    dx = cx - nearest_x
    dy = cy * scale - nearest_y
    return dx * dx + dy * dy <= rx * rx


def _draw_gate(surface: pygame.Surface, field: tuple[int, int, int, int], gate: Gate) -> None:
    top, bottom = _gate_rects(gate, field)
    _draw_limb(surface, top, gate.seed, gap_side="down")
    _draw_limb(surface, bottom, gate.seed + 17, gap_side="up")


def _draw_limb(
    surface: pygame.Surface,
    rect: tuple[float, float, float, float],
    seed: int,
    gap_side: str,
) -> None:
    left, top, width, height = rect
    if width < 2.0 or height < 2.0:
        return
    box = pygame.Rect(int(left), int(top), max(1, int(width)), max(1, int(height)))
    previous = surface.get_clip()
    surface.set_clip(box.clip(previous))
    pygame.draw.rect(surface, LEAF_DARK, box)
    rng = random.Random(seed)
    stem = max(3, box.width // 8)
    pygame.draw.line(surface, BARK, (box.centerx, box.top), (box.centerx, box.bottom), stem)
    blobs = 6 + seed % 3
    for index in range(blobs):
        blob_w = max(4, int(box.width * rng.uniform(0.45, 0.92)))
        blob_h = max(4, int(min(box.height * 0.16, box.width * 0.85)))
        blob_x = box.x + int(rng.uniform(0, max(1, box.width - blob_w)))
        span = max(1, box.height - blob_h)
        if gap_side == "down":
            blob_y = box.y + int(span * rng.uniform(0.45, 1.0))
        else:
            blob_y = box.y + int(span * rng.uniform(0.0, 0.55))
        color = LEAF_LIGHT if index % 2 == 0 else LEAF
        pygame.draw.ellipse(surface, color, (blob_x, blob_y, blob_w, blob_h))
    # A few short sticks, kept inside the limb so the painted edge and
    # the hit box stay the same opening.
    for _ in range(3):
        along = rng.uniform(0.2, 0.8)
        sy = box.top + int(box.height * along)
        reach = int(box.width * rng.uniform(0.18, 0.42))
        direction = -1 if rng.random() < 0.5 else 1
        end_x = min(box.right - 1, max(box.left + 1, box.centerx + direction * reach))
        end_y = min(box.bottom - 1, max(box.top + 1, sy + int(box.height * rng.uniform(-0.08, 0.08))))
        pygame.draw.line(surface, BARK, (box.centerx, sy), (end_x, end_y), max(2, stem // 2))
    edge = pygame.Rect(box.x, box.bottom - 5, box.width, 5) if gap_side == "down" else pygame.Rect(
        box.x, box.top, box.width, 5
    )
    pygame.draw.rect(surface, LEAF_LIGHT, edge)
    # A round cap on the opening, clipped to the limb so the painted
    # edge and the hit box stay the same gap.
    cap_h = max(8, min(box.width, box.height // 2))
    if gap_side == "down":
        cap = pygame.Rect(box.x, box.bottom - cap_h, box.width, cap_h * 2)
    else:
        cap = pygame.Rect(box.x, box.top - cap_h, box.width, cap_h * 2)
    pygame.draw.ellipse(surface, LEAF, cap)
    surface.set_clip(previous)


def _paint_pterodactyl(
    surface: pygame.Surface,
    cx: float,
    cy: float,
    rx: float,
    wing: float,
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    frozen: bool,
) -> None:
    """One membrane wing on one spar. `wing` is -1 (down) to 1 (up)."""
    body, dark, membrane, belly = colors
    if frozen:
        body, dark, membrane, belly = (_ice(body), _ice(dark), _ice(membrane), _ice(belly))
    ry = rx * 0.58
    _paint_wing(surface, cx, cy, rx, ry, -wing * 0.55, dark)
    tail = [
        (cx - rx * 0.28, cy + ry * 0.05),
        (cx - rx * 0.98, cy - ry * 0.22),
        (cx - rx * 1.02, cy + ry * 0.42),
        (cx - rx * 0.22, cy + ry * 0.28),
    ]
    pygame.draw.polygon(surface, dark, _ints(tail))
    body_rect = pygame.Rect(0, 0, max(2, int(rx * 1.15)), max(2, int(ry * 1.65)))
    body_rect.center = (int(cx - rx * 0.02), int(cy + ry * 0.06))
    pygame.draw.ellipse(surface, body, body_rect)
    belly_rect = body_rect.inflate(-max(2, int(rx * 0.38)), -max(2, int(ry * 0.5)))
    belly_rect.y += int(ry * 0.16)
    if belly_rect.width > 2 and belly_rect.height > 2:
        pygame.draw.ellipse(surface, belly, belly_rect)
    _paint_wing(surface, cx, cy, rx, ry, wing, membrane)
    tip = _wing_tip(cx, cy, rx, ry, wing)
    shoulder = (int(cx + rx * 0.02), int(cy - ry * 0.08))
    pygame.draw.line(surface, dark, shoulder, tip, max(2, int(rx * 0.07)))
    head_r = max(3, int(ry * 0.62))
    head = (int(cx + rx * 0.42), int(cy - ry * 0.32))
    pygame.draw.circle(surface, body, head, head_r)
    crest = [
        (head[0] - head_r * 0.2, head[1] - head_r * 0.4),
        (head[0] - head_r * 0.55, head[1] - head_r * 2.5),
        (head[0] + head_r * 0.35, head[1] - head_r * 1.15),
        (head[0] + head_r * 0.55, head[1] - head_r * 0.15),
    ]
    pygame.draw.polygon(surface, dark, _ints(crest))
    beak = [
        (head[0] + head_r * 0.25, head[1] - head_r * 0.12),
        (head[0] + int(head_r * 1.55), head[1] + head_r * 0.08),
        (head[0] + head_r * 0.15, head[1] + head_r * 0.42),
    ]
    pygame.draw.polygon(surface, BEAK if not frozen else _ice(BEAK), _ints(beak))
    eye = (head[0] + max(1, head_r // 6), head[1] - max(1, head_r // 8))
    pygame.draw.circle(surface, EYE, eye, max(2, head_r // 3))
    pygame.draw.circle(surface, PUPIL, eye, max(1, head_r // 6))
    if frozen:
        pygame.draw.circle(
            surface,
            ICE_RING,
            (int(cx), int(cy)),
            max(4, int(rx * 1.2)),
            max(2, int(rx * 0.08)),
        )


def _paint_wing(
    surface: pygame.Surface,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    wing: float,
    color: tuple[int, int, int],
) -> None:
    tip = _wing_tip(cx, cy, rx, ry, wing)
    # One membrane on one spar. The sweep is tall so the flap reads,
    # and it is not a fan of fingers the camera could call a hand.
    shape = [
        (cx + rx * 0.08, cy - ry * 0.05),
        (cx - rx * 0.95, cy + ry * 0.35),
        tip,
        (cx + rx * 0.48, cy - ry * 0.28),
    ]
    pygame.draw.polygon(surface, color, _ints(shape))


def _wing_tip(cx: float, cy: float, rx: float, ry: float, wing: float) -> tuple[int, int]:
    tip_x = cx - rx * 0.22
    tip_y = cy - ry * (0.15 + 2.35 * wing)
    return int(tip_x), int(tip_y)


def _ints(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in points]


def _ice(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(channel * 0.35 + ice * 0.65) for channel, ice in zip(color, ICE))


def _sky_tiles(width: int, height: int) -> tuple[pygame.Surface, pygame.Surface]:
    key = (width, height)
    cached = _SKY_CACHE.get(key)
    if cached is not None:
        return cached
    image = pygame.image.load(str(SKY_PATH)).convert()
    sky = pygame.transform.smoothscale(image, (width, height))
    # The photo does not tile. Mirroring it makes the wrap meet itself.
    mirror = pygame.transform.flip(sky, True, False)
    _SKY_CACHE[key] = (sky, mirror)
    return sky, mirror


def _draw_arrow(surface: pygame.Surface, rect: pygame.Rect) -> None:
    radius = max(8, int(rect.height * 0.28))
    center = (rect.x + radius + 10, rect.centery)
    box = pygame.Rect(0, 0, radius * 2, radius * 2)
    box.center = center
    pygame.draw.arc(surface, RESET_TEXT, box, math.radians(40), math.radians(320), max(2, radius // 6))
    tip = (center[0] + int(radius * 0.15), center[1] - radius + 2)
    head = [
        (tip[0] - 6, tip[1] + 2),
        (tip[0] + 5, tip[1] - 4),
        (tip[0] + 2, tip[1] + 7),
    ]
    pygame.draw.polygon(surface, RESET_TEXT, head)


def _blit_label(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    bg: tuple[int, int, int],
) -> None:
    """OpenCV text. pygame.font crashes on the source build for Python 3.14."""
    thickness = 2 if scale >= 0.9 else 1
    (width, height), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 4
    # The array OpenCV draws into is BGR. The swap below is what pygame shows.
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
    "FLIGHT_COLORS",
    "FREEZE_S",
    "RESET_HOLD_S",
    "SKY_PATH",
    "WIN_SCORE",
    "FlightGame",
    "reset_bounds",
    "speed_for_score",
]
