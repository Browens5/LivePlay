"""Two-goal soccer on the table.

Each interaction point is a hand. A puck sits on that point and is what
hits the ball. The ball bounces off the top and bottom of the playfield,
and off the side walls outside the goals. The center of the ball crossing
a goal line scores for the player on the other side, then the ball sits
in the middle for a moment. During that pause the puck still follows the
hand, and it does not move the ball.

The camera is looking at this drawing. The field stays the calibration
gray. Goals, the ball, the score, and the puck are blue, green, white, or
cyan so they miss the skin gate in vision.py. The puck is a disc, not a
hand shape, so the MediaPipe tracker should not treat it as another hand.
"""

from __future__ import annotations

import math

from liveplay import sdl_env  # noqa: F401  # before pygame
import pygame

from liveplay.display import draw_score
from liveplay.points import InteractionPoint

# RGB. Hue sits outside the skin bands (OpenCV hue 0–20 and 170–180).
GOAL_LEFT = (50, 150, 255)
GOAL_RIGHT = (50, 220, 120)
BALL = (248, 248, 248)
BALL_MARK = (40, 170, 255)
LINE = (206, 206, 206)
# Cyan disc and a dark-blue core. Both sit outside the skin hue bands.
PUCK = (40, 210, 255)
PUCK_CORE = (8, 40, 120)

_RESTITUTION = 0.96
_MAX_SPEED = 2200.0
_MIN_BAT = 520.0
_GOAL_HOLD_S = 0.7
_GOAL_HEIGHT = 0.42


class SoccerGame:
    """One ball, two goals, a score. No camera and no window of its own."""

    def __init__(self) -> None:
        self.score_left = 0
        self.score_right = 0
        self.field: tuple[int, int, int, int] | None = None
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_radius = 24.0
        self.puck_radius = 36.0
        self.pucks: list[tuple[float, float]] = []
        self.hold = 0.0
        self._ready = False

    def kickoff(self) -> None:
        """Park the ball in the middle. The score stays."""
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.hold = 0.0
        if self.field is not None:
            self._place_center()

    def update(
        self,
        points: list[InteractionPoint],
        field: tuple[int, int, int, int],
        dt: float,
    ) -> None:
        dt = max(0.0, min(float(dt), 0.05))
        self.field = (int(field[0]), int(field[1]), int(field[2]), int(field[3]))
        self.ball_radius = _ball_radius(self.field)
        self.puck_radius = _puck_radius(self.field)
        # The puck is glued to the smoothed hand center. One puck per hand.
        self.pucks = [(float(point.x), float(point.y)) for point in points]
        if not self._ready:
            self._place_center()
            self._ready = True
            return
        if self.hold > 0.0:
            self.hold = max(0.0, self.hold - dt)
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            self._place_center()
            return
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt
        if self._walls():
            return
        self._bat(points)
        self._cap_speed()

    def draw(self, surface: pygame.Surface, gray: int) -> None:
        if self.field is None:
            return
        x, y, width, height = self.field
        goal_top, goal_bot = _goal_span(y, height)
        depth = _goal_depth(width)
        left_rect = (x, int(goal_top), depth, int(goal_bot - goal_top))
        right_rect = (x + width - depth, int(goal_top), depth, int(goal_bot - goal_top))
        pygame.draw.rect(surface, GOAL_LEFT, left_rect)
        pygame.draw.rect(surface, GOAL_RIGHT, right_rect)
        post = max(4, depth // 5)
        for side_x, sign in ((x + depth, 1), (x + width - depth, -1)):
            pygame.draw.line(
                surface, LINE,
                (side_x, int(goal_top)),
                (side_x + sign * post, int(goal_top)),
                4,
            )
            pygame.draw.line(
                surface, LINE,
                (side_x, int(goal_bot)),
                (side_x + sign * post, int(goal_bot)),
                4,
            )
        mid_x = x + width // 2
        pygame.draw.line(surface, LINE, (mid_x, y), (mid_x, y + height), 3)
        circle_r = max(12, int(min(width, height) * 0.12))
        pygame.draw.circle(surface, LINE, (mid_x, y + height // 2), circle_r, 3)
        self._draw_pucks(surface)
        center = (int(self.ball_x), int(self.ball_y))
        radius = max(4, int(self.ball_radius))
        pygame.draw.circle(surface, BALL, center, radius)
        pygame.draw.circle(surface, BALL_MARK, center, max(3, radius // 3))
        draw_score(
            surface,
            self.score_left,
            self.score_right,
            self.field,
            GOAL_LEFT,
            GOAL_RIGHT,
            gray,
        )

    def _draw_pucks(self, surface: pygame.Surface) -> None:
        radius = max(4, int(round(self.puck_radius)))
        core = max(3, radius // 3)
        for px, py in self.pucks:
            center = (int(round(px)), int(round(py)))
            pygame.draw.circle(surface, PUCK, center, radius)
            pygame.draw.circle(surface, PUCK_CORE, center, core)

    def _place_center(self) -> None:
        assert self.field is not None
        x, y, width, height = self.field
        self.ball_x = x + width / 2.0
        self.ball_y = y + height / 2.0

    def _walls(self) -> bool:
        """Bounce off the window. Return True when a goal reset the ball."""
        assert self.field is not None
        x, y, width, height = self.field
        radius = self.ball_radius
        top = y + radius
        bottom = y + height - radius
        if self.ball_y < top:
            self.ball_y = top
            self.ball_vy = abs(self.ball_vy) * _RESTITUTION
        elif self.ball_y > bottom:
            self.ball_y = bottom
            self.ball_vy = -abs(self.ball_vy) * _RESTITUTION
        goal_top, goal_bot = _goal_span(y, height)
        in_mouth = goal_top <= self.ball_y <= goal_bot
        left = float(x)
        right = float(x + width)
        if self.ball_x - radius < left:
            if in_mouth and self.ball_x <= left:
                # Center crossed the left goal line. The other side scores.
                self._score("right")
                return True
            if not in_mouth:
                self.ball_x = left + radius
                self.ball_vx = abs(self.ball_vx) * _RESTITUTION
        elif self.ball_x + radius > right:
            if in_mouth and self.ball_x >= right:
                self._score("left")
                return True
            if not in_mouth:
                self.ball_x = right - radius
                self.ball_vx = -abs(self.ball_vx) * _RESTITUTION
        return False

    def _score(self, attacker: str) -> None:
        if attacker == "left":
            self.score_left += 1
        else:
            self.score_right += 1
        print(f"[liveplay] goal {self.score_left}-{self.score_right}")
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self._place_center()
        self.hold = _GOAL_HOLD_S

    def _bat(self, points: list[InteractionPoint]) -> None:
        radius = self.ball_radius
        ordered = sorted(
            points,
            key=lambda point: (point.x - self.ball_x) ** 2 + (point.y - self.ball_y) ** 2,
        )
        for point in ordered:
            dx = self.ball_x - point.x
            dy = self.ball_y - point.y
            dist = math.hypot(dx, dy)
            # The drawn puck is the collider, not the raw blob radius.
            reach = radius + self.puck_radius
            if dist >= reach:
                continue
            if dist < 1.0:
                dx, dy, dist = 1.0, 0.0, 1.0
            nx, ny = dx / dist, dy / dist
            self.ball_x = point.x + nx * reach
            self.ball_y = point.y + ny * reach
            inward = self.ball_vx * nx + self.ball_vy * ny
            if inward < 0.0:
                self.ball_vx -= inward * nx
                self.ball_vy -= inward * ny
            self.ball_vx += point.vx
            self.ball_vy += point.vy
            outward = self.ball_vx * nx + self.ball_vy * ny
            if outward < _MIN_BAT:
                self.ball_vx += (_MIN_BAT - outward) * nx
                self.ball_vy += (_MIN_BAT - outward) * ny
            self._cap_speed()

    def _cap_speed(self) -> None:
        speed = math.hypot(self.ball_vx, self.ball_vy)
        if speed > _MAX_SPEED:
            scale = _MAX_SPEED / speed
            self.ball_vx *= scale
            self.ball_vy *= scale


def _ball_radius(field: tuple[int, int, int, int]) -> float:
    _x, _y, width, height = field
    return max(16.0, min(52.0, 0.04 * min(width, height)))


def _puck_radius(field: tuple[int, int, int, int]) -> float:
    _x, _y, width, height = field
    return max(28.0, min(80.0, 0.055 * float(min(width, height))))


def _goal_depth(width: int) -> int:
    return max(28, int(width * 0.055))


def _goal_span(y: int, height: int) -> tuple[float, float]:
    mouth = height * _GOAL_HEIGHT
    mid = y + height / 2.0
    return mid - mouth / 2.0, mid + mouth / 2.0
