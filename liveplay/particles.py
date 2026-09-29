"""Dumb particle rules driven only by interaction points.

Spawn on a point, drift, pull toward the nearest point, fade out.
Nothing in here reads the camera or the screen contents.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from liveplay.config import ParticleConfig
from liveplay.points import InteractionPoint


@dataclass(frozen=True)
class _Style:
    radius: tuple[float, float]
    speed: tuple[float, float]
    life: tuple[float, float]
    fade_scale: float
    spawn_scale: float
    colors: tuple[tuple[int, int, int], ...]


_STYLES: dict[str, _Style] = {
    "sparks": _Style(
        radius=(2.0, 6.0),
        speed=(40.0, 180.0),
        life=(0.35, 0.85),
        fade_scale=1.0,
        spawn_scale=1.0,
        colors=(
            (255, 90, 150),
            (80, 220, 255),
            (255, 210, 70),
            (140, 255, 160),
            (190, 140, 255),
        ),
    ),
    "blobs": _Style(
        radius=(16.0, 36.0),
        speed=(10.0, 60.0),
        life=(0.7, 1.4),
        fade_scale=0.55,
        spawn_scale=0.5,
        colors=(
            (255, 120, 90),
            (90, 190, 255),
            (255, 200, 90),
            (120, 230, 160),
            (220, 140, 255),
        ),
    ),
    "trails": _Style(
        radius=(3.0, 8.0),
        speed=(30.0, 110.0),
        life=(0.9, 1.7),
        fade_scale=0.45,
        spawn_scale=0.75,
        colors=(
            (120, 230, 255),
            (255, 160, 210),
            (230, 255, 160),
            (180, 180, 255),
        ),
    ),
}


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    radius: float
    color: tuple[int, int, int]


class ParticleSystem:
    def __init__(self, cfg: ParticleConfig, rng: random.Random | None = None) -> None:
        if cfg.style not in _STYLES:
            raise ValueError(f"Unknown particle style {cfg.style!r}")
        self.cfg = cfg
        self.style_name = cfg.style
        self.spec = _STYLES[cfg.style]
        self.rng = rng or random.Random()
        self.particles: list[Particle] = []
        self.fade_per_second = cfg.fade_per_second * self.spec.fade_scale
        if cfg.spawn_per_point <= 0:
            self.spawn_per_point = 0
        else:
            # Blobs/trails scale the count down, but a request for "some"
            # particles should still spawn at least one.
            scaled = int(round(cfg.spawn_per_point * self.spec.spawn_scale))
            self.spawn_per_point = max(1, scaled)

    def clear(self) -> None:
        self.particles.clear()

    def update(self, points: list[InteractionPoint], dt: float) -> None:
        dt = max(0.0, min(float(dt), 0.05))
        attract = self.cfg.attract
        alive: list[Particle] = []
        for particle in self.particles:
            if points and attract > 0:
                nearest = _nearest(particle.x, particle.y, points)
                dx = nearest.x - particle.x
                dy = nearest.y - particle.y
                dist = math.hypot(dx, dy) + 1e-3
                if dist < 420:
                    particle.vx += (dx / dist) * attract * dt
                    particle.vy += (dy / dist) * attract * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vx *= 0.985
            particle.vy *= 0.985
            particle.life -= self.fade_per_second * dt
            if particle.life > 0:
                alive.append(particle)
        self.particles = alive
        for point in points:
            count = self.spawn_per_point
            if point.is_new:
                count += self.cfg.splash
            self._spawn(point, count)

    def draw(self, surface: pygame.Surface, gray: int) -> None:
        # Fade toward the playfield gray, not toward black. A black speck
        # on the TV is a dark blob to the overhead camera (see vision.py).
        width, height = surface.get_size()
        for particle in self.particles:
            x = int(particle.x)
            y = int(particle.y)
            if x < -40 or y < -40 or x > width + 40 or y > height + 40:
                continue
            life = max(0.0, min(1.0, particle.life))
            color = _toward_gray(particle.color, life, gray)
            radius = max(1, int(particle.radius * (0.45 + 0.55 * life)))
            if self.style_name == "trails":
                x2 = int(particle.x - particle.vx * 0.06)
                y2 = int(particle.y - particle.vy * 0.06)
                pygame.draw.line(surface, color, (x, y), (x2, y2), max(1, radius // 2))
            if self.style_name == "blobs":
                outer = _toward_gray(particle.color, life * 0.45, gray)
                pygame.draw.circle(surface, outer, (x, y), radius)
                pygame.draw.circle(surface, color, (x, y), max(1, int(radius * 0.55)))
            else:
                pygame.draw.circle(surface, color, (x, y), radius)

    def _spawn(self, point: InteractionPoint, count: int) -> None:
        radius_lo, radius_hi = self.spec.radius
        speed_lo, speed_hi = self.spec.speed
        life_lo, life_hi = self.spec.life
        colors = self.spec.colors
        for _ in range(count):
            if len(self.particles) >= self.cfg.max_count:
                return
            angle = self.rng.random() * math.tau
            speed = self.rng.uniform(float(speed_lo), float(speed_hi))
            jitter = 14.0 if self.style_name == "sparks" else 6.0
            self.particles.append(
                Particle(
                    x=point.x + self.rng.uniform(-jitter, jitter),
                    y=point.y + self.rng.uniform(-jitter, jitter),
                    vx=math.cos(angle) * speed + point.vx * 0.35,
                    vy=math.sin(angle) * speed + point.vy * 0.35,
                    life=self.rng.uniform(float(life_lo), float(life_hi)),
                    radius=self.rng.uniform(float(radius_lo), float(radius_hi)),
                    color=colors[self.rng.randrange(len(colors))],
                )
            )


def _nearest(x: float, y: float, points: list[InteractionPoint]) -> InteractionPoint:
    best = points[0]
    best_d = (best.x - x) ** 2 + (best.y - y) ** 2
    for point in points[1:]:
        dist = (point.x - x) ** 2 + (point.y - y) ** 2
        if dist < best_d:
            best = point
            best_d = dist
    return best


def _toward_gray(color: tuple[int, int, int], life: float, gray: int) -> tuple[int, int, int]:
    return tuple(int(gray + (channel - gray) * life) for channel in color)  # type: ignore[return-value]
