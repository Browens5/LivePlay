"""Side-view truck, drawn from the parts the hands have fitted.

Coordinates are fractions of the truck rectangle. Y grows downward.
The nose points right. The same anchors are the weld hover targets.
"""

from __future__ import annotations

import math

import pygame

FRAME_PARTS = ("rail", "towers", "cage", "bed")
WELD_NAMES = ("rear", "front", "cage", "bed")
PANEL_PARTS = ("nose", "cabin", "tail")
KITS = ("classic", "wedge", "tube")

# Joints, as fractions of the truck rect. Shared with the hover spots.
WELD_AT = {
    "rear": (0.20, 0.79),
    "front": (0.82, 0.79),
    "cage": (0.42, 0.74),
    "bed": (0.16, 0.48),
}

PAINTS = {
    "red": (204, 36, 32),
    "yellow": (242, 186, 28),
    "blue": (28, 108, 204),
    "green": (24, 156, 78),
    "orange": (232, 98, 24),
    "white": (236, 238, 236),
}

PRIMER = (176, 170, 160)
INK = (16, 14, 18)
STEEL = (132, 140, 150)
STEEL_DARK = (62, 68, 76)
CHROME = (214, 220, 228)
WINDOW = (28, 44, 66)


def shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)  # type: ignore[return-value]


def draw_body(
    surface: pygame.Surface,
    rect: tuple[float, float, float, float],
    angle: float,
    *,
    fitted: set[str],
    welds: set[str],
    panels: set[str],
    kit: str | None,
    body: tuple[int, int, int] | None,
    accent: tuple[int, int, int] | None,
    decal: str | None,
    hot: dict[str, float],
) -> None:
    """Blit the built truck, rotated the same way as a pygame sprite."""
    left, top, tw, th = rect
    width = max(8, int(tw))
    height = max(8, int(th))
    canvas = pygame.Surface((width, height), pygame.SRCALPHA)
    _paint_truck(
        canvas,
        fitted=fitted,
        welds=welds,
        panels=panels,
        kit=kit or "classic",
        body=body,
        accent=accent,
        decal=decal,
        hot=hot,
        show_kit=kit is not None or bool(panels),
    )
    if abs(angle) > 0.4:
        canvas = pygame.transform.rotate(canvas, angle)
    dest = canvas.get_rect(center=(int(left + tw * 0.5), int(top + th * 0.5)))
    surface.blit(canvas, dest)


def draw_icon(
    surface: pygame.Surface,
    kind: str,
    cx: float,
    cy: float,
    size: float,
    time: float,
    color: tuple[int, int, int] | None = None,
) -> None:
    """A part, torch, or paint can, centered on a station or a hand."""
    if kind == "torch":
        _torch(surface, cx, cy, size, time)
        return
    if kind in PAINTS:
        _can(surface, cx, cy, size, color or PAINTS[kind])
        return
    _part_icon(surface, kind, cx, cy, size, color or STEEL)


def _paint_truck(
    canvas: pygame.Surface,
    *,
    fitted: set[str],
    welds: set[str],
    panels: set[str],
    kit: str,
    body: tuple[int, int, int] | None,
    accent: tuple[int, int, int] | None,
    decal: str | None,
    hot: dict[str, float],
    show_kit: bool,
) -> None:
    width, height = canvas.get_size()
    line = max(2, height // 78)
    if "rail" not in fitted:
        _ghost_rail(canvas, width, height, line)
    _frame(canvas, width, height, fitted, line)
    # Beads stay on the bare frame. Panels cover the joints.
    if not panels:
        _beads(canvas, width, height, welds, hot)
    if show_kit and panels:
        _panels(canvas, width, height, kit, panels, body, accent, decal, line)
    if kit == "tube" and "cage" in fitted:
        _cage_tubes(canvas, width, height, (232, 176, 42), line + 1)
    if "towers" in fitted:
        _springs(canvas, width, height)
    if "cabin" in panels or "cage" in fitted:
        _exhaust(canvas, width, height, kit)


def _frame(canvas: pygame.Surface, width: int, height: int, fitted: set[str], line: int) -> None:
    if "rail" in fitted:
        _rail(canvas, width, height, STEEL, line)
        _rivets(canvas, width, height)
    if "towers" in fitted:
        for ax in (0.20, 0.82):
            _box(canvas, (ax - 0.025) * width, 0.60 * height, 0.05 * width, 0.28 * height, STEEL_DARK, 4, line)
            pygame.draw.line(
                canvas,
                STEEL,
                (int((ax - 0.06) * width), int(0.78 * height)),
                (int((ax + 0.06) * width), int(0.64 * height)),
                line,
            )
    if "cage" in fitted:
        _cage_tubes(canvas, width, height, STEEL, line)
    if "bed" in fitted:
        _box(canvas, 0.06 * width, 0.46 * height, 0.026 * width, 0.28 * height, STEEL_DARK, 3, line)
        _box(canvas, 0.30 * width, 0.46 * height, 0.026 * width, 0.28 * height, STEEL_DARK, 3, line)
        _box(canvas, 0.06 * width, 0.46 * height, 0.266 * width, 0.045 * height, STEEL, 3, line)


def _ghost_rail(canvas: pygame.Surface, width: int, height: int, line: int) -> None:
    rect = pygame.Rect(int(0.05 * width), int(0.735 * height), int(0.90 * width), int(0.10 * height))
    pygame.draw.rect(canvas, (186, 194, 204, 170), rect, width=max(3, line), border_radius=6)


def _rail(canvas: pygame.Surface, width: int, height: int, color: tuple[int, ...], line: int) -> None:
    _box(canvas, 0.05 * width, 0.735 * height, 0.90 * width, 0.10 * height, color, 6, line)
    gleam = color[:3] if len(color) == 4 else color
    if len(color) == 3:
        _box(canvas, 0.07 * width, 0.75 * height, 0.86 * width, 0.025 * height, shade(gleam, 1.25), 3, 0)


def _rivets(canvas: pygame.Surface, width: int, height: int) -> None:
    y = int(0.785 * height)
    x = 0.10
    while x < 0.92:
        pygame.draw.circle(canvas, (40, 44, 50), (int(x * width), y), max(2, height // 90))
        x += 0.08


def _cage_tubes(canvas: pygame.Surface, width: int, height: int, color: tuple[int, int, int], line: int) -> None:
    posts = (
        ((0.40, 0.30), (0.40, 0.75)),
        ((0.64, 0.30), (0.64, 0.75)),
        ((0.40, 0.30), (0.64, 0.30)),
        ((0.40, 0.50), (0.64, 0.50)),
        ((0.40, 0.30), (0.64, 0.50)),
    )
    for (x0, y0), (x1, y1) in posts:
        pygame.draw.line(
            canvas,
            INK,
            (int(x0 * width), int(y0 * height)),
            (int(x1 * width), int(y1 * height)),
            line + 3,
        )
        pygame.draw.line(
            canvas,
            color,
            (int(x0 * width), int(y0 * height)),
            (int(x1 * width), int(y1 * height)),
            line + 1,
        )


def _springs(canvas: pygame.Surface, width: int, height: int) -> None:
    for ax in (0.20, 0.82):
        points = []
        coils = 6
        for index in range(coils * 2 + 1):
            y = 0.80 + (0.12 * index / (coils * 2))
            x = ax + (0.028 if index % 2 else -0.028)
            points.append((int(x * width), int(y * height)))
        pygame.draw.lines(canvas, (210, 170, 40), False, points, max(2, height // 90))


def _panels(
    canvas: pygame.Surface,
    width: int,
    height: int,
    kit: str,
    panels: set[str],
    body: tuple[int, int, int] | None,
    accent: tuple[int, int, int] | None,
    decal: str | None,
    line: int,
) -> None:
    color = body or PRIMER
    dark = shade(color, 0.76)
    light = shade(color, 1.18)
    stripe = accent or shade(color, 1.22)
    if kit == "wedge":
        _wedge(canvas, width, height, panels, color, dark, light, stripe, decal, line)
    elif kit == "tube":
        _tube(canvas, width, height, panels, color, dark, light, stripe, decal, line)
    else:
        _classic(canvas, width, height, panels, color, dark, light, stripe, decal, line)


def _classic(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    stripe: tuple[int, int, int],
    decal: str | None,
    line: int,
) -> None:
    if "tail" in panels:
        _box(canvas, 0.055 * width, 0.40 * height, 0.30 * width, 0.35 * height, color, 8, line)
        _box(canvas, 0.08 * width, 0.43 * height, 0.25 * width, 0.055 * height, light, 4, 0)
        _lamp(canvas, 0.08 * width, 0.62 * height, height, (196, 32, 36))
    if "cabin" in panels:
        _box(canvas, 0.36 * width, 0.20 * height, 0.28 * width, 0.55 * height, color, 10, line)
        _box(canvas, 0.39 * width, 0.225 * height, 0.22 * width, 0.045 * height, light, 4, 0)
        _window(canvas, 0.40 * width, 0.29 * height, 0.20 * width, 0.15 * height, line)
        _box(canvas, 0.36 * width, 0.58 * height, 0.28 * width, 0.055 * height, stripe, 2, 0)
    if "nose" in panels:
        _fill(
            canvas,
            [
                (0.63 * width, 0.46 * height),
                (0.90 * width, 0.46 * height),
                (0.97 * width, 0.60 * height),
                (0.94 * width, 0.75 * height),
                (0.63 * width, 0.75 * height),
            ],
            color,
            line,
        )
        _fill(
            canvas,
            [
                (0.66 * width, 0.48 * height),
                (0.88 * width, 0.48 * height),
                (0.92 * width, 0.55 * height),
                (0.66 * width, 0.55 * height),
            ],
            light,
            0,
        )
        pygame.draw.rect(canvas, stripe, pygame.Rect(int(0.63 * width), int(0.58 * height), int(0.32 * width), int(0.055 * height)))
        _lamp(canvas, 0.88 * width, 0.64 * height, height, (255, 236, 170))
        if decal == "flames":
            _flames(canvas, width, height, 0.70, 0.66)
    if decal == "bolt" and "cabin" in panels:
        _bolt(canvas, width, height, stripe)
    _fenders(canvas, width, height)


def _wedge(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    stripe: tuple[int, int, int],
    decal: str | None,
    line: int,
) -> None:
    if "tail" in panels:
        _box(canvas, 0.05 * width, 0.48 * height, 0.26 * width, 0.27 * height, color, 6, line)
        _box(canvas, 0.04 * width, 0.14 * height, 0.28 * width, 0.07 * height, dark, 4, line)
        pygame.draw.line(canvas, INK, (int(0.12 * width), int(0.21 * height)), (int(0.14 * width), int(0.48 * height)), line)
        pygame.draw.line(canvas, INK, (int(0.24 * width), int(0.21 * height)), (int(0.22 * width), int(0.48 * height)), line)
        _lamp(canvas, 0.08 * width, 0.62 * height, height, (196, 32, 36))
    if "cabin" in panels:
        _box(canvas, 0.30 * width, 0.30 * height, 0.22 * width, 0.45 * height, color, 8, line)
        _window(canvas, 0.33 * width, 0.34 * height, 0.16 * width, 0.12 * height, line)
        _box(canvas, 0.30 * width, 0.58 * height, 0.22 * width, 0.05 * height, stripe, 2, 0)
    if "nose" in panels:
        _fill(
            canvas,
            [
                (0.50 * width, 0.32 * height),
                (0.97 * width, 0.56 * height),
                (0.97 * width, 0.75 * height),
                (0.50 * width, 0.75 * height),
            ],
            color,
            line,
        )
        _fill(
            canvas,
            [
                (0.56 * width, 0.40 * height),
                (0.90 * width, 0.56 * height),
                (0.56 * width, 0.56 * height),
            ],
            light,
            0,
        )
        pygame.draw.rect(canvas, stripe, pygame.Rect(int(0.52 * width), int(0.60 * height), int(0.44 * width), int(0.045 * height)))
        _lamp(canvas, 0.90 * width, 0.66 * height, height, (255, 236, 170))
        if decal == "flames":
            _flames(canvas, width, height, 0.72, 0.64)
    if decal == "bolt" and "cabin" in panels:
        _bolt(canvas, width, height, stripe)
    _fenders(canvas, width, height)


def _tube(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    stripe: tuple[int, int, int],
    decal: str | None,
    line: int,
) -> None:
    if "tail" in panels:
        _box(canvas, 0.06 * width, 0.62 * height, 0.30 * width, 0.12 * height, color, 4, line)
        _box(canvas, 0.08 * width, 0.64 * height, 0.26 * width, 0.035 * height, light, 2, 0)
    if "cabin" in panels:
        _box(canvas, 0.40 * width, 0.46 * height, 0.22 * width, 0.28 * height, dark, 4, line)
        _box(canvas, 0.42 * width, 0.50 * height, 0.18 * width, 0.06 * height, stripe, 2, 0)
    if "nose" in panels:
        _box(canvas, 0.66 * width, 0.56 * height, 0.28 * width, 0.18 * height, color, 5, line)
        _lamp(canvas, 0.88 * width, 0.64 * height, height, (255, 236, 170))
        if decal == "flames":
            _flames(canvas, width, height, 0.74, 0.62)
    if decal == "bolt" and "cabin" in panels:
        _bolt(canvas, width, height, stripe)
    _fenders(canvas, width, height)


def _window(canvas: pygame.Surface, x: float, y: float, w: float, h: float, line: int) -> None:
    rect = pygame.Rect(int(x), int(y), max(1, int(w)), max(1, int(h)))
    pygame.draw.rect(canvas, WINDOW, rect, border_radius=5)
    pygame.draw.rect(canvas, INK, rect, width=line, border_radius=5)
    gleam = pygame.Rect(rect.x + 4, rect.y + 3, max(2, rect.w // 3), max(2, rect.h // 4))
    pygame.draw.rect(canvas, (150, 190, 214), gleam, border_radius=2)


def _lamp(canvas: pygame.Surface, x: float, y: float, height: int, color: tuple[int, int, int]) -> None:
    radius = max(4, height // 38)
    pygame.draw.circle(canvas, INK, (int(x), int(y)), radius + 2)
    pygame.draw.circle(canvas, color, (int(x), int(y)), radius)


def _fenders(canvas: pygame.Surface, width: int, height: int) -> None:
    radius = int(height * 0.20)
    thick = max(5, height // 28)
    for ax in (0.20, 0.82):
        cx = int(ax * width)
        cy = int(0.91 * height)
        box = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
        pygame.draw.arc(canvas, INK, box, 0.15, math.pi - 0.15, thick + 3)
        pygame.draw.arc(canvas, (42, 46, 52), box, 0.15, math.pi - 0.15, thick)


def _flames(canvas: pygame.Surface, width: int, height: int, x: float, y: float) -> None:
    tongues = ((0.00, 0.10), (0.06, 0.16), (0.11, 0.09))
    for ox, reach in tongues:
        pts = [
            (int((x + ox) * width), int((y + 0.06) * height)),
            (int((x + ox + 0.03) * width), int((y - reach) * height)),
            (int((x + ox + 0.07) * width), int((y + 0.06) * height)),
        ]
        pygame.draw.polygon(canvas, (232, 96, 24), pts)
        inner = [
            (int((x + ox + 0.02) * width), int((y + 0.05) * height)),
            (int((x + ox + 0.035) * width), int((y - reach * 0.45) * height)),
            (int((x + ox + 0.05) * width), int((y + 0.05) * height)),
        ]
        pygame.draw.polygon(canvas, (255, 214, 64), inner)


def _bolt(canvas: pygame.Surface, width: int, height: int, color: tuple[int, int, int]) -> None:
    points = [
        (int(0.56 * width), int(0.28 * height)),
        (int(0.48 * width), int(0.48 * height)),
        (int(0.54 * width), int(0.48 * height)),
        (int(0.44 * width), int(0.70 * height)),
        (int(0.58 * width), int(0.46 * height)),
        (int(0.51 * width), int(0.46 * height)),
    ]
    pygame.draw.polygon(canvas, INK, points)
    inner = [(x + (2 if x > 0.50 * width else -1), y) for x, y in points]
    pygame.draw.polygon(canvas, color, inner)


def _beads(
    canvas: pygame.Surface,
    width: int,
    height: int,
    welds: set[str],
    hot: dict[str, float],
) -> None:
    for name in welds:
        fx, fy = WELD_AT[name]
        heat = hot.get(name, 0.0)
        center = (int(fx * width), int(fy * height))
        radius = max(4, height // 42)
        if heat > 0.08:
            pygame.draw.circle(canvas, (255, 150, 40), center, int(radius * (1.2 + heat)))
            pygame.draw.circle(canvas, (255, 244, 210), center, max(2, int(radius * heat)))
        else:
            pygame.draw.circle(canvas, (150, 96, 48), center, radius)
            pygame.draw.circle(canvas, (80, 52, 32), center, max(2, radius // 2))


def _exhaust(canvas: pygame.Surface, width: int, height: int, kit: str) -> None:
    pipes = (0.33,) if kit == "wedge" else (0.315, 0.355)
    for ax in pipes:
        rect = pygame.Rect(int(ax * width), int(0.16 * height), max(5, int(0.03 * width)), int(0.36 * height))
        pygame.draw.rect(canvas, INK, rect.inflate(4, 4), border_radius=5)
        pygame.draw.rect(canvas, (150, 158, 168), rect, border_radius=4)
        pygame.draw.rect(canvas, CHROME, pygame.Rect(rect.x + 2, rect.y + 3, max(2, rect.w // 3), rect.h - 8))
        cap = pygame.Rect(rect.x - 1, rect.y - 5, rect.w + 2, max(7, rect.w))
        pygame.draw.ellipse(canvas, (24, 26, 30), cap)
        pygame.draw.ellipse(canvas, (70, 76, 84), cap.inflate(-3, -2))


def _box(
    canvas: pygame.Surface,
    x: float,
    y: float,
    w: float,
    h: float,
    color: tuple[int, ...],
    radius: int,
    line: int,
) -> None:
    rect = pygame.Rect(int(x), int(y), max(1, int(w)), max(1, int(h)))
    pygame.draw.rect(canvas, color, rect, border_radius=radius)
    if line > 0 and len(color) == 3:
        pygame.draw.rect(canvas, INK, rect, width=line, border_radius=radius)


def _fill(
    canvas: pygame.Surface,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    line: int,
) -> None:
    pix = [(int(x), int(y)) for x, y in points]
    pygame.draw.polygon(canvas, color, pix)
    if line > 0:
        pygame.draw.polygon(canvas, INK, pix, line)


def _part_icon(
    surface: pygame.Surface,
    kind: str,
    cx: float,
    cy: float,
    size: float,
    color: tuple[int, int, int],
) -> None:
    line = max(2, int(size * 0.06))
    if kind == "rail":
        _box(surface, cx - size * 0.42, cy - size * 0.08, size * 0.84, size * 0.16, color, 4, line)
    elif kind == "towers":
        _box(surface, cx - size * 0.28, cy - size * 0.28, size * 0.12, size * 0.56, color, 3, line)
        _box(surface, cx + size * 0.16, cy - size * 0.28, size * 0.12, size * 0.56, color, 3, line)
    elif kind == "cage":
        pygame.draw.rect(
            surface,
            color,
            pygame.Rect(int(cx - size * 0.28), int(cy - size * 0.32), int(size * 0.56), int(size * 0.64)),
            line + 1,
            border_radius=4,
        )
    elif kind == "bed":
        pygame.draw.lines(
            surface,
            color,
            False,
            [
                (int(cx - size * 0.34), int(cy - size * 0.16)),
                (int(cx - size * 0.34), int(cy + size * 0.28)),
                (int(cx + size * 0.34), int(cy + size * 0.28)),
                (int(cx + size * 0.34), int(cy - size * 0.16)),
            ],
            line + 1,
        )
    elif kind == "nose":
        _fill(
            surface,
            [
                (cx - size * 0.30, cy - size * 0.05),
                (cx + size * 0.20, cy - size * 0.28),
                (cx + size * 0.36, cy + size * 0.22),
                (cx - size * 0.30, cy + size * 0.22),
            ],
            color,
            line,
        )
    elif kind == "cabin":
        _box(surface, cx - size * 0.26, cy - size * 0.34, size * 0.52, size * 0.64, color, 6, line)
        _window(surface, cx - size * 0.16, cy - size * 0.24, size * 0.32, size * 0.22, line)
    elif kind == "tail":
        _box(surface, cx - size * 0.34, cy - size * 0.22, size * 0.68, size * 0.46, color, 5, line)
    else:
        _box(surface, cx - size * 0.22, cy - size * 0.22, size * 0.44, size * 0.44, color, 6, line)


def _torch(surface: pygame.Surface, cx: float, cy: float, size: float, time: float) -> None:
    handle = pygame.Rect(int(cx - size * 0.08), int(cy - size * 0.02), int(size * 0.16), int(size * 0.46))
    pygame.draw.rect(surface, (32, 34, 38), handle, border_radius=4)
    band = pygame.Rect(int(cx - size * 0.12), int(cy + size * 0.08), int(size * 0.24), int(size * 0.12))
    pygame.draw.rect(surface, (210, 150, 36), band, border_radius=3)
    tip = (cx + size * 0.02, cy - size * 0.08)
    nozzle = (cx + size * 0.10, cy - size * 0.42)
    pygame.draw.line(surface, (170, 130, 50), (int(tip[0]), int(tip[1])), (int(nozzle[0]), int(nozzle[1])), max(3, int(size * 0.05)))
    flick = 0.75 + 0.25 * math.sin(time * 28.0)
    flame = [
        (nozzle[0] - size * 0.06, nozzle[1]),
        (nozzle[0] + size * 0.02, nozzle[1] - size * 0.28 * flick),
        (nozzle[0] + size * 0.10, nozzle[1] + size * 0.02),
    ]
    pygame.draw.polygon(surface, (90, 190, 255), [(int(x), int(y)) for x, y in flame])
    core = [
        (nozzle[0] - size * 0.02, nozzle[1]),
        (nozzle[0] + size * 0.02, nozzle[1] - size * 0.14 * flick),
        (nozzle[0] + size * 0.05, nozzle[1]),
    ]
    pygame.draw.polygon(surface, (255, 250, 230), [(int(x), int(y)) for x, y in core])


def _can(surface: pygame.Surface, cx: float, cy: float, size: float, color: tuple[int, int, int]) -> None:
    body = pygame.Rect(int(cx - size * 0.16), int(cy - size * 0.10), int(size * 0.32), int(size * 0.40))
    pygame.draw.rect(surface, color, body, border_radius=5)
    pygame.draw.rect(surface, INK, body, width=max(2, int(size * 0.04)), border_radius=5)
    top = pygame.Rect(int(cx - size * 0.20), int(cy - size * 0.22), int(size * 0.40), int(size * 0.14))
    pygame.draw.rect(surface, shade(color, 1.15), top, border_radius=3)
    pygame.draw.rect(surface, INK, top, width=max(2, int(size * 0.035)), border_radius=3)
    nozzle = (cx + size * 0.18, cy - size * 0.16)
    pygame.draw.line(surface, (40, 40, 44), (int(cx + size * 0.08), int(cy - size * 0.16)), (int(nozzle[0]), int(nozzle[1])), max(2, int(size * 0.04)))
    pygame.draw.circle(surface, (40, 40, 44), (int(nozzle[0]), int(nozzle[1])), max(3, int(size * 0.05)))
