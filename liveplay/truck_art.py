"""Side-view monster trucks, drawn from the parts the hands have fitted.

Coordinates are fractions of the truck rectangle. Y grows downward.
The nose points right. Weld anchors are the hover targets.

Each kit is its own frame and its own skin. Paint stays primer until a
color is sprayed onto the truck.
"""

from __future__ import annotations

import math

import pygame

FRAME_PARTS = ("rail", "towers", "arms", "cage", "bed", "hoop")
WELD_NAMES = ("rear", "front", "cage", "bed", "hoop", "arm")
PANEL_PARTS = ("nose", "cabin", "tail", "skirt", "crest")
KITS = ("megalodon", "mutt", "digger", "kraken", "dragon")
STICKERS = ("flames", "bolt", "star", "flag")
# Drop spots on the truck, as fractions of the truck rectangle.
STICKER_AT = {
    "hood": (0.76, 0.50),
    "door": (0.50, 0.46),
    "bed": (0.24, 0.48),
    "skirt": (0.42, 0.68),
}

# A part can drop on the jig only after these are already fitted.
PART_NEEDS = {
    "rail": (),
    "towers": ("rail",),
    "arms": ("towers",),
    "cage": ("rail",),
    "bed": ("rail",),
    "hoop": ("rail",),
}

WELD_AT = {
    "rear": (0.20, 0.83),
    "front": (0.80, 0.83),
    "cage": (0.52, 0.32),
    "bed": (0.12, 0.46),
    "hoop": (0.92, 0.62),
    "arm": (0.36, 0.70),
}

PAINTS = {
    "red": (204, 36, 32),
    "yellow": (242, 186, 28),
    "blue": (28, 108, 204),
    "green": (24, 156, 78),
    "orange": (232, 98, 24),
    "white": (236, 238, 236),
}

# Card art only. The truck in the bay stays primer until paint is sprayed.
SHOWCASE = {
    "megalodon": ((18, 92, 176), (226, 236, 242)),
    "mutt": ((156, 96, 46), (236, 196, 120)),
    "digger": ((16, 108, 54), (214, 210, 196)),
    "kraken": ((92, 36, 156), (42, 196, 176)),
    "dragon": ((176, 36, 28), (242, 176, 32)),
}

KIT_LABEL = {
    "megalodon": "MEGALODON",
    "mutt": "MUTT",
    "digger": "DIGGER",
    "kraken": "KRAKEN",
    "dragon": "DRAGON",
}
HOOP_LABEL = {
    "megalodon": "JAW",
    "mutt": "SNOUT",
    "digger": "GATE",
    "kraken": "CURL",
    "dragon": "HORNS",
}
CREST_LABEL = {
    "megalodon": "FIN",
    "mutt": "EARS",
    "digger": "STONE",
    "kraken": "CURLS",
    "dragon": "WING",
}

PRIMER = (176, 170, 160)
INK = (16, 14, 18)
STEEL = (138, 146, 156)
STEEL_DARK = (58, 64, 72)
CHROME = (214, 220, 228)
TOOTH = (244, 240, 228)
EYE = (248, 248, 244)


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
    stickers: dict[str, str] | None = None,
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
        kit=kit if kit in KITS else "megalodon",
        body=body,
        accent=accent,
        decal=decal,
        hot=hot,
        show_skin=bool(panels),
        themed=kit in KITS,
        stickers=stickers or {},
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
    if kind == "livery" or kind in PAINTS:
        _can(surface, cx, cy, size, color or PAINTS.get(kind, (204, 36, 32)))
        return
    if kind in STICKERS:
        _sticker(surface, kind, cx, cy, size)
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
    show_skin: bool,
    themed: bool,
    stickers: dict[str, str],
) -> None:
    width, height = canvas.get_size()
    line = max(3, height // 58)
    if "rail" not in fitted:
        _ghost_rail(canvas, width, height, line)
    _frame(canvas, width, height, fitted, panels, kit if themed else "megalodon", line, themed)
    if not panels:
        _beads(canvas, width, height, welds, hot)
    if show_skin and themed:
        _skin(canvas, width, height, kit, panels, body, accent, decal, line)
        _placed_stickers(canvas, stickers)
    if "towers" in fitted:
        _springs(canvas, width, height)
    if themed and kit == "digger" and ("cabin" in panels or "cage" in fitted):
        _exhaust(canvas, width, height)


def _frame(
    canvas: pygame.Surface,
    width: int,
    height: int,
    fitted: set[str],
    panels: set[str],
    kit: str,
    line: int,
    themed: bool,
) -> None:
    if "rail" in fitted:
        _ladder(canvas, width, height, line)
        if themed and kit in ("megalodon", "dragon") and not panels:
            _spine(canvas, width, height, line, tall=kit == "dragon")
    if "towers" in fitted:
        _towers(canvas, width, height, line)
    if "arms" in fitted:
        _arms(canvas, width, height, line)
    # The skin replaces the kit frame it bolts onto. The chassis stays.
    if "bed" in fitted and themed and "tail" not in panels:
        _bed_frame(canvas, width, height, kit, line)
    if "hoop" in fitted and themed and "nose" not in panels:
        _hoop(canvas, width, height, kit, line)
    if "cage" in fitted and themed and "crest" not in panels:
        _cage(canvas, width, height, kit, line)


def _ladder(canvas: pygame.Surface, width: int, height: int, line: int) -> None:
    _box(canvas, 0.05 * width, 0.805 * height, 0.90 * width, 0.07 * height, STEEL, 5, line)
    _box(canvas, 0.08 * width, 0.818 * height, 0.84 * width, 0.018 * height, shade(STEEL, 1.28), 2, 0)
    _box(canvas, 0.07 * width, 0.69 * height, 0.86 * width, 0.055 * height, STEEL, 4, line)
    _box(canvas, 0.09 * width, 0.70 * height, 0.82 * width, 0.014 * height, shade(STEEL, 1.3), 2, 0)
    for x in (0.16, 0.32, 0.48, 0.64, 0.78):
        _box(canvas, x * width, 0.735 * height, 0.028 * width, 0.08 * height, STEEL_DARK, 2, line)
    y = int(0.838 * height)
    x = 0.12
    while x < 0.90:
        pygame.draw.circle(canvas, (36, 40, 46), (int(x * width), y), max(2, height // 80))
        x += 0.08


def _ghost_rail(canvas: pygame.Surface, width: int, height: int, line: int) -> None:
    color = (186, 194, 204, 180)
    thick = max(3, line)
    upper = pygame.Rect(int(0.07 * width), int(0.69 * height), int(0.86 * width), int(0.055 * height))
    lower = pygame.Rect(int(0.05 * width), int(0.805 * height), int(0.90 * width), int(0.07 * height))
    pygame.draw.rect(canvas, color, upper, width=thick, border_radius=4)
    pygame.draw.rect(canvas, color, lower, width=thick, border_radius=5)


def _spine(canvas: pygame.Surface, width: int, height: int, line: int, *, tall: bool) -> None:
    reach = 0.07 if tall else 0.045
    x = 0.18
    while x < 0.72:
        _fill(
            canvas,
            [(x * width, 0.70 * height), ((x + 0.025) * width, (0.70 - reach) * height), ((x + 0.05) * width, 0.70 * height)],
            STEEL_DARK,
            line,
        )
        x += 0.07


def _towers(canvas: pygame.Surface, width: int, height: int, line: int) -> None:
    for ax in (0.20, 0.82):
        _box(canvas, (ax - 0.028) * width, 0.56 * height, 0.056 * width, 0.32 * height, STEEL_DARK, 3, line)
        _box(canvas, (ax - 0.046) * width, 0.545 * height, 0.092 * width, 0.04 * height, STEEL, 3, line)
        _fill(
            canvas,
            [
                ((ax - 0.02) * width, 0.78 * height),
                ((ax + 0.07) * width, 0.70 * height),
                ((ax + 0.02) * width, 0.78 * height),
            ],
            STEEL,
            line,
        )


def _arms(canvas: pygame.Surface, width: int, height: int, line: int) -> None:
    pairs = (
        ((0.20, 0.60), (0.46, 0.66)),
        ((0.20, 0.78), (0.46, 0.74)),
        ((0.82, 0.60), (0.56, 0.66)),
        ((0.82, 0.78), (0.56, 0.74)),
    )
    for start, end in pairs:
        _stroke(canvas, width, height, (start, end), STEEL, line + 1)
    for ax in (0.34, 0.66):
        _box(canvas, (ax - 0.012) * width, 0.62 * height, 0.024 * width, 0.16 * height, CHROME, 2, line)
        _box(canvas, (ax - 0.02) * width, 0.66 * height, 0.04 * width, 0.07 * height, (210, 168, 40), 2, 0)


def _bed_frame(canvas: pygame.Surface, width: int, height: int, kit: str, line: int) -> None:
    if kit == "megalodon":
        _stroke(canvas, width, height, ((0.08, 0.72), (0.04, 0.48), (0.16, 0.42), (0.22, 0.70)), STEEL, line + 2)
    elif kit == "mutt":
        _stroke(
            canvas,
            width,
            height,
            ((0.10, 0.62), (0.06, 0.40), (0.14, 0.28), (0.22, 0.40), (0.18, 0.66)),
            STEEL,
            line + 2,
        )
    elif kit == "digger":
        for x, top in ((0.08, 0.40), (0.16, 0.32), (0.24, 0.42)):
            _tomb_frame(canvas, width, height, x, top, line)
    elif kit == "kraken":
        _tentacle(canvas, width, height, ((0.28, 0.70), (0.16, 0.55), (0.08, 0.38), (0.14, 0.28), (0.22, 0.40)), STEEL, line, False)
    else:
        _stroke(canvas, width, height, ((0.08, 0.72), (0.05, 0.50), (0.14, 0.46)), STEEL, line + 1)
        for x in (0.08, 0.14, 0.20):
            _fill(canvas, [(x * width, 0.50 * height), ((x + 0.02) * width, 0.40 * height), ((x + 0.04) * width, 0.50 * height)], STEEL_DARK, line)


def _hoop(canvas: pygame.Surface, width: int, height: int, kit: str, line: int) -> None:
    if kit == "megalodon":
        _stroke(canvas, width, height, ((0.78, 0.58), (0.96, 0.52), (0.98, 0.66), (0.80, 0.74)), STEEL, line + 2)
        _stroke(canvas, width, height, ((0.80, 0.78), (0.96, 0.80), (0.94, 0.70)), STEEL, line + 1)
        x = 0.84
        while x < 0.96:
            _fill(canvas, [(x * width, 0.66 * height), ((x + 0.012) * width, 0.74 * height), ((x + 0.026) * width, 0.66 * height)], STEEL_DARK, max(1, line - 1))
            x += 0.035
    elif kit == "mutt":
        _box(canvas, 0.78 * width, 0.62 * height, 0.16 * width, 0.08 * height, STEEL, 4, line)
        for ax in (0.84, 0.92):
            radius = max(6, height // 28)
            pygame.draw.circle(canvas, INK, (int(ax * width), int(0.74 * height)), radius + 2)
            pygame.draw.circle(canvas, STEEL, (int(ax * width), int(0.74 * height)), radius)
    elif kit == "digger":
        _stroke(canvas, width, height, ((0.78, 0.78), (0.78, 0.48), (0.88, 0.40), (0.98, 0.48), (0.98, 0.78)), STEEL, line + 2)
        _stroke(canvas, width, height, ((0.88, 0.46), (0.88, 0.74)), STEEL, line)
    elif kit == "kraken":
        _tentacle(canvas, width, height, ((0.78, 0.74), (0.90, 0.62), (0.98, 0.48), (0.90, 0.40), (0.82, 0.50)), STEEL, line, False)
    else:
        _stroke(canvas, width, height, ((0.76, 0.66), (0.96, 0.60), (0.98, 0.72)), STEEL, line + 2)
        _fill(canvas, [(0.90 * width, 0.52 * height), (0.94 * width, 0.36 * height), (0.98 * width, 0.54 * height)], STEEL_DARK, line)
        _fill(canvas, [(0.82 * width, 0.56 * height), (0.86 * width, 0.42 * height), (0.90 * width, 0.58 * height)], STEEL_DARK, line)


def _cage(canvas: pygame.Surface, width: int, height: int, kit: str, line: int) -> None:
    if kit == "megalodon":
        _stroke(canvas, width, height, ((0.46, 0.70), (0.58, 0.06), (0.70, 0.70)), STEEL, line + 1)
        _stroke(canvas, width, height, ((0.52, 0.70), (0.58, 0.16), (0.64, 0.70)), STEEL, line)
    elif kit == "mutt":
        _stroke(canvas, width, height, ((0.38, 0.70), (0.38, 0.40), (0.52, 0.24), (0.66, 0.40), (0.66, 0.70)), STEEL, line + 1)
        _stroke(canvas, width, height, ((0.38, 0.48), (0.66, 0.48)), STEEL, line)
    elif kit == "digger":
        for x in (0.40, 0.46, 0.52, 0.58, 0.64):
            _stroke(canvas, width, height, ((x, 0.72), (x, 0.34), (x + 0.02, 0.28), (x + 0.04, 0.34)), STEEL, line)
        _stroke(canvas, width, height, ((0.40, 0.46), (0.68, 0.46)), STEEL, line)
    elif kit == "kraken":
        _tentacle(canvas, width, height, ((0.40, 0.72), (0.36, 0.48), (0.46, 0.28), (0.58, 0.18)), STEEL, line, False)
        _tentacle(canvas, width, height, ((0.66, 0.72), (0.72, 0.46), (0.62, 0.30), (0.54, 0.22)), STEEL, line, False)
    else:
        _stroke(canvas, width, height, ((0.46, 0.66), (0.40, 0.28), (0.48, 0.16)), STEEL, line + 2)
        _stroke(canvas, width, height, ((0.60, 0.66), (0.70, 0.26), (0.64, 0.14)), STEEL, line + 2)
        _stroke(canvas, width, height, ((0.42, 0.40), (0.66, 0.36)), STEEL, line)


def _tomb_frame(canvas: pygame.Surface, width: int, height: int, x: float, top: float, line: int) -> None:
    _stroke(canvas, width, height, ((x, 0.74), (x, top), (x + 0.03, top - 0.04), (x + 0.06, top), (x + 0.06, 0.74)), STEEL, line)


def _skin(
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
    dark = shade(color, 0.72)
    light = shade(color, 1.16)
    if kit == "mutt":
        _mutt(canvas, width, height, panels, color, dark, light, accent, line)
    elif kit == "digger":
        _digger(canvas, width, height, panels, color, dark, light, accent, line)
    elif kit == "kraken":
        _kraken(canvas, width, height, panels, color, dark, light, accent, line)
    elif kit == "dragon":
        _dragon(canvas, width, height, panels, color, dark, light, accent, line)
    else:
        _shark(canvas, width, height, panels, color, dark, light, accent, line)
    if "skirt" in panels or "nose" in panels:
        _fenders(canvas, width, height)
    if accent is not None and ("cabin" in panels or "nose" in panels):
        pygame.draw.rect(
            canvas,
            accent,
            pygame.Rect(int(0.34 * width), int(0.60 * height), int(0.58 * width), int(0.045 * height)),
        )
    if decal == "flames" and "nose" in panels:
        _flames(canvas, width, height, 0.72, 0.58)
    elif decal == "bolt" and "cabin" in panels:
        _bolt(canvas, width, height, accent or shade(color, 1.25))


def _shark(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int] | None,
    line: int,
) -> None:
    if "tail" in panels:
        _fill(
            canvas,
            [
                (0.32 * width, 0.50 * height),
                (0.18 * width, 0.46 * height),
                (0.02 * width, 0.24 * height),
                (0.14 * width, 0.52 * height),
                (0.05 * width, 0.76 * height),
                (0.20 * width, 0.64 * height),
                (0.32 * width, 0.74 * height),
            ],
            color,
            line,
        )
    if "cabin" in panels:
        _box(canvas, 0.40 * width, 0.30 * height, 0.26 * width, 0.42 * height, color, 8, line)
        _box(canvas, 0.44 * width, 0.34 * height, 0.16 * width, 0.10 * height, (32, 48, 68), 4, line)
        _eye(canvas, 0.56 * width, 0.52 * height, height * 0.055)
        for index in range(3):
            y = 0.40 + index * 0.06
            pygame.draw.line(canvas, INK, (int(0.44 * width), int(y * height)), (int(0.50 * width), int((y + 0.03) * height)), line)
    if "nose" in panels:
        _fill(canvas, [(0.64 * width, 0.46 * height), (0.96 * width, 0.56 * height), (0.94 * width, 0.74 * height), (0.64 * width, 0.74 * height)], color, line)
        _fill(canvas, [(0.70 * width, 0.50 * height), (0.92 * width, 0.57 * height), (0.70 * width, 0.58 * height)], light, 0)
        pygame.draw.polygon(canvas, (24, 28, 34), [(int(0.72 * width), int(0.64 * height)), (int(0.94 * width), int(0.66 * height)), (int(0.90 * width), int(0.72 * height)), (int(0.74 * width), int(0.72 * height))])
        x = 0.74
        while x < 0.92:
            _fill(canvas, [(x * width, 0.66 * height), ((x + 0.012) * width, 0.76 * height), ((x + 0.026) * width, 0.66 * height)], TOOTH, max(1, line - 1))
            x += 0.032
    if "skirt" in panels:
        _box(canvas, 0.22 * width, 0.72 * height, 0.62 * width, 0.045 * height, light, 3, line)
    if "crest" in panels:
        _fill(canvas, [(0.46 * width, 0.62 * height), (0.58 * width, 0.04 * height), (0.72 * width, 0.62 * height)], color, line)
        _stroke(canvas, width, height, ((0.46, 0.62), (0.58, 0.04)), dark, line + 1)
        if accent is not None:
            _stroke(canvas, width, height, ((0.54, 0.50), (0.60, 0.16)), accent, line)


def _mutt(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int] | None,
    line: int,
) -> None:
    if "tail" in panels:
        _box(canvas, 0.08 * width, 0.42 * height, 0.26 * width, 0.30 * height, color, 12, line)
        _tube(canvas, width, height, ((0.14, 0.46), (0.06, 0.34), (0.04, 0.20), (0.12, 0.14), (0.18, 0.26)), color, max(8, height // 22))
        _exhaust_pair(canvas, width, height, (0.20, 0.24))
    if "cabin" in panels:
        _box(canvas, 0.36 * width, 0.32 * height, 0.30 * width, 0.40 * height, color, 12, line)
        _eye(canvas, 0.46 * width, 0.46 * height, height * 0.045)
        _eye(canvas, 0.58 * width, 0.46 * height, height * 0.045)
        pygame.draw.arc(canvas, INK, pygame.Rect(int(0.44 * width), int(0.50 * height), int(0.16 * width), int(0.10 * height)), math.pi, math.tau, line)
        if accent is not None:
            _paw(canvas, 0.50 * width, 0.62 * height, height * 0.05, accent)
    if "nose" in panels:
        _box(canvas, 0.64 * width, 0.46 * height, 0.30 * width, 0.26 * height, color, 12, line)
        _box(canvas, 0.66 * width, 0.48 * height, 0.22 * width, 0.06 * height, light, 3, 0)
        pygame.draw.ellipse(canvas, (24, 22, 26), pygame.Rect(int(0.86 * width), int(0.52 * height), int(0.07 * width), int(0.10 * height)))
    if "crest" in panels:
        ear = accent or dark
        _fill(
            canvas,
            [
                (0.38 * width, 0.34 * height),
                (0.30 * width, 0.16 * height),
                (0.40 * width, 0.10 * height),
                (0.48 * width, 0.34 * height),
            ],
            color,
            line,
        )
        _fill(canvas, [(0.36 * width, 0.30 * height), (0.34 * width, 0.18 * height), (0.42 * width, 0.16 * height)], ear, 0)
        _fill(
            canvas,
            [
                (0.56 * width, 0.32 * height),
                (0.64 * width, 0.08 * height),
                (0.74 * width, 0.16 * height),
                (0.66 * width, 0.40 * height),
            ],
            color,
            line,
        )
        _fill(canvas, [(0.62 * width, 0.28 * height), (0.66 * width, 0.14 * height), (0.70 * width, 0.20 * height)], ear, 0)
    if "skirt" in panels:
        _box(canvas, 0.24 * width, 0.70 * height, 0.58 * width, 0.05 * height, dark, 3, line)


def _digger(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int] | None,
    line: int,
) -> None:
    stone = accent or shade(color, 0.82)
    if "tail" in panels:
        _box(canvas, 0.05 * width, 0.48 * height, 0.32 * width, 0.26 * height, dark, 4, line)
        _tomb(canvas, width, height, 0.08, 0.36, 0.07, 0.28, color, line, stone)
        _tomb(canvas, width, height, 0.16, 0.28, 0.08, 0.36, color, line, stone)
        _tomb(canvas, width, height, 0.25, 0.38, 0.06, 0.26, color, line, stone)
    if "cabin" in panels:
        _box(canvas, 0.38 * width, 0.22 * height, 0.26 * width, 0.52 * height, color, 6, line)
        for x, top in ((0.34, 0.06), (0.42, 0.00), (0.50, 0.05), (0.58, 0.02)):
            _fill(
                canvas,
                [
                    (x * width, 0.24 * height),
                    ((x + 0.012) * width, top * height),
                    ((x + 0.04) * width, top * height),
                    ((x + 0.052) * width, 0.24 * height),
                ],
                color,
                line,
            )
        _tomb(canvas, width, height, 0.44, 0.28, 0.14, 0.18, (28, 42, 60), line, None)
        _lamp(canvas, 0.58 * width, 0.62 * height, height, (255, 214, 120))
    if "nose" in panels:
        _box(canvas, 0.64 * width, 0.42 * height, 0.30 * width, 0.32 * height, color, 4, line)
        _box(canvas, 0.66 * width, 0.44 * height, 0.24 * width, 0.05 * height, light, 2, 0)
        _lamp(canvas, 0.88 * width, 0.58 * height, height, (255, 220, 140))
    if "crest" in panels:
        _tomb(canvas, width, height, 0.46, 0.06, 0.12, 0.28, color, line, stone)
    if "skirt" in panels:
        x = 0.22
        while x < 0.84:
            _fill(canvas, [(x * width, 0.76 * height), ((x + 0.02) * width, 0.68 * height), ((x + 0.04) * width, 0.76 * height)], dark, max(1, line - 1))
            x += 0.05


def _kraken(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int] | None,
    line: int,
) -> None:
    sucker = accent or shade(color, 0.55)
    if "tail" in panels:
        _tentacle(canvas, width, height, ((0.34, 0.62), (0.20, 0.48), (0.08, 0.32), (0.16, 0.22)), color, line + 7, True, sucker)
        _tentacle(canvas, width, height, ((0.36, 0.70), (0.18, 0.66), (0.08, 0.52), (0.12, 0.40)), color, line + 6, True, sucker)
        _tentacle(canvas, width, height, ((0.30, 0.74), (0.16, 0.78), (0.06, 0.66)), dark, line + 4, True, sucker)
    if "cabin" in panels:
        pygame.draw.ellipse(canvas, color, pygame.Rect(int(0.36 * width), int(0.28 * height), int(0.28 * width), int(0.42 * height)))
        pygame.draw.ellipse(canvas, INK, pygame.Rect(int(0.36 * width), int(0.28 * height), int(0.28 * width), int(0.42 * height)), line)
        _box(canvas, 0.42 * width, 0.34 * height, 0.16 * width, 0.08 * height, light, 3, 0)
    if "nose" in panels:
        pygame.draw.ellipse(canvas, color, pygame.Rect(int(0.60 * width), int(0.40 * height), int(0.34 * width), int(0.32 * height)))
        pygame.draw.ellipse(canvas, INK, pygame.Rect(int(0.60 * width), int(0.40 * height), int(0.34 * width), int(0.32 * height)), line)
        _eye(canvas, 0.78 * width, 0.52 * height, height * 0.07)
        pygame.draw.polygon(canvas, (24, 22, 28), [(int(0.88 * width), int(0.60 * height)), (int(0.96 * width), int(0.66 * height)), (int(0.86 * width), int(0.68 * height))])
    if "crest" in panels:
        _tentacle(canvas, width, height, ((0.48, 0.40), (0.42, 0.22), (0.50, 0.08), (0.58, 0.20)), color, line + 6, True, sucker)
        _tentacle(canvas, width, height, ((0.60, 0.42), (0.68, 0.20), (0.60, 0.08), (0.54, 0.22)), color, line + 6, True, sucker)
    if "skirt" in panels:
        _tentacle(canvas, width, height, ((0.40, 0.72), (0.32, 0.80), (0.24, 0.74)), color, line + 1, True, sucker)
        _tentacle(canvas, width, height, ((0.70, 0.72), (0.78, 0.82), (0.88, 0.74)), color, line + 1, True, sucker)


def _dragon(
    canvas: pygame.Surface,
    width: int,
    height: int,
    panels: set[str],
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int] | None,
    line: int,
) -> None:
    bone = accent or dark
    if "crest" in panels:
        _fill(
            canvas,
            [
                (0.48 * width, 0.58 * height),
                (0.22 * width, 0.08 * height),
                (0.34 * width, 0.22 * height),
                (0.10 * width, 0.30 * height),
                (0.32 * width, 0.36 * height),
                (0.16 * width, 0.46 * height),
                (0.40 * width, 0.48 * height),
                (0.54 * width, 0.68 * height),
            ],
            color,
            line,
        )
        _stroke(canvas, width, height, ((0.50, 0.58), (0.22, 0.08)), bone, line)
        _stroke(canvas, width, height, ((0.48, 0.62), (0.10, 0.30)), bone, line)
        _stroke(canvas, width, height, ((0.46, 0.66), (0.16, 0.46)), bone, line)
    if "tail" in panels:
        _fill(canvas, [(0.04 * width, 0.62 * height), (0.10 * width, 0.48 * height), (0.32 * width, 0.55 * height), (0.34 * width, 0.74 * height), (0.06 * width, 0.74 * height)], color, line)
        for x in (0.08, 0.16, 0.24):
            _fill(canvas, [(x * width, 0.55 * height), ((x + 0.02) * width, 0.40 * height), ((x + 0.045) * width, 0.55 * height)], dark, line)
    if "cabin" in panels:
        _box(canvas, 0.40 * width, 0.34 * height, 0.24 * width, 0.38 * height, color, 8, line)
        _eye(canvas, 0.56 * width, 0.48 * height, height * 0.04)
        pygame.draw.line(canvas, INK, (int(0.52 * width), int(0.46 * height)), (int(0.60 * width), int(0.50 * height)), max(2, line))
        _fill(canvas, [(0.42 * width, 0.36 * height), (0.46 * width, 0.18 * height), (0.50 * width, 0.36 * height)], dark, line)
    if "nose" in panels:
        _fill(canvas, [(0.62 * width, 0.42 * height), (0.92 * width, 0.50 * height), (0.98 * width, 0.62 * height), (0.90 * width, 0.74 * height), (0.62 * width, 0.74 * height)], color, line)
        pygame.draw.polygon(canvas, (28, 22, 24), [(int(0.78 * width), int(0.62 * height)), (int(0.96 * width), int(0.64 * height)), (int(0.88 * width), int(0.72 * height))])
        x = 0.80
        while x < 0.94:
            _fill(canvas, [(x * width, 0.64 * height), ((x + 0.01) * width, 0.72 * height), ((x + 0.022) * width, 0.64 * height)], TOOTH, 0)
            x += 0.03
        pygame.draw.circle(canvas, INK, (int(0.86 * width), int(0.54 * height)), max(3, height // 40))
        _fill(canvas, [(0.70 * width, 0.44 * height), (0.76 * width, 0.28 * height), (0.80 * width, 0.46 * height)], dark, line)
    if "skirt" in panels:
        x = 0.24
        while x < 0.86:
            pygame.draw.arc(
                canvas,
                dark,
                pygame.Rect(int(x * width), int(0.68 * height), int(0.06 * width), int(0.08 * height)),
                0,
                math.pi,
                line,
            )
            x += 0.055


def _tomb(
    canvas: pygame.Surface,
    width: int,
    height: int,
    x: float,
    y: float,
    bw: float,
    bh: float,
    color: tuple[int, int, int],
    line: int,
    mark: tuple[int, int, int] | None,
) -> None:
    rect = pygame.Rect(int(x * width), int((y + bh * 0.35) * height), int(bw * width), int(bh * 0.65 * height))
    cap = pygame.Rect(rect.x, int(y * height), rect.w, int(bh * 0.55 * height))
    pygame.draw.rect(canvas, color, rect)
    pygame.draw.ellipse(canvas, color, cap)
    pygame.draw.rect(canvas, INK, rect, width=line)
    pygame.draw.ellipse(canvas, INK, cap, width=line)
    if mark is not None:
        pygame.draw.line(canvas, mark, (rect.centerx, cap.centery), (rect.centerx, rect.bottom - 4), max(2, line - 1))


def _tentacle(
    canvas: pygame.Surface,
    width: int,
    height: int,
    pts: tuple[tuple[float, float], ...],
    color: tuple[int, int, int],
    line: int,
    suckers: bool,
    sucker: tuple[int, int, int] | None = None,
) -> None:
    _tube(canvas, width, height, pts, color, max(6, line + 4))
    if not suckers:
        return
    ink = sucker or shade(color, 0.6)
    for index, (x, y) in enumerate(pts[1:-1]):
        radius = max(3, height // 32)
        center = (int(x * width), int(y * height))
        pygame.draw.circle(canvas, ink, center, radius)
        pygame.draw.circle(canvas, INK, center, max(1, radius // 2))


def _tube(
    canvas: pygame.Surface,
    width: int,
    height: int,
    pts: tuple[tuple[float, float], ...],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    pix = [(int(x * width), int(y * height)) for x, y in pts]
    samples: list[tuple[int, int]] = []
    for start, end in zip(pix, pix[1:]):
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(dist / 3))
        for index in range(steps):
            blend = index / steps
            samples.append(
                (
                    int(start[0] + (end[0] - start[0]) * blend),
                    int(start[1] + (end[1] - start[1]) * blend),
                )
            )
    if pix:
        samples.append(pix[-1])
    for point in samples:
        pygame.draw.circle(canvas, INK, point, radius + 2)
    for point in samples:
        pygame.draw.circle(canvas, color, point, radius)


def _eye(canvas: pygame.Surface, x: float, y: float, radius: float) -> None:
    center = (int(x), int(y))
    outer = max(4, int(radius))
    pygame.draw.circle(canvas, INK, center, outer + 2)
    pygame.draw.circle(canvas, EYE, center, outer)
    pygame.draw.circle(canvas, (20, 22, 28), center, max(2, outer // 2))
    pygame.draw.circle(canvas, (255, 255, 255), (center[0] - outer // 3, center[1] - outer // 3), max(1, outer // 5))


def _paw(canvas: pygame.Surface, x: float, y: float, radius: float, color: tuple[int, int, int]) -> None:
    pygame.draw.circle(canvas, color, (int(x), int(y)), int(radius))
    for ox, oy in ((-0.7, -0.9), (-0.2, -1.15), (0.35, -1.05), (0.8, -0.7)):
        pygame.draw.circle(canvas, color, (int(x + ox * radius), int(y + oy * radius)), max(2, int(radius * 0.38)))


def _lamp(canvas: pygame.Surface, x: float, y: float, height: int, color: tuple[int, int, int]) -> None:
    radius = max(5, height // 32)
    pygame.draw.circle(canvas, INK, (int(x), int(y)), radius + 2)
    pygame.draw.circle(canvas, color, (int(x), int(y)), radius)
    pygame.draw.circle(canvas, (255, 255, 230), (int(x - radius * 0.3), int(y - radius * 0.3)), max(2, radius // 4))


def _fenders(canvas: pygame.Surface, width: int, height: int) -> None:
    radius = int(height * 0.20)
    thick = max(6, height // 26)
    for ax in (0.20, 0.82):
        box = pygame.Rect(int(ax * width) - radius, int(0.91 * height) - radius, radius * 2, radius * 2)
        pygame.draw.arc(canvas, INK, box, 0.15, math.pi - 0.15, thick + 4)
        pygame.draw.arc(canvas, (36, 40, 48), box, 0.15, math.pi - 0.15, thick)


def _springs(canvas: pygame.Surface, width: int, height: int) -> None:
    for ax in (0.20, 0.82):
        points = []
        coils = 7
        for index in range(coils * 2 + 1):
            y = 0.84 + (0.10 * index / (coils * 2))
            x = ax + (0.026 if index % 2 else -0.026)
            points.append((int(x * width), int(y * height)))
        pygame.draw.lines(canvas, (40, 36, 28), False, points, max(3, height // 70))
        pygame.draw.lines(canvas, (214, 170, 42), False, points, max(2, height // 90))


def _exhaust_pair(canvas: pygame.Surface, width: int, height: int, origin: tuple[float, float]) -> None:
    ax, top = origin
    saved = []
    for shift in (0.0, 0.04):
        saved.append(ax + shift)
    for axle in saved:
        rect = pygame.Rect(int(axle * width), int(top * height), max(6, int(0.028 * width)), int(0.22 * height))
        pygame.draw.rect(canvas, INK, rect.inflate(4, 4), border_radius=5)
        pygame.draw.rect(canvas, (148, 156, 166), rect, border_radius=4)
        cap = pygame.Rect(rect.x - 1, rect.y - 5, rect.w + 2, max(8, rect.w))
        pygame.draw.ellipse(canvas, (24, 26, 30), cap)


def _exhaust(canvas: pygame.Surface, width: int, height: int) -> None:
    for ax in (0.33, 0.37):
        rect = pygame.Rect(int(ax * width), int(0.16 * height), max(6, int(0.028 * width)), int(0.36 * height))
        pygame.draw.rect(canvas, INK, rect.inflate(4, 4), border_radius=5)
        pygame.draw.rect(canvas, (148, 156, 166), rect, border_radius=4)
        pygame.draw.rect(canvas, CHROME, pygame.Rect(rect.x + 2, rect.y + 3, max(2, rect.w // 3), rect.h - 8))
        cap = pygame.Rect(rect.x - 1, rect.y - 5, rect.w + 2, max(8, rect.w))
        pygame.draw.ellipse(canvas, (24, 26, 30), cap)
        pygame.draw.ellipse(canvas, (70, 76, 84), cap.inflate(-3, -2))


def _beads(canvas: pygame.Surface, width: int, height: int, welds: set[str], hot: dict[str, float]) -> None:
    for name in welds:
        fx, fy = WELD_AT[name]
        heat = hot.get(name, 0.0)
        center = (int(fx * width), int(fy * height))
        radius = max(4, height // 40)
        if heat > 0.08:
            pygame.draw.circle(canvas, (255, 150, 40), center, int(radius * (1.2 + heat)))
            pygame.draw.circle(canvas, (255, 244, 210), center, max(2, int(radius * heat)))
        else:
            pygame.draw.circle(canvas, (150, 96, 48), center, radius)
            pygame.draw.circle(canvas, (80, 52, 32), center, max(2, radius // 2))


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
    if len(pix) < 3:
        return
    pygame.draw.polygon(canvas, color, pix)
    if line > 0:
        pygame.draw.polygon(canvas, INK, pix, line)


def _stroke(
    canvas: pygame.Surface,
    width: int,
    height: int,
    pts: tuple[tuple[float, float], ...],
    color: tuple[int, int, int],
    line: int,
) -> None:
    pix = [(int(x * width), int(y * height)) for x, y in pts]
    if len(pix) < 2:
        return
    thick = max(2, line)
    pygame.draw.lines(canvas, INK, False, pix, thick + 3)
    pygame.draw.lines(canvas, color, False, pix, thick)


def _flames(canvas: pygame.Surface, width: int, height: int, x: float, y: float) -> None:
    for ox, reach in ((0.00, 0.10), (0.06, 0.16), (0.11, 0.09)):
        pygame.draw.polygon(
            canvas,
            (232, 96, 24),
            [
                (int((x + ox) * width), int((y + 0.05) * height)),
                (int((x + ox + 0.03) * width), int((y - reach) * height)),
                (int((x + ox + 0.07) * width), int((y + 0.05) * height)),
            ],
        )
        pygame.draw.polygon(
            canvas,
            (255, 214, 64),
            [
                (int((x + ox + 0.02) * width), int((y + 0.04) * height)),
                (int((x + ox + 0.035) * width), int((y - reach * 0.45) * height)),
                (int((x + ox + 0.05) * width), int((y + 0.04) * height)),
            ],
        )


def _placed_stickers(canvas: pygame.Surface, stickers: dict[str, str]) -> None:
    width, height = canvas.get_size()
    for spot, name in stickers.items():
        if name not in STICKERS or spot not in STICKER_AT:
            continue
        fx, fy = STICKER_AT[spot]
        _sticker(canvas, name, fx * width, fy * height, height * 0.18)


def _sticker(surface: pygame.Surface, name: str, cx: float, cy: float, size: float) -> None:
    if name == "flames":
        reach = size * 0.42
        for ox, tall in ((-0.22, 0.7), (0.0, 1.0), (0.22, 0.65)):
            points = [
                (int(cx + ox * size), int(cy + size * 0.28)),
                (int(cx + (ox + 0.08) * size), int(cy - reach * tall)),
                (int(cx + (ox + 0.2) * size), int(cy + size * 0.28)),
            ]
            pygame.draw.polygon(surface, (232, 96, 24), points)
            pygame.draw.polygon(surface, INK, points, max(2, int(size * 0.045)))
            core = [
                (int(cx + (ox + 0.05) * size), int(cy + size * 0.22)),
                (int(cx + (ox + 0.08) * size), int(cy - reach * tall * 0.42)),
                (int(cx + (ox + 0.12) * size), int(cy + size * 0.22)),
            ]
            pygame.draw.polygon(surface, (255, 214, 64), core)
        return
    if name == "bolt":
        points = [
            (cx - size * 0.02, cy - size * 0.36),
            (cx - size * 0.22, cy + size * 0.02),
            (cx - size * 0.04, cy + size * 0.02),
            (cx - size * 0.16, cy + size * 0.36),
            (cx + size * 0.16, cy - size * 0.04),
            (cx + size * 0.02, cy - size * 0.04),
        ]
        pygame.draw.polygon(surface, INK, [(int(x), int(y)) for x, y in points])
        pygame.draw.polygon(surface, (255, 214, 48), [(int(x + 1), int(y)) for x, y in points])
        return
    if name == "star":
        points = []
        for index in range(10):
            radius = size * (0.38 if index % 2 == 0 else 0.16)
            angle = -math.pi / 2 + index * math.pi / 5
            points.append((int(cx + math.cos(angle) * radius), int(cy + math.sin(angle) * radius)))
        pygame.draw.polygon(surface, INK, points)
        inner = []
        for index in range(10):
            radius = size * (0.30 if index % 2 == 0 else 0.12)
            angle = -math.pi / 2 + index * math.pi / 5
            inner.append((int(cx + math.cos(angle) * radius), int(cy + math.sin(angle) * radius)))
        pygame.draw.polygon(surface, (255, 248, 236), inner)
        return
    pole = pygame.Rect(int(cx - size * 0.28), int(cy - size * 0.34), max(3, int(size * 0.06)), int(size * 0.7))
    pygame.draw.rect(surface, (236, 236, 236), pole)
    pygame.draw.rect(surface, INK, pole, width=max(1, int(size * 0.02)))
    flag = pygame.Rect(int(cx - size * 0.22), int(cy - size * 0.34), int(size * 0.5), int(size * 0.32))
    pygame.draw.rect(surface, (236, 236, 236), flag)
    cell_w = max(2, flag.w // 4)
    cell_h = max(2, flag.h // 3)
    for row in range(3):
        for col in range(4):
            if (row + col) % 2 == 0:
                pygame.draw.rect(surface, INK, pygame.Rect(flag.x + col * cell_w, flag.y + row * cell_h, cell_w, cell_h))


def _bolt(canvas: pygame.Surface, width: int, height: int, color: tuple[int, int, int]) -> None:
    points = [
        (int(0.56 * width), int(0.30 * height)),
        (int(0.48 * width), int(0.48 * height)),
        (int(0.54 * width), int(0.48 * height)),
        (int(0.44 * width), int(0.68 * height)),
        (int(0.58 * width), int(0.46 * height)),
        (int(0.51 * width), int(0.46 * height)),
    ]
    pygame.draw.polygon(canvas, INK, points)
    pygame.draw.polygon(canvas, color, [(px + 1, py) for px, py in points])


def _part_icon(surface: pygame.Surface, kind: str, cx: float, cy: float, size: float, color: tuple[int, int, int]) -> None:
    line = max(2, int(size * 0.06))
    if kind == "rail":
        _box(surface, cx - size * 0.42, cy - size * 0.16, size * 0.84, size * 0.10, color, 3, line)
        _box(surface, cx - size * 0.42, cy + size * 0.06, size * 0.84, size * 0.10, color, 3, line)
    elif kind == "towers":
        _box(surface, cx - size * 0.28, cy - size * 0.30, size * 0.12, size * 0.60, color, 3, line)
        _box(surface, cx + size * 0.16, cy - size * 0.30, size * 0.12, size * 0.60, color, 3, line)
    elif kind == "arms":
        pygame.draw.line(surface, color, (int(cx - size * 0.34), int(cy - size * 0.16)), (int(cx + size * 0.34), int(cy + size * 0.04)), line + 2)
        pygame.draw.line(surface, color, (int(cx - size * 0.34), int(cy + size * 0.16)), (int(cx + size * 0.34), int(cy - size * 0.02)), line + 2)
    elif kind == "cage":
        pygame.draw.rect(surface, color, pygame.Rect(int(cx - size * 0.26), int(cy - size * 0.30), int(size * 0.52), int(size * 0.60)), line + 1, border_radius=4)
    elif kind == "bed":
        pygame.draw.lines(
            surface,
            color,
            False,
            [
                (int(cx - size * 0.30), int(cy - size * 0.10)),
                (int(cx - size * 0.34), int(cy + size * 0.24)),
                (int(cx + size * 0.34), int(cy + size * 0.24)),
                (int(cx + size * 0.30), int(cy - size * 0.10)),
            ],
            line + 1,
        )
    elif kind == "hoop":
        pygame.draw.arc(surface, color, pygame.Rect(int(cx - size * 0.28), int(cy - size * 0.28), int(size * 0.56), int(size * 0.56)), 0.4, math.pi - 0.2, line + 2)
    elif kind == "nose":
        _fill(surface, [(cx - size * 0.28, cy), (cx + size * 0.16, cy - size * 0.22), (cx + size * 0.34, cy + size * 0.18), (cx - size * 0.28, cy + size * 0.18)], color, line)
    elif kind == "cabin":
        _box(surface, cx - size * 0.24, cy - size * 0.30, size * 0.48, size * 0.58, color, 6, line)
    elif kind == "tail":
        _box(surface, cx - size * 0.32, cy - size * 0.18, size * 0.64, size * 0.40, color, 5, line)
    elif kind == "skirt":
        _box(surface, cx - size * 0.40, cy - size * 0.06, size * 0.80, size * 0.14, color, 3, line)
    elif kind == "crest":
        _fill(surface, [(cx, cy - size * 0.36), (cx - size * 0.28, cy + size * 0.28), (cx + size * 0.28, cy + size * 0.28)], color, line)
    else:
        _box(surface, cx - size * 0.2, cy - size * 0.2, size * 0.4, size * 0.4, color, 4, line)


def _torch(surface: pygame.Surface, cx: float, cy: float, size: float, time: float) -> None:
    handle = pygame.Rect(int(cx - size * 0.08), int(cy - size * 0.02), int(size * 0.16), int(size * 0.46))
    pygame.draw.rect(surface, (32, 34, 38), handle, border_radius=4)
    band = pygame.Rect(int(cx - size * 0.12), int(cy + size * 0.08), int(size * 0.24), int(size * 0.12))
    pygame.draw.rect(surface, (210, 150, 36), band, border_radius=3)
    nozzle = (cx + size * 0.10, cy - size * 0.42)
    pygame.draw.line(surface, (170, 130, 50), (int(cx), int(cy - size * 0.05)), (int(nozzle[0]), int(nozzle[1])), max(3, int(size * 0.05)))
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
