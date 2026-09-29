"""Fullscreen (or windowed) table display.

The playfield is flat gray on purpose. The overhead camera sees this
window. A picture, gradient, or bright background would be subtracted as
a hand. Particles are drawn on top by the game; this module only owns
the window, alignment guides, and operator messages.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from liveplay.config import AppConfig, DisplayConfig
from liveplay.errors import DisplayError
from liveplay.points import InteractionPoint


def list_displays() -> int:
    pygame.init()
    try:
        sizes = pygame.display.get_desktop_sizes()
    except pygame.error as exc:
        print(f"[liveplay] error: could not list displays: {exc}")
        return 1
    finally:
        pygame.quit()
    if not sizes:
        print("[liveplay] no displays reported.")
        return 1
    for index, (width, height) in enumerate(sizes):
        print(f"[liveplay] display {index}: {width}x{height}")
    return 0


def create_display(cfg: AppConfig) -> pygame.Surface:
    if not pygame.get_init():
        pygame.init()
    pygame.display.init()
    index = resolve_display_index(cfg.display)
    size = (cfg.display.width, cfg.display.height)
    screen = _open_window(size, index, cfg.display.fullscreen)
    pygame.display.set_caption("Live Play")
    pygame.mouse.set_visible(not cfg.display.fullscreen)
    pygame.key.set_repeat(180, 40)
    sizes = _desktop_sizes()
    native = sizes[index] if index < len(sizes) else None
    mode = "fullscreen" if cfg.display.fullscreen else "windowed"
    native_text = f"{native[0]}x{native[1]}" if native else "unknown"
    print(
        f"[liveplay] {mode} {size[0]}x{size[1]} on display {index} "
        f"(desktop {native_text})"
    )
    if native is not None and native != size:
        print(
            f"[liveplay] warning: display {index} is {native_text}, not "
            f"{size[0]}x{size[1]}. The table TV should be 1080p. "
            f"Pass --display to pick a different screen."
        )
    return screen


def resolve_display_index(display: DisplayConfig) -> int:
    return choose_display(
        _desktop_sizes(),
        index=display.index,
        prefer_external=display.prefer_external,
        target=(display.width, display.height),
    )


def choose_display(
    sizes: list[tuple[int, int]],
    *,
    index: int | None,
    prefer_external: bool,
    target: tuple[int, int],
) -> int:
    """Pick the table TV.

    An explicit index wins. Otherwise, if more than one display is connected,
    prefer a non-primary display that is already 1920x1080 (the Vizio), and
    fall back to the last display. macOS usually lists the built-in panel
    first and the HDMI output last.
    """
    if index is not None:
        if not sizes or index < 0 or index >= len(sizes):
            if sizes:
                listing = ", ".join(f"{i}: {w}x{h}" for i, (w, h) in enumerate(sizes))
            else:
                listing = "none"
            raise DisplayError(
                f"Display index {index} is out of range. "
                f"Found {len(sizes)} ({listing}). Run --list-displays."
            )
        return index
    if not sizes:
        return 0
    if prefer_external and len(sizes) > 1:
        for candidate, size in enumerate(sizes):
            if candidate == 0:
                continue
            if size == target:
                return candidate
        return len(sizes) - 1
    return 0


def fill_playfield(surface: pygame.Surface, gray: int) -> None:
    surface.fill((gray, gray, gray))


def blit_bgr(surface: pygame.Surface, frame_bgr) -> None:
    """Copy a BGR numpy frame onto the pygame surface. Sizes must match."""
    import numpy as np

    if frame_bgr.shape[1] != surface.get_width() or frame_bgr.shape[0] != surface.get_height():
        raise DisplayError(
            f"Camera frame is {frame_bgr.shape[1]}x{frame_bgr.shape[0]} but the "
            f"window is {surface.get_width()}x{surface.get_height()}."
        )
    # surfarray wants width, height, RGB. Copying here is cheaper than a
    # pitch bug in frombuffer when the width is not a multiple of 4.
    rgb = np.ascontiguousarray(np.transpose(frame_bgr[:, :, ::-1], (1, 0, 2)))
    surface.blit(pygame.surfarray.make_surface(rgb), (0, 0))


def draw_alignment_guides(surface: pygame.Surface) -> None:
    """Corner ticks for adult alignment. Not drawn during play or snapshot."""
    width, height = surface.get_size()
    color = (240, 240, 240)
    length = max(24, min(width, height) // 18)
    thickness = 3
    corners = (
        (0, 0, 1, 1),
        (width - 1, 0, -1, 1),
        (0, height - 1, 1, -1),
        (width - 1, height - 1, -1, -1),
    )
    for x, y, sx, sy in corners:
        pygame.draw.line(surface, color, (x, y), (x + sx * length, y), thickness)
        pygame.draw.line(surface, color, (x, y), (x, y + sy * length), thickness)
    cx, cy = width // 2, height // 2
    pygame.draw.line(surface, color, (cx - 16, cy), (cx + 16, cy), 1)
    pygame.draw.line(surface, color, (cx, cy - 16), (cx, cy + 16), 1)


def draw_points(surface: pygame.Surface, points: list[InteractionPoint]) -> None:
    for point in points:
        color = (255, 210, 70) if point.is_new else (80, 255, 140)
        center = (int(point.x), int(point.y))
        radius = max(8, int(point.size))
        pygame.draw.circle(surface, color, center, radius, 3)
        pygame.draw.circle(surface, color, center, 4)


def draw_message(surface: pygame.Surface, message: str, gray: int = 90) -> None:
    surface.fill((gray, gray, gray))
    title_font = pygame.font.Font(None, 72)
    body_font = pygame.font.Font(None, 40)
    lines = _wrap(message, body_font, surface.get_width() - 160)
    y = max(80, surface.get_height() // 2 - (len(lines) + 2) * 24)
    title = title_font.render("Live Play", True, (250, 250, 250))
    surface.blit(title, (80, y))
    y += 90
    for line in lines:
        rendered = body_font.render(line, True, (230, 230, 230))
        surface.blit(rendered, (80, y))
        y += 48


def present_error(screen: pygame.Surface, message: str, wait_s: float = 20.0) -> None:
    print(f"[liveplay] error: {message}")
    draw_message(screen, message)
    pygame.display.flip()
    if wait_s <= 0:
        return
    deadline = pygame.time.get_ticks() + int(wait_s * 1000)
    while pygame.time.get_ticks() < deadline:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                return
        pygame.time.wait(40)


def _open_window(size: tuple[int, int], index: int, fullscreen: bool) -> pygame.Surface:
    attempts: list[tuple[int, dict]] = []
    if fullscreen:
        attempts.append((pygame.FULLSCREEN, {"display": index, "vsync": 1}))
        attempts.append((pygame.FULLSCREEN, {"display": index}))
        attempts.append((pygame.FULLSCREEN, {}))
    else:
        attempts.append((pygame.SCALED, {"display": index}))
        attempts.append((pygame.SCALED, {}))
        attempts.append((0, {}))
    last_error: Exception | None = None
    for flags, kwargs in attempts:
        try:
            return pygame.display.set_mode(size, flags, **kwargs)
        except TypeError as exc:
            last_error = exc
        except pygame.error as exc:
            last_error = exc
    raise DisplayError(f"Could not open the display: {last_error}")


def _desktop_sizes() -> list[tuple[int, int]]:
    try:
        return list(pygame.display.get_desktop_sizes())
    except pygame.error:
        return []


def _wrap(message: str, font: pygame.font.Font, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in message.splitlines():
        if paragraph == "":
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if font.size(trial)[0] <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [""]
