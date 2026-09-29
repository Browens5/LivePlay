"""Fullscreen (or windowed) table display.

The playfield is flat gray on purpose. The overhead camera sees this
window. A picture, gradient, or bright background would be subtracted as
a hand. Particles are drawn on top by the game; this module only owns
the window, alignment guides, and operator messages.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from liveplay import sdl_env  # noqa: F401  # before pygame
import pygame

from liveplay.config import AppConfig, DisplayConfig
from liveplay.errors import DisplayError
from liveplay.points import InteractionPoint

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_warned_pygame = False


def init_video() -> None:
    """Start the window system, not the audio mixer or pygame.font.

    pygame.font is not used. On Python 3.14 the stock pygame package has
    no wheels, and the source build crashes in pygame.font.init().
    """
    pygame.display.init()
    _warn_pygame_build()


def _warn_pygame_build() -> None:
    global _warned_pygame
    if _warned_pygame or sys.version_info < (3, 14):
        return
    if getattr(pygame, "IS_CE", False):
        return
    _warned_pygame = True
    print(
        "[liveplay] warning: this Python is 3.14 or newer and the installed "
        "pygame is not pygame-ce. The original pygame package has no 3.14 "
        "wheels, so pip built it from source and pygame.font crashes. "
        "Operator text does not use that module. If the window itself fails, run:\n"
        "  pip uninstall -y pygame\n"
        "  pip install -r requirements.txt"
    )


def list_displays() -> int:
    try:
        init_video()
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
    init_video()
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


def fill_visible(surface: pygame.Surface, visible: tuple[int, int, int, int], gray: int) -> None:
    """Gray only where the TV can show pixels. The cropped margin stays black."""
    surface.fill((0, 0, 0))
    x, y, width, height = visible
    if width > 0 and height > 0:
        pygame.draw.rect(surface, (gray, gray, gray), (x, y, width, height))


def draw_visible_border(surface: pygame.Surface, visible: tuple[int, int, int, int]) -> None:
    x, y, width, height = visible
    if width > 1 and height > 1:
        pygame.draw.rect(surface, (245, 245, 245), (x, y, width, height), 4)


# RGB. Chosen so the overhead camera's HSV reading misses the skin bands
# in vision.py (hue 0–20 and 170–180). Yellow and pink sit on those edges.
POSE_OUTLINE = (40, 210, 255)
POSE_PALM = (50, 230, 110)
POSE_TIP = (90, 150, 255)


def draw_poses(surface: pygame.Surface, poses) -> None:
    """Project a hand outline, palm, and fingertips onto the glass.

    The camera sees this drawing. The colors fail the skin gate, so the
    pose itself is less likely to be tracked as another hand.
    """
    outline = POSE_OUTLINE
    palm_color = POSE_PALM
    tip_color = POSE_TIP
    for pose in poses:
        palm = (int(pose.x), int(pose.y))
        if len(pose.contour) >= 3:
            points = [(int(x), int(y)) for x, y in pose.contour]
            pygame.draw.lines(surface, outline, True, points, 3)
        pygame.draw.circle(surface, palm_color, palm, max(10, int(pose.size * 0.35)), 3)
        for tip_x, tip_y in pose.fingertips:
            tip = (int(tip_x), int(tip_y))
            pygame.draw.line(surface, outline, palm, tip, 2)
            pygame.draw.circle(surface, tip_color, tip, 7)


def draw_caption(
    surface: pygame.Surface,
    text: str,
    box: tuple[int, int, int, int] | None = None,
    gray: int = 90,
) -> None:
    """A short line on the bottom edge of `box`. The middle of the glass stays clear.

    The pose check used to paint a full-screen card over the hand outline.
    This leaves the tracking graphics visible and only labels the bottom edge.
    """
    if not text:
        return
    if box is None:
        x, y, width, height = 0, 0, surface.get_width(), surface.get_height()
    else:
        x, y, width, height = (int(v) for v in box)
    if width < 8 or height < 8:
        return
    scale, thickness, pad = 0.7, 2, 8
    max_width = max(40, width - pad * 2)
    lines = _wrap(text, scale, thickness, max_width)[:2]
    line_h, baseline = _line_metrics(scale, thickness)
    step = line_h + baseline + 4
    text_w = max(_text_width(line or " ", scale, thickness) for line in lines)
    image_w = min(width, text_w + pad * 2)
    image_h = min(height, pad * 2 + step * len(lines))
    image = np.full((image_h, image_w, 3), gray, dtype=np.uint8)
    cursor = pad + line_h
    for line in lines:
        if line:
            cv2.putText(
                image,
                line,
                (pad, cursor),
                _FONT,
                scale,
                (245, 245, 245),
                thickness,
                cv2.LINE_AA,
            )
        cursor += step
    pos_x = int(x) + max(0, (width - image_w) // 2)
    pos_y = int(y) + max(0, height - image_h - 6)
    _blit_bgr_at(surface, image, (pos_x, pos_y))


def draw_points(surface: pygame.Surface, points: list[InteractionPoint]) -> None:
    for point in points:
        color = (255, 210, 70) if point.is_new else (80, 255, 140)
        center = (int(point.x), int(point.y))
        radius = max(8, int(point.size))
        pygame.draw.circle(surface, color, center, radius, 3)
        pygame.draw.circle(surface, color, center, 4)


def draw_message(surface: pygame.Surface, message: str, gray: int = 90) -> None:
    """Operator text. OpenCV's built-in font, not pygame.font.

    pygame.font crashes on the source build pip produces for Python 3.14.
    """
    width, height = surface.get_size()
    image = np.full((height, width, 3), gray, dtype=np.uint8)
    margin = min(80, max(16, width // 12))
    max_width = max(40, width - margin * 2)
    title_scale, body_scale = 2.0, 1.15
    thickness = 2
    lines = _wrap(message, body_scale, thickness, max_width)
    title_h, title_base = _line_metrics(title_scale, thickness + 1)
    body_h, body_base = _line_metrics(body_scale, thickness)
    body_step = body_h + body_base + 16
    block = title_h + title_base + 40 + max(1, len(lines)) * body_step
    baseline = max(margin + title_h, (height - block) // 2 + title_h)
    cv2.putText(
        image,
        "Live Play",
        (margin, baseline),
        _FONT,
        title_scale,
        (250, 250, 250),
        thickness + 1,
        cv2.LINE_AA,
    )
    baseline += title_base + 40 + body_h
    for line in lines:
        if line:
            cv2.putText(
                image,
                line,
                (margin, baseline),
                _FONT,
                body_scale,
                (235, 235, 235),
                thickness,
                cv2.LINE_AA,
            )
        baseline += body_step
    blit_bgr(surface, image)


def draw_status(surface: pygame.Surface, text: str) -> None:
    """One warning line over the camera view. No pygame.font."""
    if not text:
        return
    scale, thickness, pad = 0.75, 2, 12
    max_width = max(40, surface.get_width() - 48 - pad * 2)
    lines = _wrap(text, scale, thickness, max_width)
    line_h, baseline = _line_metrics(scale, thickness)
    step = line_h + baseline + 6
    text_w = max(_text_width(line or " ", scale, thickness) for line in lines)
    image_w = min(surface.get_width() - 32, text_w + pad * 2)
    image_h = pad * 2 + step * len(lines)
    image = np.full((image_h, image_w, 3), 20, dtype=np.uint8)
    y = pad + line_h
    for line in lines:
        if line:
            cv2.putText(
                image,
                line,
                (pad, y),
                _FONT,
                scale,
                (160, 230, 255),
                thickness,
                cv2.LINE_AA,
            )
        y += step
    _blit_bgr_at(surface, image, (24, 24))


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


def _wrap(message: str, scale: float, thickness: int, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in message.splitlines():
        if paragraph == "":
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            trial = word if not current else f"{current} {word}"
            if _text_width(trial, scale, thickness) <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [""]


def _line_metrics(scale: float, thickness: int) -> tuple[int, int]:
    (_width, height), baseline = cv2.getTextSize("Ag", _FONT, scale, thickness)
    return height, baseline


def _text_width(text: str, scale: float, thickness: int) -> int:
    (width, _height), _baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    return width


def _blit_bgr_at(surface: pygame.Surface, frame_bgr: np.ndarray, pos: tuple[int, int]) -> None:
    rgb = np.ascontiguousarray(np.transpose(frame_bgr[:, :, ::-1], (1, 0, 2)))
    surface.blit(pygame.surfarray.make_surface(rgb), pos)
