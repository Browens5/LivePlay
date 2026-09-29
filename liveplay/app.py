"""One process: capture → vision → points → particles → the table TV.

Modes
-----
passthrough  Camera mapped onto the TV, with corner ticks. Align the crop.
overlay      Same image plus blob circles. Check tracking before play.
play         Flat gray playfield and particles only. This is the kid mode.
calibrate    Flat gray. SPACE snapshots the empty table.

Adult keys (keyboard on the Mac, nothing drawn for kids to tap):
  Esc or Cmd/Ctrl+Q   quit
  1 / 2 / 3           passthrough / overlay / play
  C                   calibrate
  Space               snapshot, while calibrating
  Arrows              nudge the camera crop (passthrough and overlay)
  Shift+arrows        resize the crop
  S                   write the crop back to config.json
"""

from __future__ import annotations

import sys
import time

from liveplay.calibration import load_calibration, save_calibration
from liveplay.capture import nudge_roi, open_frame_source
from liveplay.config import AppConfig, load_config, parse_args
from liveplay.display import (
    blit_bgr,
    create_display,
    draw_alignment_guides,
    draw_message,
    draw_points,
    fill_playfield,
    list_displays,
    present_error,
)
from liveplay.errors import CalibrationError, CaptureError, ConfigError, LivePlayError
from liveplay.particles import ParticleSystem
from liveplay.points import PointTracker
from liveplay.vision import BlobVision, make_backend

# How long the screen stays pure gray before we trust the camera frame.
# The instruction text is itself a bright blob the webcam would memorize.
_CALIBRATION_HOLD_S = 0.75
_STALL_FRAMES = 45


class Mode:
    PASSTHROUGH = "passthrough"
    OVERLAY = "overlay"
    PLAY = "play"
    CALIBRATE = "calibrate"


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    if args.list_displays:
        return list_displays()
    try:
        cfg = load_config(args)
    except ConfigError as exc:
        print(f"[liveplay] error: {exc}", file=sys.stderr)
        return 1
    try:
        return run(cfg)
    except LivePlayError as exc:
        print(f"[liveplay] error: {exc}", file=sys.stderr)
        return 1


def run(cfg: AppConfig) -> int:
    import pygame

    try:
        pygame.init()
    except pygame.error as exc:
        print(f"[liveplay] error: could not start video: {exc}", file=sys.stderr)
        return 1
    screen = None
    source = None
    try:
        screen = create_display(cfg)
        source = open_frame_source(cfg)
        session = Session(cfg, screen, source)
        session.run()
        return 0
    except LivePlayError as exc:
        if screen is not None:
            present_error(screen, str(exc), wait_s=0 if cfg.frames else 20)
        else:
            print(f"[liveplay] error: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()
        pygame.quit()


class Session:
    def __init__(self, cfg: AppConfig, screen, source) -> None:
        import pygame

        self.pygame = pygame
        self.cfg = cfg
        self.screen = screen
        self.source = source
        self.clock = pygame.time.Clock()
        self.running = True
        self.frames = 0
        self.dt = 1.0 / 60.0
        self.fail_reads = 0
        self.mode = cfg.mode
        self.return_mode = cfg.mode if cfg.mode != Mode.CALIBRATE else Mode.PLAY
        self.arm_until: float | None = None
        self._shown_warning: str | None = None
        self._log_at = time.perf_counter()
        self._last_points = 0
        background = self._load_background()
        self.backend: BlobVision = make_backend(cfg.vision, background)
        self.tracker = PointTracker(cfg.vision.match_distance, cfg.vision.smoothing)
        self.particles = ParticleSystem(cfg.particles)
        if (
            self.mode in (Mode.PLAY, Mode.OVERLAY)
            and self.backend.needs_calibration
            and not self.backend.has_background
        ):
            print("[liveplay] no empty-table snapshot yet. Entering calibration.")
            print("[liveplay] clear the table, then press SPACE. Esc quits.")
            self.return_mode = self.mode
            self.mode = Mode.CALIBRATE
        print(
            f"[liveplay] mode {self.mode}  vision {cfg.vision.method}  "
            f"particles {cfg.particles.style}. Esc or Cmd/Ctrl+Q quits."
        )

    def run(self) -> None:
        while self.running:
            self._events()
            if not self.running:
                break
            frame = self.source.read()
            if frame is None:
                self._on_missing_frame()
                continue
            self.fail_reads = 0
            self._tick(frame)
            self.pygame.display.flip()
            self.clock.tick(60)
            self.dt = max(self.clock.get_time() / 1000.0, 1.0 / 120.0)
            self.frames += 1
            self._maybe_log()
            if self.cfg.frames and self.frames >= self.cfg.frames:
                self.running = False
            if self.cfg.display.fullscreen:
                self.pygame.mouse.set_visible(False)

    def _load_background(self):
        path = self.cfg.calibration.path
        image = load_calibration(path)
        if image is None:
            return None
        expected = (self.cfg.display.height, self.cfg.display.width, 3)
        if image.shape != expected:
            print(
                f"[liveplay] calibration {path} is {image.shape[1]}x{image.shape[0]}, "
                f"but the playfield is {self.cfg.display.width}x{self.cfg.display.height}. "
                f"Press C, clear the table, and press SPACE to recapture."
            )
            return None
        print(f"[liveplay] loaded calibration {path}")
        return image

    def _tick(self, frame) -> None:
        if self.mode == Mode.CALIBRATE:
            self._tick_calibrate(frame)
        elif self.mode == Mode.PASSTHROUGH:
            blit_bgr(self.screen, frame)
            draw_alignment_guides(self.screen)
        elif self.mode == Mode.OVERLAY:
            points = self._detect(frame)
            blit_bgr(self.screen, frame)
            draw_points(self.screen, points)
            draw_alignment_guides(self.screen)
            self._draw_warning()
        elif self.mode == Mode.PLAY:
            points = self._detect(frame)
            self.particles.update(points, self.dt)
            # Flat gray matches the calibration field. Do not put a picture
            # or gradient here; the camera would track the artwork.
            fill_playfield(self.screen, self.cfg.calibration.gray)
            self.particles.draw(self.screen, self.cfg.calibration.gray)
        else:
            raise ConfigError(f"Unknown mode {self.mode!r}.")

    def _tick_calibrate(self, frame) -> None:
        gray = self.cfg.calibration.gray
        fill_playfield(self.screen, gray)
        now = time.perf_counter()
        if self.arm_until is not None:
            if now >= self.arm_until:
                self._save_snapshot(frame)
            return
        draw_message(
            self.screen,
            "Clear hands and toys off the table.\n"
            "The screen stays flat gray so the camera has a clean background.\n"
            "Press SPACE to snapshot the empty table.\n"
            "Esc quits.",
            gray=gray,
        )

    def _save_snapshot(self, frame) -> None:
        path = self.cfg.calibration.path
        try:
            save_calibration(path, frame)
        except CalibrationError:
            self.arm_until = None
            raise
        self.backend.set_background(frame)
        self.tracker.reset()
        self.particles.clear()
        self.arm_until = None
        height, width = frame.shape[:2]
        print(f"[liveplay] saved calibration {path} ({width}x{height})")
        self.mode = self.return_mode if self.return_mode != Mode.CALIBRATE else Mode.PLAY
        print(f"[liveplay] mode {self.mode}")

    def _detect(self, frame):
        points = self.backend.detect(frame)
        points = self.tracker.update(points, self.dt)
        self._last_points = len(points)
        self._report_warning()
        return points

    def _report_warning(self) -> None:
        warning = self.backend.last_warning
        if warning and warning != self._shown_warning:
            print(f"[liveplay] {warning}")
        self._shown_warning = warning

    def _draw_warning(self) -> None:
        warning = self.backend.last_warning
        if not warning:
            return
        font = self.pygame.font.Font(None, 36)
        rendered = font.render(warning, True, (255, 230, 160))
        self.screen.blit(rendered, (24, 24))

    def _on_missing_frame(self) -> None:
        self.fail_reads += 1
        if self.fail_reads == 1:
            print("[liveplay] camera frame missing, retrying...")
        draw_message(
            self.screen,
            "The camera stopped sending frames.\n"
            "Check the USB cable and that no other app is using the webcam.",
        )
        self.pygame.display.flip()
        self.clock.tick(30)
        if self.fail_reads > _STALL_FRAMES:
            raise CaptureError(
                "The camera stopped producing frames. Check the USB cable "
                "and that another app is not using the webcam."
            )

    def _events(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            elif event.type == self.pygame.KEYDOWN:
                self._on_key(event)

    def _on_key(self, event) -> None:
        key = event.key
        mods = event.mod
        pg = self.pygame
        if key == pg.K_ESCAPE or (
            key == pg.K_q and (mods & (pg.KMOD_META | pg.KMOD_CTRL))
        ):
            self.running = False
            return
        if key == pg.K_1:
            self._set_mode(Mode.PASSTHROUGH)
        elif key == pg.K_2:
            self._set_mode(Mode.OVERLAY)
        elif key == pg.K_3:
            self._set_mode(Mode.PLAY)
        elif key == pg.K_c:
            self._set_mode(Mode.CALIBRATE)
        elif key == pg.K_SPACE and self.mode == Mode.CALIBRATE and self.arm_until is None:
            self.arm_until = time.perf_counter() + _CALIBRATION_HOLD_S
            print("[liveplay] holding flat gray so the camera can catch up...")
        elif key == pg.K_s and self.mode in (Mode.PASSTHROUGH, Mode.OVERLAY):
            self._save_roi()
        elif key in (pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN):
            self._nudge(key, bool(mods & pg.KMOD_SHIFT))

    def _set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        if mode == Mode.CALIBRATE:
            self.return_mode = self.mode if self.mode != Mode.CALIBRATE else Mode.PLAY
            self.arm_until = None
            print("[liveplay] calibration. Clear the table, then press SPACE.")
        if mode == Mode.PLAY:
            self.particles.clear()
        self.mode = mode
        print(f"[liveplay] mode {self.mode}")

    def _save_roi(self) -> None:
        try:
            self.cfg.save()
        except ConfigError as exc:
            print(f"[liveplay] could not save ROI: {exc}")

    def _nudge(self, key: int, resize: bool) -> None:
        if self.mode not in (Mode.PASSTHROUGH, Mode.OVERLAY):
            return
        camera = self.source
        raw_size = getattr(camera, "raw_size", None)
        if raw_size is None or not hasattr(camera, "roi"):
            print("[liveplay] ROI nudge needs a real camera frame.")
            return
        if self.cfg.perspective is not None:
            print("[liveplay] perspective is set; ROI nudge is ignored. Edit config.json.")
            return
        dx = dy = dw = dh = 0
        pg = self.pygame
        if key == pg.K_LEFT:
            dw, dx = (-1, 0) if resize else (0, -1)
        elif key == pg.K_RIGHT:
            dw, dx = (1, 0) if resize else (0, 1)
        elif key == pg.K_UP:
            dh, dy = (-1, 0) if resize else (0, -1)
        elif key == pg.K_DOWN:
            dh, dy = (1, 0) if resize else (0, 1)
        camera.roi = nudge_roi(camera.roi, raw_size[0], raw_size[1], dx, dy, dw, dh)
        self.cfg.roi = camera.roi
        print(f"[liveplay] roi x={camera.roi[0]} y={camera.roi[1]} w={camera.roi[2]} h={camera.roi[3]}")

    def _maybe_log(self) -> None:
        now = time.perf_counter()
        if now - self._log_at < 2.0:
            return
        self._log_at = now
        fps = self.clock.get_fps()
        print(
            f"[liveplay] {fps:.0f} fps  mode={self.mode}  points={self._last_points}"
        )


__all__ = ["main", "run", "Session", "Mode"]
