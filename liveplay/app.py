"""One process: capture → vision → points → a game → the table TV.

Modes
-----
geometry     Gray-code scan. Finds the framebuffer rectangle the TV shows.
play         Soccer. A puck follows each hand and knocks the ball.
forest       A dinosaur walks through the trees, following a hand.
calibrate    Flat gray. SPACE snapshots the empty table.
passthrough  Camera mapped onto the TV, with corner ticks.
overlay      Live feed plus blob circles.

Adult keys (keyboard on the Mac, nothing drawn for kids to tap):
  Esc or Cmd/Ctrl+Q   quit
  1 / 2 / 3 / 4       passthrough / overlay / play / forest
  C                   empty-table snapshot
  G                   measure the TV again
  Space               snapshot, or retry a failed scan
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
    draw_status,
    fill_playfield,
    fill_visible,
    init_video,
    list_displays,
    present_error,
)
from liveplay.geometry import DisplayGeometry, DisplayScan, load_geometry, save_geometry
from liveplay.errors import CalibrationError, CaptureError, ConfigError, LivePlayError
from liveplay.forest import ForestGame
from liveplay.points import PointTracker
from liveplay.soccer import SoccerGame
from liveplay.vision import make_backend

# How long the screen stays pure gray before we trust the camera frame.
# The instruction text is itself a bright blob the webcam would memorize.
_CALIBRATION_HOLD_S = 0.75
_STALL_FRAMES = 45


class Mode:
    PASSTHROUGH = "passthrough"
    OVERLAY = "overlay"
    PLAY = "play"
    FOREST = "forest"
    CALIBRATE = "calibrate"
    GEOMETRY = "geometry"
    POSE = "pose"


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
        init_video()
    except pygame.error as exc:
        print(f"[liveplay] error: could not start video: {exc}", file=sys.stderr)
        return 1
    screen = None
    source = None
    session = None
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
        if session is not None:
            session.close()
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
        self.scan: DisplayScan | None = None
        self._scan_error_printed = False
        frame_size = (screen.get_width(), screen.get_height())
        self.geometry = self._load_geometry(frame_size)
        if cfg.fake_camera and self.geometry is None:
            # The fake camera never sees the stripes, so it cannot measure
            # overscan. Treat its image as already screen-sized.
            self.geometry = DisplayGeometry.full_frame(*frame_size)
        background = self._load_background()
        self.backend = make_backend(cfg.vision, background)
        self.tracker = PointTracker(cfg.vision.match_distance, cfg.vision.smoothing)
        self.soccer = SoccerGame()
        self.forest = ForestGame()
        self._enter_startup_mode()
        print(
            f"[liveplay] mode {self.mode}  vision {cfg.vision.method}. "
            f"Esc or Cmd/Ctrl+Q quits."
        )

    def run(self) -> None:
        while self.running:
            self._events()
            if not self.running:
                break
            raw = self.source.read_raw()
            if raw is None:
                self._on_missing_frame()
                continue
            self.fail_reads = 0
            self._tick(raw)
            self.pygame.display.flip()
            self.clock.tick(60)
            self.dt = max(self.clock.get_time() / 1000.0, 1.0 / 120.0)
            self.frames += 1
            self._maybe_log()
            if self.cfg.frames and self.frames >= self.cfg.frames:
                self.running = False
            if self.cfg.display.fullscreen:
                self.pygame.mouse.set_visible(False)

    def _load_geometry(self, frame_size: tuple[int, int]) -> DisplayGeometry | None:
        try:
            geometry = load_geometry(self.cfg.geometry.path, frame_size)
        except CalibrationError as exc:
            print(f"[liveplay] {exc}")
            return None
        if geometry is not None:
            x, y, w, h = geometry.visible
            print(f"[liveplay] loaded display geometry visible x={x} y={y} w={w} h={h}")
        return geometry

    def _enter_startup_mode(self) -> None:
        # A real webcam measures the TV before play. The fake camera cannot
        # see the stripes, so it skips straight to the game.
        if self.mode == Mode.POSE:
            self.mode = Mode.PLAY
        if self.cfg.fake_camera:
            if self.mode == Mode.GEOMETRY:
                self.mode = Mode.PLAY
            self._maybe_need_background()
            return
        if self.mode == Mode.GEOMETRY or (
            self.mode in (Mode.PLAY, Mode.FOREST) and self.geometry is None
        ):
            if self.mode in (Mode.PLAY, Mode.FOREST):
                self.return_mode = self.mode
            self._begin_geometry()
            return
        self._maybe_need_background()

    def _maybe_need_background(self) -> None:
        if self.mode not in (Mode.PLAY, Mode.FOREST, Mode.OVERLAY):
            return
        if self.backend.needs_calibration and not self.backend.has_background:
            print("[liveplay] no empty-table snapshot yet. Entering calibration.")
            print("[liveplay] clear the table, then press SPACE. Esc quits.")
            self.return_mode = self.mode
            self.mode = Mode.CALIBRATE

    def _load_background(self):
        path = self.cfg.calibration.path
        image = load_calibration(path)
        if image is None:
            return None
        expected = self._background_shape()
        if expected is not None and image.shape != expected:
            print(
                f"[liveplay] calibration {path} is {image.shape[1]}x{image.shape[0]}, "
                f"but vision expects {expected[1]}x{expected[0]}. "
                f"Press C, clear the table, and press SPACE to recapture."
            )
            return None
        print(f"[liveplay] loaded calibration {path}")
        return image

    def _background_shape(self) -> tuple[int, int, int] | None:
        # Vision runs in framebuffer pixels (the warp), so the snapshot is
        # the screen size even when the camera itself is 720p.
        return (self.cfg.display.height, self.cfg.display.width, 3)

    def _begin_geometry(self) -> None:
        size = (self.screen.get_width(), self.screen.get_height())
        geo = self.cfg.geometry
        self.scan = DisplayScan(size, geo.bits, geo.settle, geo.inset)
        self.tracker.reset()
        self._scan_error_printed = False
        self.mode = Mode.GEOMETRY
        print(
            f"[liveplay] measuring the TV. {self.scan.total} patterns, "
            f"about {self.scan.total * geo.settle:.0f}s. Keep the glass clear. Esc quits."
        )

    def _tick(self, raw) -> None:
        self.screen.set_clip(None)
        if self.mode == Mode.GEOMETRY:
            self._tick_geometry(raw)
        elif self.mode == Mode.CALIBRATE:
            self._tick_calibrate(raw)
        elif self.mode == Mode.PASSTHROUGH:
            blit_bgr(self.screen, self.source.map_to_screen(raw))
            draw_alignment_guides(self.screen)
        elif self.mode == Mode.OVERLAY:
            mapped = self.source.map_to_screen(raw)
            # Overlay is the crop check. Circles sit on the stretched
            # camera image, not on the measured TV rectangle.
            points, _poses = self._detect(raw, align_to_camera=True)
            blit_bgr(self.screen, mapped)
            draw_points(self.screen, points)
            draw_alignment_guides(self.screen)
            self._draw_warning()
        elif self.mode == Mode.PLAY:
            points, _poses = self._detect(raw)
            field = self._field_rect()
            self.soccer.update(points, field, self.dt)
            # Flat gray, only inside the rectangle the TV actually shows.
            # The puck is a cyan disc on the hand center. Goals, the ball,
            # and that disc miss the skin gate, and the disc is not a hand
            # shape. See liveplay/soccer.py and liveplay/hands.py.
            clip = self._paint_field()
            self.screen.set_clip(clip)
            self.soccer.draw(self.screen, self.cfg.calibration.gray)
        elif self.mode == Mode.FOREST:
            points, _poses = self._detect(raw)
            field = self._field_rect()
            self.forest.update(points, field, self.dt)
            # The picture fills the glass. MediaPipe still looks for a hand
            # shape. diff+skin would see the whole forest as a hand.
            # See liveplay/forest.py.
            clip = self._paint_field()
            self.screen.set_clip(clip)
            self.forest.draw(self.screen)
        else:
            raise ConfigError(f"Unknown mode {self.mode!r}.")

    def _field_rect(self) -> tuple[int, int, int, int]:
        geometry = self.geometry
        if geometry is None or geometry.direct_screen:
            return (0, 0, self.screen.get_width(), self.screen.get_height())
        return geometry.visible

    def _paint_field(self):
        """Gray where the TV can light a pixel. Returns that rectangle, if known."""
        gray = self.cfg.calibration.gray
        geometry = self.geometry
        if geometry is None or geometry.direct_screen:
            fill_playfield(self.screen, gray)
            return None
        fill_visible(self.screen, geometry.visible, gray)
        return geometry.visible

    def _draw_inset_message(self, message: str) -> None:
        if self.geometry is None:
            draw_message(self.screen, message, gray=self.cfg.calibration.gray)
            return
        x, y, width, height = self.geometry.visible
        # Keep the words inside the measured rectangle so overscan cannot
        # push them off the glass.
        pad = 24
        sw, sh = self.screen.get_size()
        x0 = max(0, min(sw - 1, int(x) + pad))
        y0 = max(0, min(sh - 1, int(y) + pad))
        x1 = max(x0 + 1, min(sw, int(x) + int(width) - pad))
        y1 = max(y0 + 1, min(sh, int(y) + int(height) - pad))
        if (x1 - x0) < 40 or (y1 - y0) < 40:
            draw_message(self.screen, message, gray=self.cfg.calibration.gray)
            return
        region = self.screen.subsurface((x0, y0, x1 - x0, y1 - y0))
        draw_message(region, message, gray=self.cfg.calibration.gray)

    def _tick_geometry(self, raw) -> None:
        if self.scan is None:
            self._begin_geometry()
        assert self.scan is not None
        if self.scan.error:
            self._show_scan_error()
            return
        pattern, geometry = self.scan.tick(raw, time.perf_counter())
        if self.scan.error:
            self._show_scan_error()
            return
        if geometry is not None:
            self._finish_geometry(geometry)
            return
        if pattern is not None:
            blit_bgr(self.screen, pattern)

    def _show_scan_error(self) -> None:
        assert self.scan is not None and self.scan.error
        if not self._scan_error_printed:
            print(f"[liveplay] {self.scan.error}")
            self._scan_error_printed = True
        draw_message(
            self.screen,
            self.scan.error + "\nPress SPACE to measure again. Esc quits.",
            gray=self.cfg.calibration.gray,
        )

    def _finish_geometry(self, geometry: DisplayGeometry) -> None:
        save_geometry(self.cfg.geometry.path, geometry)
        self.geometry = geometry
        self.scan = None
        self.tracker.reset()
        # The snapshot has to be the warped view of this playfield. A photo
        # from before the scan is a different mapping, even when the pixel
        # size happens to match.
        self.backend.set_background(None)
        if self.return_mode not in (Mode.PLAY, Mode.FOREST):
            self.return_mode = Mode.PLAY
        self.mode = Mode.CALIBRATE
        print("[liveplay] TV area measured. Clear the table, then press SPACE.")

    def _tick_calibrate(self, raw) -> None:
        gray = self.cfg.calibration.gray
        clip = self._paint_field()
        now = time.perf_counter()
        if self.arm_until is not None:
            if now >= self.arm_until:
                self._save_snapshot(self._vision_frame(raw))
            return
        message = (
            "Clear hands and toys off the table.\n"
            "The screen stays flat gray so the camera has a clean background.\n"
            "Press SPACE to snapshot the empty table.\n"
            "Esc quits."
        )
        if clip is None:
            draw_message(self.screen, message, gray=gray)
        else:
            self._draw_inset_message(message)

    def _save_snapshot(self, frame) -> None:
        path = self.cfg.calibration.path
        try:
            save_calibration(path, frame)
        except CalibrationError:
            self.arm_until = None
            raise
        self.backend.set_background(frame)
        self.tracker.reset()
        self.soccer.kickoff()
        self.arm_until = None
        height, width = frame.shape[:2]
        print(f"[liveplay] saved calibration {path} ({width}x{height})")
        self.mode = self.return_mode if self.return_mode != Mode.CALIBRATE else Mode.PLAY
        print(f"[liveplay] mode {self.mode}")

    def _vision_frame(self, raw):
        geometry = self.geometry
        if geometry is not None and not geometry.direct_screen:
            return geometry.warp(raw, self.cfg.calibration.gray)
        return self.source.map_to_screen(raw)

    def _detect(self, raw, *, align_to_camera: bool = False):
        if align_to_camera:
            frame = self.source.map_to_screen(raw)
        else:
            frame = self._vision_frame(raw)
        points = self.backend.detect(frame)
        poses = list(self.backend.last_poses)
        points = self.tracker.update(points, self.dt)
        self._last_points = len(points)
        self._report_warning()
        return points, poses

    def _report_warning(self) -> None:
        warning = self.backend.last_warning
        if warning and warning != self._shown_warning:
            print(f"[liveplay] {warning}")
        self._shown_warning = warning

    def _draw_warning(self) -> None:
        warning = self.backend.last_warning
        if not warning:
            return
        draw_status(self.screen, warning)

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
        # A scan in progress only listens for quit. Space retries a failed scan.
        if self.mode == Mode.GEOMETRY and self.scan is not None and not self.scan.error:
            return
        if key == pg.K_1:
            self._set_mode(Mode.PASSTHROUGH)
        elif key == pg.K_2:
            self._set_mode(Mode.OVERLAY)
        elif key == pg.K_3:
            self._set_mode(Mode.PLAY)
        elif key == pg.K_4:
            self._set_mode(Mode.FOREST)
        elif key == pg.K_g:
            if self.cfg.fake_camera:
                print("[liveplay] the fake camera cannot see the TV. Skipping the scan.")
                return
            if self.mode in (Mode.PLAY, Mode.FOREST):
                self.return_mode = self.mode
            else:
                self.return_mode = Mode.PLAY
            self._begin_geometry()
        elif key == pg.K_c:
            self._set_mode(Mode.CALIBRATE)
        elif key == pg.K_SPACE and self.mode == Mode.GEOMETRY and self.scan is not None and self.scan.error:
            self._begin_geometry()
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
        if mode in (Mode.PLAY, Mode.FOREST):
            if self.geometry is None and not self.cfg.fake_camera:
                self.return_mode = mode
                self._begin_geometry()
                return
            if mode == Mode.PLAY:
                self.soccer.kickoff()
        self.tracker.reset()
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

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if closer is not None:
            closer()

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
