"""A monster truck gets two tires, six lug nuts, and a race."""

import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import pygame

from liveplay.config import load_config, parse_args
from liveplay.garage import (
    ASSET_DIR,
    GRAB_S,
    GRACE_S,
    LUGS_PER_WHEEL,
    PLACE_S,
    RACE_S,
    RESET_AFTER_S,
    RESET_S,
    TIGHTEN_S,
    TRUCK_ASPECT,
    WHEEL_COUNT,
    GarageGame,
    Wheel,
    axle_center,
    cutscene_pose,
    fault_of,
    spots_for,
)
from liveplay.points import InteractionPoint
from liveplay.sound import TableAudio

FIELD = (0, 0, 1000, 600)


def _hand(nx: float, ny: float, size: float = 40.0) -> InteractionPoint:
    return InteractionPoint(FIELD[0] + nx * FIELD[2], FIELD[1] + ny * FIELD[3], size)


def _hold(game: GarageGame, hand: InteractionPoint, seconds: float) -> None:
    left = seconds
    while left > 1e-9:
        step = min(0.05, left)
        game.update([hand], FIELD, step)
        left -= step


def _at(game: GarageGame, spot: str) -> InteractionPoint:
    center = spots_for(FIELD)[spot]
    return InteractionPoint(center.cx, center.cy, 40.0)


def _away() -> InteractionPoint:
    return _hand(0.40, 0.10)


class GarageRepairTest(unittest.TestCase):
    def test_garage_mode_is_a_real_mode(self) -> None:
        cfg = load_config(parse_args(["--mode", "garage", "--config", "missing-garage.json"]))
        self.assertEqual(cfg.mode, "garage")

    def test_the_truck_starts_bare(self) -> None:
        game = GarageGame(random.Random(0))
        self.assertEqual(len(game.wheels), WHEEL_COUNT)
        self.assertTrue(all(not wheel.mounted for wheel in game.wheels))
        self.assertTrue(all(set(wheel.lugs) == {0} for wheel in game.wheels))
        self.assertEqual(len(game.wheels[0].lugs), LUGS_PER_WHEEL)
        self.assertIsNone(game.held)
        self.assertFalse(game.ready)
        self.assertEqual(fault_of(game.wheels), "tire")
        self.assertIn("tire", game.hint)

    def test_a_tire_is_grabbed_after_one_second_and_not_before(self) -> None:
        game = GarageGame(random.Random(0))
        game.update([_at(game, "pile")], FIELD, 0.016)
        _hold(game, _at(game, "pile"), GRAB_S - 0.08)
        self.assertIsNone(game.held)
        self.assertGreater(game.hover_t, 0.5)
        _hold(game, _at(game, "pile"), 0.12)
        self.assertEqual(game.held, "tire")

    def test_leaving_the_pile_resets_the_hover_after_a_short_grace(self) -> None:
        game = GarageGame(random.Random(0))
        _hold(game, _at(game, "pile"), 0.55)
        saved = game.hover_t
        _hold(game, _away(), GRACE_S * 0.4)
        self.assertEqual(game.hover_id, "pile")
        self.assertAlmostEqual(game.hover_t, saved, places=2)
        _hold(game, _away(), GRACE_S)
        self.assertIsNone(game.hover_id)
        self.assertEqual(game.hover_t, 0.0)
        self.assertIsNone(game.held)

    def test_a_tire_fits_an_open_axle_and_not_a_full_one(self) -> None:
        game = GarageGame(random.Random(0))
        self._grab(game, "pile", "tire")
        _hold(game, _at(game, "axle-0"), PLACE_S + 0.05)
        self.assertTrue(game.wheels[0].mounted)
        self.assertFalse(game.wheels[1].mounted)
        self.assertIsNone(game.held)
        self._grab(game, "pile", "tire")
        # The rear axle is full, so hovering it does not take the second tire.
        _hold(game, _at(game, "axle-0"), PLACE_S + 0.1)
        self.assertEqual(game.held, "tire")
        self.assertFalse(game.wheels[1].mounted)
        _hold(game, _at(game, "axle-1"), PLACE_S + 0.05)
        self.assertTrue(game.wheels[1].mounted)
        self.assertIsNone(game.held)
        self.assertEqual(game.counts()[0], 2)

    def test_a_nut_needs_a_tire_and_then_fills_the_next_hole(self) -> None:
        game = GarageGame(random.Random(0))
        _hold(game, _at(game, "bucket"), GRAB_S + 0.1)
        self.assertIsNone(game.held)
        self._mount_both(game)
        self._grab(game, "bucket", "nut")
        _hold(game, _at(game, "axle-0"), PLACE_S + 0.05)
        self.assertEqual(game.wheels[0].lugs, [1, 0, 0])
        self.assertEqual(game.wheels[1].lugs, [0, 0, 0])
        self.assertIsNone(game.held)
        self._grab(game, "bucket", "nut")
        _hold(game, _at(game, "axle-1"), PLACE_S + 0.05)
        self.assertEqual(game.wheels[1].lugs, [1, 0, 0])

    def test_the_wrench_tightens_one_nut_per_second_and_stays_in_hand(self) -> None:
        game = GarageGame(random.Random(0))
        self._mount_both(game)
        _hold(game, _at(game, "wrench"), GRAB_S + 0.1)
        self.assertIsNone(game.held)
        self._place_all_nuts(game)
        self.assertEqual(fault_of(game.wheels), "loose")
        self._grab(game, "wrench", "wrench")
        _hold(game, _at(game, "axle-0"), TIGHTEN_S + 0.05)
        self.assertEqual(game.wheels[0].lugs, [2, 1, 1])
        self.assertEqual(game.held, "wrench")
        _hold(game, _at(game, "axle-0"), TIGHTEN_S * 2 + 0.05)
        self.assertEqual(game.wheels[0].lugs, [2, 2, 2])
        self.assertEqual(game.wheels[1].lugs, [1, 1, 1])
        self.assertEqual(game.held, "wrench")

    def test_a_carried_part_goes_back_only_after_the_hand_leaves_and_returns(self) -> None:
        game = GarageGame(random.Random(0))
        self._grab(game, "pile", "tire")
        # The hand is still on the pile. That must not immediately put the tire back.
        _hold(game, _at(game, "pile"), GRAB_S + 0.2)
        self.assertEqual(game.held, "tire")
        _hold(game, _away(), 0.3)
        _hold(game, _at(game, "pile"), GRAB_S + 0.05)
        self.assertIsNone(game.held)

    def test_the_hand_carrying_the_part_is_the_one_that_can_drop_it(self) -> None:
        game = GarageGame(random.Random(0))
        self._grab(game, "pile", "tire")
        pile = _at(game, "pile")
        axle = _at(game, "axle-0")
        # The tire stays with the hand on the pile. The other hand is on the axle.
        left = PLACE_S + 0.2
        while left > 1e-9:
            step = min(0.05, left)
            game.update([pile, axle], FIELD, step)
            left -= step
        self.assertEqual(game.held, "tire")
        self.assertFalse(game.wheels[0].mounted)
        _hold(game, axle, PLACE_S + 0.1)
        self.assertTrue(game.wheels[0].mounted)
        self.assertIsNone(game.held)

    def test_racing_early_crashes_and_a_finished_truck_wins(self) -> None:
        game = GarageGame(random.Random(0))
        self._race(game)
        self.assertEqual(game.phase, "cutscene")
        self.assertEqual(game.outcome, "crash")
        self.assertEqual(game.fault, "tire")
        self.assertIsNone(game.held)
        pose = cutscene_pose(3.4, "crash", "tire")
        self.assertEqual(pose.title, "CRASH")
        self.assertLess(pose.angle, -40.0)
        self.assertIsNone(pose.detach)

        game.reset()
        self._mount_both(game)
        self._race(game)
        self.assertEqual(game.fault, "lug")
        loose = cutscene_pose(2.0, "crash", "lug", problem=0)
        self.assertEqual(loose.detach, 0)
        self.assertGreater(loose.detach_nx, loose.nx)

        game.reset()
        self._mount_both(game)
        self._place_all_nuts(game)
        self._race(game)
        self.assertEqual(game.fault, "loose")
        self.assertEqual(game.outcome, "crash")

        game.reset()
        self._finish(game)
        self.assertTrue(game.ready)
        self.assertIsNone(fault_of(game.wheels))
        self.assertIn("RACE", game.hint)
        self._race(game)
        self.assertEqual(game.outcome, "win")
        self.assertIsNone(game.fault)
        jumped = cutscene_pose(2.2, "win", None)
        landed = cutscene_pose(0.4, "win", None)
        self.assertLess(jumped.ny, landed.ny - 0.15)
        self.assertEqual(cutscene_pose(4.2, "win", None).title, "WINNER")
        # Clockwise, so the top of the tire moves the same way as the truck.
        self.assertLess(cutscene_pose(1.0, "win", None).spin, 0.0)
        self.assertLess(cutscene_pose(1.2, "win", None).spin, cutscene_pose(0.4, "win", None).spin)
        self.assertGreater(abs(cutscene_pose(3.6, "crash", "loose").angle), 50.0)

    def test_reset_waits_until_the_finish_and_then_clears_the_bay(self) -> None:
        game = GarageGame(random.Random(0))
        self._mount_both(game)
        # RESET is not a target while the truck is still in the shop.
        _hold(game, _at(game, "reset"), RESET_S + 0.2)
        self.assertTrue(game.wheels[0].mounted)
        self.assertEqual(game.phase, "repair")
        self._race(game)
        _hold(game, _away(), RESET_S + 0.2)
        self.assertEqual(game.phase, "cutscene")
        self.assertLess(game.cut_t, RESET_AFTER_S)
        _hold(game, _away(), RESET_AFTER_S)
        self.assertGreaterEqual(game.cut_t, RESET_AFTER_S)
        self.assertEqual(game.phase, "cutscene")
        self.assertIn("RESET", game.hint)
        _hold(game, _at(game, "reset"), RESET_S - 0.12)
        self.assertEqual(game.phase, "cutscene")
        _hold(game, _at(game, "reset"), 0.2)
        self.assertEqual(game.phase, "repair")
        self.assertIsNone(game.outcome)
        self.assertFalse(any(wheel.mounted for wheel in game.wheels))
        self.assertEqual(game.counts(), (0, 0, 0))

    def test_a_full_repair_uses_two_tires_and_six_nuts(self) -> None:
        game = GarageGame(random.Random(0))
        self._finish(game)
        self.assertEqual(game.counts(), (2, 6, 6))
        self.assertTrue(all(wheel.mounted and wheel.lugs == [2, 2, 2] for wheel in game.wheels))
        # Nothing left to pick up. The race button is the next hover.
        _hold(game, _at(game, "pile"), GRAB_S + 0.1)
        _hold(game, _at(game, "bucket"), GRAB_S + 0.1)
        _hold(game, _at(game, "wrench"), GRAB_S + 0.1)
        self.assertIsNone(game.held)

    def test_one_big_frame_does_not_skip_the_hover(self) -> None:
        game = GarageGame(random.Random(0))
        game.update([_at(game, "pile")], FIELD, 5.0)
        self.assertIsNone(game.held)
        self.assertLess(game.hover_t, GRAB_S)

    def test_sprites_are_local_and_the_truck_aspect_matches_the_art(self) -> None:
        names = ("shop.png", "track.png", "truck_body.png", "tire.png", "lug.png", "wrench.png", "bucket.png")
        for name in names:
            path = ASSET_DIR / name
            self.assertTrue(path.is_file(), name)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(image, name)
            assert image is not None
            if name in ("shop.png", "track.png"):
                self.assertEqual(image.shape[2], 3)
            else:
                self.assertEqual(image.shape[2], 4, name)
                self.assertGreater(int((image[:, :, 3] == 0).sum()), 100, name)
        truck = cv2.imread(str(ASSET_DIR / "truck_body.png"), cv2.IMREAD_UNCHANGED)
        assert truck is not None
        aspect = truck.shape[1] / truck.shape[0]
        self.assertAlmostEqual(aspect, TRUCK_ASPECT, places=2)
        rear, front = axle_center(FIELD, 0), axle_center(FIELD, 1)
        self.assertLess(rear[0], front[0])

    def _grab(self, game: GarageGame, spot: str, item: str) -> None:
        _hold(game, _away(), 0.25)
        _hold(game, _at(game, spot), GRAB_S + 0.05)
        self.assertEqual(game.held, item)

    def _mount_both(self, game: GarageGame) -> None:
        for axle in ("axle-0", "axle-1"):
            self._grab(game, "pile", "tire")
            _hold(game, _away(), 0.25)
            _hold(game, _at(game, axle), PLACE_S + 0.05)
            self.assertIsNone(game.held)

    def _place_all_nuts(self, game: GarageGame) -> None:
        for axle in ("axle-0", "axle-1"):
            for _lug in range(LUGS_PER_WHEEL):
                self._grab(game, "bucket", "nut")
                _hold(game, _away(), 0.25)
                _hold(game, _at(game, axle), PLACE_S + 0.05)
                self.assertIsNone(game.held)

    def _finish(self, game: GarageGame) -> None:
        self._mount_both(game)
        self._place_all_nuts(game)
        self._grab(game, "wrench", "wrench")
        for axle in ("axle-0", "axle-1"):
            _hold(game, _away(), 0.25)
            _hold(game, _at(game, axle), TIGHTEN_S * LUGS_PER_WHEEL + 0.1)
        self.assertIsNone(fault_of(game.wheels))
        # Put the wrench back so the bay looks finished.
        _hold(game, _away(), 0.25)
        _hold(game, _at(game, "wrench"), GRAB_S + 0.05)
        self.assertIsNone(game.held)

    def _race(self, game: GarageGame) -> None:
        _hold(game, _away(), 0.25)
        _hold(game, _at(game, "race"), RACE_S + 0.05)


class GarageDrawTest(unittest.TestCase):
    def test_the_shop_shows_the_truck_and_a_mounted_tire_changes_the_axle(self) -> None:
        pygame.display.init()
        try:
            pygame.display.set_mode((FIELD[2], FIELD[3]))
            surface = pygame.Surface((FIELD[2], FIELD[3]))
            game = GarageGame(random.Random(0))
            game.update([], FIELD, 0.05)
            game.time = 0.314
            game.draw(surface)
            bare = _pixels(surface)
            button = spots_for(FIELD)["race"].rect
            assert button is not None
            fill = surface.get_at((int(button[0] + 10), int(button[1] + 10)))[:3]
            self.assertEqual(fill, (22, 104, 176))
            rear = axle_center(FIELD, 0)
            before = surface.get_at((int(rear[0]), int(rear[1])))[:3]
            game.wheels[0].mounted = True
            game.wheels[0].lugs = [1, 0, 0]
            game.draw(surface)
            self.assertNotEqual(_pixels(surface), bare)
            after = surface.get_at((int(rear[0]), int(rear[1])))[:3]
            self.assertNotEqual(before, after)
            game.wheels[0].lugs = [2, 2, 2]
            game.wheels[1] = Wheel(mounted=True, lugs=[2, 2, 2])
            game.draw(surface)
            ready = spots_for(FIELD)["race"].rect
            assert ready is not None
            green = surface.get_at((int(ready[0] + 10), int(ready[1] + 10)))[:3]
            self.assertGreater(green[1], green[2])
            self.assertGreater(green[1], 80)
        finally:
            pygame.display.quit()

    def test_the_finish_draws_a_winner_and_a_crash(self) -> None:
        pygame.display.init()
        try:
            pygame.display.set_mode((FIELD[2], FIELD[3]))
            surface = pygame.Surface((FIELD[2], FIELD[3]))
            game = GarageGame(random.Random(1))
            game.update([], FIELD, 0.05)
            game.phase = "cutscene"
            game.outcome = "win"
            game.fault = None
            game.wheels = [Wheel(mounted=True, lugs=[2, 2, 2]) for _ in range(2)]
            game.cut_t = 4.4
            game.draw(surface)
            win = _pixels(surface)
            game.outcome = "crash"
            game.fault = "tire"
            game.wheels = [Wheel(), Wheel()]
            game.cut_t = 3.6
            game.draw(surface)
            self.assertNotEqual(_pixels(surface), win)
        finally:
            pygame.display.quit()


def _pixels(surface: pygame.Surface) -> bytes:
    return pygame.image.tobytes(surface, "RGB")


class _Ear:
    """Records cues without opening a speaker."""

    def __init__(self) -> None:
        self.played: list[str] = []
        self.song: str | None = "silent"
        self.stopped = False

    def play(self, name: str) -> None:
        self.played.append(name)

    def music(self, name: str | None) -> None:
        self.song = name

    def stop(self) -> None:
        self.stopped = True
        self.song = None


class GarageAudioTest(unittest.TestCase):
    def test_building_the_game_stays_quiet_until_the_bay_is_shown(self) -> None:
        ear = _Ear()
        game = GarageGame(random.Random(0), audio=ear)
        self.assertEqual(ear.song, "silent")
        self.assertEqual(ear.played, [])
        game.update([], FIELD, 0.05)
        self.assertEqual(ear.song, "shop")

    def test_each_repair_step_has_its_own_effect(self) -> None:
        ear = _Ear()
        game = GarageGame(random.Random(0), audio=ear)
        _hold(game, _at(game, "pile"), GRAB_S + 0.05)
        self.assertEqual(ear.played, ["pickup"])
        _hold(game, _away(), 0.3)
        _hold(game, _at(game, "pile"), GRAB_S + 0.05)
        self.assertEqual(ear.played, ["pickup", "putback"])
        _hold(game, _away(), 0.3)
        _hold(game, _at(game, "pile"), GRAB_S + 0.05)
        _hold(game, _at(game, "axle-0"), PLACE_S + 0.05)
        self.assertEqual(ear.played[-1], "tire")
        _hold(game, _away(), 0.3)
        _hold(game, _at(game, "bucket"), GRAB_S + 0.05)
        _hold(game, _at(game, "axle-0"), PLACE_S + 0.05)
        self.assertEqual(ear.played[-1], "nut")
        _hold(game, _away(), 0.3)
        _hold(game, _at(game, "wrench"), GRAB_S + 0.05)
        _hold(game, _at(game, "axle-0"), TIGHTEN_S + 0.05)
        self.assertEqual(ear.played[-1], "wrench")

    def test_a_crash_cuts_the_music_and_a_win_cheers(self) -> None:
        ear = _Ear()
        game = GarageGame(random.Random(0), audio=ear)
        _hold(game, _at(game, "race"), RACE_S + 0.05)
        self.assertEqual(ear.played[-1], "rev")
        self.assertEqual(ear.song, "shop")
        _hold(game, _away(), 1.5)
        self.assertIn("crash", ear.played)
        self.assertIsNone(ear.song)

        ear = _Ear()
        game = GarageGame(random.Random(0), audio=ear)
        game.wheels = [Wheel(mounted=True, lugs=[2, 2, 2]) for _ in range(2)]
        _hold(game, _at(game, "race"), RACE_S + 0.05)
        self.assertEqual(ear.song, "race")
        self.assertNotIn("fanfare", ear.played)
        _hold(game, _away(), 4.0)
        self.assertIn("fanfare", ear.played)
        self.assertEqual(ear.song, "race")

    def test_leaving_the_garage_stops_the_loop(self) -> None:
        ear = _Ear()
        game = GarageGame(random.Random(0), audio=ear)
        game.update([], FIELD, 0.05)
        game.quiet()
        self.assertTrue(ear.stopped)
        self.assertIsNone(ear.song)

    def test_samples_play_on_the_dummy_driver(self) -> None:
        audio = TableAudio()
        for name in ("pickup", "putback", "tire", "nut", "wrench", "rev", "crash", "fanfare"):
            audio.play(name)
        audio.music("shop")
        self.assertTrue(audio.enabled)
        self.assertEqual(audio._music_name, "shop")
        channel = audio._music_channel
        assert channel is not None
        self.assertTrue(channel.get_busy())
        audio.music("shop")
        self.assertTrue(channel.get_busy())
        audio.play("tire")
        self.assertTrue(channel.get_busy())
        audio.music("race")
        self.assertEqual(audio._music_name, "race")
        self.assertTrue(channel.get_busy())
        audio.music(None)
        self.assertIsNone(audio._music_name)
        self.assertFalse(channel.get_busy())
        audio.play("missing")
        audio.stop()

    def test_a_missing_device_stays_silent(self) -> None:
        audio = TableAudio()
        real_init = pygame.mixer.init
        real_get = pygame.mixer.get_init

        def no_device(*_args: object, **_kwargs: object) -> None:
            raise pygame.error("no device")

        pygame.mixer.init = no_device  # type: ignore[method-assign]
        pygame.mixer.get_init = lambda: None  # type: ignore[method-assign]
        try:
            audio.play("crash")
            audio.music("shop")
            audio.stop()
        finally:
            pygame.mixer.init = real_init  # type: ignore[method-assign]
            pygame.mixer.get_init = real_get  # type: ignore[method-assign]
        self.assertFalse(audio.enabled)
        audio.play("tire")
        audio.music("shop")
        self.assertIsNone(audio._music_name)


if __name__ == "__main__":
    unittest.main()
