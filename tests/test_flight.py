"""Two pterodactyls follow two hands through gaps in the branches."""

import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np
import pygame

from liveplay.config import load_config, parse_args
from liveplay.flight import (
    FLIGHT_COLORS,
    FREEZE_S,
    RESET_HOLD_S,
    SKY_PATH,
    WIN_SCORE,
    FlightGame,
    Gate,
    _ice,
    reset_bounds,
    speed_for_score,
)
from liveplay.points import InteractionPoint
from liveplay.vision import BlobVision

FIELD = (0, 0, 1000, 600)


def _hand(nx: float, ny: float, size: float = 36.0) -> InteractionPoint:
    return InteractionPoint(nx * FIELD[2], ny * FIELD[3], size)


def _hold(game: FlightGame, hand: InteractionPoint, seconds: float) -> None:
    left = seconds
    while left > 1e-9:
        step = min(0.05, left)
        game.update([hand], FIELD, step)
        left -= step


def _vision():
    from liveplay.config import VisionConfig

    return VisionConfig(
        method="skin",
        diff_threshold=28,
        min_area=80,
        max_area_fraction=0.4,
        max_blobs=6,
        track_dark_blobs=False,
        dark_delta=35,
        morph_kernel=3,
        scale=1.0,
        smoothing=1.0,
        match_distance=300,
    )


class FlightCourseTest(unittest.TestCase):
    def test_flight_mode_is_a_real_mode(self) -> None:
        cfg = load_config(parse_args(["--mode", "flight", "--config", "missing-flight.json"]))
        self.assertEqual(cfg.mode, "flight")

    def test_a_pterodactyl_sits_on_the_hand(self) -> None:
        game = FlightGame(random.Random(0))
        game.update([_hand(0.31, 0.44)], FIELD, 0.016)
        player = game.players[0]
        self.assertTrue(player.active)
        self.assertFalse(game.players[1].active)
        self.assertAlmostEqual(player.nx, 0.31)
        self.assertAlmostEqual(player.ny, 0.44)
        game.update([_hand(0.58, 0.22)], FIELD, 0.016)
        self.assertAlmostEqual(player.nx, 0.58)
        self.assertAlmostEqual(player.ny, 0.22)

    def test_two_hands_fly_two_pterodactyls(self) -> None:
        game = FlightGame(random.Random(0))
        game.update([_hand(0.22, 0.30), _hand(0.74, 0.62)], FIELD, 0.016)
        self.assertTrue(game.players[0].active)
        self.assertTrue(game.players[1].active)
        self.assertAlmostEqual(game.players[0].nx, 0.22)
        self.assertAlmostEqual(game.players[1].nx, 0.74)
        self.assertEqual(game.players[0].score, 0)
        self.assertEqual(game.players[1].score, 0)

    def test_each_player_keeps_their_hand_when_they_cross(self) -> None:
        game = FlightGame(random.Random(0))
        game.update([_hand(0.20, 0.40), _hand(0.80, 0.40)], FIELD, 0.016)
        game.players[0].score = 4
        game.players[1].score = 7
        game.update([_hand(0.34, 0.42), _hand(0.66, 0.38)], FIELD, 0.016)
        self.assertLess(game.players[0].nx, 0.5)
        self.assertGreater(game.players[1].nx, 0.5)
        self.assertEqual(game.players[0].score, 4)
        self.assertEqual(game.players[1].score, 7)

    def test_passing_an_opening_scores_one_point(self) -> None:
        game = FlightGame(random.Random(0))
        hand = _hand(0.40, 0.50)
        game.update([hand], FIELD, 0.016)
        game.gates = [Gate(x=0.08, gap_y=0.50, gap_h=0.90, width=0.08, seed=1)]
        game.update([hand], FIELD, 0.016)
        self.assertEqual(game.players[0].score, 1)
        game.update([hand], FIELD, 0.016)
        self.assertEqual(game.players[0].score, 1)

    def test_a_hit_freezes_that_player_and_the_other_still_scores(self) -> None:
        game = FlightGame(random.Random(1))
        high = _hand(0.30, 0.12)
        low = _hand(0.30, 0.55)
        game.update([high, low], FIELD, 0.016)
        # Top limb covers the upper hand. The lower hand is in the gap.
        game.gates = [Gate(x=0.18, gap_y=0.62, gap_h=0.28, width=0.22, seed=2)]
        game.update([high, low], FIELD, 0.016)
        self.assertGreater(game.players[0].freeze, FREEZE_S - 0.05)
        self.assertEqual(game.players[0].score, 0)
        self.assertEqual(game.players[1].score, 0)
        self.assertFalse(game.won)
        stuck = (game.players[0].nx, game.players[0].ny)
        # The frozen body stays on the limb. The other hand has not crossed.
        game.update([_hand(0.62, 0.12), low], FIELD, 0.016)
        self.assertAlmostEqual(game.players[0].nx, stuck[0])
        self.assertAlmostEqual(game.players[0].ny, stuck[1])
        self.assertEqual(game.players[0].score, 0)
        self.assertEqual(game.players[1].score, 0)
        # The opening is already behind them. Only the flying player scores.
        game.gates = [Gate(x=0.02, gap_y=0.55, gap_h=0.70, width=0.06, seed=3)]
        game.update([_hand(0.62, 0.12), _hand(0.40, 0.55)], FIELD, 0.016)
        self.assertEqual(game.players[0].score, 0)
        self.assertEqual(game.players[1].score, 1)
        self.assertFalse(game.won)

    def test_freeze_ends_after_five_seconds_and_the_hand_leads_again(self) -> None:
        game = FlightGame(random.Random(0))
        game.update([_hand(0.30, 0.40)], FIELD, 0.016)
        player = game.players[0]
        player.freeze = 0.2
        player.nx, player.ny = 0.20, 0.20
        game.gates = []
        # The hand walks off the frozen body. The sprite stays until the
        # five seconds are up, then it is back on the palm.
        for nx in (0.40, 0.58, 0.74):
            game.update([_hand(nx, 0.46)], FIELD, 0.02)
        self.assertGreater(player.freeze, 0.0)
        self.assertAlmostEqual(player.nx, 0.20)
        self.assertAlmostEqual(player.hand_nx, 0.74, places=2)
        while player.freeze > 0.0:
            game.update([_hand(0.74, 0.46)], FIELD, 0.05)
        self.assertEqual(player.freeze, 0.0)
        self.assertAlmostEqual(player.nx, 0.74, places=2)
        self.assertAlmostEqual(player.ny, 0.46, places=2)

    def test_limbs_speed_up_as_the_score_climbs_toward_100(self) -> None:
        self.assertLess(speed_for_score(0), speed_for_score(1))
        self.assertLess(speed_for_score(1), speed_for_score(50))
        self.assertLess(speed_for_score(50), speed_for_score(99))
        self.assertEqual(speed_for_score(99), speed_for_score(100))
        self.assertGreater(speed_for_score(99), speed_for_score(0) * 3)

        slow = _travel(0)
        fast = _travel(90)
        self.assertGreater(fast, slow * 2)

    def test_one_hundred_ends_the_course_with_confetti(self) -> None:
        game = FlightGame(random.Random(0))
        hand = _hand(0.42, 0.50)
        game.update([hand], FIELD, 0.016)
        game.players[0].score = WIN_SCORE - 1
        game.gates = [Gate(x=0.05, gap_y=0.50, gap_h=0.90, width=0.06, seed=4)]
        game.update([hand], FIELD, 0.016)
        self.assertTrue(game.won)
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.players[0].score, WIN_SCORE)
        self.assertGreater(len(game.confetti), 40)
        parked = game.gates[0].x
        game.update([hand], FIELD, 0.05)
        self.assertEqual(game.gates[0].x, parked)
        self.assertEqual(game.players[0].score, WIN_SCORE)

    def test_hovering_reset_for_three_seconds_clears_the_course(self) -> None:
        game = FlightGame(random.Random(0))
        game.update([_hand(0.40, 0.50)], FIELD, 0.016)
        game.players[0].score = 12
        game.won = True
        game.winner = 0
        left, top, width, height = reset_bounds(FIELD)
        button = InteractionPoint(left + width * 0.5, top + height * 0.5, 20)
        outside = _hand(0.50, 0.50)
        _hold(game, button, 2.0)
        self.assertGreater(game.reset_hold, 1.5)
        self.assertEqual(game.players[0].score, 12)
        game.update([outside], FIELD, 0.05)
        self.assertEqual(game.reset_hold, 0.0)
        self.assertEqual(game.players[0].score, 12)
        _hold(game, button, RESET_HOLD_S + 0.1)
        self.assertEqual(game.players[0].score, 0)
        self.assertEqual(game.players[1].score, 0)
        self.assertFalse(game.won)
        self.assertIsNone(game.winner)
        self.assertEqual(game.confetti, [])

    def test_the_sky_and_the_painted_colors_miss_the_skin_gate(self) -> None:
        colors = list(FLIGHT_COLORS) + [_ice(color) for color in FLIGHT_COLORS]
        for rgb in colors:
            block = np.full((80, 80, 3), (rgb[2], rgb[1], rgb[0]), dtype=np.uint8)
            backend = BlobVision(_vision(), None)
            self.assertEqual(backend.detect(block), [], rgb)
        sky = cv2.imread(str(SKY_PATH))
        self.assertIsNotNone(sky)
        assert sky is not None
        hsv = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin = ((hue <= 20) | (hue >= 170)) & (sat >= 40) & (val >= 50)
        self.assertEqual(int(skin.sum()), 0)


class FlightDrawTest(unittest.TestCase):
    def test_the_scene_shows_a_scrolling_sky_and_a_flapping_wing(self) -> None:
        pygame.display.init()
        try:
            pygame.display.set_mode((FIELD[2], FIELD[3]))
            surface = pygame.Surface((FIELD[2], FIELD[3]))
            game = FlightGame(random.Random(0))
            game.update([_hand(0.50, 0.48, size=80)], FIELD, 0.05)
            game.players[0].flap = 0.0
            game.draw(surface)
            still = _pixels(surface)
            game.players[0].flap = 0.2
            game.draw(surface)
            flapped = _pixels(surface)
            self.assertNotEqual(still, flapped)
            game.parallax += 180
            game.draw(surface)
            scrolled = _pixels(surface)
            self.assertNotEqual(flapped, scrolled)
            # The body sits on the hand, and it is teal rather than sky blue.
            body = surface.get_at((500, int(0.48 * FIELD[3])))[:3]
            sky = surface.get_at((12, 12))[:3]
            self.assertGreater(body[1], body[0])
            self.assertNotEqual(body, sky)
            left, top, width, height = reset_bounds(FIELD)
            fill = (16, 58, 140)
            found = False
            for py in range(int(top) + 2, int(top + height) - 2):
                for px in range(int(left) + 2, int(left + width) - 2):
                    if surface.get_at((px, py))[:3] == fill:
                        found = True
                        break
                if found:
                    break
            self.assertTrue(found)
        finally:
            pygame.display.quit()


def _travel(score: int) -> float:
    game = FlightGame(random.Random(0))
    game.update([], FIELD, 0.016)
    game.players[0].score = score
    start = game.gates[0].x
    game.update([], FIELD, 0.05)
    return start - game.gates[0].x


def _pixels(surface: pygame.Surface) -> bytes:
    return pygame.image.tobytes(surface, "RGB")


if __name__ == "__main__":
    unittest.main()
