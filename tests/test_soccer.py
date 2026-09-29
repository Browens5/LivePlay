"""Soccer rules: bounce, bat, score, reset. No camera."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np

from liveplay.config import VisionConfig
from liveplay.points import InteractionPoint
from liveplay.soccer import BALL, GOAL_LEFT, GOAL_RIGHT, SoccerGame
from liveplay.vision import BlobVision


FIELD = (0, 0, 400, 200)


def _ready() -> SoccerGame:
    game = SoccerGame()
    game.update([], FIELD, 1.0 / 60.0)
    return game


class SoccerRulesTest(unittest.TestCase):
    def test_side_wall_bounces_outside_the_goal(self) -> None:
        game = _ready()
        game.ball_x = 24
        game.ball_y = 16
        game.ball_vx = -500
        game.ball_vy = 0
        game.update([], FIELD, 0.05)
        self.assertGreater(game.ball_vx, 0)
        self.assertEqual((game.score_left, game.score_right), (0, 0))
        self.assertGreaterEqual(game.ball_x, game.ball_radius)

    def test_top_and_bottom_bounce(self) -> None:
        game = _ready()
        game.ball_x = 200
        game.ball_y = 20
        game.ball_vx = 0
        game.ball_vy = -400
        game.update([], FIELD, 0.05)
        self.assertGreater(game.ball_vy, 0)
        game.ball_y = 180
        game.ball_vy = 400
        game.update([], FIELD, 0.05)
        self.assertLess(game.ball_vy, 0)

    def test_right_goal_scores_for_the_left_player_and_resets(self) -> None:
        game = _ready()
        game.ball_x = 390
        game.ball_y = 100
        game.ball_vx = 800
        game.ball_vy = 0
        game.update([], FIELD, 0.05)
        self.assertEqual(game.score_left, 1)
        self.assertEqual(game.score_right, 0)
        self.assertAlmostEqual(game.ball_x, 200)
        self.assertAlmostEqual(game.ball_y, 100)
        self.assertEqual(game.ball_vx, 0)
        # The ball stays put while the goal is shown.
        game.update([], FIELD, 0.05)
        self.assertEqual(game.score_left, 1)
        self.assertAlmostEqual(game.ball_x, 200)

    def test_left_goal_scores_for_the_right_player(self) -> None:
        game = _ready()
        game.ball_x = 10
        game.ball_y = 100
        game.ball_vx = -800
        game.ball_vy = 0
        game.update([], FIELD, 0.05)
        self.assertEqual((game.score_left, game.score_right), (0, 1))
        self.assertAlmostEqual(game.ball_x, 200)

    def test_hand_bats_the_ball(self) -> None:
        game = _ready()
        game.ball_x = 200
        game.ball_y = 100
        game.ball_vx = 0
        game.ball_vy = 0
        hand = InteractionPoint(x=170, y=100, size=40, vx=900, vy=40)
        game.update([hand], FIELD, 1.0 / 60.0)
        self.assertGreater(game.ball_x, 200)
        self.assertGreater(game.ball_vx, 0)

    def test_goal_colors_are_not_skin(self) -> None:
        frame = np.full((80, 240, 3), 90, dtype=np.uint8)
        for index, rgb in enumerate((GOAL_LEFT, GOAL_RIGHT, BALL)):
            bgr = (rgb[2], rgb[1], rgb[0])
            origin = 10 + index * 70
            cv2.rectangle(frame, (origin, 10), (origin + 50, 70), bgr, thickness=-1)
        backend = BlobVision(
            VisionConfig(
                method="skin",
                diff_threshold=28,
                min_area=80,
                max_area_fraction=0.5,
                max_blobs=6,
                track_dark_blobs=False,
                dark_delta=40,
                morph_kernel=3,
                scale=1.0,
                smoothing=1.0,
                match_distance=300,
            ),
            None,
        )
        self.assertEqual(backend.detect(frame), [])

    def test_score_is_drawn_at_the_top_middle(self) -> None:
        import pygame

        pygame.display.init()
        try:
            surface = pygame.display.set_mode((400, 200))
            surface.fill((90, 90, 90))
            game = _ready()
            game.score_left = 2
            game.score_right = 1
            game.draw(surface, 90)
            # Off the center line and above the ball, the field stays gray.
            self.assertEqual(surface.get_at((160, 70))[:3], (90, 90, 90))
            # Left goal is blue. Right goal is green.
            self.assertEqual(surface.get_at((8, 100))[:3], GOAL_LEFT)
            self.assertEqual(surface.get_at((392, 100))[:3], GOAL_RIGHT)
            top = [surface.get_at((px, 24))[:3] for px in range(140, 260)]
            self.assertTrue(any(pixel != (90, 90, 90) for pixel in top))
        finally:
            pygame.display.quit()


if __name__ == "__main__":
    unittest.main()
