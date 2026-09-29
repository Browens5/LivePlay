"""The dinosaur walks toward a hand and plays at each place in the forest."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from liveplay.config import load_config, parse_args
from liveplay.forest import (
    FALLS,
    FALLS_BRIGHT,
    GRASS,
    GRASS_LIGHT,
    LAKE,
    SPOTS,
    DINO,
    ForestGame,
    waterfall_bright,
)
from liveplay.points import InteractionPoint

FIELD = (0, 0, 1000, 600)


def _hand(nx: float, ny: float) -> InteractionPoint:
    return InteractionPoint(nx * FIELD[2], ny * FIELD[3], 20)


def _until_pose(game: ForestGame, nx: float, ny: float, pose: str) -> bool:
    hand = _hand(nx, ny)
    for _ in range(160):
        game.update([hand], FIELD, 0.05)
        if game.pose == pose:
            return True
    return False


class ForestWalkTest(unittest.TestCase):
    def test_without_a_hand_the_dinosaur_stays_put(self) -> None:
        game = ForestGame()
        game.update([], FIELD, 0.05)
        start = (game.nx, game.ny)
        for _ in range(40):
            game.update([], FIELD, 0.05)
        self.assertEqual(game.pose, "idle")
        self.assertAlmostEqual(game.nx, start[0])
        self.assertAlmostEqual(game.ny, start[1])

    def test_the_dinosaur_walks_toward_the_hand(self) -> None:
        game = ForestGame()
        game.update([], FIELD, 0.05)
        start = game.nx
        for _ in range(40):
            game.update([_hand(0.92, 0.74)], FIELD, 0.05)
        self.assertGreater(game.nx, start + 0.15)
        self.assertEqual(game.facing, 1)

    def test_the_dinosaur_stays_inside_the_meadow(self) -> None:
        game = ForestGame()
        for _ in range(80):
            game.update([_hand(-0.2, -0.2)], FIELD, 0.05)
        self.assertGreaterEqual(game.nx, 0.04)
        self.assertLessEqual(game.nx, 0.96)
        self.assertGreaterEqual(game.ny, 0.36)
        self.assertLessEqual(game.ny, 0.94)

    def test_each_place_has_its_own_animation(self) -> None:
        for spot in SPOTS:
            game = ForestGame()
            self.assertTrue(
                _until_pose(game, spot.x, spot.y, spot.pose),
                f"{spot.name} stayed on {game.pose} at {(game.nx, game.ny)}",
            )
            self.assertLess(abs(game.nx - spot.x) + abs(game.ny - spot.y), 0.08)

    def test_waterfall_stripes_scroll(self) -> None:
        self.assertNotEqual(waterfall_bright(10, 0.0), waterfall_bright(10, 0.1))

    def test_forest_mode_is_a_real_mode(self) -> None:
        cfg = load_config(parse_args(["--mode", "forest", "--config", "missing-forest.json"]))
        self.assertEqual(cfg.mode, "forest")


class ForestDrawTest(unittest.TestCase):
    def test_the_scene_shows_water_grass_and_the_dinosaur(self) -> None:
        pygame.display.init()
        try:
            surface = pygame.display.set_mode((640, 360))
            game = ForestGame()
            game.update([], (0, 0, 640, 360), 0.05)
            game.time = 0.0
            game.draw(surface)
            self.assertEqual(surface.get_at((200, 100))[:3], _falls_color(100, 0.0))
            game.time = 0.1
            game.draw(surface)
            self.assertEqual(surface.get_at((200, 100))[:3], _falls_color(100, 0.1))
            self.assertNotEqual(_falls_color(100, 0.0), _falls_color(100, 0.1))
            lake = surface.get_at((220, 200))[:3]
            self.assertIn(lake, (LAKE, FALLS, FALLS_BRIGHT))
            ground = surface.get_at((430, 300))[:3]
            self.assertIn(ground, (GRASS, GRASS_LIGHT, DINO))
            feet_x, feet_y = game.pixel()
            colors = [
                surface.get_at((x, y))[:3]
                for y in range(feet_y - 40, feet_y)
                for x in range(feet_x - 24, feet_x + 24)
                if 0 <= x < 640 and 0 <= y < 360
            ]
            self.assertIn(DINO, colors)
        finally:
            pygame.display.quit()


def _falls_color(y: int, time: float):
    return FALLS_BRIGHT if waterfall_bright(y, time) else FALLS


if __name__ == "__main__":
    unittest.main()
