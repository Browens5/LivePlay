import random
import unittest

from liveplay.config import ParticleConfig
from liveplay.particles import Particle, ParticleSystem, _toward_gray
from liveplay.points import InteractionPoint


def _particles(**overrides: object) -> ParticleConfig:
    values: dict[str, object] = {
        "style": "sparks",
        "max_count": 50,
        "spawn_per_point": 0,
        "fade_per_second": 0.0,
        "attract": 2500.0,
        "splash": 0,
    }
    values.update(overrides)
    return ParticleConfig(**values)  # type: ignore[arg-type]


class ParticleTest(unittest.TestCase):
    def test_particles_are_pulled_toward_a_point(self) -> None:
        system = ParticleSystem(_particles(), rng=random.Random(0))
        system.particles.append(Particle(20, 100, 0, 0, life=5, radius=4, color=(255, 40, 40)))
        point = InteractionPoint(180, 100, 20, 0, 0, False)
        for _ in range(25):
            system.update([point], dt=1 / 30)
        self.assertGreater(system.particles[0].x, 50)

    def test_particles_fade_out(self) -> None:
        system = ParticleSystem(_particles(fade_per_second=1.0), rng=random.Random(0))
        system.fade_per_second = 5.0
        system.particles.append(Particle(10, 10, 0, 0, life=0.2, radius=4, color=(255, 0, 0)))
        system.update([], dt=0.05)
        self.assertEqual(system.particles, [])

    def test_new_point_spawns_a_splash(self) -> None:
        system = ParticleSystem(
            _particles(spawn_per_point=2, splash=8, style="sparks"),
            rng=random.Random(1),
        )
        system.update([InteractionPoint(50, 50, 12, 0, 0, True)], dt=1 / 60)
        self.assertGreaterEqual(len(system.particles), 8)

    def test_fade_color_lands_on_the_playfield_gray(self) -> None:
        # Dying particles should match the gray field so the camera does
        # not see them as dark objects.
        self.assertEqual(_toward_gray((255, 0, 0), 0, 90), (90, 90, 90))
        self.assertEqual(_toward_gray((255, 0, 0), 1, 90), (255, 0, 0))

    def test_styles_accept_blobs_and_trails(self) -> None:
        for style in ("blobs", "trails", "sparks"):
            system = ParticleSystem(_particles(style=style, spawn_per_point=1, splash=0))
            system.update([InteractionPoint(10, 10, 8, 0, 0, True)], dt=1 / 60)
            self.assertGreater(len(system.particles), 0)


if __name__ == "__main__":
    unittest.main()
