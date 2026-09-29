import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from liveplay.app import main
from liveplay.config import load_config, parse_args
from liveplay.display import choose_display
from liveplay.errors import ConfigError, DisplayError


class ConfigTest(unittest.TestCase):
    def test_cli_overrides_and_roi_save_leaves_other_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text((ROOT / "config.json").read_text())
            cfg = load_config(
                parse_args(
                    [
                        "--config",
                        str(path),
                        "--mode",
                        "passthrough",
                        "--camera",
                        "3",
                        "--particle-style",
                        "blobs",
                        "--vision",
                        "skin",
                        "--roi",
                        "4,5,6,7",
                    ]
                )
            )
            self.assertEqual(cfg.mode, "passthrough")
            self.assertEqual(cfg.camera.index, 3)
            self.assertEqual(cfg.particles.style, "blobs")
            self.assertEqual(cfg.vision.method, "skin")
            self.assertEqual(cfg.roi, (4, 5, 6, 7))
            cfg.save()
            saved = json.loads(path.read_text())
            self.assertEqual(saved["mode"], "play")
            self.assertEqual(saved["camera"]["index"], 0)
            self.assertEqual(saved["roi"], {"x": 4, "y": 5, "w": 6, "h": 7})

    def test_unknown_key_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"nope": 1}\n')
            with self.assertRaises(ConfigError):
                load_config(parse_args(["--config", str(path)]))

    def test_windowed_and_fullscreen_conflict(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(parse_args(["--windowed", "--fullscreen"]))

    def test_main_rejects_broken_config_without_opening_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{")
            self.assertEqual(main(["--config", str(path)]), 1)
            path.write_text('{"extra": true}')
            self.assertEqual(main(["--config", str(path)]), 1)

    def test_relative_calibration_path_follows_the_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.json"
            path.write_text((ROOT / "config.json").read_text())
            cfg = load_config(parse_args(["--config", str(path)]))
            self.assertEqual(cfg.calibration.path, root / "calib" / "empty_table.png")
            self.assertEqual(cfg.geometry.path, root / "calib" / "display_geometry.json")
            self.assertEqual(cfg.geometry.bits, 7)


class DisplayChoiceTest(unittest.TestCase):
    def test_single_display_is_zero(self) -> None:
        index = choose_display(
            [(1920, 1080)],
            index=None,
            prefer_external=True,
            target=(1920, 1080),
        )
        self.assertEqual(index, 0)

    def test_external_1080p_is_preferred(self) -> None:
        sizes = [(2560, 1440), (1280, 720), (1920, 1080)]
        index = choose_display(sizes, index=None, prefer_external=True, target=(1920, 1080))
        self.assertEqual(index, 2)

    def test_last_display_when_none_match_1080p(self) -> None:
        sizes = [(2560, 1440), (1280, 720)]
        index = choose_display(sizes, index=None, prefer_external=True, target=(1920, 1080))
        self.assertEqual(index, 1)

    def test_explicit_index_out_of_range(self) -> None:
        with self.assertRaises(DisplayError):
            choose_display([(1920, 1080)], index=2, prefer_external=True, target=(1920, 1080))


if __name__ == "__main__":
    unittest.main()
