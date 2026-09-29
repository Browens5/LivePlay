"""Run the real loop against a fake camera and a dummy SDL window."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from liveplay.app import main
from liveplay.calibration import load_calibration, save_calibration
from liveplay.errors import CalibrationError


def _write_config(directory: Path, width: int = 640, height: int = 360) -> Path:
    data = json.loads((ROOT / "config.json").read_text())
    data["display"]["width"] = width
    data["display"]["height"] = height
    data["display"]["fullscreen"] = False
    data["vision"]["min_area"] = 800
    data["vision"]["max_area_fraction"] = 0.5
    path = directory / "config.json"
    path.write_text(json.dumps(data))
    return path


class SmokeTest(unittest.TestCase):
    def test_each_mode_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_config(root)
            calibration = root / "empty.png"
            image = np.full((360, 640, 3), 90, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(calibration), image))
            for mode in ("passthrough", "overlay", "play", "calibrate"):
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    code = main(
                        [
                            "--config",
                            str(config),
                            "--calibration",
                            str(calibration),
                            "--fake-camera",
                            "--windowed",
                            "--frames",
                            "5",
                            "--mode",
                            mode,
                        ]
                    )
                self.assertEqual(code, 0, output.getvalue())
                self.assertIn(f"mode {mode}", output.getvalue())

    def test_missing_calibration_asks_instead_of_hanging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_config(root)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                code = main(
                    [
                        "--config",
                        str(config),
                        "--calibration",
                        str(root / "missing.png"),
                        "--fake-camera",
                        "--windowed",
                        "--frames",
                        "4",
                        "--mode",
                        "play",
                    ]
                )
            text = output.getvalue()
            self.assertEqual(code, 0, text)
            self.assertIn("no empty-table snapshot", text)

    def test_bad_camera_index_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(Path(tmp))
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                code = main(
                    [
                        "--config",
                        str(config),
                        "--windowed",
                        "--camera",
                        "99",
                        "--frames",
                        "2",
                        "--mode",
                        "passthrough",
                    ]
                )
            text = output.getvalue()
            self.assertEqual(code, 1, text)
            self.assertIn("No camera at index 99", text)

    def test_list_displays_returns(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            code = main(["--list-displays"])
        self.assertIn(code, (0, 1), output.getvalue())


class CalibrationFileTest(unittest.TestCase):
    def test_roundtrip_and_unreadable_file(self) -> None:
        image = np.full((12, 16, 3), 90, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.png"
            save_calibration(path, image)
            loaded = load_calibration(path)
            assert loaded is not None
            self.assertTrue(np.array_equal(loaded, image))
            path.write_text("not a png")
            with self.assertRaises(CalibrationError):
                load_calibration(path)
        self.assertIsNone(load_calibration(Path(tmp) / "missing.png"))


if __name__ == "__main__":
    unittest.main()
