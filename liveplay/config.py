"""Load config.json and apply CLI overrides.

Paths in the file are relative to the config file's directory so the app
can be started from another working directory.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from liveplay.errors import ConfigError

MODES = ("passthrough", "overlay", "play", "calibrate", "geometry", "pose")
VISION_METHODS = ("diff", "skin", "diff+skin")
PARTICLE_STYLES = ("sparks", "blobs", "trails")

DEFAULTS: dict = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
    "roi": {"x": 0, "y": 0, "w": 0, "h": 0},
    "perspective": None,
    "undistort": {"enabled": False, "camera_matrix": None, "dist_coeffs": None},
    "display": {
        "width": 1920,
        "height": 1080,
        "fullscreen": True,
        "index": None,
        "prefer_external": True,
    },
    "calibration": {"path": "calib/empty_table.png", "gray": 90},
    "vision": {
        "method": "diff+skin",
        "diff_threshold": 28,
        "min_area": 2200,
        "max_area_fraction": 0.2,
        "max_blobs": 6,
        "track_dark_blobs": False,
        "dark_delta": 40,
        "morph_kernel": 5,
        "scale": 0.5,
        "smoothing": 0.55,
        "match_distance": 300,
    },
    "particles": {
        "style": "sparks",
        "max_count": 450,
        "spawn_per_point": 3,
        "fade_per_second": 0.7,
        "attract": 900,
        "splash": 16,
    },
    "geometry": {
        "path": "calib/display_geometry.json",
        "bits": 7,
        "settle": 0.32,
        "inset": 0,
    },
    "mode": "play",
}


@dataclass
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int


@dataclass
class UndistortConfig:
    enabled: bool
    camera_matrix: list | None
    dist_coeffs: list | None


@dataclass
class DisplayConfig:
    width: int
    height: int
    fullscreen: bool
    index: int | None
    prefer_external: bool


@dataclass
class CalibrationConfig:
    path: Path
    gray: int


@dataclass
class GeometryConfig:
    path: Path
    bits: int
    settle: float
    inset: int


@dataclass
class VisionConfig:
    method: str
    diff_threshold: int
    min_area: int
    max_area_fraction: float
    max_blobs: int
    track_dark_blobs: bool
    dark_delta: int
    morph_kernel: int
    scale: float
    smoothing: float
    match_distance: float


@dataclass
class ParticleConfig:
    style: str
    max_count: int
    spawn_per_point: int
    fade_per_second: float
    attract: float
    splash: int


@dataclass
class AppConfig:
    camera: CameraConfig
    roi: tuple[int, int, int, int]
    perspective: list | None
    undistort: UndistortConfig
    display: DisplayConfig
    calibration: CalibrationConfig
    geometry: GeometryConfig
    vision: VisionConfig
    particles: ParticleConfig
    mode: str
    fake_camera: bool
    frames: int
    config_path: Path | None
    raw: dict = field(repr=False)

    def save(self) -> None:
        if self.config_path is None:
            raise ConfigError("No config path to save. Pass --config.")
        self.raw["roi"] = {
            "x": self.roi[0],
            "y": self.roi[1],
            "w": self.roi[2],
            "h": self.roi[3],
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.raw, indent=2) + "\n"
        self.config_path.write_text(text)
        print(f"[liveplay] saved ROI to {self.config_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liveplay",
        description="Live Play v0 — local interactive table (no network).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m liveplay --list-displays\n"
            "  python -m liveplay --mode passthrough\n"
            "  python -m liveplay --mode play --display 1\n"
            "  python -m liveplay --camera 0 --particle-style blobs\n"
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--camera", type=int, help="Webcam index.")
    parser.add_argument(
        "--display",
        type=int,
        help="Display index. Default: an external 1920x1080 if one is connected.",
    )
    parser.add_argument(
        "--list-displays",
        action="store_true",
        help="Print displays and exit.",
    )
    parser.add_argument("--windowed", action="store_true", help="Window instead of fullscreen.")
    parser.add_argument("--fullscreen", action="store_true", help="Force fullscreen.")
    parser.add_argument("--roi", help="Camera crop x,y,w,h. w or h of 0 uses the full frame.")
    parser.add_argument("--calibration", type=Path, help="Empty-table snapshot path.")
    parser.add_argument("--particle-style", choices=PARTICLE_STYLES)
    parser.add_argument("--vision", choices=VISION_METHODS)
    parser.add_argument(
        "--fake-camera",
        action="store_true",
        help="Synthetic camera for development without a webcam.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Exit after this many frames. 0 runs until Esc or Cmd/Ctrl+Q.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_config(args: argparse.Namespace) -> AppConfig:
    path = args.config.expanduser()
    base_dir = path.parent if str(path.parent) not in ("", ".") else Path(".")
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a JSON object.")
        raw = _merge(DEFAULTS, loaded, "")
        if path.parent != Path(""):
            base_dir = path.parent
    else:
        print(f"[liveplay] no config at {path}; using defaults.")
        raw = copy.deepcopy(DEFAULTS)
        base_dir = Path(".")

    try:
        cfg = _from_raw(raw, base_dir, path)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value in {path}: {exc}") from exc
    _apply_cli(cfg, args)
    _validate(cfg)
    return cfg


def _apply_cli(cfg: AppConfig, args: argparse.Namespace) -> None:
    """CLI flags change this run only. Saving the ROI does not store them."""
    if args.mode is not None:
        cfg.mode = args.mode
    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.display is not None:
        cfg.display.index = args.display
    if args.windowed and args.fullscreen:
        raise ConfigError("Pass only one of --windowed and --fullscreen.")
    if args.windowed:
        cfg.display.fullscreen = False
    if args.fullscreen:
        cfg.display.fullscreen = True
    if args.roi is not None:
        cfg.roi = _parse_roi(args.roi)
    if args.calibration is not None:
        cal_path = args.calibration.expanduser()
        if not cal_path.is_absolute():
            base = cfg.config_path.parent if cfg.config_path is not None else Path(".")
            if str(base) == "":
                base = Path(".")
            cal_path = base / cal_path
        cfg.calibration.path = cal_path
    if args.particle_style is not None:
        cfg.particles.style = args.particle_style
    if args.vision is not None:
        cfg.vision.method = args.vision
    cfg.fake_camera = bool(args.fake_camera)
    cfg.frames = int(args.frames)
    if cfg.frames < 0:
        raise ConfigError("--frames must be 0 or more.")


def _from_raw(raw: dict, base_dir: Path, config_path: Path) -> AppConfig:
    cam = raw["camera"]
    und = raw["undistort"]
    disp = raw["display"]
    cal = raw["calibration"]
    geo = raw["geometry"]
    vis = raw["vision"]
    parts = raw["particles"]
    roi = raw["roi"]
    cal_path = _resolve_path(base_dir, cal["path"])
    geo_path = _resolve_path(base_dir, geo["path"])
    return AppConfig(
        camera=CameraConfig(
            index=int(cam["index"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            fps=int(cam["fps"]),
        ),
        roi=(int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])),
        perspective=raw["perspective"],
        undistort=UndistortConfig(
            enabled=bool(und["enabled"]),
            camera_matrix=und["camera_matrix"],
            dist_coeffs=und["dist_coeffs"],
        ),
        display=DisplayConfig(
            width=int(disp["width"]),
            height=int(disp["height"]),
            fullscreen=bool(disp["fullscreen"]),
            index=None if disp["index"] is None else int(disp["index"]),
            prefer_external=bool(disp["prefer_external"]),
        ),
        calibration=CalibrationConfig(path=cal_path, gray=int(cal["gray"])),
        geometry=GeometryConfig(
            path=geo_path,
            bits=int(geo["bits"]),
            settle=float(geo["settle"]),
            inset=int(geo["inset"]),
        ),
        vision=VisionConfig(
            method=str(vis["method"]),
            diff_threshold=int(vis["diff_threshold"]),
            min_area=int(vis["min_area"]),
            max_area_fraction=float(vis["max_area_fraction"]),
            max_blobs=int(vis["max_blobs"]),
            track_dark_blobs=bool(vis["track_dark_blobs"]),
            dark_delta=int(vis["dark_delta"]),
            morph_kernel=int(vis["morph_kernel"]),
            scale=float(vis["scale"]),
            smoothing=float(vis["smoothing"]),
            match_distance=float(vis["match_distance"]),
        ),
        particles=ParticleConfig(
            style=str(parts["style"]),
            max_count=int(parts["max_count"]),
            spawn_per_point=int(parts["spawn_per_point"]),
            fade_per_second=float(parts["fade_per_second"]),
            attract=float(parts["attract"]),
            splash=int(parts["splash"]),
        ),
        mode=str(raw["mode"]),
        fake_camera=False,
        frames=0,
        config_path=config_path,
        raw=raw,
    )


def _validate(cfg: AppConfig) -> None:
    if cfg.mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}.")
    if cfg.vision.method not in VISION_METHODS:
        raise ConfigError(f"vision.method must be one of {VISION_METHODS}.")
    if cfg.particles.style not in PARTICLE_STYLES:
        raise ConfigError(f"particles.style must be one of {PARTICLE_STYLES}.")
    if cfg.camera.index < 0:
        raise ConfigError("camera.index must be 0 or more.")
    if cfg.camera.fps <= 0 or cfg.camera.width < 16 or cfg.camera.height < 16:
        raise ConfigError("camera width, height, and fps must be positive.")
    if cfg.display.width < 160 or cfg.display.height < 120:
        raise ConfigError("display width/height are too small.")
    if cfg.display.index is not None and cfg.display.index < 0:
        raise ConfigError("display.index must be 0 or more.")
    if not 0 <= cfg.calibration.gray <= 255:
        raise ConfigError("calibration.gray must be 0..255.")
    if not 0 < cfg.vision.scale <= 1:
        raise ConfigError("vision.scale must be in (0, 1].")
    if cfg.vision.morph_kernel < 1 or cfg.vision.morph_kernel % 2 == 0:
        raise ConfigError("vision.morph_kernel must be a positive odd integer.")
    if cfg.vision.min_area < 1 or cfg.vision.max_blobs < 1:
        raise ConfigError("vision min_area and max_blobs must be at least 1.")
    if not 0 < cfg.vision.max_area_fraction <= 1:
        raise ConfigError("vision.max_area_fraction must be in (0, 1].")
    if any(v < 0 for v in cfg.roi):
        raise ConfigError("roi values must be 0 or more.")
    if cfg.perspective is not None:
        _validate_perspective(cfg.perspective)
    if cfg.undistort.enabled:
        if not cfg.undistort.camera_matrix or not cfg.undistort.dist_coeffs:
            raise ConfigError(
                "undistort.enabled is true but camera_matrix or dist_coeffs is missing."
            )
    if cfg.particles.max_count < 1 or cfg.particles.spawn_per_point < 0:
        raise ConfigError("particle counts are invalid.")
    if cfg.particles.fade_per_second < 0 or cfg.particles.attract < 0:
        raise ConfigError("particle fade and attract must be 0 or more.")
    if not 4 <= cfg.geometry.bits <= 9:
        raise ConfigError("geometry.bits must be from 4 to 9.")
    if cfg.geometry.settle <= 0:
        raise ConfigError("geometry.settle must be greater than 0.")
    if cfg.geometry.inset < 0:
        raise ConfigError("geometry.inset must be 0 or more.")


def _validate_perspective(points: object) -> None:
    if not isinstance(points, list) or len(points) != 4:
        raise ConfigError("perspective must be null or four [x, y] points.")
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ConfigError("each perspective point must be [x, y].")
        try:
            float(point[0])
            float(point[1])
        except (TypeError, ValueError) as exc:
            raise ConfigError("perspective coordinates must be numbers.") from exc


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_roi(text: str) -> tuple[int, int, int, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ConfigError("--roi expects x,y,w,h.")
    try:
        values = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ConfigError("--roi values must be integers.") from exc
    return values  # type: ignore[return-value]


def _merge(defaults: dict, override: dict, path: str) -> dict:
    if not isinstance(override, dict):
        raise ConfigError(f"{path or 'config'} must be an object.")
    unknown = sorted(set(override) - set(defaults))
    if unknown:
        where = path or "config"
        raise ConfigError(f"Unknown {where} keys: {', '.join(unknown)}.")
    merged: dict = {}
    for key, default_value in defaults.items():
        child = f"{path}{key}"
        if key not in override:
            merged[key] = copy.deepcopy(default_value)
            continue
        value = override[key]
        if isinstance(default_value, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{child} must be an object.")
            merged[key] = _merge(default_value, value, f"{child}.")
        else:
            merged[key] = value
    return merged
