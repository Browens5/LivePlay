# Live Play v0

A local prototype for a kids’ table: a TV under plexiglass, a Mac Studio, and a fixed overhead webcam. Hands (and, optionally, dark toys) over the glass become bright particles on the TV. Nothing leaves the machine. There is no account, no network API, and no scene generator.

```
        [webcam]
           |
           v
    +---------------+   plexiglass
    |  hands / toys |
    +---------------+
    |   Vizio 1080p |   particles drawn here
    +---------------+
           |  HDMI + USB
     [Mac Studio]       this process only
```

## Architecture

```
Webcam → Capture → Vision (local) → Interaction points → Game loop → Fullscreen on Vizio
```

One Python process. The stages are separate modules so a second mode can be added later without rewriting capture or the window.

| Stage | Module | What it does |
| --- | --- | --- |
| Capture | `liveplay/capture.py` | Open the webcam (AVFoundation on macOS), drop the frame queue, crop or warp so the image lines up with the TV. |
| Vision | `liveplay/vision.py` | `VisionBackend`: one mapped frame in, a list of points out. v0 is background subtraction plus an optional skin-color gate. |
| Points | `liveplay/points.py` | `{x, y, size, vx, vy}` in TV pixels. `x, y` are the screen. Velocity is pixels per second. |
| Game | `liveplay/particles.py` | Spawn, attract, fade. The only input is that point list. |
| Display | `liveplay/display.py`, `liveplay/app.py` | Fullscreen 1920×1080 on the table, or a window for setup. |

`make_backend()` in `liveplay/vision.py` is the extension point. A later MediaPipe hands backend, or a toy-scene mode, should implement `VisionBackend.detect()` and be chosen there. A second game should branch in `Session._tick` and consume `InteractionPoint` only. It should not open the camera itself.

## Hardware setup

Defaults assume this table:

- Display: ~36" 1080p Vizio, lying flat, HDMI from the Mac Studio. Output is 1920×1080.
- Compute: Mac Studio (Apple Silicon is fine).
- Camera: one fixed USB webcam above the TV, aimed at the whole panel. Default index `0`, requested at 1280×720 / 30 fps, then mapped to 1920×1080.
- Surface: plexiglass over the panel with a small gap so hands are not pressing the LCD. Matte sheet if you can get it.

On the Vizio, turn off the processing that adds lag and crops the picture:

- Picture mode **Game** or **Computer** if the set has it.
- Turn **motion smoothing / ClearAction** off.
- Turn **overscan** off (Just Scan, Dot by Dot, or 1:1 — the name varies). A cropped HDMI image makes fingers miss the particles.
- The TV’s own processing is often a bigger delay than this program. The in-app budget is about one camera frame plus a few milliseconds of blob detection. A TV in cinema mode can spend the whole 100 ms by itself.

On the Mac:

- System Settings → Privacy & Security → Camera → allow the terminal you launch from.
- Displays: the Vizio is the table. If a second monitor is connected, this app prefers a non-primary 1920×1080, otherwise the last display. Override with `--display`.
- Energy: set the display to not sleep while the table is on. `caffeinate -d` (below) holds it for one run.
- The keyboard stays with the adult. Kids only see the glass. There are no on-screen buttons in play mode.

## Dependencies

Python 3.10 or newer. No GPU runtime and no model download.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runtime imports are `numpy`, `opencv-python-headless`, and `pygame-ce` (imported as `pygame`). Headless OpenCV avoids a second GUI toolkit. If a Mac OpenCV build cannot see the webcam, install `opencv-python` instead of `opencv-python-headless`.

`pygame-ce` is a drop-in pygame build with macOS wheels for Python 3.14. The original `pygame` package has none. On 3.14, pip compiles that package from source and `pygame.font` then crashes at startup with `cannot import name 'Font'` / `font module not available`. Operator text is drawn with OpenCV so that crash is not on the startup path, but the window still needs a real pygame build:

```bash
pip uninstall -y pygame
pip install -r requirements.txt
```

The program does not open a socket. You can unplug Ethernet.

## Run

From this directory:

```bash
python -m liveplay --list-displays
caffeinate -d python -m liveplay
```

`caffeinate` is optional. It keeps the panel awake for that run.

Quit with **Esc** or **Cmd+Q** (Ctrl+Q on a non-Mac keyboard). The pointer is hidden while fullscreen. Fullscreen is the default.

Useful flags (they override `config.json` for one run and are not written back, except the ROI when you press S):

```bash
python -m liveplay --mode passthrough
python -m liveplay --mode play --display 1
python -m liveplay --camera 0 --particle-style blobs
python -m liveplay --vision skin
python -m liveplay --windowed          # setup on the Mac’s own screen
```

`--fake-camera` draws a moving blob instead of a webcam, for development without the table.

## Setup procedure

Do these in order the first time. Keys work when the table window is focused.

### 1. Passthrough, so you can see the alignment

```bash
python -m liveplay --mode passthrough
```

The webcam is cropped and stretched to the full TV. White corner ticks mark the screen. A finger on a corner of the glass should show up on that tick.

- **Arrow keys** move the crop.
- **Shift+arrows** resize it.
- **S** writes the crop to `config.json`.

`roi` of `0,0,0,0` means the full camera frame. If the TV is tilted in the image, set `perspective` to four camera-pixel corners in order: top-left, top-right, bottom-right, bottom-left. That warp replaces the crop. If `undistort.enabled` is on, those points are in the undistorted image.

### 2. Empty-table calibration, then the blob overlay

The camera sees the TV. Calibration paints the screen flat gray (`calibration.gray`, default 90) and stores what “nothing on the glass” looks like, including static glare.

```bash
python -m liveplay --mode calibrate
```

Clear hands and toys. Press **Space**. The instructions disappear and the screen stays gray for a moment so the snapshot does not memorize the text. The file is `calib/empty_table.png` (gitignored).

Then:

```bash
python -m liveplay --mode overlay
```

Or press **2**. Green circles should sit on a hand and not on the empty glass. Press **C** any time to snapshot again. Recalibrate when you change lights, the crop, the gray level, or the camera.

Play and overlay refuse to guess. If the snapshot is missing or the wrong size, the screen switches to calibration and says so. A corrupt file is an on-screen error, not a hang.

### 3. Particles

```bash
python -m liveplay --mode play
```

Or press **3**. Passthrough turns off. The field is the same flat gray as calibration, and particles follow the tracked blobs. Styles: `sparks` (default), `blobs`, `trails`.

```bash
python -m liveplay --particle-style blobs
```

### 4. Leave it in kiosk

```bash
caffeinate -d python -m liveplay --display 1
```

Fullscreen, hidden cursor, no menu. **Esc** or **Cmd+Q** quits. **1 / 2 / 3 / C** are adult shortcuts to the modes above.

## Config

`config.json` next to this file. Paths inside it are relative to the file, not the shell’s current directory.

| Key | Default | Role |
| --- | --- | --- |
| `camera.index` | `0` | USB camera index. CLI: `--camera`. |
| `camera.width`, `height`, `fps` | `1280`, `720`, `30` | Requested capture size. Lower than 1080p to keep the USB queue short. |
| `roi` | `0,0,0,0` | Crop in camera pixels before the stretch to 1080p. `w` or `h` of 0 means full frame. CLI: `--roi x,y,w,h`. |
| `perspective` | `null` | Four `[x, y]` corners of the TV in the camera image. Overrides `roi`. |
| `undistort` | off | Optional OpenCV camera matrix (3×3) and distortion coefficients. |
| `display.width`, `height` | `1920`, `1080` | Playfield size. Should match the Vizio. |
| `display.fullscreen` | `true` | CLI: `--windowed` or `--fullscreen`. |
| `display.index` | `null` | `null` auto-picks. CLI: `--display`. |
| `display.prefer_external` | `true` | With several displays, prefer a non-primary 1920×1080, else the last one. |
| `calibration.path` | `calib/empty_table.png` | Empty-table snapshot. CLI: `--calibration`. |
| `calibration.gray` | `90` | Flat playfield and calibration screen, 0–255. Change it and snapshot again. |
| `vision.method` | `diff+skin` | `diff`, `skin`, or `diff+skin`. CLI: `--vision`. |
| `vision.diff_threshold` | `28` | How different a pixel must be from the snapshot. |
| `vision.min_area` | `2200` | Smallest blob, in full-screen pixels. Hands pass; particle specks do not. |
| `vision.track_dark_blobs` | `false` | Also track objects darker than the empty table (dark toys). |
| `particles.style` | `sparks` | `sparks`, `blobs`, or `trails`. CLI: `--particle-style`. |
| `mode` | `play` | Startup mode. CLI: `--mode`. |

`skin` does not need a snapshot (color only). `diff` and `diff+skin` do. `diff` alone will also track bright graphics on the TV. The default `diff+skin` is there because the camera is looking at its own output.

## Feedback loop

The webcam is pointed at the display. A few rules keep the particles from being detected as hands:

- Calibrate on flat gray, and keep the playfield that same gray. A photo or a gradient would differ from the snapshot everywhere.
- Prefer `diff+skin`. Sparks are bright but not skin-colored, so they fail the gate. The ranges live in `liveplay/vision.py` (`_SKIN_LOW_1` and the second red wrap).
- Ignore blobs smaller than `vision.min_area`.
- Fade particles toward that gray, not toward black, so they don’t become fake dark toys.
- The capture code asks the camera for manual exposure. Many webcams ignore it. If blobs drift after a few minutes, lock exposure in the camera’s own tool and recalibrate.

More detail is commented in `liveplay/vision.py`, `liveplay/capture.py`, and `liveplay/display.py`.

## Keys

| Key | When | Action |
| --- | --- | --- |
| Esc, Cmd+Q, Ctrl+Q | always | Quit |
| 1 / 2 / 3 | always | Passthrough / overlay / play |
| C | always | Calibration (flat gray) |
| Space | calibration | Snapshot the empty table |
| Arrows | passthrough, overlay | Move the crop |
| Shift+arrows | passthrough, overlay | Resize the crop |
| S | passthrough, overlay | Save the crop into `config.json` |

## Failures

These print `[liveplay] error: …` and, once the window exists, show the same text on the TV:

- No camera, or the camera stops sending frames (USB, permission, or another app holding it).
- Display index out of range (`--list-displays`).
- Calibration file present but unreadable.
- Broken `config.json`.

A missing snapshot is not fatal: play/overlay switch to the calibration screen and wait for Space. The process does not sit there with a blank terminal.

## Known limitations

- **Glare and reflections.** Ceiling lights bounce off the plexi and can look like blobs. Skin gating rejects many of them. A large washed-out patch can still win. Dim the room, use matte plexi, and raise `vision.diff_threshold` or `vision.min_area` if you get ghosts.
- **Kids blocking the camera.** Points vanish and the particles fade. That is the whole model; there is no memory of a hand it cannot see.
- **Not fingertips.** v0 tracks a blob center, not a pose. MediaPipe would be another `VisionBackend`, not a change to the game loop.
- **Skin color is a guess.** Colored gloves, very warm light, or a hand in deep shadow can miss. `--vision diff` is the fallback and will also see bright particles.
- **Dark toys** are off until `vision.track_dark_blobs` is true. Shadows can then count too.
- **Auto exposure and auto focus** still move on some UVC cameras. The props we set are best-effort.
- **Alignment** is a crop or a four-point warp, not a full lens model, unless you fill in `undistort`.
- **One mode, one machine, no installer.** Packaging, accounts, cloud vision, and generative fill are out of scope.
- **Latency.** Target is under about 100 ms of camera-to-glass delay from *this* process. The panel can add more if motion smoothing is on.

## Tests

No webcam and no window required:

```bash
python -m unittest discover -s tests -v
```

The smoke test runs passthrough, overlay, play, and calibrate on a fake camera with SDL’s dummy driver.

## Project layout

```
config.json                 camera, ROI, display, vision, particles
liveplay/capture.py         webcam, ROI, perspective, undistort
liveplay/vision.py          VisionBackend and the blob detector
liveplay/points.py          InteractionPoint and the frame-to-frame tracker
liveplay/particles.py       spawn / attract / fade
liveplay/calibration.py     empty-table snapshot
liveplay/display.py         fullscreen window, guides, errors
liveplay/app.py             modes and the main loop
liveplay/config.py          config file and CLI
```
