"""Shop music and repair sound effects, synthesized locally.

Nothing is downloaded. The samples are built in memory the first time
the mixer opens. A machine with no audio device keeps playing the
picture; the mixer error is reported once and then ignored.
"""

from __future__ import annotations

import numpy as np

from liveplay import sdl_env  # noqa: F401  # before pygame
import pygame

RATE = 22050

# Shop loop, 100 BPM. Two bars of a stomping riff.
_SHOP_BPM = 100.0
# Race loop, faster, for the winning finish.
_RACE_BPM = 138.0


class TableAudio:
    """One mixer. `play` is a one-shot. `music` loops until replaced."""

    def __init__(self) -> None:
        self.enabled = False
        self._opened = False
        self._effects: dict[str, pygame.mixer.Sound] = {}
        self._loops: dict[str, pygame.mixer.Sound] = {}
        self._music_channel: pygame.mixer.Channel | None = None
        self._music_name: str | None = None

    def play(self, name: str) -> None:
        self._open()
        sound = self._effects.get(name)
        if sound is None:
            return
        sound.play()

    def music(self, name: str | None) -> None:
        """Loop `name`, or stop the loop when `name` is None."""
        self._open()
        if name == self._music_name:
            return
        self._stop_music()
        if not name:
            return
        loop = self._loops.get(name)
        channel = self._music_channel
        if loop is None or channel is None:
            return
        channel.play(loop, loops=-1)
        self._music_name = name

    def stop(self) -> None:
        self._stop_music()
        if self.enabled:
            pygame.mixer.stop()

    def _stop_music(self) -> None:
        if self._music_channel is not None:
            self._music_channel.stop()
        self._music_name = None

    def _open(self) -> None:
        if self._opened:
            return
        self._opened = True
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=RATE, size=-16, channels=2, buffer=1024)
            # Channel 0 stays with the loop. Effects use the rest, so a
            # thud cannot knock the song off the mixer.
            pygame.mixer.set_reserved(1)
            self._music_channel = pygame.mixer.Channel(0)
            self._effects = {name: _sound(clip, 0.9) for name, clip in _effects().items()}
            self._loops = {name: _sound(clip, 0.42) for name, clip in _songs().items()}
            self.enabled = True
        except pygame.error as exc:
            print(f"[liveplay] audio unavailable: {exc}")
            self.enabled = False


def _sound(samples: np.ndarray, volume: float) -> pygame.mixer.Sound:
    stereo = np.column_stack((samples, samples))
    pcm = np.clip(stereo * (32767 * volume), -32767, 32767).astype(np.int16)
    sound = pygame.mixer.Sound(buffer=np.ascontiguousarray(pcm).tobytes())
    return sound


def _effects() -> dict[str, np.ndarray]:
    return {
        "pickup": _blip(420, 760, 0.12),
        "putback": _blip(640, 280, 0.12),
        "tire": _thud(),
        "nut": _ping(740),
        "wrench": _ratchet(),
        "rev": _sweep(70, 160, 0.55),
        "crash": _crash(),
        "fanfare": _fanfare(),
        "clank": _clank(),
        "weld": _weld(),
        "spray": _spray(),
        "stamp": _blip(520, 880, 0.16),
    }


def _songs() -> dict[str, np.ndarray]:
    return {"shop": _groove(_SHOP_BPM, shop=True), "race": _groove(_RACE_BPM, shop=False)}


def _groove(bpm: float, *, shop: bool) -> np.ndarray:
    beat = 60.0 / bpm
    bars = 2
    beats = 4 * bars
    length = beat * beats
    mix = np.zeros(int(RATE * length) + 8, np.float32)
    # Kick on every beat. Snare on 2 and 4. Hats on the off-eighths.
    for i in range(beats):
        at = i * beat
        _add(mix, _kick(), at)
        if i % 2 == 1:
            _add(mix, _snare(), at)
        _add(mix, _hat(), at + beat * 0.5)
    bass = (82.4, 82.4, 98.0, 110.0, 82.4, 73.4, 110.0, 82.4) if shop else (
        82.4, 123.5, 98.0, 146.8, 110.0, 82.4, 98.0, 164.8,
    )
    step = length / len(bass)
    for i, freq in enumerate(bass):
        _add(mix, _tone(freq, step * 0.92, 0.28, "saw", decay=1.2), i * step)
        if not shop:
            _add(mix, _tone(freq * 1.5, step * 0.4, 0.08, "square", decay=3.0), i * step)
    melody = (329.6, 392.0, 440.0, 392.0, 329.6, 293.7, 329.6, 246.9) if shop else (
        523.3, 659.3, 587.3, 784.0, 659.3, 523.3, 587.3, 880.0,
    )
    note = length / len(melody)
    for i, freq in enumerate(melody):
        _add(mix, _tone(freq, note * 0.85, 0.10, "square", decay=2.4), i * note)
    return _clip(mix)


def _blip(start: float, end: float, seconds: float) -> np.ndarray:
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    freq = np.linspace(start, end, n)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    wave = np.sin(phase) * _envelope(n, attack=0.005, release=0.04)
    return _clip(wave * 0.55)


def _thud() -> np.ndarray:
    body = _kick() * 1.1
    slap = _tone(90, 0.2, 0.35, "sine", decay=8.0)
    mix = np.zeros(max(len(body), len(slap)), np.float32)
    _add(mix, body, 0)
    _add(mix, slap, 0)
    return _clip(mix)


def _ping(freq: float) -> np.ndarray:
    n = int(RATE * 0.18)
    t = np.arange(n) / RATE
    wave = (
        np.sin(2 * np.pi * freq * t) * 0.55
        + np.sin(2 * np.pi * freq * 2.01 * t) * 0.25
    )
    return _clip(wave * np.exp(-t * 14) * 0.7)


def _ratchet() -> np.ndarray:
    mix = np.zeros(int(RATE * 0.28), np.float32)
    for i in range(3):
        _add(mix, _ping(180 + i * 40) * 0.45, i * 0.07)
        click = _noise(0.03, 7 + i) * 0.5
        _add(mix, click, i * 0.07)
    return _clip(mix)


def _sweep(start: float, end: float, seconds: float) -> np.ndarray:
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    freq = np.linspace(start, end, n)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    saw = 2.0 * ((np.cumsum(freq) / RATE) % 1.0) - 1.0
    wave = (np.sin(phase) * 0.6 + saw * 0.3) * _envelope(n, attack=0.02, release=0.12)
    return _clip(wave * 0.55)


def _crash() -> np.ndarray:
    boom = _tone(55, 0.7, 0.7, "sine", decay=4.0)
    noise = _noise(0.55, 11) * 0.8
    mix = np.zeros(max(len(boom), len(noise)), np.float32)
    _add(mix, boom, 0)
    _add(mix, noise, 0)
    return _clip(mix)


def _clank() -> np.ndarray:
    mix = np.zeros(int(RATE * 0.24), np.float32)
    _add(mix, _ping(220) * 0.85, 0.0)
    _add(mix, _ping(130) * 0.55, 0.045)
    _add(mix, _noise(0.08, 3) * 0.4, 0.0)
    return _clip(mix)


def _weld() -> np.ndarray:
    n = int(RATE * 0.22)
    t = np.arange(n) / RATE
    noise = _noise_samples(n, 9) * np.exp(-t * 7)
    buzz = np.sign(np.sin(2 * np.pi * 80 * t)) * np.exp(-t * 8) * 0.18
    return _clip((noise * 0.6 + buzz).astype(np.float32))


def _spray() -> np.ndarray:
    n = int(RATE * 0.32)
    t = np.arange(n) / RATE
    env = np.sin(np.clip(t / 0.32, 0, 1) * np.pi)
    return _clip((_noise_samples(n, 2) * env * 0.4).astype(np.float32))


def _fanfare() -> np.ndarray:
    mix = np.zeros(int(RATE * 0.9), np.float32)
    for i, freq in enumerate((523.3, 659.3, 784.0, 1046.5)):
        _add(mix, _tone(freq, 0.28, 0.28, "square", decay=3.0), i * 0.12)
    return _clip(mix)


def _kick() -> np.ndarray:
    n = int(RATE * 0.16)
    t = np.arange(n) / RATE
    freq = 160 * np.exp(-t * 16) + 48
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    return (np.sin(phase) * np.exp(-t * 12) * 0.95).astype(np.float32)


def _snare() -> np.ndarray:
    n = int(RATE * 0.14)
    t = np.arange(n) / RATE
    noise = _noise_samples(n, 4) * np.exp(-t * 16)
    tone = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 20)
    return (noise * 0.55 + tone * 0.3).astype(np.float32)


def _hat() -> np.ndarray:
    n = int(RATE * 0.035)
    t = np.arange(n) / RATE
    noise = np.diff(_noise_samples(n + 1, 5))
    return (noise * np.exp(-t * 70) * 0.35).astype(np.float32)


def _noise(seconds: float, seed: int) -> np.ndarray:
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    return (_noise_samples(n, seed) * np.exp(-t * 6)).astype(np.float32)


def _noise_samples(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _tone(freq: float, seconds: float, volume: float, kind: str, decay: float) -> np.ndarray:
    n = max(1, int(RATE * seconds))
    t = np.arange(n) / RATE
    if kind == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif kind == "saw":
        wave = 2.0 * ((freq * t) % 1.0) - 1.0
    else:
        wave = np.sin(2 * np.pi * freq * t)
    env = _envelope(n, attack=0.008, release=min(0.08, seconds * 0.4))
    if decay:
        env = env * np.exp(-decay * t)
    return (wave * env * volume).astype(np.float32)


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n, np.float32)
    a = min(n, max(1, int(attack * RATE)))
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    r = min(n, max(1, int(release * RATE)))
    env[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return env


def _add(buffer: np.ndarray, clip: np.ndarray, at: float) -> None:
    start = int(at * RATE)
    if start >= len(buffer):
        return
    end = min(len(buffer), start + len(clip))
    buffer[start:end] += clip[: end - start]


def _clip(samples: np.ndarray) -> np.ndarray:
    return np.tanh(samples).astype(np.float32)


_AUDIO: TableAudio | None = None


def default_audio() -> TableAudio:
    global _AUDIO
    if _AUDIO is None:
        _AUDIO = TableAudio()
    return _AUDIO


__all__ = ["TableAudio", "default_audio"]
