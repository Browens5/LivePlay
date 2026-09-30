"""Must be imported before pygame.

Python 3.14 has no wheels for the original pygame package. pip then builds
it from source, and that build's font fallback circular-imports and warns
on `import pygame`. We do not use pygame.font (OpenCV draws operator
text). The filter keeps that known warning from looking like a crash.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Do not force the dummy audio driver. The garage plays music, and a
# missing device is handled when the mixer opens. Tests set the dummy
# driver themselves.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*pygame\.font.*")
