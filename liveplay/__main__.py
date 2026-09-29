import os

# Quiet library banners before pygame/cv2 import via app.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from liveplay.app import main

if __name__ == "__main__":
    raise SystemExit(main())
