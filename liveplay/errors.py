"""Failures the operator should see, as opposed to programmer errors."""


class LivePlayError(Exception):
    """Expected failure with a message safe to print and show on the TV."""


class ConfigError(LivePlayError):
    pass


class CaptureError(LivePlayError):
    pass


class CalibrationError(LivePlayError):
    pass


class DisplayError(LivePlayError):
    pass
