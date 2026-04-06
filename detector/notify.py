"""
Cross-platform OS-level notifications.

macOS  → native NSUserNotification via plyer (or osascript fallback)
Linux  → libnotify (notify-send) via plyer
Windows → Windows Toast via plyer
"""

import logging
import platform
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

_PLYER_AVAILABLE = False
try:
    from plyer import notification as _plyer_notification
    _PLYER_AVAILABLE = True
except ImportError:
    pass


def _send_macos(title: str, message: str) -> None:
    """AppleScript fallback — always available on macOS."""
    script = (
        f'display notification "{message}" '
        f'with title "{title}" '
        f'sound name "Basso"'
    )
    subprocess.run(["osascript", "-e", script], check=False)


def _send_linux(title: str, message: str) -> None:
    """notify-send fallback."""
    subprocess.run(
        ["notify-send", "--urgency=critical", "--icon=dialog-warning", title, message],
        check=False,
    )


def send_alert(
    title: str,
    message: str,
    *,
    app_name: str = "Deepfake Detector",
    timeout: int = 10,
) -> None:
    """Send an OS-level notification. Never raises — logs errors instead."""
    try:
        if _PLYER_AVAILABLE:
            _plyer_notification.notify(
                title=title,
                message=message,
                app_name=app_name,
                timeout=timeout,
            )
            return

        system = platform.system()
        if system == "Darwin":
            _send_macos(title, message)
        elif system == "Linux":
            _send_linux(title, message)
        else:
            # Windows without plyer — nothing we can do without extra deps
            logger.warning("plyer not installed; cannot send Windows notification.")
    except Exception as exc:
        logger.error("Notification failed: %s", exc)


class NotificationGate:
    """Rate-limits notifications so the user isn't spammed."""

    def __init__(self, cooldown_seconds: float = 30.0) -> None:
        self._cooldown = cooldown_seconds
        self._last_sent: dict[str, float] = {}

    def maybe_alert(
        self,
        channel: str,           # "video" or "audio"
        title: str,
        message: str,
        confidence: Optional[float] = None,
    ) -> bool:
        """Send alert if cooldown has elapsed. Returns True when sent."""
        now = time.monotonic()
        if now - self._last_sent.get(channel, 0.0) < self._cooldown:
            return False

        if confidence is not None:
            message = f"{message}\nConfidence: {confidence:.0%}"

        send_alert(title, message)
        self._last_sent[channel] = now
        logger.warning("[ALERT] %s — %s", title, message)
        return True
