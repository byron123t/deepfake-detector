"""
Screen-based frame sampler.

Captures frames from the display (covering any video-call app) rather than
trying to intercept a specific app's video stream.  This lets the detector
work with Zoom, Meet, FaceTime, Teams, and browser-based calls without any
per-app integration.

Strategy
--------
Phase 1 — Early sampling (first `sample_window` seconds):
    Capture `sample_count` frames evenly spaced across the window.
    These are the most important: deepfakes are often pre-recorded with a
    static background, and artefacts are most visible before the remote
    party has adapted their lighting / expression.

Phase 2 — Ongoing monitoring:
    After the early window, capture a new burst every `ongoing_interval`
    seconds to catch mid-call swaps (e.g. face-swap filters turned on later).
"""

import logging
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mss
    import mss.tools
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False
    logger.warning("mss not installed — screen capture unavailable. Run: pip install mss")


class FrameSampler:
    """
    Samples BGR frames from the screen at controlled intervals.

    Parameters
    ----------
    sample_count : int
        Number of frames to collect per sampling burst.
    sample_window : float
        Seconds over which to spread the early burst.
    ongoing_interval : float
        Seconds between subsequent bursts after the early window.
    capture_region : tuple | None
        (left, top, width, height) in screen pixels.  None → primary monitor.
    """

    def __init__(
        self,
        sample_count: int = 8,
        sample_window: float = 30.0,
        ongoing_interval: float = 60.0,
        capture_region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.sample_count = sample_count
        self.sample_window = sample_window
        self.ongoing_interval = ongoing_interval
        self.capture_region = capture_region
        self._start_time: Optional[float] = None
        self._last_burst_time: float = 0.0
        self._early_done: bool = False

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._early_done = False
        self._last_burst_time = self._start_time

    def _grab_frame(self, sct: "mss.base.MSSBase") -> Optional[np.ndarray]:
        """Capture one frame and return as an HxWx3 BGR uint8 array."""
        if self.capture_region is not None:
            l, t, w, h = self.capture_region
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]  # primary monitor

        raw = sct.grab(monitor)
        # mss returns BGRA; drop alpha channel
        frame = np.frombuffer(raw.raw, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        return frame[:, :, :3].copy()  # BGR

    def collect_burst(self) -> List[np.ndarray]:
        """
        Grab `sample_count` frames spread evenly over `sample_window` seconds.
        Blocks until all frames are collected.
        """
        if not _MSS_AVAILABLE:
            logger.error("mss not available; cannot capture frames.")
            return []

        frames: List[np.ndarray] = []
        interval = self.sample_window / max(self.sample_count - 1, 1)

        with mss.mss() as sct:
            for i in range(self.sample_count):
                frame = self._grab_frame(sct)
                if frame is not None:
                    frames.append(frame)
                    logger.debug("Captured frame %d/%d", i + 1, self.sample_count)
                if i < self.sample_count - 1:
                    time.sleep(interval)

        return frames

    def collect_single(self) -> Optional[np.ndarray]:
        """Grab one frame immediately."""
        if not _MSS_AVAILABLE:
            return None
        with mss.mss() as sct:
            return self._grab_frame(sct)

    def should_run_burst(self) -> bool:
        """
        Returns True when it's time for a new sampling burst.
        Call this in a polling loop; it handles both early and ongoing phases.
        """
        if self._start_time is None:
            return False

        now = time.monotonic()
        elapsed = now - self._start_time

        if not self._early_done:
            # Run the early burst immediately at start
            if elapsed >= 0:
                self._early_done = True
                self._last_burst_time = now
                return True
        else:
            if now - self._last_burst_time >= self.ongoing_interval:
                self._last_burst_time = now
                return True

        return False
