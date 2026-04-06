"""
Voice Activity Detection for the user's own microphone.

Uses Google's WebRTC VAD — the same detector used in Chrome/Firefox for
echo cancellation and noise suppression.  It operates on 10/20/30 ms frames
at 8, 16, 32, or 48 kHz.

Purpose
-------
We only want to analyze *incoming* audio (the remote party's voice).
By monitoring the user's microphone with VAD, we can detect when the user
is silent and therefore the audio on the line is entirely from the remote
side — a clean window for deepfake analysis.

Reference
---------
WebRTC project:  https://webrtc.org/
Python binding:  https://github.com/wiseman/py-webrtcvad
"""

import collections
import logging
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)

try:
    import webrtcvad
    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    _WEBRTCVAD_AVAILABLE = False
    logger.warning("webrtcvad not installed. Run: pip install webrtcvad")


class MicVAD:
    """
    Tracks whether the local user is currently speaking.

    Uses a sliding window of VAD decisions to smooth out transient noise.

    Parameters
    ----------
    sample_rate : int
        Must be 8000, 16000, 32000, or 48000.
    frame_ms : int
        Frame length in milliseconds.  Must be 10, 20, or 30.
    aggressiveness : int
        0 = least aggressive (more speech labelled as speech),
        3 = most aggressive (more silence labelled as silence).
    smoothing_frames : int
        Number of consecutive frames in the sliding window.
        Voice state is set to the majority label in this window.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        smoothing_frames: int = 5,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.smoothing_frames = smoothing_frames
        self._ring: Deque[bool] = collections.deque(maxlen=smoothing_frames)
        self._vad = None
        if _WEBRTCVAD_AVAILABLE:
            self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, pcm_float32: np.ndarray) -> bool:
        """
        Decide whether `pcm_float32` contains speech.

        Parameters
        ----------
        pcm_float32 : np.ndarray, shape (frame_samples,), dtype float32, range [-1, 1]
        """
        if self._vad is None:
            # Fall back to RMS energy threshold
            return float(np.sqrt(np.mean(pcm_float32 ** 2))) > 0.01

        # WebRTC VAD requires 16-bit PCM bytes
        pcm_int16 = (pcm_float32 * 32767).astype(np.int16)
        raw = pcm_int16.tobytes()

        try:
            result = self._vad.is_speech(raw, self.sample_rate)
        except Exception:
            result = False

        self._ring.append(result)
        # Majority vote over the sliding window
        return sum(self._ring) > len(self._ring) / 2

    def reset(self) -> None:
        self._ring.clear()
