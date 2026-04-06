"""
Audio capture for the deepfake detector.

Two streams run concurrently:
  1. Microphone monitor stream — used *only* for Voice Activity Detection.
     We never store or analyse this audio for deepfakes.
  2. System audio (loopback) stream — the incoming call audio from the
     remote party.  This is the stream fed to the deepfake classifier.

Platform notes
--------------
macOS
  Install BlackHole (https://github.com/ExistentialAudio/BlackHole) or
  Soundflower and set it as a loopback device.  Then pass its device index
  via AudioConfig.system_audio_device.

Linux
  PulseAudio exposes a "monitor" source for each sink.  Run
    python scripts/list_audio_devices.py
  to find it.  It typically has "monitor" in its name.

Windows
  sounddevice on Windows automatically exposes WASAPI loopback sources.
  They appear with "(loopback)" in their name in the device list.
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    logger.warning("sounddevice not installed. Run: pip install sounddevice")


class AudioCapture:
    """
    Dual-stream audio capture.

    Parameters
    ----------
    sample_rate : int
        Target sample rate for both streams (resampled internally if needed).
    mic_device : int | None
        sounddevice device index for the microphone.  None = OS default.
    system_audio_device : int | None
        sounddevice device index for system audio / loopback.
        None = same as microphone (useful if your loopback source is
        the default input).
    clip_duration : float
        Seconds of audio to buffer before making a clip available.
    on_user_silent_clip : Callable[[np.ndarray], None] | None
        Callback invoked with a float32 audio array (shape: [samples,])
        each time a clip of system audio is ready *and* the user is silent.
    is_user_speaking_fn : Callable[[np.ndarray], bool] | None
        Function that returns True when the given mic PCM contains speech.
        Typically provided by MicVAD.is_speech.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mic_device: Optional[int] = None,
        system_audio_device: Optional[int] = None,
        clip_duration: float = 3.0,
        on_user_silent_clip: Optional[Callable[[np.ndarray], None]] = None,
        is_user_speaking_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mic_device = mic_device
        self.system_audio_device = system_audio_device
        self.clip_duration = clip_duration
        self._on_clip = on_user_silent_clip
        self._is_speaking = is_user_speaking_fn

        self._mic_speaking = False
        self._system_buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._clip_samples = int(clip_duration * sample_rate)

        self._mic_stream: Optional["sd.InputStream"] = None
        self._sys_stream: Optional["sd.InputStream"] = None
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Stream callbacks (called from sounddevice's audio thread)
    # ------------------------------------------------------------------

    def _mic_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        mono = indata[:, 0].copy()
        if self._is_speaking is not None:
            speaking = self._is_speaking(mono)
        else:
            # Simple RMS energy gate
            speaking = float(np.sqrt(np.mean(mono ** 2))) > 0.015
        with self._lock:
            self._mic_speaking = speaking

    def _sys_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        mono = indata[:, 0].copy()

        with self._lock:
            user_talking = self._mic_speaking

        if user_talking:
            # Flush accumulated buffer — mixed audio is not clean for analysis
            self._system_buffer.clear()
            self._buffer_samples = 0
            return

        # Accumulate silence windows into the clip buffer
        self._system_buffer.append(mono)
        self._buffer_samples += len(mono)

        if self._buffer_samples >= self._clip_samples:
            clip = np.concatenate(self._system_buffer[: ], axis=0)[: self._clip_samples]
            self._system_buffer.clear()
            self._buffer_samples = 0
            if self._on_clip is not None:
                try:
                    self._on_clip(clip)
                except Exception as exc:
                    logger.error("on_clip callback error: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not _SD_AVAILABLE:
            logger.error("sounddevice not available; audio capture disabled.")
            return
        if self._running:
            return

        block_size = int(self.sample_rate * 0.030)  # 30 ms blocks

        self._mic_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.mic_device,
            blocksize=block_size,
            callback=self._mic_callback,
        )

        # If system_audio_device is the same as the mic (or None), open only
        # one stream and use it for both; otherwise open a second stream.
        sys_device = self.system_audio_device

        self._sys_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=sys_device,
            blocksize=block_size,
            callback=self._sys_callback,
        )

        self._mic_stream.start()
        self._sys_stream.start()
        self._running = True
        logger.info(
            "Audio capture started — mic device=%s, system device=%s",
            self.mic_device,
            sys_device,
        )

    def stop(self) -> None:
        self._running = False
        for stream in (self._mic_stream, self._sys_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._mic_stream = None
        self._sys_stream = None
        logger.info("Audio capture stopped.")
