"""
Orchestrator — ties together the video and audio pipelines.

Lifecycle
---------
1. Call  orchestrator.start()  when a call begins (or at detection time).
2. The video pipeline runs in a background thread: it collects frame bursts
   and runs batch inference.
3. The audio pipeline streams continuously in sounddevice callbacks: VAD
   gates the analysis so only remote-party audio is examined.
4. When either pipeline flags a deepfake, a rate-limited OS notification fires.
5. Call  orchestrator.stop()  to clean up.
"""

import logging
import queue
import threading
import time
from typing import Optional

from config import Config
from detector.notify import NotificationGate
from detector.video.frame_sampler import FrameSampler
from detector.video.face_extractor import FaceExtractor
from detector.video.model import VideoDeepfakeDetector
from detector.audio.capture import AudioCapture
from detector.audio.vad import MicVAD
from detector.audio.model import AudioDeepfakeDetector

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordinates video and audio deepfake detection for a live call.

    Parameters
    ----------
    config : Config
        Full configuration object.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self._gate = NotificationGate(self.cfg.notification_cooldown)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        # Video components (lazy — only built if video enabled)
        self._sampler: Optional[FrameSampler] = None
        self._extractor: Optional[FaceExtractor] = None
        self._video_model: Optional[VideoDeepfakeDetector] = None

        # Audio components
        self._vad: Optional[MicVAD] = None
        self._audio_capture: Optional[AudioCapture] = None
        self._audio_model: Optional[AudioDeepfakeDetector] = None
        self._audio_clip_queue: queue.Queue = queue.Queue(maxsize=8)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_video(self) -> None:
        vc = self.cfg.video
        self._sampler = FrameSampler(
            sample_count=vc.sample_count,
            sample_window=vc.sample_window,
            ongoing_interval=vc.ongoing_interval,
            capture_region=vc.capture_region,
        )
        self._extractor = FaceExtractor(
            min_confidence=vc.face_min_confidence,
            padding=vc.face_padding,
            output_size=vc.frame_size,
        )
        self._video_model = VideoDeepfakeDetector(
            backend=vc.model_backend,
            model_path=vc.model_path,
            threshold=vc.deepfake_threshold,
        )
        logger.info("Video pipeline initialised (backend=%s).", vc.model_backend)

    def _init_audio(self) -> None:
        ac = self.cfg.audio
        self._vad = MicVAD(
            sample_rate=ac.sample_rate,
            frame_ms=ac.vad_frame_ms,
            aggressiveness=ac.vad_aggressiveness,
        )
        self._audio_model = AudioDeepfakeDetector(
            model_path=ac.model_path,
            threshold=ac.deepfake_threshold,
            sample_rate=ac.sample_rate,
            n_mels=ac.n_mels,
            n_fft=ac.n_fft,
            hop_length=ac.hop_length,
        )
        self._audio_capture = AudioCapture(
            sample_rate=ac.sample_rate,
            mic_device=ac.mic_device,
            system_audio_device=ac.system_audio_device,
            clip_duration=ac.clip_duration,
            on_user_silent_clip=self._on_audio_clip,
            is_user_speaking_fn=self._vad.is_speech,
        )
        logger.info("Audio pipeline initialised.")

    # ------------------------------------------------------------------
    # Audio callback (called from sounddevice thread)
    # ------------------------------------------------------------------

    def _on_audio_clip(self, clip) -> None:
        """Non-blocking: drop the clip if the inference queue is full."""
        try:
            self._audio_clip_queue.put_nowait(clip)
        except queue.Full:
            pass  # inference is behind; skip this clip

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def _video_worker(self) -> None:
        """Runs frame sampling and inference in a background thread."""
        assert self._sampler is not None
        assert self._extractor is not None
        assert self._video_model is not None

        self._sampler.start()
        logger.info("Video worker started.")

        while not self._stop_event.is_set():
            if not self._sampler.should_run_burst():
                time.sleep(0.5)
                continue

            logger.info("Collecting video frame burst (%d frames)…", self.cfg.video.sample_count)
            frames = self._sampler.collect_burst()

            face_crops = []
            for frame in frames:
                crop = self._extractor.extract_largest_face(frame)
                if crop is not None:
                    face_crops.append(crop)

            if not face_crops:
                logger.info("No faces detected in burst — skipping inference.")
                continue

            probs, is_fake = self._video_model.predict(face_crops)
            avg_prob = sum(probs) / len(probs)
            logger.info(
                "Video inference: %d faces, avg_fake_prob=%.2f, is_fake=%s",
                len(face_crops), avg_prob, is_fake,
            )

            if is_fake:
                self._gate.maybe_alert(
                    channel="video",
                    title="⚠️ Deepfake Detected — Video",
                    message=(
                        "The video in this call may be AI-generated or manipulated.\n"
                        f"Analysed {len(face_crops)} face(s)."
                    ),
                    confidence=avg_prob,
                )

        if self._extractor:
            self._extractor.close()
        logger.info("Video worker stopped.")

    def _audio_inference_worker(self) -> None:
        """Pulls audio clips from the queue and runs the classifier."""
        assert self._audio_model is not None
        logger.info("Audio inference worker started.")

        while not self._stop_event.is_set():
            try:
                clip = self._audio_clip_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            prob, is_fake = self._audio_model.predict(clip)
            logger.info("Audio inference: fake_prob=%.2f, is_fake=%s", prob, is_fake)

            if is_fake:
                self._gate.maybe_alert(
                    channel="audio",
                    title="⚠️ Deepfake Detected — Audio",
                    message=(
                        "The voice on this call may be AI-synthesised or cloned.\n"
                        "Treat this call with caution."
                    ),
                    confidence=prob,
                )

        logger.info("Audio inference worker stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all enabled detection pipelines."""
        self._stop_event.clear()

        if self.cfg.video.enabled:
            try:
                self._init_video()
                t = threading.Thread(
                    target=self._video_worker,
                    name="video-worker",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            except Exception as exc:
                logger.error("Video pipeline failed to start: %s", exc)

        if self.cfg.audio.enabled:
            try:
                self._init_audio()
                t = threading.Thread(
                    target=self._audio_inference_worker,
                    name="audio-inference",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
                self._audio_capture.start()
            except Exception as exc:
                logger.error("Audio pipeline failed to start: %s", exc)

        logger.info(
            "Deepfake detector running — video=%s, audio=%s.",
            self.cfg.video.enabled,
            self.cfg.audio.enabled,
        )

    def stop(self) -> None:
        """Signal all workers to stop and wait for clean shutdown."""
        logger.info("Stopping deepfake detector…")
        self._stop_event.set()

        if self._audio_capture is not None:
            self._audio_capture.stop()

        for t in self._threads:
            t.join(timeout=5.0)

        self._threads.clear()
        logger.info("Deepfake detector stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
