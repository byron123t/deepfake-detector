"""
Configuration for the real-time deepfake detector.

Video pipeline references:
  - FaceForensics++ (Rössler et al., ICCV 2019)
  - EfficientNet-based DFDC detector (Tan et al., 2019 + DFDC 2020)
  - MesoNet (Afchar et al., ICCVW 2018)

Audio pipeline references:
  - AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks (Jung et al., ICASSP 2022)
  - RawNet2 (Tak et al., Interspeech 2021)
  - ASVspoof 2019/2021 challenge baselines
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class VideoConfig:
    enabled: bool = True

    # Frame sampling — only analyze the early portion of a call.
    # Sampling N frames spread across the first `sample_window` seconds
    # keeps the pipeline fast while covering the initial handshake period
    # most likely to use a prepared deepfake.
    sample_count: int = 8
    sample_window: float = 30.0       # seconds to spread samples across
    ongoing_interval: float = 60.0    # seconds between subsequent sample bursts

    # Face detection (MediaPipe BlazeFace)
    face_min_confidence: float = 0.7
    face_padding: float = 0.2         # fractional padding around detected face

    # Classification
    deepfake_threshold: float = 0.55  # posterior probability to flag as fake
    frame_size: int = 224             # model input resolution

    # Screen capture region — None means full primary monitor.
    # Tune this to the region showing the remote participant's face.
    capture_region: Optional[Tuple[int, int, int, int]] = None  # (left, top, width, height)

    # Model backend: "mesonet" (default — pretrained weights available via
    # convert_mesonet_keras.py) or "efficientnet" (requires FF++ fine-tuning
    # for deepfake-specific accuracy; falls back to ImageNet pretrain only)
    model_backend: str = "mesonet"
    model_path: Optional[str] = None   # override default path under models/


@dataclass
class AudioConfig:
    enabled: bool = True

    # Audio parameters
    sample_rate: int = 16000           # Hz — matches ASVspoof / AASIST training
    clip_duration: float = 3.0         # seconds per inference clip
    hop_duration: float = 1.5          # seconds between clip windows (50% overlap)

    # Voice Activity Detection (WebRTC VAD)
    # Aggressiveness 0 = least aggressive (more false negatives),
    # 3 = most aggressive (more false positives on speech as silence)
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 30             # VAD frame length in ms (10, 20, or 30)

    # User silence gate: only analyze incoming audio when the user's own
    # microphone RMS energy is below this threshold.
    user_silence_rms: float = 0.015

    # Classification
    deepfake_threshold: float = 0.60
    n_mels: int = 64                   # mel filter banks for spectrogram model
    n_fft: int = 512
    hop_length: int = 160

    # Device indices — None = OS default.
    # On macOS use BlackHole/Soundflower for system audio loopback;
    # on Linux use a PulseAudio monitor source;
    # on Windows WASAPI loopback is selected automatically.
    mic_device: Optional[int] = None
    system_audio_device: Optional[int] = None

    model_path: Optional[str] = None


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    models_dir: str = "models"

    # Minimum seconds between repeated notifications for the same modality
    notification_cooldown: float = 30.0

    # Confidence required across multiple frames/clips before alerting
    # (majority vote across the sampled batch)
    consensus_threshold: float = 0.5   # fraction of samples that must flag fake

    log_level: str = "INFO"
