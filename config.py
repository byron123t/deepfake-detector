"""
Configuration for the real-time deepfake detector.

Video pipeline
--------------
Primary:  CommunityForensics ViT (CVPR 2025, MIT)
          Park & Owens, arXiv 2411.04125
          HuggingFace: buildborderless/CommunityForensics-DeepfakeDet-ViT
          Trained on 2.7M images from 4,803 generators; best cross-generator
          generalisation of any open model.

Alternate: GenConViT (2023, GPL-3.0)
          Deressa et al., arXiv 2307.07036
          HuggingFace: Deressa/GenConViT
          ConvNeXt + Swin Transformer + VAE; 95.8% avg on DFDC/FF++/Celeb-DF.

Lightweight: MesoNet-4 (ICCVW 2018)
          Afchar et al., arXiv 1809.00888
          ~28K params; run scripts/convert_mesonet_keras.py for weights.

Audio pipeline
--------------
Primary:  wav2vec2-large-xlsr deepfake detector (Apache 2.0)
          Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
          EER 4.01% on ASVspoof 2019 LA; 150k downloads/month.

Alternate: HuBERT-base in-the-wild deepfake detector (Apache 2.0)
          abhishtagatya/hubert-base-960h-itw-deepfake
          Claimed EER 1.43% (training data undocumented).

State-of-the-art references not yet integrated:
  AASIST (Jung et al., ICASSP 2022):  github.com/clovaai/aasist  EER 0.83%
  SSL+AASIST (Tak et al., Odyssey 2022): github.com/TakHemlata/SSL_Anti-spoofing
  WavLM ensemble (arXiv 2408.07414): top ASVspoof5 2024 system

Generalisation caveat
---------------------
Deepfake-Eval-2024 (arXiv 2503.02857) shows ~50% AUC drop across all SOTA
open models when tested on real-world 2024 deepfakes.  Detection results
should be treated as a probabilistic signal, not ground truth.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class VideoConfig:
    enabled: bool = True

    # Frame sampling: gather N frames spread over the first `sample_window`
    # seconds, then re-sample every `ongoing_interval` seconds.
    sample_count: int = 8
    sample_window: float = 30.0
    ongoing_interval: float = 60.0

    # Face detection (MediaPipe BlazeFace)
    face_min_confidence: float = 0.7
    face_padding: float = 0.2

    # Classification threshold: fake probability required to flag a face.
    # Set conservatively high to reduce false positives on legitimate calls.
    deepfake_threshold: float = 0.60

    # Screen capture region — None = full primary monitor.
    capture_region: Optional[Tuple[int, int, int, int]] = None  # (left, top, w, h)

    # Backend: "community_forensics" | "genconvit" | "mesonet"
    # community_forensics is the default: best generalisation, auto-downloads.
    model_backend: str = "community_forensics"
    model_path: Optional[str] = None   # override default; only used for mesonet


@dataclass
class AudioConfig:
    enabled: bool = True

    sample_rate: int = 16000           # 16 kHz — required by all HF models
    clip_duration: float = 3.0         # seconds per inference window
    hop_duration: float = 1.5          # overlap between windows (50%)

    # WebRTC VAD settings — gates analysis to remote-party audio only
    vad_aggressiveness: int = 2        # 0 (least) to 3 (most) aggressive
    vad_frame_ms: int = 30             # must be 10, 20, or 30

    # User silence gate: analysis only runs when mic RMS < this threshold
    user_silence_rms: float = 0.015

    deepfake_threshold: float = 0.60

    # Backend: "wav2vec2" | "hubert" | "lcnn"
    # wav2vec2 is the default: Apache 2.0, 150k downloads/month, verified EER.
    audio_backend: str = "wav2vec2"
    model_path: Optional[str] = None   # override; only used for lcnn

    # Audio device indices — None = OS default.
    # Run: python scripts/list_audio_devices.py
    mic_device: Optional[int] = None
    system_audio_device: Optional[int] = None

    # LCNN-specific feature extraction (only used when backend="lcnn")
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    models_dir: str = "models"
    notification_cooldown: float = 30.0   # min seconds between repeat alerts
    consensus_threshold: float = 0.5      # fraction of samples that must flag fake
    log_level: str = "INFO"
