"""
Audio deepfake classifiers — literature-grounded, all weights auto-downloadable.

Backends (newest/best first)
-----------------------------

wav2vec2  [DEFAULT]
  Model: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
  HuggingFace: https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
  Architecture: wav2vec2-large-xlsr-53 (300 M params) fine-tuned for binary
                real/fake classification on ASVspoof 2019 LA.
  License: Apache 2.0.  Downloads: ~150k/month.
  EER: 4.01% on ASVspoof 2019 LA eval set.
  Weights: auto-downloaded via transformers on first run (~1.2 GB).

hubert
  Model: abhishtagatya/hubert-base-960h-itw-deepfake
  HuggingFace: https://huggingface.co/abhishtagatya/hubert-base-960h-itw-deepfake
  Architecture: HuBERT-Base (94 M params) fine-tuned on "in-the-wild" deepfake audio.
  License: Apache 2.0.  Downloads: ~123k/month.
  Reported EER: 1.43%, accuracy 98.7%  (training data undocumented — treat with caution).
  Weights: auto-downloaded via transformers (~360 MB).

lcnn  [trainable fallback]
  Architecture: Light CNN (LCNN-9) with Max-Feature-Map activations on log-mel
                spectrograms. Follows Lavrentyeva et al., Interspeech 2017
                (https://arxiv.org/abs/1701.04224) adapted for ASVspoof challenges.
  Trainable on ASVspoof 2019 LA with scripts/train_audio_lcnn.py.
  No pretrained weights provided — requires local training.

SSL+AASIST reference (not integrated here — for future upgrade)
  "Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2Vec 2.0"
  Tak et al., Speaker Odyssey 2022.
  GitHub: https://github.com/TakHemlata/SSL_Anti-spoofing  (MIT, 165 stars)
  Architecture: wav2vec2-XLSR-300M frontend + AASIST backend.
  EER: 0.82% (LA track), 2.85% (DF track) on ASVspoof 2021.
  Weights: Google Drive link in the repo README.

AASIST reference (not integrated here — for future upgrade)
  "Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"
  Jung et al., ICASSP 2022.  arXiv: https://arxiv.org/abs/2110.01200
  GitHub: https://github.com/clovaai/aasist  (MIT, 275 stars)
  EER: 0.83% (AASIST), 0.99% (AASIST-L lightweight variant).
  Weights: included in the repo (ASVspoof 2019 LA checkpoint).

Generalisation caveat (important)
-----------------------------------
The NII Yamagishilab wav2vec2 model (EER ~2% on standard benchmarks) jumps to
30% EER on Deepfake-Eval-2024 real-world samples.  All audio models degrade
substantially on out-of-distribution, post-2022 voice cloning systems.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend 1: wav2vec2-large-xlsr  (Apache 2.0, 150k downloads/month)
# ---------------------------------------------------------------------------

class Wav2VecAudioDetector:
    """
    Loads Gustking/wav2vec2-large-xlsr-deepfake-audio-classification.
    Weights auto-downloaded to ~/.cache/huggingface on first run (~1.2 GB).
    """

    HF_MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

    def __init__(self, device: torch.device, sample_rate: int = 16000) -> None:
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError:
            raise ImportError("pip install transformers")

        logger.info("Loading wav2vec2-xlsr audio detector from HuggingFace (~1.2 GB on first run)…")
        self.extractor = AutoFeatureExtractor.from_pretrained(self.HF_MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(self.HF_MODEL_ID)
        self.model.eval().to(device)
        self.device = device
        self.sample_rate = sample_rate

        id2label = self.model.config.id2label
        self._fake_idx = next(
            (k for k, v in id2label.items() if "fake" in v.lower() or "spoof" in v.lower()), 1
        )
        logger.info("wav2vec2 audio detector loaded. fake_label_idx=%d", self._fake_idx)

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> float:
        inputs = self.extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)
        return probs[0, self._fake_idx].item()


# ---------------------------------------------------------------------------
# Backend 2: HuBERT-base  (Apache 2.0, 123k downloads/month)
# ---------------------------------------------------------------------------

class HuBERTAudioDetector:
    """
    Loads abhishtagatya/hubert-base-960h-itw-deepfake.
    Weights auto-downloaded (~360 MB on first run).

    Caution: training data for this model is undocumented ("in-the-wild");
    EER claims (1.43%) should be treated with scepticism until independently
    verified.  Use wav2vec2 if reproducibility matters.
    """

    HF_MODEL_ID = "abhishtagatya/hubert-base-960h-itw-deepfake"

    def __init__(self, device: torch.device, sample_rate: int = 16000) -> None:
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError:
            raise ImportError("pip install transformers")

        logger.info("Loading HuBERT audio detector from HuggingFace (~360 MB on first run)…")
        self.extractor = AutoFeatureExtractor.from_pretrained(self.HF_MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(self.HF_MODEL_ID)
        self.model.eval().to(device)
        self.device = device
        self.sample_rate = sample_rate

        id2label = self.model.config.id2label
        self._fake_idx = next(
            (k for k, v in id2label.items() if "fake" in v.lower() or "spoof" in v.lower()), 1
        )
        logger.info("HuBERT audio detector loaded.")

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> float:
        inputs = self.extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)
        return probs[0, self._fake_idx].item()


# ---------------------------------------------------------------------------
# Backend 3: LCNN on log-mel spectrograms  (trainable fallback)
# ---------------------------------------------------------------------------

class MaxFeatureMap(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


class LCNN(nn.Module):
    """
    LCNN-9 with Max-Feature-Map activations on log-mel spectrograms.
    Follows Lavrentyeva et al., Interspeech 2017.
    Input: (N, 1, T, F) where F = n_mels.  Output: (N, 2) logits.
    Train with scripts/train_audio_lcnn.py on ASVspoof 2019 LA.
    """

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()

        def cmfm(in_ch, out_ch, k=3, p=1):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch * 2, k, padding=p, bias=False), MaxFeatureMap())

        self.features = nn.Sequential(
            cmfm(1,   32), nn.MaxPool2d(2, 2),
            cmfm(32,  32),
            cmfm(32,  64), nn.MaxPool2d(2, 2),
            cmfm(64,  64),
            cmfm(64, 128), nn.MaxPool2d(2, 2),
            cmfm(128, 128),
            cmfm(128, 128), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256), MaxFeatureMap(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def extract_log_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    target_frames: int = 300,
) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels, fmax=sample_rate // 2,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T  # (T, F)
    if log_mel.shape[0] < target_frames:
        log_mel = np.pad(log_mel, ((0, target_frames - log_mel.shape[0]), (0, 0)))
    else:
        log_mel = log_mel[:target_frames]
    mn, mx = log_mel.min(), log_mel.max()
    if mx > mn:
        log_mel = 2.0 * (log_mel - mn) / (mx - mn) - 1.0
    return log_mel[np.newaxis]  # (1, T, F)


# ---------------------------------------------------------------------------
# Unified public wrapper
# ---------------------------------------------------------------------------

class AudioDeepfakeDetector:
    """
    Unified wrapper for all audio backends.

    backend options
    ---------------
    "wav2vec2"  Apache 2.0; 150k downloads/month; EER 4.01% on ASVspoof2019
    "hubert"    Apache 2.0; 123k downloads/month; claimed EER 1.43% (unverified)
    "lcnn"      Trainable fallback — no pretrained weights; needs ASVspoof 2019 data
    """

    def __init__(
        self,
        backend: str = "wav2vec2",
        model_path: Optional[str] = None,   # only used for lcnn
        threshold: float = 0.60,
        sample_rate: int = 16000,
        device: Optional[str] = None,
    ) -> None:
        self.backend     = backend.lower()
        self.threshold   = threshold
        self.sample_rate = sample_rate
        self.device      = self._pick_device(device)
        self._impl       = self._build(model_path)

    @staticmethod
    def _pick_device(req: Optional[str]) -> torch.device:
        if req:
            return torch.device(req)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build(self, model_path: Optional[str]):
        if self.backend == "wav2vec2":
            return Wav2VecAudioDetector(self.device, self.sample_rate)

        elif self.backend == "hubert":
            return HuBERTAudioDetector(self.device, self.sample_rate)

        elif self.backend == "lcnn":
            net = LCNN()
            path = model_path or "models/lcnn_asvspoof.pth"
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                net.load_state_dict(state, strict=False)
                logger.info("LCNN weights loaded from %s", path)
            else:
                logger.warning(
                    "LCNN weights not found at %s. "
                    "Train with scripts/train_audio_lcnn.py on ASVspoof 2019 LA.", path
                )
            net.eval().to(self.device)
            return net

        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}. "
                "Choose: wav2vec2 | hubert | lcnn"
            )

    def predict(self, waveform: np.ndarray) -> Tuple[float, bool]:
        """
        Parameters
        ----------
        waveform : np.ndarray, float32, shape (N,), range [-1, 1]

        Returns
        -------
        fake_prob : float  — probability of synthetic/spoofed speech
        is_deepfake : bool
        """
        if self.backend in ("wav2vec2", "hubert"):
            prob = self._impl.predict(waveform)
        else:
            # LCNN — log-mel feature extraction
            try:
                feat = extract_log_mel(waveform, sample_rate=self.sample_rate)
            except Exception as exc:
                logger.error("Log-mel extraction failed: %s", exc)
                return 0.0, False
            x = torch.from_numpy(feat).unsqueeze(0).float().to(self.device)
            logits = self._impl(x)
            prob = F.softmax(logits, dim=1)[0, 1].item()

        return prob, prob >= self.threshold
