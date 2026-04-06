"""
Audio deepfake classifier.

Architecture: Light CNN (LCNN) on log-mel spectrograms.

Background
----------
The dominant approach in the ASVspoof community (2019–2021 challenges) is to
extract a time-frequency representation and pass it through a CNN.  We use:

  - Log-mel spectrogram (64 bins, 25 ms window, 10 ms hop)  →  time×freq tensor
  - LCNN: alternating Conv2d + Max-Feature-Map (MFM) activations
    (Wu et al., "Light CNN for Deep Face Representation", ICCV 2015;
     adapted for anti-spoofing by Lavrentyeva et al., Interspeech 2017)

State-of-the-art alternatives (referenced for context)
-------------------------------------------------------
- AASIST (Jung et al., ICASSP 2022): graph attention over raw waveform.
  ~300 K params.  Reference impl: https://github.com/clovaai/aasist
- RawNet2 (Tak et al., Interspeech 2021): sinc-layer + GRU on raw waveform.
  Reference impl: https://github.com/asvspoof-challenge/2021

The LCNN here offers a good balance of accuracy and speed for real-time use.
Swap in AASIST weights via model_path once downloaded.

Model weights
-------------
Run  python scripts/download_models.py  to fetch weights.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Max-Feature-Map activation  (replaces ReLU in LCNN)
# ---------------------------------------------------------------------------

class MaxFeatureMap(nn.Module):
    """Split channels in half and take the element-wise max."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


# ---------------------------------------------------------------------------
# Light CNN (LCNN-9)
# ---------------------------------------------------------------------------

class LCNN(nn.Module):
    """
    9-layer LCNN adapted for anti-spoofing on log-mel spectrograms.

    Input  : (N, 1, T, F)  where T ≈ time frames, F = n_mels (64)
    Output : (N, 2)         logits [real, fake]
    """

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()

        def conv_mfm(in_ch, out_ch, k=3, s=1, p=1):
            # out_ch doubles so MFM halves it back
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 2, k, stride=s, padding=p, bias=False),
                MaxFeatureMap(),
            )

        self.features = nn.Sequential(
            conv_mfm(1,   32),                          # → 32ch
            nn.MaxPool2d(2, 2),
            conv_mfm(32,  32),                          # → 32ch
            conv_mfm(32,  64),                          # → 64ch
            nn.MaxPool2d(2, 2),
            conv_mfm(64,  64),                          # → 64ch
            conv_mfm(64, 128),                          # → 128ch
            nn.MaxPool2d(2, 2),
            conv_mfm(128, 128),                         # → 128ch
            conv_mfm(128, 128),                         # → 128ch
            nn.MaxPool2d(2, 2),
        )
        # After 4 poolings with factor 2: F/16, T/16
        # For n_mels=64, F→4; for ~3 s at 16kHz hop=160: T=300→18
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            MaxFeatureMap(),   # → 128
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_log_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    target_frames: int = 300,
) -> np.ndarray:
    """
    Convert a float32 waveform to a log-mel spectrogram.

    Returns shape (1, target_frames, n_mels) suitable for the model.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=sample_rate // 2,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
    log_mel = log_mel.T  # (T, n_mels)

    # Pad or truncate to fixed length
    if log_mel.shape[0] < target_frames:
        pad = target_frames - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="constant")
    else:
        log_mel = log_mel[:target_frames]

    # Normalise to [-1, 1]
    min_v, max_v = log_mel.min(), log_mel.max()
    if max_v > min_v:
        log_mel = 2.0 * (log_mel - min_v) / (max_v - min_v) - 1.0

    return log_mel[np.newaxis]  # (1, T, F)


# ---------------------------------------------------------------------------
# Public detector wrapper
# ---------------------------------------------------------------------------

class AudioDeepfakeDetector:
    """
    Wraps the LCNN model and exposes a `predict` method.

    Parameters
    ----------
    model_path : str | None
        Path to checkpoint.  If None uses models/lcnn_asvspoof.pth.
    threshold : float
        Probability of "fake" class above which audio is flagged.
    sample_rate : int
    n_mels : int
    device : str | None
    """

    DEFAULT_WEIGHTS = "models/lcnn_asvspoof.pth"

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.60,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        device: Optional[str] = None,
    ) -> None:
        self.threshold   = threshold
        self.sample_rate = sample_rate
        self.n_mels      = n_mels
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.device      = self._pick_device(device)

        self._model = self._load(model_path)
        self._model.eval()
        self._model.to(self.device)

    @staticmethod
    def _pick_device(requested: Optional[str]) -> torch.device:
        if requested:
            return torch.device(requested)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load(self, model_path: Optional[str]) -> nn.Module:
        net = LCNN(n_mels=self.n_mels)
        path = model_path or self.DEFAULT_WEIGHTS
        if path and os.path.exists(path):
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                net.load_state_dict(state, strict=False)
                logger.info("Loaded audio model weights from %s", path)
            except Exception as exc:
                logger.warning("Could not load weights from %s: %s", path, exc)
        else:
            logger.warning(
                "No audio model weights at %s. "
                "Run  python scripts/download_models.py. "
                "Predictions will be random until weights are loaded.",
                path,
            )
        return net

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> Tuple[float, bool]:
        """
        Classify a single audio clip.

        Parameters
        ----------
        waveform : np.ndarray, dtype float32, shape (N,)

        Returns
        -------
        fake_prob : float  — probability the audio is synthetic/spoofed
        is_deepfake : bool — True when fake_prob >= threshold
        """
        try:
            feat = extract_log_mel(
                waveform,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        except Exception as exc:
            logger.error("Feature extraction failed: %s", exc)
            return 0.0, False

        # Shape: (1, 1, T, F)
        x = torch.from_numpy(feat).unsqueeze(0).float().to(self.device)
        logits = self._model(x)
        prob = F.softmax(logits, dim=1)[0, 1].item()
        return prob, prob >= self.threshold
