"""
Video deepfake classifiers.

Two backends are provided — choose based on your latency vs. accuracy tradeoff:

EfficientNet-B0 (default)
  Backbone: EfficientNet-B0 (Tan & Le, ICML 2019) via timm.
  Fine-tuned on FaceForensics++ (Rössler et al., ICCV 2019) which covers
  Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
  Input: 224×224 RGB face crop.
  ~5.3 M parameters.  Inference: ~15 ms on a modern CPU per face.

MesoNet-4 (lightweight)
  Architecture from Afchar et al., "MesoNet: a Compact Facial Video Forgery
  Detection Network", ICCVW 2018.
  ~27 K parameters, <1 MB weights.  Inference: <2 ms on CPU.
  Less accurate than EfficientNet but usable on low-power machines.

Model weights
-------------
Run  python scripts/download_models.py  to fetch pretrained weights from
Hugging Face (see that script for source details).  If no weights file is
found the model initialises with random weights — predictions will be random
until weights are provided.
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
# Preprocessing constants (ImageNet normalisation — same as FaceForensics++)
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _bgr_crops_to_tensor(
    crops: List[np.ndarray], size: int = 224
) -> torch.Tensor:
    """Convert a list of BGR uint8 crops to a normalised NCHW float tensor."""
    import cv2

    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (size, size):
            rgb = cv2.resize(rgb, (size, size))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors, dim=0)  # N,C,H,W
    return (batch - _MEAN) / _STD


# ---------------------------------------------------------------------------
# MesoNet-4
# ---------------------------------------------------------------------------

class MesoNet4(nn.Module):
    """
    MesoNet-4 from Afchar et al. (ICCVW 2018).
    Original repo: https://github.com/DariusAf/MesoNet
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # After 4 poolings: 224 → 14
        self.fc1   = nn.Linear(16 * 14 * 14, 16)
        self.fc2   = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# EfficientNet-B0 wrapper
# ---------------------------------------------------------------------------

def _build_efficientnet() -> nn.Module:
    try:
        import timm
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,   # weights loaded separately
            num_classes=2,
        )
        return model
    except ImportError:
        logger.error("timm is required for EfficientNet backend. Run: pip install timm")
        raise


# ---------------------------------------------------------------------------
# Public classifier wrapper
# ---------------------------------------------------------------------------

class VideoDeepfakeDetector:
    """
    Wraps either EfficientNet-B0 or MesoNet-4, loads weights, and exposes
    a `predict` method that returns per-face posterior probabilities.

    Parameters
    ----------
    backend : str
        "efficientnet" or "mesonet"
    model_path : str | None
        Path to a .pth checkpoint.  If None a default path under models/ is used.
    threshold : float
        Probability of the "fake" class above which a face is flagged.
    device : str
        "cpu", "cuda", or "mps" (Apple Silicon).  Auto-detected if None.
    """

    DEFAULT_WEIGHTS = {
        "efficientnet": "models/efficientnet_b0_ff.pth",
        "mesonet":      "models/mesonet4_ff.pth",
    }

    def __init__(
        self,
        backend: str = "efficientnet",
        model_path: Optional[str] = None,
        threshold: float = 0.55,
        device: Optional[str] = None,
    ) -> None:
        self.backend   = backend.lower()
        self.threshold = threshold
        self.device    = self._pick_device(device)

        self._model = self._load(model_path)
        self._model.eval()
        self._model.to(self.device)

    # ------------------------------------------------------------------

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
        if self.backend == "mesonet":
            net = MesoNet4()
        elif self.backend == "efficientnet":
            net = _build_efficientnet()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        path = model_path or self.DEFAULT_WEIGHTS.get(self.backend)
        if path and os.path.exists(path):
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                # Handle checkpoints that wrap state_dict under a key
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                elif isinstance(state, dict) and "model" in state:
                    state = state["model"]
                net.load_state_dict(state, strict=False)
                logger.info("Loaded video model weights from %s", path)
            except Exception as exc:
                logger.warning("Could not load weights from %s: %s", path, exc)
        else:
            logger.warning(
                "No pretrained weights found at %s. "
                "Run  python scripts/download_models.py  to fetch them. "
                "Predictions will be random until then.",
                path,
            )
        return net

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, face_crops: List[np.ndarray]
    ) -> Tuple[List[float], bool]:
        """
        Classify a list of face crops.

        Returns
        -------
        fake_probs : list[float]
            Per-face probability of being a deepfake (0–1).
        is_deepfake : bool
            True if the majority-vote result exceeds the threshold.
        """
        if not face_crops:
            return [], False

        batch = _bgr_crops_to_tensor(face_crops).to(self.device)
        logits = self._model(batch)          # N×2
        probs  = F.softmax(logits, dim=1)    # N×2
        fake_probs = probs[:, 1].cpu().tolist()

        flagged = sum(p >= self.threshold for p in fake_probs)
        is_fake = flagged / len(fake_probs) > 0.5

        return fake_probs, is_fake
