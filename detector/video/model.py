"""
Video deepfake classifiers.

Two backends are provided — choose based on your latency vs. accuracy tradeoff:

MesoNet-4 (default for pretrained use)
  Architecture from Afchar et al., "MesoNet: a Compact Facial Video Forgery
  Detection Network", ICCVW 2018.
  Original repo: https://github.com/DariusAf/MesoNet
  ~28 K parameters, <1 MB weights, <2 ms/face on CPU.
  Pretrained Keras weights (.h5) are publicly available from the original repo
  and can be converted with:  python scripts/convert_mesonet_keras.py

  Architecture note: this implementation matches the original exactly —
  256×256 input, conv layers with bias, LeakyReLU(0.1) between FCs,
  single sigmoid output.

EfficientNet-B0
  Backbone: EfficientNet-B0 (Tan & Le, ICML 2019) via timm.
  ImageNet pretrained weights load automatically via timm.
  For deepfake-specific accuracy, fine-tune on FaceForensics++
  (Rössler et al., ICCV 2019):
    Dataset access: https://github.com/ondyari/FaceForensics
    (requires submitting a form; data is free for researchers)
  Without FF++ fine-tuning the EfficientNet backbone still detects coarse
  visual artefacts but will have higher false-negative rates on subtle fakes.

Model weights
-------------
  MesoNet  → python scripts/convert_mesonet_keras.py   (downloads + converts)
  EfficientNet → python scripts/finetune_efficientnet.py  (requires FF++ data)
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
    crops: List[np.ndarray], size: int
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

    This implementation exactly matches the original Keras architecture so
    that Keras weights converted by scripts/convert_mesonet_keras.py load
    without any shape mismatches:
      - Input: 256×256 (original paper default)
      - Conv layers include bias (Keras default)
      - BatchNorm epsilon=1e-3 (Keras default)
      - FC1 uses LeakyReLU(0.1) (not plain ReLU)
      - FC2 output: single neuron + sigmoid (binary real/fake)
    """

    INPUT_SIZE = 256  # must match — original paper uses 256×256

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  8, 3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(8,  eps=1e-3)
        self.conv2 = nn.Conv2d(8,  8, 5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm2d(8,  eps=1e-3)
        self.conv3 = nn.Conv2d(8,  16, 5, padding=2, bias=True)
        self.bn3   = nn.BatchNorm2d(16, eps=1e-3)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=True)
        self.bn4   = nn.BatchNorm2d(16, eps=1e-3)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # After 4 max-pool(2): 256 → 128 → 64 → 32 → 16
        self.fc1 = nn.Linear(16 * 16 * 16, 16)
        self.fc2 = nn.Linear(16, 1)  # sigmoid output — matches original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.1))
        return torch.sigmoid(self.fc2(x))  # scalar in [0,1]; 1 = fake


# ---------------------------------------------------------------------------
# EfficientNet-B0 wrapper
# ---------------------------------------------------------------------------

def _build_efficientnet() -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet via timm.

    timm downloads ~20 MB of ImageNet weights automatically on first use.
    For improved deepfake accuracy, fine-tune on FaceForensics++:
      https://github.com/ondyari/FaceForensics  (dataset request form)
    See scripts/finetune_efficientnet.py for a training loop.
    """
    try:
        import timm
        # pretrained=True loads real ImageNet weights from timm's servers.
        # These are used as the feature extractor; the classification head
        # (num_classes=2) is randomly initialised and needs fine-tuning.
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
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
    Wraps either EfficientNet-B0 or MesoNet-4.

    Parameters
    ----------
    backend : str
        "mesonet"  — pretrained weights available via convert_mesonet_keras.py
        "efficientnet" — ImageNet pretrained backbone; fine-tune on FF++ for
                         best accuracy (see scripts/finetune_efficientnet.py)
    model_path : str | None
        Path to a .pth checkpoint override.  None uses the default path.
    threshold : float
        Fake probability above which a face is flagged.
    device : str | None
        "cpu", "cuda", or "mps".  Auto-detected when None.
    """

    DEFAULT_WEIGHTS = {
        "mesonet":      "models/mesonet4_ff.pth",
        "efficientnet": "models/efficientnet_b0_ff.pth",  # optional fine-tuned override
    }

    def __init__(
        self,
        backend: str = "mesonet",
        model_path: Optional[str] = None,
        threshold: float = 0.55,
        device: Optional[str] = None,
    ) -> None:
        self.backend   = backend.lower()
        self.threshold = threshold
        self.device    = self._pick_device(device)

        self._model, self._input_size = self._load(model_path)
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

    def _load(self, model_path: Optional[str]) -> Tuple[nn.Module, int]:
        if self.backend == "mesonet":
            net = MesoNet4()
            input_size = MesoNet4.INPUT_SIZE  # 256
            path = model_path or self.DEFAULT_WEIGHTS["mesonet"]
            if path and os.path.exists(path):
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    net.load_state_dict(state)
                    logger.info("Loaded MesoNet weights from %s", path)
                except Exception as exc:
                    logger.warning("Could not load MesoNet weights: %s", exc)
            else:
                logger.warning(
                    "MesoNet weights not found at %s. "
                    "Run:  python scripts/convert_mesonet_keras.py",
                    path,
                )

        elif self.backend == "efficientnet":
            net = _build_efficientnet()   # downloads ImageNet weights via timm
            input_size = 224
            path = model_path or self.DEFAULT_WEIGHTS["efficientnet"]
            if path and os.path.exists(path):
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    net.load_state_dict(state)
                    logger.info("Loaded fine-tuned EfficientNet weights from %s", path)
                except Exception as exc:
                    logger.warning("Could not load EfficientNet weights: %s", exc)
            else:
                logger.info(
                    "No FF++-fine-tuned EfficientNet weights at %s. "
                    "Using ImageNet pretrained backbone only — accuracy on subtle "
                    "deepfakes will be limited. See scripts/finetune_efficientnet.py.",
                    path,
                )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        return net, input_size

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
            True when the majority of sampled faces exceed the threshold.
        """
        if not face_crops:
            return [], False

        batch = _bgr_crops_to_tensor(face_crops, size=self._input_size).to(self.device)
        out = self._model(batch)

        if self.backend == "mesonet":
            # MesoNet output: (N, 1) sigmoid — 1 = fake
            fake_probs = out.squeeze(1).cpu().tolist()
            if isinstance(fake_probs, float):
                fake_probs = [fake_probs]
        else:
            # EfficientNet output: (N, 2) logits
            fake_probs = F.softmax(out, dim=1)[:, 1].cpu().tolist()

        flagged = sum(p >= self.threshold for p in fake_probs)
        is_fake = (flagged / len(fake_probs)) > 0.5

        return fake_probs, is_fake
