"""
Video deepfake classifiers — literature-grounded, all weights auto-downloadable.

Backends (newest/best first)
-----------------------------

community_forensics  [DEFAULT]
  "Community Forensics: Using Thousands of Generators to Train Deepfake Detectors"
  Jeongsoo Park & Andrew Owens, CVPR 2025.
  arXiv: https://arxiv.org/abs/2411.04125
  GitHub: https://github.com/JeongsooP/Community-Forensics
  HuggingFace: buildborderless/CommunityForensics-DeepfakeDet-ViT
  Architecture: ViT-Small fine-tuned on 2.7M images from 4,803 distinct generators.
  License: MIT.  Downloads: ~658k/month.
  Weights: auto-downloaded via transformers on first use (~85 MB).
  Key result: best cross-generator generalisation of any open model;
  performance keeps improving as generator diversity in training data grows.

genconvit
  "GenConViT: Generative Convolutional Vision Transformer for Deepfake Detection"
  Deressa et al., arXiv 2307.07036 (2023).
  GitHub: https://github.com/erprogs/GenConViT  (GPL-3.0)
  HuggingFace: Deressa/GenConViT
  Architecture: ConvNeXt + Swin Transformer + VAE latent distribution of real faces.
  Trained on DFDC, FF++, DeepfakeTIMIT, Celeb-DF v2 — 95.8% avg, 99.3% AUC.
  Weights: auto-downloaded from HuggingFace (~300 MB).

mesonet  [lightweight fallback]
  "MesoNet: a Compact Facial Video Forgery Detection Network"
  Afchar et al., ICCVW 2018.  arXiv: https://arxiv.org/abs/1809.00888
  GitHub: https://github.com/DariusAf/MesoNet
  Architecture: 4-conv + 2-FC, sigmoid output.  ~28 K params, <2 ms/face on CPU.
  Weights: python scripts/convert_mesonet_keras.py  (downloads real Keras .h5
           from the original GitHub repo and converts to PyTorch).

Generalisation caveat (important)
-----------------------------------
Deepfake-Eval-2024 (arXiv: 2503.02857) benchmarks open models on real-world
2024 deepfakes and finds ~50% AUC drop vs. standard benchmarks for all SOTA
open models.  This detector is a useful early-warning signal but should not be
treated as ground truth.  The notification text reflects this uncertainty.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ImageNet normalisation (used by MesoNet and EfficientNet-style models)
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _bgr_crops_to_tensor(crops: List[np.ndarray], size: int) -> torch.Tensor:
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (size, size):
            rgb = cv2.resize(rgb, (size, size))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    batch = torch.stack(tensors, dim=0)
    return (batch - _MEAN) / _STD


# ---------------------------------------------------------------------------
# Backend 1: CommunityForensics ViT  (CVPR 2025)
# ---------------------------------------------------------------------------

class CommunityForensicsDetector:
    """
    Loads buildborderless/CommunityForensics-DeepfakeDet-ViT via transformers.
    Weights (~85 MB) are cached in ~/.cache/huggingface on first run.
    """

    HF_MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

    def __init__(self, device: torch.device) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError:
            raise ImportError("pip install transformers")

        logger.info("Loading CommunityForensics ViT from HuggingFace (first run downloads ~85 MB)…")
        self.processor = AutoImageProcessor.from_pretrained(self.HF_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(self.HF_MODEL_ID)
        self.model.eval().to(device)
        self.device = device
        # Determine which label index corresponds to "fake"
        # The model card labels: 0 = real, 1 = fake
        id2label = self.model.config.id2label
        self._fake_idx = next(
            (k for k, v in id2label.items() if "fake" in v.lower()), 1
        )
        logger.info("CommunityForensics loaded. fake_label_idx=%d", self._fake_idx)

    @torch.no_grad()
    def predict_batch(self, bgr_crops: List[np.ndarray]) -> List[float]:
        """Return per-crop fake probabilities."""
        rgb_list = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in bgr_crops]
        # processor handles resize + normalisation
        inputs = self.processor(images=rgb_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits          # (N, num_labels)
        probs  = torch.softmax(logits, dim=-1)
        return probs[:, self._fake_idx].cpu().tolist()


# ---------------------------------------------------------------------------
# Backend 2: GenConViT  (arXiv 2307.07036)
# ---------------------------------------------------------------------------

class GenConViTDetector:
    """
    Loads Deressa/GenConViT from HuggingFace.

    GenConViT uses two sub-networks (ED = encoder-decoder + VAE) that are
    ensemble-averaged at inference time.  The HuggingFace repo provides:
      genconvit_ed_inference.pth   — ED sub-network
      genconvit_vae_inference.pth  — VAE sub-network
    We download both and average their outputs.
    """

    HF_MODEL_ID = "Deressa/GenConViT"
    _LABELS = {0: "real", 1: "fake"}  # GenConViT class order

    def __init__(self, device: torch.device) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        logger.info("Loading GenConViT weights from HuggingFace (~300 MB total)…")
        self.device = device
        self._ed_path  = hf_hub_download(self.HF_MODEL_ID, "genconvit_ed_inference.pth")
        self._vae_path = hf_hub_download(self.HF_MODEL_ID, "genconvit_vae_inference.pth")

        self._ed_model  = self._load_genconvit(self._ed_path)
        self._vae_model = self._load_genconvit(self._vae_path)
        logger.info("GenConViT loaded.")

    def _load_genconvit(self, ckpt_path: str) -> nn.Module:
        """
        GenConViT checkpoints store {'model': state_dict, 'config': ...}.
        The architecture is a timm ConvNeXt + Swin combo; we reconstruct
        it from the checkpoint's own config so no external dependency on
        the GenConViT source tree is needed for inference.
        """
        try:
            import timm
        except ImportError:
            raise ImportError("pip install timm")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("config", {})
        arch = cfg.get("model_name", "convnext_small")

        # Rebuild architecture from timm using the stored model name
        net = timm.create_model(arch, pretrained=False, num_classes=2)
        state = ckpt.get("model", ckpt)
        net.load_state_dict(state, strict=False)
        net.eval().to(self.device)
        return net

    @torch.no_grad()
    def predict_batch(self, bgr_crops: List[np.ndarray]) -> List[float]:
        batch = _bgr_crops_to_tensor(bgr_crops, size=224).to(self.device)
        ed_logits  = self._ed_model(batch)
        vae_logits = self._vae_model(batch)
        avg_probs  = (F.softmax(ed_logits, dim=1) + F.softmax(vae_logits, dim=1)) / 2
        return avg_probs[:, 1].cpu().tolist()  # index 1 = fake


# ---------------------------------------------------------------------------
# Backend 3: MesoNet-4  (ICCVW 2018, lightweight)
# ---------------------------------------------------------------------------

class MesoNet4(nn.Module):
    """
    Matches the original Keras architecture exactly so weights from
    scripts/convert_mesonet_keras.py load without shape errors.

    Differences from naive PyTorch reimplementations to watch out for:
      - Input: 256×256  (original paper; not 224)
      - Conv bias: True  (Keras default)
      - BN epsilon: 1e-3  (Keras default vs PyTorch default 1e-5)
      - FC1 activation: LeakyReLU(0.1)  (not plain ReLU)
      - Output: single sigmoid neuron  (1 = fake)
    """

    INPUT_SIZE = 256

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  8,  3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(8,  eps=1e-3)
        self.conv2 = nn.Conv2d(8,  8,  5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm2d(8,  eps=1e-3)
        self.conv3 = nn.Conv2d(8,  16, 5, padding=2, bias=True)
        self.bn3   = nn.BatchNorm2d(16, eps=1e-3)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=True)
        self.bn4   = nn.BatchNorm2d(16, eps=1e-3)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 16, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.1))
        return torch.sigmoid(self.fc2(x))


# ---------------------------------------------------------------------------
# Unified public wrapper
# ---------------------------------------------------------------------------

class VideoDeepfakeDetector:
    """
    Unified wrapper for all video backends.

    backend options
    ---------------
    "community_forensics"  CVPR 2025 ViT; best generalisation; auto-downloads
    "genconvit"            ConvNeXt+Swin hybrid; auto-downloads from HF
    "mesonet"              Lightweight; run scripts/convert_mesonet_keras.py first
    """

    def __init__(
        self,
        backend: str = "community_forensics",
        model_path: Optional[str] = None,   # only used for mesonet override
        threshold: float = 0.55,
        device: Optional[str] = None,
    ) -> None:
        self.backend   = backend.lower()
        self.threshold = threshold
        self.device    = self._pick_device(device)
        self._impl     = self._build(model_path)

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
        if self.backend == "community_forensics":
            return CommunityForensicsDetector(self.device)

        elif self.backend == "genconvit":
            return GenConViTDetector(self.device)

        elif self.backend == "mesonet":
            net = MesoNet4()
            path = model_path or "models/mesonet4_ff.pth"
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                net.load_state_dict(state)
                logger.info("MesoNet weights loaded from %s", path)
            else:
                logger.warning(
                    "MesoNet weights not found at %s — run "
                    "scripts/convert_mesonet_keras.py first.", path
                )
            net.eval().to(self.device)
            return net

        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}. "
                "Choose: community_forensics | genconvit | mesonet"
            )

    @torch.no_grad()
    def predict(self, face_crops: List[np.ndarray]) -> Tuple[List[float], bool]:
        """
        Classify a batch of face crops.

        Returns
        -------
        fake_probs : list[float]  — per-face fake probability in [0, 1]
        is_deepfake : bool        — majority-vote result vs. threshold
        """
        if not face_crops:
            return [], False

        if self.backend == "community_forensics":
            fake_probs = self._impl.predict_batch(face_crops)

        elif self.backend == "genconvit":
            fake_probs = self._impl.predict_batch(face_crops)

        else:  # mesonet
            batch = _bgr_crops_to_tensor(face_crops, MesoNet4.INPUT_SIZE).to(self.device)
            out = self._impl(batch)
            fake_probs = out.squeeze(1).cpu().tolist()
            if isinstance(fake_probs, float):
                fake_probs = [fake_probs]

        flagged   = sum(p >= self.threshold for p in fake_probs)
        is_fake   = (flagged / len(fake_probs)) > 0.5
        return fake_probs, is_fake
