#!/usr/bin/env python3
"""
Model weight setup guide.

This script handles the one model that can be fetched and converted automatically
(MesoNet-4), and prints honest instructions for the others.

┌─────────────────────┬───────────────────────┬──────────────────────────────────┐
│ Model               │ Status                │ Action                           │
├─────────────────────┼───────────────────────┼──────────────────────────────────┤
│ MesoNet-4           │ Auto (Keras→PyTorch)  │ python scripts/download_models.py│
│ EfficientNet-B0     │ ImageNet auto via timm │ timm downloads on first run      │
│ EfficientNet FF++   │ Manual (form request) │ See instructions below           │
│ LCNN audio          │ Manual (ASVspoof data)│ See instructions below           │
└─────────────────────┴───────────────────────┴──────────────────────────────────┘

Usage
-----
  python scripts/download_models.py           # fetch + convert MesoNet (recommended)
  python scripts/download_models.py --info    # print all model source info and exit
"""

import argparse
import subprocess
import sys
from pathlib import Path


INFO = """
═══════════════════════════════════════════════════════════════
  Deepfake Detector — Model Weight Sources
═══════════════════════════════════════════════════════════════

VIDEO — MesoNet-4  (recommended starting point)
  Paper : Afchar et al., ICCVW 2018
          https://arxiv.org/abs/1809.00888
  Repo  : https://github.com/DariusAf/MesoNet
  Weights: publicly available Keras .h5 files in the repo
  Setup : python scripts/convert_mesonet_keras.py
          (downloads Meso4_DF.h5 from GitHub and converts to .pth)

VIDEO — EfficientNet-B0  (higher accuracy after fine-tuning)
  Paper : Tan & Le, ICML 2019  https://arxiv.org/abs/1905.11946
  Backbone pretrained on ImageNet is downloaded automatically by timm
  on first use (~20 MB).
  For deepfake-specific accuracy, fine-tune on FaceForensics++:
    Dataset: https://github.com/ondyari/FaceForensics
             Fill out the form; data is free for research use.
    Training: python scripts/finetune_efficientnet.py --data /path/to/ff++
  Without FF++ fine-tuning the backbone still flags coarse visual
  artefacts but will miss subtle, high-quality deepfakes.

AUDIO — LCNN on log-mel spectrograms
  Paper : Lavrentyeva et al., Interspeech 2017
          https://arxiv.org/abs/1701.04224
  The LCNN architecture used here follows the ASVspoof 2019/2021 baselines:
    https://github.com/asvspoof-challenge/2021
  Training data: ASVspoof 2019 Logical Access (LA) partition
    https://datashare.ed.ac.uk/handle/10283/3336
    (free registration required; ~10 GB download)
  Once you have the data:
    python scripts/train_audio_lcnn.py --data /path/to/asvspoof2019
  State-of-the-art audio anti-spoofing (for later upgrade):
    AASIST (Jung et al., ICASSP 2022): https://github.com/clovaai/aasist
    RawNet2 (Tak et al., Interspeech 2021): https://github.com/asvspoof-challenge/2021

═══════════════════════════════════════════════════════════════
"""


def run_conversion() -> None:
    """Run the MesoNet Keras→PyTorch converter as a subprocess."""
    script = Path(__file__).parent / "convert_mesonet_keras.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path(__file__).parent.parent,
    )
    sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up deepfake detector model weights.")
    ap.add_argument(
        "--info",
        action="store_true",
        help="Print model source information and exit without downloading.",
    )
    ap.add_argument(
        "--variant",
        choices=["df", "f2f"],
        default="df",
        help="MesoNet weight variant: 'df' = Deepfakes (default), 'f2f' = Face2Face.",
    )
    args = ap.parse_args()

    print(INFO)

    if args.info:
        return

    print("Fetching MesoNet-4 weights (Deepfakes variant) …")
    print("Source: https://github.com/DariusAf/MesoNet\n")

    script = Path(__file__).parent / "convert_mesonet_keras.py"
    result = subprocess.run(
        [sys.executable, str(script), "--variant", args.variant],
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        print("\n[ERROR] Conversion failed. See output above.")
        sys.exit(1)

    print("\nMesoNet weights ready.")
    print("Next steps for higher accuracy:")
    print("  • EfficientNet FF++: request dataset at https://github.com/ondyari/FaceForensics")
    print("  • Audio LCNN: register at https://datashare.ed.ac.uk/handle/10283/3336")


if __name__ == "__main__":
    main()
