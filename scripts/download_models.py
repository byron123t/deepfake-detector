#!/usr/bin/env python3
"""
Model weight setup.

The primary models (CommunityForensics ViT + wav2vec2) download automatically
via HuggingFace transformers on first use — you do NOT need to run this script
to start the detector.

This script handles the one model requiring a manual conversion step (MesoNet-4),
and prints honest source information for all models.

  python scripts/download_models.py           # run MesoNet conversion + print info
  python scripts/download_models.py --info    # print model info only (no download)
  python scripts/download_models.py --mesonet # MesoNet conversion only
"""

import argparse
import subprocess
import sys
from pathlib import Path

INFO = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              Deepfake Detector — Model Reference Card                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  VIDEO                                                                       ║
║                                                                              ║
║  community_forensics  [DEFAULT — auto-downloads ~85 MB on first run]        ║
║    Model   : buildborderless/CommunityForensics-DeepfakeDet-ViT             ║
║    Paper   : Park & Owens, CVPR 2025. arXiv:2411.04125                      ║
║    Arch    : ViT-Small fine-tuned on 2.7M images from 4,803 generators      ║
║    License : MIT                                                             ║
║    Why use : Best cross-generator generalisation of any open model;         ║
║              658k downloads/month.                                          ║
║                                                                              ║
║  genconvit  [auto-downloads ~300 MB on first run]                           ║
║    Model   : Deressa/GenConViT on HuggingFace                               ║
║    Paper   : Deressa et al., arXiv:2307.07036 (2023)                        ║
║    Arch    : ConvNeXt + Swin Transformer + VAE ensemble                     ║
║    License : GPL-3.0                                                        ║
║    Result  : 95.8% avg accuracy on DFDC/FF++/DeepfakeTIMIT/Celeb-DF v2     ║
║                                                                              ║
║  mesonet  [requires conversion — run: python scripts/convert_mesonet_keras.py]
║    Weights : DariusAf/MesoNet on GitHub (real .h5 Keras weights)            ║
║    Paper   : Afchar et al., ICCVW 2018. arXiv:1809.00888                    ║
║    Arch    : 4-conv + 2-FC, 28K params, <2ms/face on CPU                   ║
║    License : Original repo doesn't specify (research code)                  ║
║                                                                              ║
║  SOTA references (not yet integrated):                                       ║
║    SBI (CVPR 2022 Oral): github.com/mapooon/SelfBlendedImages               ║
║    LAA-Net (CVPR 2024): github.com/10Ring/LAA-Net                           ║
║    DeepfakeBench (NeurIPS 2023): github.com/SCLBD/DeepfakeBench             ║
║      → 36 detector implementations with weights                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  AUDIO                                                                       ║
║                                                                              ║
║  wav2vec2  [DEFAULT — auto-downloads ~1.2 GB on first run]                  ║
║    Model   : Gustking/wav2vec2-large-xlsr-deepfake-audio-classification     ║
║    Arch    : wav2vec2-large-xlsr-53 fine-tuned on ASVspoof 2019 LA          ║
║    License : Apache 2.0.  Downloads: 150k/month                             ║
║    EER     : 4.01% on ASVspoof 2019 LA eval set                             ║
║                                                                              ║
║  hubert  [auto-downloads ~360 MB on first run]                              ║
║    Model   : abhishtagatya/hubert-base-960h-itw-deepfake                    ║
║    Arch    : HuBERT-Base fine-tuned on "in-the-wild" data (undocumented)    ║
║    License : Apache 2.0.  Downloads: 123k/month                             ║
║    EER     : Claimed 1.43% / 98.7% acc (training data not documented)       ║
║                                                                              ║
║  SOTA references (not yet integrated):                                       ║
║    AASIST (ICASSP 2022): github.com/clovaai/aasist  EER 0.83%  MIT         ║
║    SSL+AASIST (Odyssey 2022): github.com/TakHemlata/SSL_Anti-spoofing       ║
║      → wav2vec2-XLSR + AASIST backend; EER 0.82% on ASVspoof 2021 LA       ║
║      → Weights on Google Drive (link in repo README)                        ║
║    WavLM ensemble (arXiv:2408.07414): top ASVspoof5 2024 system             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  GENERALISATION WARNING                                                      ║
║    Deepfake-Eval-2024 (arXiv:2503.02857): real-world 2024 deepfakes from    ║
║    social media cause ~50% AUC drop in ALL SOTA open models.                ║
║    Treat detection output as a probabilistic early warning, not fact.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def run_mesonet_conversion(variant: str = "df") -> None:
    script = Path(__file__).parent / "convert_mesonet_keras.py"
    result = subprocess.run(
        [sys.executable, str(script), "--variant", variant],
        cwd=Path(__file__).parent.parent,
    )
    sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--info",    action="store_true", help="Print model info and exit.")
    ap.add_argument("--mesonet", action="store_true", help="Run MesoNet Keras conversion.")
    ap.add_argument("--variant", choices=["df", "f2f"], default="df",
                    help="MesoNet variant: df=Deepfakes (default), f2f=Face2Face.")
    args = ap.parse_args()

    print(INFO)

    if args.info:
        return

    if args.mesonet:
        print("Converting MesoNet Keras weights…")
        run_mesonet_conversion(args.variant)
        return

    print("Primary models (CommunityForensics ViT + wav2vec2) download automatically")
    print("via HuggingFace transformers on first run — no manual action required.")
    print()
    print("To convert MesoNet weights (lightweight fallback):")
    print("  python scripts/convert_mesonet_keras.py")
    print()
    print("To run the detector now:")
    print("  python main.py")


if __name__ == "__main__":
    main()
