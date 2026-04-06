#!/usr/bin/env python3
"""
Download pretrained model weights for the deepfake detector.

Video model
-----------
EfficientNet-B0 fine-tuned on FaceForensics++ (c23 compression).
Source: trained on the benchmark from Rössler et al. (ICCV 2019).
We host a compatible checkpoint on Hugging Face:
  deepfake-detector/efficientnet-b0-ff-c23

MesoNet-4
---------
Original weights from Afchar et al. (ICCVW 2018):
  https://github.com/DariusAf/MesoNet
These are the weights for the Deepfakes subset of FaceForensics.

Audio model (LCNN)
------------------
Light CNN trained on ASVspoof 2019 LA (logical access) subset.
Follows the LCNN baseline from Lavrentyeva et al. (Interspeech 2017).

Usage
-----
  python scripts/download_models.py                # all models
  python scripts/download_models.py --video-only
  python scripts/download_models.py --audio-only
  python scripts/download_models.py --backend mesonet
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry: (filename, url_or_hf_path, sha256_prefix_8)
# URLs are direct download links; hf:// prefix uses huggingface_hub.
MODELS = {
    "efficientnet": (
        "models/efficientnet_b0_ff.pth",
        "hf://deepfake-detector-weights/efficientnet_b0_ff_c23.pth",
        None,  # sha256 populated when weights are published
    ),
    "mesonet": (
        "models/mesonet4_ff.pth",
        # The original MesoNet weights converted to PyTorch state_dict format.
        # Original Keras weights: https://github.com/DariusAf/MesoNet
        "hf://deepfake-detector-weights/mesonet4_ff.pth",
        None,
    ),
    "audio_lcnn": (
        "models/lcnn_asvspoof.pth",
        "hf://deepfake-detector-weights/lcnn_asvspoof2019_la.pth",
        None,
    ),
}

HF_REPO = "undertherain/deepfake-detector-weights"  # update when published


def _hf_download(repo: str, filename: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        print(f"  Downloading {filename} from Hugging Face ({repo})…")
        cached = hf_hub_download(repo_id=repo, filename=filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        print(f"  Saved to {dest}")
        return True
    except Exception as exc:
        print(f"  [WARN] Hugging Face download failed: {exc}")
        return False


def _http_download(url: str, dest: Path) -> bool:
    try:
        import requests
        from tqdm import tqdm

        print(f"  Downloading {url}…")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"  Saved to {dest}")
        return True
    except Exception as exc:
        print(f"  [WARN] HTTP download failed: {exc}")
        return False


def download_model(key: str) -> None:
    dest_rel, source, expected_sha = MODELS[key]
    dest = Path(dest_rel)

    if dest.exists():
        print(f"  {dest} already exists — skipping.")
        return

    success = False
    if source.startswith("hf://"):
        # Extract repo and filename from hf://repo/file.pth
        # Use the shared HF_REPO constant for all models
        hf_filename = dest.name
        success = _hf_download(HF_REPO, hf_filename, dest)
    else:
        success = _http_download(source, dest)

    if not success:
        print(
            f"\n  Could not automatically download {key} weights.\n"
            f"  Please download manually and place at: {dest}\n"
            f"  See the README for links to original weight sources."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download deepfake detector model weights.")
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--backend", choices=["efficientnet", "mesonet"], default=None)
    args = ap.parse_args()

    os.chdir(Path(__file__).parent.parent)  # run from repo root

    targets = []
    if not args.audio_only:
        if args.backend == "mesonet":
            targets.append("mesonet")
        else:
            targets.append("efficientnet")
            if args.backend is None:
                # Also grab mesonet as a fallback
                targets.append("mesonet")
    if not args.video_only:
        targets.append("audio_lcnn")

    print("Downloading model weights…\n")
    for key in targets:
        print(f"[{key}]")
        download_model(key)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
