#!/usr/bin/env python3
"""
Download and convert the original MesoNet-4 Keras weights to PyTorch format.

Source
------
Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection Network",
ICCVW 2018.  https://github.com/DariusAf/MesoNet

The .h5 weights file is hosted in the original GitHub repo under:
  weights/Meso4_DF.h5        — trained on Deepfakes subset of FaceForensics
  weights/Meso4_F2F.h5       — trained on Face2Face subset

This script:
  1. Downloads Meso4_DF.h5 (or Meso4_F2F.h5) from the GitHub repo.
  2. Reads the Keras weights using h5py (no TensorFlow/Keras needed).
  3. Re-maps them to our PyTorch MesoNet4 state_dict.
  4. Saves models/mesonet4_ff.pth ready for main.py --video-backend mesonet.

Usage
-----
  pip install h5py requests
  python scripts/convert_mesonet_keras.py              # Deepfakes weights
  python scripts/convert_mesonet_keras.py --variant f2f   # Face2Face weights
  python scripts/convert_mesonet_keras.py --h5 /path/to/local/Meso4_DF.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# GitHub raw URLs (official DariusAf/MesoNet repo — these are real files)
# ---------------------------------------------------------------------------
URLS = {
    "df":  "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5",
    "f2f": "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_F2F.h5",
}

OUTPUT_PATH = Path("models/mesonet4_ff.pth")


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_h5(url: str, dest: Path) -> None:
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Install dependencies first:  pip install requests tqdm")
        sys.exit(1)

    print(f"Downloading {url} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Saved to {dest}")


# ---------------------------------------------------------------------------
# Keras h5 → PyTorch weight mapping
# ---------------------------------------------------------------------------
# Keras stores weights inside an HDF5 hierarchy.  For a Sequential/Functional
# model the typical path is:
#   /model_weights/<layer_name>/<layer_name>/<weight_name>
# but for older Keras versions it may be:
#   /model_weights/<layer_name>/<weight_name>
# We handle both cases below.

def _get_weight(h5_root, layer_name: str, weight_name: str) -> np.ndarray:
    """Try both old and new Keras h5 layout."""
    try:
        import h5py
    except ImportError:
        print("h5py is required:  pip install h5py")
        sys.exit(1)

    mw = h5_root["model_weights"]

    # New layout: model_weights/layer/layer/kernel:0
    if layer_name in mw and layer_name in mw[layer_name]:
        node = mw[layer_name][layer_name]
    elif layer_name in mw:
        node = mw[layer_name]
    else:
        raise KeyError(f"Layer '{layer_name}' not found in h5 model_weights.")

    # Weight name may or may not have ':0' suffix
    for candidate in [weight_name, weight_name + ":0", weight_name.replace(":0", "")]:
        if candidate in node:
            return np.array(node[candidate])

    raise KeyError(f"Weight '{weight_name}' not found in layer '{layer_name}'.")


def convert(h5_path: Path) -> dict:
    """
    Read Keras weights from h5_path and return a PyTorch state_dict that
    matches our MesoNet4 architecture exactly.

    Keras Conv2D kernel shape:  (kH, kW, C_in, C_out)
    PyTorch Conv2d weight shape: (C_out, C_in, kH, kW)   ← needs transpose

    Keras Dense kernel shape:   (in_features, out_features)
    PyTorch Linear weight shape: (out_features, in_features) ← needs transpose

    Keras BatchNormalization stores: gamma, beta, moving_mean, moving_variance.
    PyTorch BatchNorm2d stores:      weight, bias, running_mean, running_var.
    """
    try:
        import h5py
    except ImportError:
        print("h5py is required:  pip install h5py")
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        def get(layer, w):
            return _get_weight(f, layer, w)

        sd = {}

        # ---- Conv layers ----
        # Layer names in Keras Sequential model: conv2d, conv2d_1, conv2d_2, conv2d_3
        conv_map = [
            ("conv2d",   "conv1"),
            ("conv2d_1", "conv2"),
            ("conv2d_2", "conv3"),
            ("conv2d_3", "conv4"),
        ]
        for keras_name, pt_name in conv_map:
            k = get(keras_name, "kernel")          # (kH, kW, C_in, C_out)
            b = get(keras_name, "bias")
            sd[f"{pt_name}.weight"] = torch.from_numpy(k.transpose(3, 2, 0, 1).copy())
            sd[f"{pt_name}.bias"]   = torch.from_numpy(b.copy())

        # ---- BatchNorm layers ----
        bn_map = [
            ("batch_normalization",   "bn1"),
            ("batch_normalization_1", "bn2"),
            ("batch_normalization_2", "bn3"),
            ("batch_normalization_3", "bn4"),
        ]
        for keras_name, pt_name in bn_map:
            sd[f"{pt_name}.weight"]       = torch.from_numpy(get(keras_name, "gamma").copy())
            sd[f"{pt_name}.bias"]         = torch.from_numpy(get(keras_name, "beta").copy())
            sd[f"{pt_name}.running_mean"] = torch.from_numpy(get(keras_name, "moving_mean").copy())
            sd[f"{pt_name}.running_var"]  = torch.from_numpy(get(keras_name, "moving_variance").copy())
            # num_batches_tracked is not in Keras; default 0
            sd[f"{pt_name}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

        # ---- Dense / FC layers ----
        k1 = get("dense", "kernel")    # (4096, 16) → PyTorch: (16, 4096)
        b1 = get("dense", "bias")
        sd["fc1.weight"] = torch.from_numpy(k1.T.copy())
        sd["fc1.bias"]   = torch.from_numpy(b1.copy())

        k2 = get("dense_1", "kernel")  # (16, 1) → PyTorch: (1, 16)
        b2 = get("dense_1", "bias")
        sd["fc2.weight"] = torch.from_numpy(k2.T.copy())
        sd["fc2.bias"]   = torch.from_numpy(b2.copy())

    return sd


# ---------------------------------------------------------------------------
# Verify the converted weights load cleanly
# ---------------------------------------------------------------------------

def verify(state_dict: dict) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from detector.video.model import MesoNet4

    net = MesoNet4()
    missing, unexpected = net.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}")
        print(f"[WARN] Unexpected keys: {unexpected}")
    else:
        print("State dict loaded with no missing or unexpected keys.")

    # Quick shape check
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 1), f"Unexpected output shape: {out.shape}"
    print(f"Forward pass OK — output shape {tuple(out.shape)}, value {out.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MesoNet Keras weights → PyTorch.")
    ap.add_argument(
        "--variant",
        choices=["df", "f2f"],
        default="df",
        help="Weight variant: 'df' = Deepfakes (default), 'f2f' = Face2Face.",
    )
    ap.add_argument(
        "--h5",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a locally downloaded .h5 file (skips download).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        metavar="PATH",
        help=f"Output .pth path (default: {OUTPUT_PATH}).",
    )
    args = ap.parse_args()

    # Change to repo root so relative paths work
    import os
    os.chdir(Path(__file__).parent.parent)

    h5_path = args.h5
    if h5_path is None:
        h5_path = Path(f"models/Meso4_{args.variant.upper()}.h5")
        if not h5_path.exists():
            download_h5(URLS[args.variant], h5_path)

    print(f"Converting {h5_path} …")
    state_dict = convert(h5_path)

    print("Verifying converted weights …")
    verify(state_dict)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"\nSaved PyTorch weights → {args.output}")
    print("You can now run:  python main.py --video-backend mesonet")


if __name__ == "__main__":
    main()
