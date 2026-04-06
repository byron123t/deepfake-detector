#!/usr/bin/env python3
"""
Fine-tune EfficientNet-B0 on FaceForensics++ for video deepfake detection.

Dataset
-------
FaceForensics++ (Rössler et al., ICCV 2019)
  Request access: https://github.com/ondyari/FaceForensics
  Fill out the form; the dataset is free for research use.
  Download the c23 (light compression) split for best model quality.

Expected directory layout after downloading + extracting:
  <data_dir>/
    original_sequences/youtube/c23/frames/<video_id>/<frame>.png
    manipulated_sequences/Deepfakes/c23/frames/<video_id>/<frame>.png
    manipulated_sequences/Face2Face/c23/frames/<video_id>/<frame>.png
    manipulated_sequences/FaceSwap/c23/frames/<video_id>/<frame>.png
    manipulated_sequences/NeuralTextures/c23/frames/<video_id>/<frame>.png

The official frame-extraction script is at:
  https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py

Usage
-----
  python scripts/finetune_efficientnet.py --data /path/to/faceforensics
  python scripts/finetune_efficientnet.py --data /path/to/faceforensics --epochs 10 --batch-size 32
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, metavar="DIR",
                    help="Root directory of FaceForensics++ frames.")
    ap.add_argument("--output", type=Path, default=Path("models/efficientnet_b0_ff.pth"),
                    help="Where to save the fine-tuned checkpoint.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None,
                    help="'cpu', 'cuda', or 'mps'. Auto-detected if omitted.")
    return ap.parse_args()


def build_dataset(data_dir: Path):
    """
    Build a PyTorch ImageFolder-compatible dataset from the FF++ frame layout.
    Returns (train_dataset, val_dataset).
    """
    from torchvision import datasets, transforms

    real_dir  = data_dir / "original_sequences" / "youtube" / "c23" / "frames"
    fake_dirs = [
        data_dir / "manipulated_sequences" / m / "c23" / "frames"
        for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    ]

    # Verify at least one directory exists
    if not real_dir.exists():
        logger.error("Real frames directory not found: %s", real_dir)
        logger.error("Expected FaceForensics++ layout — see script docstring.")
        sys.exit(1)

    import os, shutil, tempfile

    # Symlink real and fake frames into a temp ImageFolder layout
    # Layout: tmp/real/<frames>, tmp/fake/<frames>
    tmp = Path(tempfile.mkdtemp(prefix="ff_finetune_"))
    (tmp / "real").mkdir()
    (tmp / "fake").mkdir()

    def link_frames(src_root: Path, dst: Path) -> int:
        count = 0
        for frame in src_root.rglob("*.png"):
            dst_path = dst / f"{frame.parent.name}_{frame.name}"
            if not dst_path.exists():
                os.symlink(frame, dst_path)
            count += 1
        return count

    n_real = link_frames(real_dir, tmp / "real")
    n_fake = sum(link_frames(d, tmp / "fake") for d in fake_dirs if d.exists())
    logger.info("Dataset: %d real frames, %d fake frames", n_real, n_fake)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(str(tmp), transform=tfm)
    n_val = max(1, int(len(full) * 0.1))
    n_train = len(full) - n_val

    import torch
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val])
    val_ds.dataset = datasets.ImageFolder(str(tmp), transform=val_tfm)

    return train_ds, val_ds, tmp


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from detector.video.model import _build_efficientnet, _pick_device

    device = torch.device(args.device) if args.device else (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    logger.info("Device: %s", device)

    train_ds, val_ds, tmp_dir = build_dataset(args.data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2)

    model = _build_efficientnet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(images)
        scheduler.step()
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += len(images)
        val_acc = v_correct / v_total

        logger.info(
            "Epoch %d/%d — train_acc=%.3f  val_acc=%.3f  loss=%.4f",
            epoch, args.epochs, train_acc, val_acc, total_loss / total,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            logger.info("  → Saved best checkpoint to %s", args.output)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Fine-tuning complete. Best val acc: %.3f", best_val_acc)


if __name__ == "__main__":
    main()
