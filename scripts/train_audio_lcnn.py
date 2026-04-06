#!/usr/bin/env python3
"""
Train the LCNN audio deepfake classifier on ASVspoof 2019 LA.

Dataset
-------
ASVspoof 2019 Logical Access (LA) partition:
  Register and download: https://datashare.ed.ac.uk/handle/10283/3336
  (~10 GB; free for research use after registration)

Expected layout after unpacking:
  <data_dir>/
    LA/
      ASVspoof2019_LA_train/flac/*.flac
      ASVspoof2019_LA_dev/flac/*.flac
      ASVspoof2019_LA_eval/flac/*.flac
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Protocol file format (space-separated):
  SPEAKER_ID  FILENAME  ENV  SYSTEM_ID  LABEL
  e.g.: LA_0079  LA_T_1000137  -  A09  spoof

Usage
-----
  python scripts/train_audio_lcnn.py --data /path/to/asvspoof2019
  python scripts/train_audio_lcnn.py --data /path/to/asvspoof2019 --epochs 20
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, metavar="DIR",
                    help="Root of ASVspoof 2019 dataset.")
    ap.add_argument("--output", type=Path, default=Path("models/lcnn_asvspoof.pth"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--clip-seconds", type=float, default=3.0,
                    help="Fixed clip length in seconds for training.")
    return ap.parse_args()


class ASVspoofDataset:
    def __init__(self, audio_dir: Path, protocol_file: Path,
                 sample_rate: int = 16000, clip_seconds: float = 3.0):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = int(clip_seconds * sample_rate)
        self.items = []  # (filepath, label)  label: 0=real, 1=fake

        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                fname, label_str = parts[1], parts[4]
                flac_path = audio_dir / f"{fname}.flac"
                if flac_path.exists():
                    self.items.append((flac_path, 0 if label_str == "bonafide" else 1))

        logger.info("Loaded %d samples from %s", len(self.items), protocol_file.name)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import librosa
        import numpy as np
        import torch

        path, label = self.items[idx]
        wav, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)

        # Pad or truncate to fixed length
        if len(wav) < self.clip_samples:
            wav = np.pad(wav, (0, self.clip_samples - len(wav)))
        else:
            # Random crop for training
            start = np.random.randint(0, len(wav) - self.clip_samples + 1)
            wav = wav[start: start + self.clip_samples]

        return torch.from_numpy(wav), label


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from detector.audio.model import LCNN, extract_log_mel

    device = torch.device(args.device) if args.device else (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    logger.info("Device: %s", device)

    la_dir = args.data / "LA"
    proto_dir = la_dir / "ASVspoof2019_LA_cm_protocols"

    train_ds = ASVspoofDataset(
        la_dir / "ASVspoof2019_LA_train" / "flac",
        proto_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        args.sample_rate, args.clip_seconds,
    )
    dev_ds = ASVspoofDataset(
        la_dir / "ASVspoof2019_LA_dev" / "flac",
        proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        args.sample_rate, args.clip_seconds,
    )

    def collate(batch):
        """Extract log-mel on-the-fly and stack into tensors."""
        import numpy as np
        feats, labels = [], []
        for wav, lbl in batch:
            feat = extract_log_mel(wav.numpy(), sample_rate=args.sample_rate)
            feats.append(torch.from_numpy(feat))
            labels.append(lbl)
        return torch.stack(feats).unsqueeze(1), torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=4)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2)

    model = LCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.float().to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for feats, labels in dev_loader:
                feats, labels = feats.float().to(device), labels.to(device)
                out = model(feats)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += len(labels)

        val_acc = v_correct / v_total
        logger.info(
            "Epoch %d/%d — train_acc=%.3f  val_acc=%.3f  loss=%.4f",
            epoch, args.epochs, correct / total, val_acc, total_loss / total,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            logger.info("  → Saved checkpoint to %s", args.output)

    logger.info("Training complete. Best dev acc: %.3f", best_val_acc)


if __name__ == "__main__":
    main()
