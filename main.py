#!/usr/bin/env python3
"""
Real-time deepfake detector for video/audio calls.

Usage examples
--------------
# Default: monitor both video and audio
python main.py

# Audio-only call (disables video capture)
python main.py --no-video

# Video only
python main.py --no-audio

# Tune the detection window and burst size
python main.py --sample-count 12 --sample-window 45

# Restrict screen capture to the remote party's face region
python main.py --region 0 0 800 600

# List available audio devices (to find loopback / monitor source)
python main.py --list-audio-devices

# Use the lightweight MesoNet backend instead of EfficientNet
python main.py --video-backend mesonet

# Run for exactly N seconds then exit (useful for testing)
python main.py --duration 60
"""

import argparse
import logging
import signal
import sys
import time

from config import Config, VideoConfig, AudioConfig
from detector import Orchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time deepfake detector for video and audio calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--no-video", action="store_true", help="Disable video detection.")
    p.add_argument("--no-audio", action="store_true", help="Disable audio detection.")

    p.add_argument(
        "--video-backend",
        choices=["community_forensics", "genconvit", "mesonet"],
        default="community_forensics",
        help=(
            "Video classifier backend (default: community_forensics). "
            "community_forensics: CVPR 2025 ViT, auto-downloads ~85 MB, best generalisation. "
            "genconvit: ConvNeXt+Swin hybrid, auto-downloads ~300 MB. "
            "mesonet: lightweight 28K-param model, run scripts/convert_mesonet_keras.py first."
        ),
    )
    p.add_argument(
        "--audio-backend",
        choices=["wav2vec2", "hubert", "lcnn"],
        default="wav2vec2",
        help=(
            "Audio classifier backend (default: wav2vec2). "
            "wav2vec2: Apache 2.0, auto-downloads ~1.2 GB, EER 4.01%% on ASVspoof2019. "
            "hubert: Apache 2.0, auto-downloads ~360 MB, claimed EER 1.43%%. "
            "lcnn: trainable fallback — train with scripts/train_audio_lcnn.py."
        ),
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=8,
        metavar="N",
        help="Frames to sample per burst (default: 8).",
    )
    p.add_argument(
        "--sample-window",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Spread early burst across this many seconds (default: 30).",
    )
    p.add_argument(
        "--ongoing-interval",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Seconds between subsequent video bursts (default: 60).",
    )
    p.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        default=None,
        help="Screen region to capture for video (pixels). Default: full primary monitor.",
    )
    p.add_argument(
        "--mic-device",
        type=int,
        default=None,
        metavar="IDX",
        help="sounddevice index for the microphone.",
    )
    p.add_argument(
        "--system-audio-device",
        type=int,
        default=None,
        metavar="IDX",
        help="sounddevice index for system/loopback audio.",
    )
    p.add_argument(
        "--video-threshold",
        type=float,
        default=0.55,
        metavar="P",
        help="Fake-probability threshold for video (default: 0.55).",
    )
    p.add_argument(
        "--audio-threshold",
        type=float,
        default=0.60,
        metavar="P",
        help="Fake-probability threshold for audio (default: 0.60).",
    )
    p.add_argument(
        "--notification-cooldown",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Min seconds between repeat notifications (default: 30).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SEC",
        help="Run for this many seconds then exit. Default: run until Ctrl-C.",
    )
    p.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return p.parse_args()


def list_audio_devices() -> None:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\nAvailable audio devices:\n")
        for i, d in enumerate(devices):
            kind = []
            if d["max_input_channels"] > 0:
                kind.append("IN")
            if d["max_output_channels"] > 0:
                kind.append("OUT")
            print(f"  [{i:2d}] [{'/'.join(kind)}] {d['name']}")
        print(
            "\nFor system audio loopback, look for a device with 'monitor', "
            "'BlackHole', 'Soundflower', or '(loopback)' in its name."
        )
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")


def build_config(args: argparse.Namespace) -> Config:
    video_cfg = VideoConfig(
        enabled=not args.no_video,
        sample_count=args.sample_count,
        sample_window=args.sample_window,
        ongoing_interval=args.ongoing_interval,
        capture_region=tuple(args.region) if args.region else None,
        model_backend=args.video_backend,
        deepfake_threshold=args.video_threshold,
    )
    audio_cfg = AudioConfig(
        enabled=not args.no_audio,
        audio_backend=args.audio_backend,
        mic_device=args.mic_device,
        system_audio_device=args.system_audio_device,
        deepfake_threshold=args.audio_threshold,
    )
    return Config(
        video=video_cfg,
        audio=audio_cfg,
        notification_cooldown=args.notification_cooldown,
        log_level=args.log_level,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_audio_devices:
        list_audio_devices()
        return

    config = build_config(args)
    orchestrator = Orchestrator(config)

    # Graceful shutdown on Ctrl-C or SIGTERM
    _stop = threading.Event() if False else None  # type: ignore[assignment]
    interrupted = [False]

    def _handle_signal(sig, frame):
        interrupted[0] = True

    import threading
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("Deepfake Detector — starting…")
    print("  Video:", "enabled" if config.video.enabled else "disabled")
    print("  Audio:", "enabled" if config.audio.enabled else "disabled")
    print("  Press Ctrl-C to stop.\n")

    orchestrator.start()

    try:
        start = time.monotonic()
        while not interrupted[0]:
            if args.duration and (time.monotonic() - start) >= args.duration:
                break
            time.sleep(0.2)
    finally:
        orchestrator.stop()
        print("\nDetector stopped.")


if __name__ == "__main__":
    main()
